//! Radiation: the force the water's memory exerts on a moving hull.
//!
//! # What this module is and is not
//!
//! It is the *memory* half of radiation. The other half — added mass at
//! infinite frequency — is not a force and is not here: it multiplies
//! acceleration, so it belongs in the mass matrix, and
//! [`crate::rigid_body::RigidBody::add_added_mass`] is where it goes. Splitting
//! them this way is Cummins' decomposition, and the split is what makes the
//! remainder integrable: what is left after taking the constant out is a
//! convolution over velocity history, and [`crate::cummins::FluidMemory`]
//! replaces that with a few states.
//!
//! # Any set of coupled modes, and why not one hard-wired pair
//!
//! Strip theory gives a symmetric matrix of coefficients over whichever modes
//! share a symmetry. Two such sets exist for a hull symmetric about its
//! centreline, and they do not couple to each other: the *vertical* pair, heave
//! and pitch, and the *lateral* triple, sway, roll and yaw. That is the whole
//! reason this module is written over a list of modes rather than around the
//! vertical pair it started as — the lateral triple is nine convolutions from
//! six fitted models, and hand-writing that after hand-writing the vertical four
//! would have been the same code twice.
//!
//! The memory is therefore the matrix convolution
//!
//! ```text
//! μ_i = Σ_j K_ij * ν_j
//! ```
//!
//! over the chosen modes, which needs `n²` state vectors from `n(n+1)/2` fitted
//! models: `K_ij` and `K_ji` are the same transfer function, but they are driven
//! by different velocities and therefore cannot share their states.
//!
//! # Why the coupling terms are not optional
//!
//! On the YD-41 the vertical coupling is not a correction — `A₃₅ / A₃₃` is 4.9 m,
//! the longitudinal centroid of the sectional added mass, about 40 % of the
//! waterline aft of the origin. Including heave and dropping the coupling would
//! be a worse model than including neither, because it would say that heaving a
//! hull whose added mass is distributed asymmetrically about the origin produces
//! no pitching moment. The lateral set is worse still: its roll-sway coupling
//! carries the whole lever arm between the waterline and the roll axis.

use crate::cummins::{FluidMemory, InfiniteAddedMass, MemoryError, MemoryOptions, Spectrum};
use crate::sim::{ForceModule, StepCtx};
use crate::telemetry::Telemetry;
use crate::wrench::Wrench;
use nalgebra::{Matrix6, Vector3};

/// Body-frame index of the sway mode.
pub const SWAY: usize = 1;
/// Body-frame index of the heave mode.
pub const HEAVE: usize = 2;
/// Body-frame index of the roll mode.
pub const ROLL: usize = 3;
/// Body-frame index of the pitch mode.
pub const PITCH: usize = 4;
/// Body-frame index of the yaw mode.
pub const YAW: usize = 5;

/// Number of entries in the upper triangle of an `n × n` symmetric matrix.
const fn triangle(n: usize) -> usize {
    n * (n + 1) / 2
}

/// Index into the row-major upper triangle of an `n × n` symmetric matrix.
const fn triangle_index(n: usize, row: usize, column: usize) -> usize {
    let (i, j) = if row <= column {
        (row, column)
    } else {
        (column, row)
    };
    // Rows above `i` contribute `n - k` entries each, for `k` in `0..i`.
    i * n - triangle(i) + i + (j - i)
}

/// A symmetric set of radiation spectra over a chosen set of body-frame modes.
///
/// `modes[i]` is the generalised coordinate that row and column `i` refer to:
/// 0, 1, 2 are surge, sway and heave, and 3, 4, 5 are roll, pitch and yaw. The
/// spectra are the upper triangle, row-major, so for modes `[a, b, c]` the order
/// is `aa, ab, ac, bb, bc, cc`.
///
/// Separate [`Spectrum`] values rather than one matrix-valued object, because
/// each is fitted independently and each carries its own [`InfiniteAddedMass`] —
/// and the caller should see every disagreement rather than an average of them.
#[derive(Debug, Clone)]
pub struct RadiationSpectra {
    modes: Vec<usize>,
    entries: Vec<Spectrum>,
}

impl RadiationSpectra {
    /// Builds a set from its modes and the upper triangle of its spectra.
    ///
    /// # Errors
    ///
    /// [`MemoryError::Singular`] if the number of spectra is not the size of the
    /// upper triangle, or if a mode is out of range or repeated. These are
    /// caller mistakes rather than physics, but they are the kind that otherwise
    /// surface as a coefficient silently landing in the wrong matrix entry.
    pub fn new(modes: Vec<usize>, entries: Vec<Spectrum>) -> Result<Self, MemoryError> {
        let count = modes.len();
        let sorted_and_distinct = modes.windows(2).all(|pair| pair[0] < pair[1]);
        if count == 0
            || entries.len() != triangle(count)
            || !sorted_and_distinct
            || modes.iter().any(|&mode| mode >= 6)
        {
            return Err(MemoryError::Singular);
        }
        Ok(Self { modes, entries })
    }

    /// The vertical pair: heave, the heave-pitch coupling, and pitch.
    ///
    /// # Errors
    ///
    /// As [`RadiationSpectra::new`], which cannot fail for this shape.
    pub fn vertical(
        heave: Spectrum,
        coupling: Spectrum,
        pitch: Spectrum,
    ) -> Result<Self, MemoryError> {
        Self::new(vec![HEAVE, PITCH], vec![heave, coupling, pitch])
    }

    /// The lateral triple, in the order [`crate::strip::lateral_spectra`] returns:
    /// `22, 24, 26, 44, 46, 66`.
    ///
    /// # Errors
    ///
    /// As [`RadiationSpectra::new`], which cannot fail for this shape.
    pub fn lateral(entries: [Spectrum; 6]) -> Result<Self, MemoryError> {
        Self::new(vec![SWAY, ROLL, YAW], entries.to_vec())
    }

    /// The modes this set covers.
    #[must_use]
    pub fn modes(&self) -> &[usize] {
        &self.modes
    }

    /// The spectrum coupling two modes, or `None` if either is not in this set.
    ///
    /// Addressed by body-frame mode rather than by row, so that a caller asks for
    /// `entry(HEAVE, PITCH)` and does not have to know this set's layout.
    /// Symmetric: the order of the arguments does not matter.
    #[must_use]
    pub fn entry(&self, one: usize, other: usize) -> Option<&Spectrum> {
        let row = self.modes.iter().position(|&mode| mode == one)?;
        let column = self.modes.iter().position(|&mode| mode == other)?;
        self.entries
            .get(triangle_index(self.modes.len(), row, column))
    }

    /// The infinite-frequency added mass of every entry.
    #[must_use]
    pub fn infinite(&self, options: crate::cummins::TransformOptions) -> RadiationInfinite {
        RadiationInfinite {
            modes: self.modes.clone(),
            entries: self
                .entries
                .iter()
                .map(|spectrum| spectrum.infinite_added_mass(options))
                .collect(),
        }
    }
}

/// The infinite-frequency added mass of a [`RadiationSpectra`], same layout.
#[derive(Debug, Clone, PartialEq)]
pub struct RadiationInfinite {
    modes: Vec<usize>,
    entries: Vec<InfiniteAddedMass>,
}

impl RadiationInfinite {
    /// The added mass as a generalized six-by-six matrix.
    ///
    /// Fills only the rows and columns of this set's modes and leaves the rest
    /// zero — ready for [`crate::rigid_body::RigidBody::add_added_mass`], and
    /// additive with another set's contribution.
    ///
    /// Symmetric by construction: each off-diagonal pair is written from one
    /// number, so they cannot drift apart.
    #[must_use]
    pub fn matrix(&self) -> Matrix6<f64> {
        let count = self.modes.len();
        let mut matrix = Matrix6::zeros();
        for (row, &i) in self.modes.iter().enumerate() {
            for (column, &j) in self.modes.iter().enumerate() {
                matrix[(i, j)] = self.entries[triangle_index(count, row, column)].value;
            }
        }
        matrix
    }

    /// The added mass coupling two modes, or `None` if either is not in this set.
    ///
    /// Addressed by body-frame mode, as [`RadiationSpectra::entry`] is, and
    /// symmetric in its arguments for the same reason.
    #[must_use]
    pub fn entry(&self, one: usize, other: usize) -> Option<InfiniteAddedMass> {
        let row = self.modes.iter().position(|&mode| mode == one)?;
        let column = self.modes.iter().position(|&mode| mode == other)?;
        self.entries
            .get(triangle_index(self.modes.len(), row, column))
            .copied()
    }

    /// Worst disagreement between the two estimators, over every entry.
    ///
    /// The one number to look at before trusting any of this: it is the agreement
    /// between two independent routes to a constant that does not depend on
    /// frequency. See [`InfiniteAddedMass::disagreement`].
    #[must_use]
    pub fn worst_disagreement(&self) -> f64 {
        self.entries
            .iter()
            .map(|entry| entry.disagreement)
            .fold(0.0, f64::max)
    }
}

/// The memory functions of one set of coupled modes, and their states.
#[derive(Debug, Clone)]
pub struct Radiation {
    modes: Vec<usize>,
    /// The full `n × n` matrix of memories, row-major: entry `(i, j)` is driven by
    /// mode `j`'s rate and felt in mode `i`'s component of the wrench.
    ///
    /// Full rather than triangular, because the states are not symmetric even
    /// though the transfer functions are. `K_ij` and `K_ji` are fitted once and
    /// cloned into both slots, which shares the model and separates the history.
    memory: Vec<FluidMemory>,
    /// What the sweep that fed the fit said about itself. See
    /// [`Radiation::with_provenance`].
    provenance: Option<Provenance>,
    /// What the last step produced, for [`ForceModule::telemetry`].
    last: Option<Wrench>,
}

/// What the load-time chain found out on the way to a fit.
///
/// Computed at every load and, until this existed, read by nothing but
/// `vela-cli radiation`: a sailing assembly that had clamped a Lewis form at the
/// bow or broken the energy identity at some frequency said nothing about it,
/// which is exactly the error `lewis` says nobody should have to rediscover.
/// Published in the module's telemetry instead, so that whoever reads the
/// simulation can see the fit's inputs beside its output.
///
/// The residuals are the worst **within the band the fit consumed** — up to
/// [`MemoryOptions::fit_ceiling`] times the damping peak — and not over the
/// whole grid, because the grid's tail is a statement about the multipole
/// series' truncation at frequencies whose damping has died, not about the
/// coefficients the boat moves on. See `strip::SweepQuality`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Provenance {
    /// Stations whose Lewis form had to be clamped to the mappable region.
    pub clamped_stations: usize,
    /// `strip::HeavePitch::slenderness`; zero for the lateral set.
    pub slenderness: f64,
    /// The frequency the residuals are judged up to, rad/s.
    pub fitted_ceiling: f64,
    /// Worst departure from the energy identity within the fitted band.
    pub worst_energy_residual: f64,
    /// Worst departure from reciprocity within the fitted band; zero for the
    /// vertical set, which has no cross-mode identity.
    pub worst_reciprocity_residual: f64,
}

impl Provenance {
    /// Reduces a sweep's per-frequency record to the band a fit consumed.
    ///
    /// `ceiling` is the highest frequency the fit read; the assembly computes
    /// it the way [`FluidMemory::fit`] does, from the same options and the same
    /// spectra, so the number published is about the coefficients that were
    /// actually used.
    #[must_use]
    pub fn within(
        clamped_stations: usize,
        sweep: &crate::strip::SweepQuality,
        ceiling: f64,
    ) -> Self {
        Self {
            clamped_stations,
            slenderness: sweep.slenderness,
            fitted_ceiling: ceiling,
            worst_energy_residual: sweep.worst_energy_residual_below(ceiling),
            worst_reciprocity_residual: sweep.worst_reciprocity_residual_below(ceiling),
        }
    }
}

impl Radiation {
    /// Fits the memory of every coefficient in the set.
    ///
    /// # Errors
    ///
    /// [`MemoryError::Singular`] if the spectra and the infinite added mass do not
    /// describe the same modes, and otherwise whatever [`FluidMemory::fit`]
    /// refuses: an unstable fit, or a spectrum too short to fit at the requested
    /// order.
    pub fn fit(
        spectra: &RadiationSpectra,
        infinite: &RadiationInfinite,
        options: MemoryOptions,
    ) -> Result<Self, MemoryError> {
        if spectra.modes != infinite.modes {
            return Err(MemoryError::Singular);
        }
        let count = spectra.modes.len();
        // Fit the triangle once, then place each model in both of its slots.
        let fitted: Vec<FluidMemory> = spectra
            .entries
            .iter()
            .zip(infinite.entries.iter())
            .map(|(spectrum, constant)| FluidMemory::fit(spectrum, *constant, options))
            .collect::<Result<_, _>>()?;
        let mut memory = Vec::with_capacity(count * count);
        for row in 0..count {
            for column in 0..count {
                memory.push(fitted[triangle_index(count, row, column)].clone());
            }
        }
        Ok(Self {
            modes: spectra.modes.clone(),
            memory,
            provenance: None,
            last: None,
        })
    }

    /// Attaches what the sweep found out about itself, for the telemetry.
    #[must_use]
    pub fn with_provenance(mut self, provenance: Provenance) -> Self {
        self.provenance = Some(provenance);
        self
    }

    /// Worst relative fit error over every model in the set.
    #[must_use]
    pub fn worst_error(&self) -> f64 {
        self.memory
            .iter()
            .map(FluidMemory::worst_error)
            .fold(0.0, f64::max)
    }

    /// Real part of the least-damped pole over every model, 1/s.
    #[must_use]
    pub fn slowest_pole(&self) -> f64 {
        self.memory
            .iter()
            .map(FluidMemory::slowest_pole)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Worst passivity violation over the **diagonal** models of the set.
    ///
    /// Diagonal only, for the reason [`FluidMemory::passivity_violation`]
    /// gives: a coupling coefficient is allowed a negative real part, a mode
    /// taken by itself is not. Zero for a set whose water only ever damps.
    #[must_use]
    pub fn passivity_violation(&self) -> f64 {
        let count = self.modes.len();
        (0..count)
            .map(|index| self.memory[index * count + index].passivity_violation())
            .fold(0.0, f64::max)
    }

    /// Forgets the history, as if the boat had always been still.
    pub fn reset(&mut self) {
        for memory in &mut self.memory {
            memory.reset();
        }
        self.last = None;
    }
}

/// The rate of one generalised coordinate, from a body state.
fn rate(state: &crate::state::BodyState, mode: usize) -> f64 {
    match mode {
        0..=2 => state.velocity[mode],
        _ => state.angular_velocity[mode - 3],
    }
}

/// Adds a generalised component into the force or moment half of a wrench.
fn accumulate(wrench: &mut Wrench, mode: usize, value: f64) {
    match mode {
        0..=2 => wrench.force[mode] += value,
        _ => wrench.moment[mode - 3] += value,
    }
}

impl ForceModule for Radiation {
    fn name(&self) -> &'static str {
        "radiation"
    }

    fn step(&mut self, ctx: &StepCtx<'_>) -> Wrench {
        let count = self.modes.len();
        let rates: Vec<f64> = self
            .modes
            .iter()
            .map(|&mode| rate(ctx.state, mode))
            .collect();
        let mut wrench = Wrench {
            force: Vector3::zeros(),
            moment: Vector3::zeros(),
        };
        for (row, &felt) in self.modes.iter().enumerate() {
            let mut total = 0.0;
            for (column, &driving) in rates.iter().enumerate() {
                total += self.memory[row * count + column].advance(driving, ctx.dt);
            }
            // The convolution sits on the left of the equation of motion, so it
            // opposes: it is subtracted from the applied load, which as a wrench
            // means the negative.
            accumulate(&mut wrench, felt, -total);
        }
        self.last = Some(wrench);
        wrench
    }

    fn telemetry(&self, out: &mut Telemetry) {
        let last = self.last.unwrap_or(Wrench {
            force: Vector3::zeros(),
            moment: Vector3::zeros(),
        });
        for &mode in &self.modes {
            let (key, value) = match mode {
                SWAY => ("radiation.sway.memory_force", last.force.y),
                HEAVE => ("radiation.heave.memory_force", last.force.z),
                ROLL => ("radiation.roll.memory_moment", last.moment.x),
                PITCH => ("radiation.pitch.memory_moment", last.moment.y),
                YAW => ("radiation.yaw.memory_moment", last.moment.z),
                _ => ("radiation.surge.memory_force", last.force.x),
            };
            // Published as the load the module applied, which is the negative of
            // the convolution — the sign a dynamometer would read.
            out.set(key, value);
        }
        // Kept as they were: two sets mount side by side and the last one to
        // publish wins these two, which is why the per-set keys below exist.
        out.set("radiation.worst_fit_error", self.worst_error());
        out.set("radiation.slowest_pole", self.slowest_pole());

        // The fit and what fed it, keyed by set so that the vertical pair and
        // the lateral triple cannot overwrite each other. A reader who finds a
        // clamped station or an energy residual in the per cents here knows
        // the fit was built on coefficients the theory did not fully own.
        let keys = if self.modes.contains(&HEAVE) {
            &VERTICAL_KEYS
        } else {
            &LATERAL_KEYS
        };
        out.set(keys.worst_fit_error, self.worst_error());
        out.set(keys.slowest_pole, self.slowest_pole());
        out.set(keys.passivity_violation, self.passivity_violation());
        if let Some(provenance) = self.provenance {
            out.set(keys.clamped_stations, provenance.clamped_stations as f64);
            out.set(keys.fitted_ceiling, provenance.fitted_ceiling);
            out.set(keys.worst_energy_residual, provenance.worst_energy_residual);
            out.set(
                keys.worst_reciprocity_residual,
                provenance.worst_reciprocity_residual,
            );
            if let Some(key) = keys.slenderness {
                out.set(key, provenance.slenderness);
            }
        }
    }
}

/// The per-set telemetry keys. Stable names: rename them never.
struct SetKeys {
    worst_fit_error: &'static str,
    slowest_pole: &'static str,
    passivity_violation: &'static str,
    clamped_stations: &'static str,
    fitted_ceiling: &'static str,
    worst_energy_residual: &'static str,
    worst_reciprocity_residual: &'static str,
    /// Only the vertical sweep computes it.
    slenderness: Option<&'static str>,
}

const VERTICAL_KEYS: SetKeys = SetKeys {
    worst_fit_error: "radiation.vertical.worst_fit_error",
    slowest_pole: "radiation.vertical.slowest_pole",
    passivity_violation: "radiation.vertical.passivity_violation",
    clamped_stations: "radiation.vertical.clamped_stations",
    fitted_ceiling: "radiation.vertical.fitted_ceiling",
    worst_energy_residual: "radiation.vertical.worst_energy_residual",
    worst_reciprocity_residual: "radiation.vertical.worst_reciprocity_residual",
    slenderness: Some("radiation.vertical.slenderness"),
};

const LATERAL_KEYS: SetKeys = SetKeys {
    worst_fit_error: "radiation.lateral.worst_fit_error",
    slowest_pole: "radiation.lateral.slowest_pole",
    passivity_violation: "radiation.lateral.passivity_violation",
    clamped_stations: "radiation.lateral.clamped_stations",
    fitted_ceiling: "radiation.lateral.fitted_ceiling",
    worst_energy_residual: "radiation.lateral.worst_energy_residual",
    worst_reciprocity_residual: "radiation.lateral.worst_reciprocity_residual",
    slenderness: None,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controls::Controls;
    use crate::cummins::TransformOptions;
    use crate::env::{StillWater, UniformWind};
    use crate::lewis::{LewisForm, SectionGeometry};
    use crate::state::BodyState;
    use crate::strip::{self, Strip};
    use crate::tasai::{SectionSolver, TasaiOptions};
    use approx::assert_relative_eq;
    use nalgebra::Complex;

    const WATER: f64 = 1025.0;
    const GRAVITY: f64 = 9.81;

    /// A yacht-like hull: seventeen stations tapering to nothing at the ends,
    /// asymmetric about the origin so that the coupling term is real.
    fn hull() -> Vec<Strip> {
        (0..17)
            .map(|i| {
                let along = i as f64 / 16.0;
                let taper = 1.0 - 0.85 * (2.0 * along - 0.9).abs().powi(2);
                Strip {
                    x: 12.0 * along,
                    form: (taper > 0.05).then(|| {
                        LewisForm::fit(&SectionGeometry {
                            beam: 3.2 * taper,
                            draft: 0.45 * taper,
                            area: 0.78 * 3.2 * 0.45 * taper * taper,
                        })
                    }),
                }
            })
            .collect()
    }

    fn built() -> (Radiation, RadiationInfinite, RadiationSpectra) {
        let strips = hull();
        let solver = SectionSolver::new(TasaiOptions::default());
        let grid: Vec<f64> = (1..=120).map(|i| 30.0 * f64::from(i) / 120.0).collect();
        let ((heave, coupling, pitch), _) =
            strip::vertical_spectra(&strips, &grid, WATER, GRAVITY, &solver)
                .expect("a hull over a grid has spectra");
        let spectra = RadiationSpectra::vertical(heave, coupling, pitch)
            .expect("three spectra make a vertical pair");
        let infinite = spectra.infinite(TransformOptions::default());
        let model = Radiation::fit(
            &spectra,
            &infinite,
            MemoryOptions {
                order: 5,
                ..MemoryOptions::default()
            },
        )
        .expect("a yacht hull admits a memory model");
        (model, infinite, spectra)
    }

    /// The added mass of the vertical modes is a possible mass matrix.
    ///
    /// A boat's own mass matrix plus this has to stay positive definite, or the
    /// factorization refuses and — worse, if it did not — the boat would
    /// accelerate against the force applied to it. `A₃₃ A₅₅ ≥ A₃₅²` is the
    /// condition, and it is Cauchy-Schwarz on the sectional distribution, so it
    /// holds for any hull. Checking it here checks that the three fits were made
    /// of the same hull.
    #[test]
    fn the_vertical_added_mass_makes_a_possible_mass_matrix() {
        let (_, infinite, _) = built();
        let matrix = infinite.matrix();

        assert_relative_eq!(matrix[(2, 4)], matrix[(4, 2)], max_relative = 1e-15);
        assert!(matrix[(2, 2)] > 0.0, "heave added mass must be positive");
        assert!(matrix[(4, 4)] > 0.0, "pitch added mass must be positive");
        let determinant = matrix[(2, 2)] * matrix[(4, 4)] - matrix[(2, 4)] * matrix[(4, 2)];
        assert!(
            determinant > 0.0,
            "the vertical block is indefinite: {determinant}"
        );
        // And the coupling is not negligible, or this test proves nothing.
        assert!(
            matrix[(2, 4)].abs() > 0.1 * matrix[(2, 2)],
            "this hull should couple heave to pitch"
        );
    }

    /// The two independent routes to each `A_∞` agree.
    ///
    /// The diagnostic that stands behind everything downstream. Checked on all
    /// three coefficients, because the coupling is the one whose sign and scale
    /// nothing else in this module would catch.
    #[test]
    fn all_three_infinite_added_masses_are_corroborated() {
        let (_, infinite, _) = built();
        assert!(
            infinite.worst_disagreement() < 0.02,
            "worst disagreement was {:.3} %: {infinite:?}",
            100.0 * infinite.worst_disagreement()
        );
    }

    /// The assembled module reproduces the frequency domain it was built from.
    ///
    /// The same forced-oscillation argument as `cummins`, but through the whole
    /// module: heave alone recovers `K₃₃` in the force and `K₅₃` in the moment,
    /// and pitch alone recovers `K₃₅` and `K₅₅`. Driving one mode at a time is
    /// what separates the four convolutions, and it is the only test that would
    /// catch the coupling being wired into the wrong component or with the wrong
    /// sign.
    #[test]
    fn the_module_reproduces_every_vertical_coefficient() {
        let (mut model, infinite, spectra) = built();
        let env = StillWater::new(UniformWind::uniform(0.0, 0.0));
        let controls = Controls::close_hauled(crate::aero::SailSet::upwind());

        let heave = spectra.entry(HEAVE, HEAVE).expect("heave is in the pair");
        let coupling = spectra.entry(HEAVE, PITCH).expect("the coupling is too");
        let pitch = spectra.entry(PITCH, PITCH).expect("and pitch");
        let at_infinity = |one, other| {
            infinite
                .entry(one, other)
                .expect("the same modes the spectra have")
                .value
        };

        for &target in &[1.0_f64, 2.5] {
            for heaving in [true, false] {
                let index = heave
                    .frequencies()
                    .iter()
                    .enumerate()
                    .min_by(|a, b| (a.1 - target).abs().total_cmp(&(b.1 - target).abs()))
                    .map_or(0, |(i, _)| i);
                let omega = heave.frequencies()[index];

                let wanted = |spectrum: &crate::cummins::Spectrum, constant: f64| {
                    Complex::new(
                        spectrum.damping()[index],
                        omega * (spectrum.added_mass()[index] - constant),
                    )
                };
                let (force_wanted, moment_wanted) = if heaving {
                    (
                        wanted(heave, at_infinity(HEAVE, HEAVE)),
                        wanted(coupling, at_infinity(HEAVE, PITCH)),
                    )
                } else {
                    (
                        wanted(coupling, at_infinity(HEAVE, PITCH)),
                        wanted(pitch, at_infinity(PITCH, PITCH)),
                    )
                };

                model.reset();
                let dt = 0.002;
                let period = std::f64::consts::TAU / omega;
                let settle = (10.0 * period / dt) as usize;
                let cycles = (6.0 * period / dt) as usize;
                let mut force = (0.0, 0.0);
                let mut moment = (0.0, 0.0);
                for step in 0..(settle + cycles) {
                    let time = step as f64 * dt;
                    let rate = omega * (omega * time).cos();
                    let state = BodyState {
                        velocity: nalgebra::Vector3::new(
                            0.0,
                            0.0,
                            if heaving { rate } else { 0.0 },
                        ),
                        angular_velocity: nalgebra::Vector3::new(
                            0.0,
                            if heaving { 0.0 } else { rate },
                            0.0,
                        ),
                        ..BodyState::default()
                    };
                    let ctx = StepCtx {
                        state: &state,
                        controls: &controls,
                        env: &env,
                        time,
                        dt,
                    };
                    let wrench = model.step(&ctx);
                    if step >= settle {
                        // The wrench opposes, so undo that sign to compare with K.
                        force.0 += -wrench.force.z * (omega * time).cos() * dt;
                        force.1 += -wrench.force.z * (omega * time).sin() * dt;
                        moment.0 += -wrench.moment.y * (omega * time).cos() * dt;
                        moment.1 += -wrench.moment.y * (omega * time).sin() * dt;
                    }
                }
                let span = cycles as f64 * dt;
                let recover = |(in_phase, quadrature): (f64, f64)| {
                    Complex::new(
                        2.0 * in_phase / span / omega,
                        -2.0 * quadrature / span / omega,
                    )
                };
                let force_got = recover(force);
                let moment_got = recover(moment);

                assert_relative_eq!(force_got.re, force_wanted.re, max_relative = 0.03);
                assert_relative_eq!(force_got.im, force_wanted.im, max_relative = 0.03);
                assert_relative_eq!(moment_got.re, moment_wanted.re, max_relative = 0.03);
                assert_relative_eq!(moment_got.im, moment_wanted.im, max_relative = 0.03);
            }
        }
    }

    /// The wrench opposes the motion.
    ///
    /// Cheap, and the one thing that would silently invert the whole model: get
    /// this backwards and the water drives the boat instead of damping it.
    #[test]
    fn the_memory_wrench_opposes_a_starting_heave() {
        let (mut model, _, _) = built();
        let env = StillWater::new(UniformWind::uniform(0.0, 0.0));
        let controls = Controls::close_hauled(crate::aero::SailSet::upwind());
        let state = BodyState {
            velocity: nalgebra::Vector3::new(0.0, 0.0, 1.0),
            ..BodyState::default()
        };
        let ctx = StepCtx {
            state: &state,
            controls: &controls,
            env: &env,
            time: 0.0,
            dt: 0.01,
        };
        // A few steps, because the memory starts empty and builds.
        let mut wrench = model.step(&ctx);
        for _ in 0..20 {
            wrench = model.step(&ctx);
        }
        assert!(
            wrench.force.z < 0.0,
            "heaving down must be resisted upward, got {}",
            wrench.force.z
        );
    }

    fn built_lateral() -> (Radiation, RadiationInfinite, RadiationSpectra) {
        let strips = hull();
        let solver = SectionSolver::new(TasaiOptions::default());
        let grid: Vec<f64> = (1..=120).map(|i| 30.0 * f64::from(i) / 120.0).collect();
        // The waterline the sections were cut at, which is what the roll and
        // coupling coefficients are levered about.
        let waterline = 0.45;
        let (entries, _) =
            strip::lateral_spectra(&strips, waterline, &grid, WATER, GRAVITY, &solver)
                .expect("a hull over a grid has lateral spectra");
        let spectra =
            RadiationSpectra::lateral(entries).expect("six spectra make a lateral triple");
        let infinite = spectra.infinite(TransformOptions::default());
        let model = Radiation::fit(&spectra, &infinite, MemoryOptions::default())
            .expect("a yacht hull admits a lateral memory model");
        (model, infinite, spectra)
    }

    /// The assembled lateral module reproduces all nine of its entries.
    ///
    /// The three-by-three twin of `the_module_reproduces_every_vertical_coefficient`,
    /// and the sharp check on the lateral chain. Driving one mode at a time and
    /// reading all three components of the wrench separates the nine convolutions,
    /// which is the only test that catches a coefficient wired into the wrong
    /// component, transposed, or with the wrong sign — and with three modes there
    /// are a great many more ways to get that wrong than with two.
    ///
    /// This is where the accuracy claim for the lateral chain lives, rather than in
    /// an integration test. A free-decay experiment cannot be this sharp: the
    /// memory's poles are no faster than the roll period, so the rolling mode is a
    /// genuinely coupled rigid-body-plus-memory mode rather than a mass-spring-damper
    /// with coefficients substituted at one frequency. Here the frequency is imposed,
    /// and the answer is exact.
    #[test]
    fn the_module_reproduces_every_lateral_coefficient() {
        let (mut model, infinite, spectra) = built_lateral();
        let env = StillWater::new(UniformWind::uniform(0.0, 0.0));
        let controls = Controls::close_hauled(crate::aero::SailSet::upwind());
        let modes = [SWAY, ROLL, YAW];

        for &target in &[1.0_f64, 2.5] {
            // Every entry shares the grid, so one index serves all nine.
            let reference = spectra.entry(SWAY, SWAY).expect("sway is in the triple");
            let index = reference
                .frequencies()
                .iter()
                .enumerate()
                .min_by(|a, b| (a.1 - target).abs().total_cmp(&(b.1 - target).abs()))
                .map_or(0, |(i, _)| i);
            let omega = reference.frequencies()[index];

            for &driven in &modes {
                model.reset();
                let dt = 0.002;
                let period = std::f64::consts::TAU / omega;
                let settle = (12.0 * period / dt) as usize;
                let cycles = (6.0 * period / dt) as usize;
                // Projections of each felt component onto cos(ωt) and sin(ωt).
                let mut projected = [(0.0_f64, 0.0_f64); 3];

                for step in 0..(settle + cycles) {
                    let time = step as f64 * dt;
                    let rate = omega * (omega * time).cos();
                    let mut state = BodyState::default();
                    match driven {
                        SWAY => state.velocity.y = rate,
                        ROLL => state.angular_velocity.x = rate,
                        _ => state.angular_velocity.z = rate,
                    }
                    let ctx = StepCtx {
                        state: &state,
                        controls: &controls,
                        env: &env,
                        time,
                        dt,
                    };
                    let wrench = model.step(&ctx);
                    if step >= settle {
                        // The wrench opposes, so undo that sign before comparing.
                        let felt = [-wrench.force.y, -wrench.moment.x, -wrench.moment.z];
                        let (cosine, sine) = (omega * time).sin_cos();
                        for (slot, value) in projected.iter_mut().zip(felt) {
                            slot.0 += value * sine * dt;
                            slot.1 += value * cosine * dt;
                        }
                    }
                }

                let scale = 2.0 / (omega * cycles as f64 * dt);
                let wanted: Vec<Complex<f64>> = modes
                    .iter()
                    .map(|&felt| {
                        let spectrum = spectra.entry(driven, felt).expect("in the triple");
                        let constant = infinite.entry(driven, felt).expect("in the triple").value;
                        Complex::new(
                            spectrum.damping()[index],
                            omega * (spectrum.added_mass()[index] - constant),
                        )
                    })
                    .collect();

                // Judged against the largest entry this driven mode produces, not
                // against each entry's own size. One row spans two orders of
                // magnitude — driving sway at 1 rad/s asks for a yaw moment of a
                // hundred alongside a sway force of ten thousand — and demanding
                // relative precision on the smallest would be demanding precision
                // on a number that does nothing. What matters is that no entry is
                // wrong by enough to matter beside the ones that do, which is what
                // catches a transposed, misplaced or sign-flipped coefficient. Three
                // per cent, matching the tolerance the vertical twin above already
                // keeps; what is left at that level is the rational fit at the low
                // end of the grid, where it is least constrained.
                let row_scale = wanted
                    .iter()
                    .map(|value| value.norm())
                    .fold(0.0_f64, f64::max);
                for (row, want) in wanted.iter().enumerate() {
                    let got = Complex::new(projected[row].0 * scale, -projected[row].1 * scale);
                    let error = (got - want).norm();
                    assert!(
                        error < 0.03 * row_scale,
                        "driving mode {driven} at {omega:.2} rad/s, the mode {} \
                         response was {got:.1} where {want:.1} was wanted — off by \
                         {error:.1}, against a row scale of {row_scale:.1}",
                        modes[row]
                    );
                }
            }
        }
    }

    /// The lateral added mass makes a mass matrix a Cholesky can survive.
    ///
    /// The three-by-three block has to be positive definite on its own, because it
    /// is added to the rigid body's and the factorisation has to succeed. Checked
    /// on the leading minors, and separately that the coupling is not negligible —
    /// without that last assertion a diagonal-only bug would pass.
    #[test]
    fn the_lateral_added_mass_makes_a_possible_mass_matrix() {
        let (_, infinite, _) = built_lateral();
        let matrix = infinite.matrix();
        let block =
            nalgebra::Matrix3::from_fn(|i, j| matrix[([SWAY, ROLL, YAW][i], [SWAY, ROLL, YAW][j])]);

        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            assert_relative_eq!(block[(i, j)], block[(j, i)], max_relative = 1e-15);
        }
        assert!(block[(0, 0)] > 0.0, "sway added mass must be positive");
        assert!(block[(1, 1)] > 0.0, "roll added inertia must be positive");
        assert!(block[(2, 2)] > 0.0, "yaw added inertia must be positive");
        let two_by_two = block[(0, 0)] * block[(1, 1)] - block[(0, 1)] * block[(1, 0)];
        assert!(
            two_by_two > 0.0,
            "the sway-roll block is indefinite: {two_by_two}"
        );
        assert!(
            block.determinant() > 0.0,
            "the lateral block is indefinite: {}",
            block.determinant()
        );
        assert!(
            block[(0, 1)].abs() > 0.1 * block[(0, 0)],
            "this hull should couple sway to roll"
        );
    }
}
