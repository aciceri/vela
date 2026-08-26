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
//! # Heave and pitch, and why both
//!
//! Strip theory gives three independent coefficients for the vertical modes:
//! `33` for heave, `55` for pitch and `35 = 53` coupling. On the YD-41 the
//! coupling is not a correction — `A_35 / A_33` is 4.9 m, which is the
//! longitudinal centroid of the sectional added mass and about 40 % of the
//! waterline aft of the origin. Including heave and dropping the coupling would
//! be a worse model than including neither, because it would say that heaving a
//! hull whose added mass is distributed asymmetrically about the origin produces
//! no pitching moment.
//!
//! So the memory here is the two-by-two convolution
//!
//! ```text
//! μ₃ = K₃₃ * ν₃ + K₃₅ * ν₅
//! μ₅ = K₅₃ * ν₃ + K₅₅ * ν₅
//! ```
//!
//! which needs *four* state vectors from three fitted models: `K₃₅` and `K₅₃`
//! are the same transfer function, but they are driven by different velocities
//! and therefore cannot share their states.
//!
//! Sway, roll and yaw are absent because the sections for them are not written.
//! They are absent rather than approximated: a zero is a visible gap, and a
//! plausible guess is not.

use crate::cummins::{FluidMemory, InfiniteAddedMass, MemoryError, MemoryOptions, Spectrum};
use crate::sim::{ForceModule, StepCtx};
use crate::telemetry::Telemetry;
use crate::wrench::Wrench;
use nalgebra::{Matrix6, Vector3};

/// The three memory functions of the vertical modes, and their states.
#[derive(Debug, Clone)]
pub struct Radiation {
    /// `K₃₃`, driven by heave velocity, felt as heave force.
    heave: FluidMemory,
    /// `K₃₅`, driven by pitch rate, felt as heave force.
    heave_from_pitch: FluidMemory,
    /// `K₅₃`, driven by heave velocity, felt as pitch moment. Same transfer
    /// function as `heave_from_pitch`, different states.
    pitch_from_heave: FluidMemory,
    /// `K₅₅`, driven by pitch rate, felt as pitch moment.
    pitch: FluidMemory,
    /// What the last step produced, for [`ForceModule::telemetry`].
    last: Option<(f64, f64)>,
}

/// The spectra strip theory produces for the vertical modes.
///
/// Three separate [`Spectrum`] values rather than one matrix-valued object,
/// because each is fitted independently and each carries its own
/// [`InfiniteAddedMass`] — and the caller should see all three disagreements
/// rather than an average of them.
#[derive(Debug, Clone)]
pub struct VerticalSpectra {
    /// Heave, `A₃₃` and `B₃₃`.
    pub heave: Spectrum,
    /// Coupling, `A₃₅` and `B₃₅`.
    pub coupling: Spectrum,
    /// Pitch, `A₅₅` and `B₅₅`.
    pub pitch: Spectrum,
}

impl Radiation {
    /// Fits the memory of all three vertical coefficients.
    ///
    /// # Errors
    ///
    /// Passes through whatever [`FluidMemory::fit`] refuses, which is an
    /// unstable fit or a spectrum too short to fit at the requested order.
    pub fn fit(
        spectra: &VerticalSpectra,
        infinite: &VerticalInfinite,
        options: MemoryOptions,
    ) -> Result<Self, MemoryError> {
        let coupling = FluidMemory::fit(&spectra.coupling, infinite.coupling, options)?;
        Ok(Self {
            heave: FluidMemory::fit(&spectra.heave, infinite.heave, options)?,
            heave_from_pitch: coupling.clone(),
            pitch_from_heave: coupling,
            pitch: FluidMemory::fit(&spectra.pitch, infinite.pitch, options)?,
            last: None,
        })
    }

    /// The infinite-frequency added mass as a generalized matrix.
    ///
    /// Heave is index 2 and pitch is index 4 in the body-frame ordering, so this
    /// fills `(2,2)`, `(2,4)`, `(4,2)` and `(4,4)` and leaves the rest alone —
    /// ready for [`crate::rigid_body::RigidBody::add_added_mass`].
    ///
    /// Symmetric by construction: the coupling is written into both off-diagonal
    /// entries from one number, so they cannot drift apart.
    #[must_use]
    pub fn added_mass_matrix(infinite: &VerticalInfinite) -> Matrix6<f64> {
        let mut matrix = Matrix6::zeros();
        matrix[(2, 2)] = infinite.heave.value;
        matrix[(2, 4)] = infinite.coupling.value;
        matrix[(4, 2)] = infinite.coupling.value;
        matrix[(4, 4)] = infinite.pitch.value;
        matrix
    }

    /// Worst relative fit error over the three models.
    #[must_use]
    pub fn worst_error(&self) -> f64 {
        self.heave
            .worst_error()
            .max(self.pitch_from_heave.worst_error())
            .max(self.pitch.worst_error())
    }

    /// Real part of the least-damped pole over the three models, 1/s.
    #[must_use]
    pub fn slowest_pole(&self) -> f64 {
        self.heave
            .slowest_pole()
            .max(self.pitch_from_heave.slowest_pole())
            .max(self.pitch.slowest_pole())
    }

    /// Forgets the history, as if the boat had always been still.
    pub fn reset(&mut self) {
        self.heave.reset();
        self.heave_from_pitch.reset();
        self.pitch_from_heave.reset();
        self.pitch.reset();
        self.last = None;
    }
}

/// The infinite-frequency added mass of the three vertical coefficients.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VerticalInfinite {
    /// `A₃₃`, kg.
    pub heave: InfiniteAddedMass,
    /// `A₃₅ = A₅₃`, kg·m.
    pub coupling: InfiniteAddedMass,
    /// `A₅₅`, kg·m².
    pub pitch: InfiniteAddedMass,
}

impl VerticalInfinite {
    /// Worst disagreement between the two estimators, over the three
    /// coefficients.
    ///
    /// The one number to look at before trusting any of this: it is the
    /// agreement between two independent routes to a constant that does not
    /// depend on frequency. See [`InfiniteAddedMass::disagreement`].
    #[must_use]
    pub fn worst_disagreement(&self) -> f64 {
        self.heave
            .disagreement
            .max(self.coupling.disagreement)
            .max(self.pitch.disagreement)
    }
}

impl ForceModule for Radiation {
    fn name(&self) -> &'static str {
        "radiation"
    }

    fn step(&mut self, ctx: &StepCtx<'_>) -> Wrench {
        // Heave is `z` in a `z`-down body frame and pitch is rotation about `y`,
        // so these are the two components the vertical modes live in.
        let heave_rate = ctx.state.velocity.z;
        let pitch_rate = ctx.state.angular_velocity.y;
        let dt = ctx.dt;

        let force =
            self.heave.advance(heave_rate, dt) + self.heave_from_pitch.advance(pitch_rate, dt);
        let moment =
            self.pitch_from_heave.advance(heave_rate, dt) + self.pitch.advance(pitch_rate, dt);

        self.last = Some((force, moment));
        // The convolution sits on the left of the equation of motion, so it
        // opposes: it is subtracted from the applied load, which as a wrench
        // means the negative.
        Wrench {
            force: Vector3::new(0.0, 0.0, -force),
            moment: Vector3::new(0.0, -moment, 0.0),
        }
    }

    fn telemetry(&self, out: &mut Telemetry) {
        let (force, moment) = self.last.unwrap_or((0.0, 0.0));
        out.set("radiation.heave.memory_force", force);
        out.set("radiation.pitch.memory_moment", moment);
        out.set("radiation.worst_fit_error", self.worst_error());
        out.set("radiation.slowest_pole", self.slowest_pole());
    }
}

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

    fn built() -> (Radiation, VerticalInfinite, VerticalSpectra) {
        let strips = hull();
        let solver = SectionSolver::new(TasaiOptions::default());
        let grid: Vec<f64> = (1..=120).map(|i| 30.0 * f64::from(i) / 120.0).collect();
        let (heave, coupling, pitch) =
            strip::vertical_spectra(&strips, &grid, WATER, GRAVITY, &solver)
                .expect("a hull over a grid has spectra");
        let spectra = VerticalSpectra {
            heave,
            coupling,
            pitch,
        };
        let transform = TransformOptions::default();
        let infinite = VerticalInfinite {
            heave: spectra.heave.infinite_added_mass(transform),
            coupling: spectra.coupling.infinite_added_mass(transform),
            pitch: spectra.pitch.infinite_added_mass(transform),
        };
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
        let matrix = Radiation::added_mass_matrix(&infinite);

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

        for &target in &[1.0_f64, 2.5] {
            for heaving in [true, false] {
                let index = spectra
                    .heave
                    .frequencies()
                    .iter()
                    .enumerate()
                    .min_by(|a, b| (a.1 - target).abs().total_cmp(&(b.1 - target).abs()))
                    .map_or(0, |(i, _)| i);
                let omega = spectra.heave.frequencies()[index];

                let wanted = |spectrum: &crate::cummins::Spectrum, constant: f64| {
                    Complex::new(
                        spectrum.damping()[index],
                        omega * (spectrum.added_mass()[index] - constant),
                    )
                };
                let (force_wanted, moment_wanted) = if heaving {
                    (
                        wanted(&spectra.heave, infinite.heave.value),
                        wanted(&spectra.coupling, infinite.coupling.value),
                    )
                } else {
                    (
                        wanted(&spectra.coupling, infinite.coupling.value),
                        wanted(&spectra.pitch, infinite.pitch.value),
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
}
