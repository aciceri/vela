//! Strip theory: sectional coefficients integrated into hull coefficients.
//!
//! # Provenance
//!
//! Journée, J.M.J., *Theoretical Manual of SEAWAY (Release 4.19)*, Delft,
//! Report 1216a, 2001, §2.5.1, which gives the zero-forward-speed integration:
//!
//! ```text
//! heave:  X_h3 = ∫ X'_h3 dx_b
//! pitch:  X_h5 = -∫ X'_h3 · x_b dx_b
//! ```
//!
//! The antisymmetric modes are §6.2, §6.4 and §6.6, which add a second moment
//! arm the vertical modes do not have:
//!
//! ```text
//! sway:  X_h2 = ∫ X'_h2 dx_b
//! roll:  X_h4 = ∫ X'_h4 dx_b + OG · X_h2
//! yaw:   X_h6 = ∫ X'_h2 · x_b dx_b
//! ```
//!
//! where `OG` is the depth of the reference point below the section origin. Yaw
//! takes `+x_b` where pitch takes `-x_b`, because `(r × F)_z = x f_y` while
//! `(r × F)_y = -x f_z`; see [`lateral_coefficients`].
//!
//! The sectional loads come from [`crate::tasai`]; this module is only the
//! integrals, the minus sign on pitch, and the vertical shift on roll.
//!
//! # What strip theory assumes
//!
//! That each section's flow is two-dimensional — that the water beside one
//! station neither knows nor cares what the neighbouring stations are doing.
//! The source states the price: *"Fundamentally, strip theory is valid for long
//! and slender bodies only. In spite of this restriction, experiments have shown
//! that strip theory can be applied successfully for floating bodies with a
//! length to breadth ratio larger than three, `L/B ≥ 3`."*
//!
//! That bound is close for a sailing yacht and worth reporting rather than
//! assuming: the YD-41 waterline is 11.90 m on 3.18 m of beam, so `L/B = 3.7` —
//! inside, but not by much. [`HeavePitch::slenderness`] carries the number so a
//! caller can see where it stands instead of trusting that someone checked.
//!
//! # Zero forward speed
//!
//! These are the zero-speed coefficients. A moving hull has additional terms,
//! and they are not a refinement: they break the symmetry between `A_35` and
//! `A_53`, which the test suite here pins as an equality. The source is
//! unusually direct about whether they are worth having — *"it appeared from
//! user's experience that for ships with moderate forward speed (`Fn ≤ 0.30`),
//! the ordinary method provides a better fit with experimental data. Thus from a
//! practical point of view, the use of the ordinary method is advised
//! generally."* A sailing yacht upwind sits at `Fn ≈ 0.3`, at the edge of that
//! advice, which is a reason to prefer the simpler model and say so, not a
//! reason to pretend the question does not exist.

use crate::cummins::Spectrum;
use crate::lewis::LewisForm;
use crate::tasai::SectionSolver;

/// A station's longitudinal position and the form fitted to it.
///
/// `x` is the body-frame longitudinal coordinate, which is the file-frame one
/// unchanged: the file-to-body rotation is about the `x` axis. Sections with
/// nothing immersed belong here too, with `form` set to `None` — they carry no
/// added mass, but they carry the length over which the coefficients taper to
/// zero, and dropping them truncates the hull at its last wet station.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Strip {
    /// Longitudinal position, m.
    pub x: f64,
    /// The Lewis form of the immersed section, or `None` if it is dry.
    pub form: Option<LewisForm>,
}

/// Hull heave and pitch coefficients at one frequency, about the body origin.
///
/// The two-by-two symmetric matrices `A` and `B` in
/// `Z = -A_33 ẇ - A_35 q̇ - B_33 w - B_35 q` and
/// `M = -A_53 ẇ - A_55 q̇ - B_53 w - B_55 q`, in the body frame where heave is
/// positive down and pitch is positive bow-up.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeavePitch {
    /// `A_33`, kg.
    pub added_mass_heave: f64,
    /// `A_35 = A_53`, kg·m.
    ///
    /// One field rather than two, because at zero forward speed they are the
    /// same integral and storing it twice would only create the opportunity for
    /// them to disagree.
    pub added_mass_coupling: f64,
    /// `A_55`, kg·m².
    pub added_mass_pitch: f64,
    /// `B_33`, kg/s.
    pub damping_heave: f64,
    /// `B_35 = B_53`, kg·m/s.
    pub damping_coupling: f64,
    /// `B_55`, kg·m²/s.
    pub damping_pitch: f64,
    /// Waterline length over greatest waterline beam, dimensionless.
    ///
    /// The source's own validity bound is `L/B ≥ 3`. Reported so that a hull
    /// which fails it produces a visible number rather than a quiet answer.
    pub slenderness: f64,
    /// The largest [`tasai::HeaveCoefficients::energy_residual`] over the
    /// stations at this frequency.
    ///
    /// Carried up from the sectional solve because the hull coefficient is only
    /// as good as its worst section, and because a caller integrating over
    /// frequency should be able to see the truncation without re-solving.
    pub worst_energy_residual: f64,
}

/// Integrates sectional heave coefficients into hull heave and pitch.
///
/// Trapezoidal over the station positions, which need not be evenly spaced.
/// Returns `None` for a non-positive frequency, or for fewer than two stations —
/// strip theory needs a length to integrate over.
///
/// Takes the solver rather than the options because a frequency sweep over a
/// hull is thousands of section solves and the quadrature rule is the same for
/// every one of them. See [`SectionSolver`].
#[must_use]
pub fn heave_pitch_coefficients(
    strips: &[Strip],
    omega: f64,
    density: f64,
    gravity: f64,
    solver: &SectionSolver,
) -> Option<HeavePitch> {
    if strips.len() < 2 || omega <= 0.0 {
        return None;
    }

    // Solve each wet section; a dry one contributes nothing but its position.
    let mut sectional: Vec<(f64, f64, f64)> = Vec::with_capacity(strips.len());
    let mut worst_energy_residual: f64 = 0.0;
    let mut widest = 0.0_f64;
    for strip in strips {
        match &strip.form {
            None => sectional.push((strip.x, 0.0, 0.0)),
            Some(form) => {
                let solved = solver.heave(form, omega, density, gravity)?;
                worst_energy_residual = worst_energy_residual.max(solved.energy_residual.abs());
                widest = widest.max(form.beam());
                sectional.push((strip.x, solved.added_mass, solved.damping));
            }
        }
    }

    // Three moments of each sectional coefficient: the zeroth gives heave, the
    // first the coupling, the second pitch. `X_h5 = -∫ X'_h3 x dx` puts the
    // minus sign on the odd one.
    let mut mass = [0.0; 3];
    let mut damping = [0.0; 3];
    for pair in sectional.windows(2) {
        let (x0, a0, b0) = pair[0];
        let (x1, a1, b1) = pair[1];
        let span = x1 - x0;
        for (power, (m, d)) in mass.iter_mut().zip(damping.iter_mut()).enumerate() {
            let weight0 = x0.powi(power as i32);
            let weight1 = x1.powi(power as i32);
            *m += 0.5 * (a0 * weight0 + a1 * weight1) * span;
            *d += 0.5 * (b0 * weight0 + b1 * weight1) * span;
        }
    }

    let length = sectional[sectional.len() - 1].0 - sectional[0].0;
    Some(HeavePitch {
        added_mass_heave: mass[0],
        added_mass_coupling: -mass[1],
        added_mass_pitch: mass[2],
        damping_heave: damping[0],
        damping_coupling: -damping[1],
        damping_pitch: damping[2],
        slenderness: if widest > 0.0 {
            length / widest
        } else {
            f64::INFINITY
        },
        worst_energy_residual,
    })
}

/// Samples all three vertical coefficients across a frequency grid.
///
/// Returns the spectra of `A₃₃/B₃₃`, `A₃₅/B₃₅` and `A₅₅/B₅₅` in that order, from
/// one pass of section solves — the three share every solve, so computing them
/// separately would triple the cost for nothing.
///
/// The grid has to reach far enough out that the damping has genuinely died,
/// because everything downstream transforms it as though it continued to
/// infinity. For a yacht that is further than ship experience suggests — see
/// [`crate::cummins::Spectrum`], which is what this feeds.
///
/// Returns `None` if any frequency has no solution, or a grid does not make a
/// valid spectrum.
#[must_use]
pub fn vertical_spectra(
    strips: &[Strip],
    frequencies: &[f64],
    density: f64,
    gravity: f64,
    solver: &SectionSolver,
) -> Option<((Spectrum, Spectrum, Spectrum), SweepQuality)> {
    let count = frequencies.len();
    let mut heave = (Vec::with_capacity(count), Vec::with_capacity(count));
    let mut coupling = (Vec::with_capacity(count), Vec::with_capacity(count));
    let mut pitch = (Vec::with_capacity(count), Vec::with_capacity(count));
    let mut quality = SweepQuality::default();
    for &frequency in frequencies {
        let solved = heave_pitch_coefficients(strips, frequency, density, gravity, solver)?;
        heave.0.push(solved.added_mass_heave);
        heave.1.push(solved.damping_heave);
        coupling.0.push(solved.added_mass_coupling);
        coupling.1.push(solved.damping_coupling);
        pitch.0.push(solved.added_mass_pitch);
        pitch.1.push(solved.damping_pitch);
        quality.slenderness = solved.slenderness;
        quality.frequencies.push(frequency);
        quality.energy_residuals.push(solved.worst_energy_residual);
        quality.reciprocity_residuals.push(0.0);
    }
    let grid = frequencies.to_vec();
    Some((
        (
            Spectrum::new(grid.clone(), heave.0, heave.1).ok()?,
            Spectrum::new(grid.clone(), coupling.0, coupling.1).ok()?,
            Spectrum::new(grid, pitch.0, pitch.1).ok()?,
        ),
        quality,
    ))
}

/// What a frequency sweep found out about itself, frequency by frequency.
///
/// These numbers are computed at every frequency of every load and used to be
/// read by nobody but `vela-cli radiation`: an assembly that swallowed a
/// clamped bow or a broken energy identity said nothing. They travel with the
/// spectra now so the radiation module can publish them, and a reader of the
/// telemetry can tell a fit that was fed good coefficients from one that was
/// not.
///
/// Kept per frequency rather than as one worst case, because the worst case
/// over a grid that reaches 30 rad/s is a number about the grid's tail and not
/// about the boat: the identity is a *relative* residual against a damping
/// that has died to a fiftieth of its peak up there, and the multipole series
/// is truncated — 27 % at 30 rad/s on the YD-41 against 1e-5 at 4 rad/s. What
/// matters is the band the memory fit consumed, and
/// [`SweepQuality::worst_energy_residual_below`] answers for that band.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SweepQuality {
    /// [`HeavePitch::slenderness`], which does not depend on frequency. Zero
    /// for a lateral sweep, which does not compute it.
    pub slenderness: f64,
    /// The grid, rad/s.
    pub frequencies: Vec<f64>,
    /// [`HeavePitch::worst_energy_residual`] or
    /// [`LateralModes::worst_energy_residual`] at each grid frequency.
    pub energy_residuals: Vec<f64>,
    /// [`LateralModes::worst_reciprocity_residual`] at each grid frequency.
    /// Zero for a vertical sweep, which has no cross-mode identity to check.
    pub reciprocity_residuals: Vec<f64>,
}

impl SweepQuality {
    /// Worst energy residual at or below `ceiling` rad/s.
    #[must_use]
    pub fn worst_energy_residual_below(&self, ceiling: f64) -> f64 {
        Self::worst_below(&self.frequencies, &self.energy_residuals, ceiling)
    }

    /// Worst reciprocity residual at or below `ceiling` rad/s.
    #[must_use]
    pub fn worst_reciprocity_residual_below(&self, ceiling: f64) -> f64 {
        Self::worst_below(&self.frequencies, &self.reciprocity_residuals, ceiling)
    }

    fn worst_below(frequencies: &[f64], residuals: &[f64], ceiling: f64) -> f64 {
        frequencies
            .iter()
            .zip(residuals)
            .filter(|(frequency, _)| **frequency <= ceiling)
            .map(|(_, residual)| residual.abs())
            .fold(0.0, f64::max)
    }
}

/// Hull sway, roll and yaw coefficients at one frequency, about the body origin.
///
/// The three-by-three symmetric matrices `A` and `B` in
/// `Y = -A_22 v̇ - A_24 ṗ - A_26 ṙ - B_22 v - B_24 p - B_26 r`,
/// `K = -A_42 v̇ - A_44 ṗ - A_46 ṙ - B_42 v - B_44 p - B_46 r` and
/// `N = -A_62 v̇ - A_64 ṗ - A_66 ṙ - B_62 v - B_64 p - B_66 r`, in the body frame
/// where sway is positive to starboard, roll is positive starboard-down and yaw
/// is positive bow-to-starboard.
///
/// Radiation only. There is no restoring term here and no viscous roll damping:
/// the engine takes roll restoring from a pressure integral over the clipped
/// mesh, which is the same number obtained without a second waterplane model to
/// keep in agreement, and viscous damping is not a potential-flow quantity at
/// all.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LateralModes {
    /// `A_22`, kg.
    pub added_mass_sway: f64,
    /// `A_24 = A_42`, kg·m.
    ///
    /// One field rather than two, for the same reason as
    /// [`HeavePitch::added_mass_coupling`]: at zero forward speed they are the
    /// same integral, and storing it twice would only create the opportunity for
    /// them to disagree.
    pub added_mass_sway_roll: f64,
    /// `A_26 = A_62`, kg·m.
    pub added_mass_sway_yaw: f64,
    /// `A_44`, kg·m².
    pub added_mass_roll: f64,
    /// `A_46 = A_64`, kg·m².
    pub added_mass_roll_yaw: f64,
    /// `A_66`, kg·m².
    pub added_mass_yaw: f64,
    /// `B_22`, kg/s.
    pub damping_sway: f64,
    /// `B_24 = B_42`, kg·m/s.
    pub damping_sway_roll: f64,
    /// `B_26 = B_62`, kg·m/s.
    pub damping_sway_yaw: f64,
    /// `B_44`, kg·m²/s.
    pub damping_roll: f64,
    /// `B_46 = B_64`, kg·m²/s.
    pub damping_roll_yaw: f64,
    /// `B_66`, kg·m²/s.
    pub damping_yaw: f64,
    /// The largest of [`tasai::LateralCoefficients::sway_energy_residual`] and
    /// [`tasai::LateralCoefficients::roll_energy_residual`], in absolute value,
    /// over the stations at this frequency.
    ///
    /// Carried up from the sectional solve because the hull coefficient is only
    /// as good as its worst section, and because a caller integrating over
    /// frequency should be able to see the truncation without re-solving. Both
    /// identities collapse into one number because they share a cause: too few
    /// multipoles for the reduced frequency this section is being asked about.
    pub worst_energy_residual: f64,
    /// The largest [`tasai::LateralCoefficients::reciprocity_residual`], in
    /// absolute value, over the stations at this frequency.
    ///
    /// The antisymmetric problem has a second, independent accuracy check that
    /// the vertical one does not: `M'_42` and `M'_24` come out of different
    /// solves and potential flow says they are equal. Reported here for the same
    /// reason as the energy residual — the assembled `A_24` and `A_46` inherit
    /// whatever disagreement the worst station had, and a caller should be able
    /// to see it without re-solving.
    pub worst_reciprocity_residual: f64,
}

/// Integrates sectional lateral coefficients into hull sway, roll and yaw.
///
/// Trapezoidal over the station positions, which need not be evenly spaced,
/// exactly as [`heave_pitch_coefficients`] — dry stations included, for the
/// reason given on [`Strip`]. Returns `None` for a non-positive frequency, or
/// for fewer than two stations.
///
/// # The vertical datum, which suddenly matters
///
/// It did not matter for heave and pitch. A sectional vertical force acts on the
/// centreline, so it has no roll lever arm and no moment about any point on the
/// centreline; moving the reference point up or down changes nothing, and
/// [`heave_pitch_coefficients`] correspondingly never asks where the waterline
/// is. A sectional *lateral* force does have a roll lever arm, and it is exactly
/// the vertical distance from the reference point to the line of action.
///
/// [`SectionSolver::lateral`] returns coefficients about the section's own
/// origin `O`, which sits on the waterline at the centreline. The engine's body
/// origin sits on the baseline (see [`crate::frames`]). So `waterline_height` is
/// the source's `OG`: the depth of the body origin below the section origin,
/// positive downward, which for a hull floating at its design waterline is the
/// design draft. Pass zero and you get the coefficients about the waterline
/// instead — which is a legitimate thing to want, and is what the parallel-axis
/// test exploits, but it is not the frame the rest of the engine works in.
///
/// # Assembly
///
/// With `h = OG`, the source's §6.2/§6.4/§6.6 coefficients at zero forward speed
/// reduce to six raw moments and a shift:
///
/// ```text
/// a22 = ∫ M'_22 dx
/// a24 = ∫ M'_42 dx + h · a22
/// a26 = ∫ M'_22 x dx
/// a44 = ∫ M'_44 dx + 2 h ∫ M'_42 dx + h² · a22
/// a46 = ∫ M'_42 x dx + h · a26
/// a66 = ∫ M'_22 x² dx
/// ```
///
/// and identically for `b` with `N'` for `M'`. The `h` terms carry `+` from
/// `M_P = M_O + (r_O - r_P) × F` with `r_O - r_P = (0, 0, -h)` and
/// `F = (0, f_y, 0)`, which gives `(h f_y, 0, 0)`. Yaw takes `+∫ ... x dx` where
/// pitch took `-∫ ... x dx`, because `(r × F)_z = x f_y` while
/// `(r × F)_y = -x f_z`.
///
/// The `a44` line is the source's `a44 = ∫M'_44 dx + OG ∫M'_42 dx + OG · a24`
/// with `a24` substituted; written out it is visibly the parallel-axis shift of
/// a roll axis moved vertically, which is what the tests check.
///
/// Takes the solver rather than the options for the same reason as
/// [`heave_pitch_coefficients`]: a sweep is thousands of section solves and the
/// quadrature rule is the same for all of them.
#[must_use]
pub fn lateral_coefficients(
    strips: &[Strip],
    waterline_height: f64,
    frequency: f64,
    density: f64,
    gravity: f64,
    solver: &SectionSolver,
) -> Option<LateralModes> {
    if strips.len() < 2 || frequency <= 0.0 {
        return None;
    }

    // Solve each wet section; a dry one contributes nothing but its position.
    // The triples are `[M'_22, M'_42, M'_44]` and `[N'_22, N'_42, N'_44]`, in
    // that order throughout this function.
    let mut sectional: Vec<(f64, [f64; 3], [f64; 3])> = Vec::with_capacity(strips.len());
    let mut worst_energy_residual: f64 = 0.0;
    let mut worst_reciprocity_residual: f64 = 0.0;
    for strip in strips {
        match &strip.form {
            None => sectional.push((strip.x, [0.0; 3], [0.0; 3])),
            Some(form) => {
                let solved = solver.lateral(form, frequency, density, gravity)?;
                worst_energy_residual = worst_energy_residual
                    .max(solved.sway_energy_residual.abs())
                    .max(solved.roll_energy_residual.abs());
                worst_reciprocity_residual =
                    worst_reciprocity_residual.max(solved.reciprocity_residual.abs());
                sectional.push((
                    strip.x,
                    [
                        solved.sway_added_mass,
                        solved.coupling_added_mass,
                        solved.roll_added_inertia,
                    ],
                    [
                        solved.sway_damping,
                        solved.coupling_damping,
                        solved.roll_damping,
                    ],
                ));
            }
        }
    }

    // Moments of the three sectional coefficients: `mass[power][index]` is
    // `∫ coefficient_index · x^power dx`. Sway needs powers up to two, the
    // coupling up to one and the roll inertia only the zeroth, but running all
    // three powers over all three coefficients costs four multiply-adds per
    // strip and keeps this loop the same shape as the vertical path's, which is
    // worth more than the arithmetic.
    let mut mass = [[0.0; 3]; 3];
    let mut damping = [[0.0; 3]; 3];
    for pair in sectional.windows(2) {
        let (x0, a0, b0) = pair[0];
        let (x1, a1, b1) = pair[1];
        let span = x1 - x0;
        for (power, (m, d)) in mass.iter_mut().zip(damping.iter_mut()).enumerate() {
            let weight0 = x0.powi(power as i32);
            let weight1 = x1.powi(power as i32);
            for (index, (m, d)) in m.iter_mut().zip(d.iter_mut()).enumerate() {
                *m += 0.5 * (a0[index] * weight0 + a1[index] * weight1) * span;
                *d += 0.5 * (b0[index] * weight0 + b1[index] * weight1) * span;
            }
        }
    }

    let height = waterline_height;
    let assemble = |moments: &[[f64; 3]; 3]| {
        let [zeroth_sway, zeroth_coupling, zeroth_roll] = moments[0];
        let [first_sway, first_coupling, _] = moments[1];
        let second_sway = moments[2][0];
        // `[a22, a24, a26, a44, a46, a66]`.
        [
            zeroth_sway,
            zeroth_coupling + height * zeroth_sway,
            first_sway,
            zeroth_roll + 2.0 * height * zeroth_coupling + height * height * zeroth_sway,
            first_coupling + height * first_sway,
            second_sway,
        ]
    };
    let a = assemble(&mass);
    let b = assemble(&damping);

    Some(LateralModes {
        added_mass_sway: a[0],
        added_mass_sway_roll: a[1],
        added_mass_sway_yaw: a[2],
        added_mass_roll: a[3],
        added_mass_roll_yaw: a[4],
        added_mass_yaw: a[5],
        damping_sway: b[0],
        damping_sway_roll: b[1],
        damping_sway_yaw: b[2],
        damping_roll: b[3],
        damping_roll_yaw: b[4],
        damping_yaw: b[5],
        worst_energy_residual,
        worst_reciprocity_residual,
    })
}

/// Samples all six lateral coefficients across a frequency grid.
///
/// Returns the spectra of `A/B` for `22`, `24`, `26`, `44`, `46` and `66`, in
/// that order, from one pass of section solves — all six share every solve, so
/// computing them separately would cost six times as much for nothing.
///
/// An array rather than a six-tuple: past three or so, positional tuple fields
/// stop being readable at the call site and `.3` tells the reader nothing, while
/// an indexable array at least invites a named constant or a loop.
///
/// The grid has to reach far enough out that the damping has genuinely died, for
/// the reason given on [`vertical_spectra`] — more so here, since the roll
/// damping of a shallow section is small and slow to fall away, so a grid that
/// looks converged in sway may not be in roll.
///
/// Returns `None` if any frequency has no solution, or a grid does not make a
/// valid spectrum.
#[must_use]
pub fn lateral_spectra(
    strips: &[Strip],
    waterline_height: f64,
    frequencies: &[f64],
    density: f64,
    gravity: f64,
    solver: &SectionSolver,
) -> Option<([Spectrum; 6], SweepQuality)> {
    let count = frequencies.len();
    let mut added_mass: [Vec<f64>; 6] = std::array::from_fn(|_| Vec::with_capacity(count));
    let mut damping: [Vec<f64>; 6] = std::array::from_fn(|_| Vec::with_capacity(count));
    let mut quality = SweepQuality::default();
    for &frequency in frequencies {
        let solved = lateral_coefficients(
            strips,
            waterline_height,
            frequency,
            density,
            gravity,
            solver,
        )?;
        let a = [
            solved.added_mass_sway,
            solved.added_mass_sway_roll,
            solved.added_mass_sway_yaw,
            solved.added_mass_roll,
            solved.added_mass_roll_yaw,
            solved.added_mass_yaw,
        ];
        let b = [
            solved.damping_sway,
            solved.damping_sway_roll,
            solved.damping_sway_yaw,
            solved.damping_roll,
            solved.damping_roll_yaw,
            solved.damping_yaw,
        ];
        for (index, column) in added_mass.iter_mut().enumerate() {
            column.push(a[index]);
        }
        for (index, column) in damping.iter_mut().enumerate() {
            column.push(b[index]);
        }
        quality.frequencies.push(frequency);
        quality.energy_residuals.push(solved.worst_energy_residual);
        quality
            .reciprocity_residuals
            .push(solved.worst_reciprocity_residual);
    }

    let grid = frequencies.to_vec();
    let mut spectra: [Option<Spectrum>; 6] = std::array::from_fn(|_| None);
    for (index, slot) in spectra.iter_mut().enumerate() {
        *slot = Some(
            Spectrum::new(
                grid.clone(),
                std::mem::take(&mut added_mass[index]),
                std::mem::take(&mut damping[index]),
            )
            .ok()?,
        );
    }
    Some((
        spectra.map(|slot| slot.expect("every slot was just filled")),
        quality,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lewis::SectionGeometry;
    use crate::tasai::TasaiOptions;
    use approx::assert_relative_eq;
    use nalgebra::Matrix3;

    const WATER: f64 = 1025.0;
    const GRAVITY: f64 = 9.81;

    /// A hull of identical sections, so that every integral is analytic.
    ///
    /// Not a boat, on purpose: a pontoon of constant section is the one hull for
    /// which the moments of the sectional coefficient can be written down, which
    /// is what makes it useful for checking that the integration is the
    /// integration the source asks for and nothing else.
    fn pontoon(stations: usize, from: f64, to: f64) -> Vec<Strip> {
        let form = LewisForm::fit(&SectionGeometry {
            beam: 3.2,
            draft: 0.6,
            area: 0.8 * 3.2 * 0.6,
        });
        (0..stations)
            .map(|i| Strip {
                x: from + (to - from) * i as f64 / (stations - 1) as f64,
                form: Some(form),
            })
            .collect()
    }

    fn solve(strips: &[Strip], omega: f64) -> HeavePitch {
        heave_pitch_coefficients(
            strips,
            omega,
            WATER,
            GRAVITY,
            &SectionSolver::new(TasaiOptions::default()),
        )
        .expect("a positive frequency over a length has a solution")
    }

    /// A hull symmetric about the origin has no heave-pitch coupling.
    ///
    /// The coupling is the first moment of a coefficient that is here constant,
    /// so it integrates to zero by symmetry. This is the sign convention's
    /// smoke test: get the minus sign wrong and this still passes, but get the
    /// *arm* wrong — measure it from an end rather than the origin — and it does
    /// not.
    #[test]
    fn a_hull_symmetric_about_the_origin_does_not_couple_heave_to_pitch() {
        let solved = solve(&pontoon(21, -6.0, 6.0), 1.0);
        assert!(solved.added_mass_heave > 0.0);
        assert!(solved.added_mass_pitch > 0.0);
        assert_relative_eq!(
            solved.added_mass_coupling / solved.added_mass_heave,
            0.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            solved.damping_coupling / solved.damping_heave,
            0.0,
            epsilon = 1e-12
        );
    }

    /// Moving the origin moves the coefficients by the parallel-axis relation.
    ///
    /// Shifting every station by `d` must give `A_35' = A_35 - d A_33` and
    /// `A_55' = A_55 - 2 d A_35 + d² A_33`, because these are the first and
    /// second moments of the same sectional distribution about a new point. It
    /// holds whatever the sections are, so it tests the integration without
    /// assuming anything about the hydrodynamics.
    #[test]
    fn shifting_the_origin_obeys_the_parallel_axis_relation() {
        let shift = 4.0;
        let here = solve(&pontoon(21, -6.0, 6.0), 1.2);
        let shifted: Vec<Strip> = pontoon(21, -6.0, 6.0)
            .into_iter()
            .map(|strip| Strip {
                x: strip.x + shift,
                ..strip
            })
            .collect();
        let there = solve(&shifted, 1.2);

        assert_relative_eq!(
            there.added_mass_heave,
            here.added_mass_heave,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            there.added_mass_coupling,
            here.added_mass_coupling - shift * here.added_mass_heave,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            there.added_mass_pitch,
            here.added_mass_pitch - 2.0 * shift * here.added_mass_coupling
                + shift * shift * here.added_mass_heave,
            max_relative = 1e-12
        );
    }

    /// The added mass matrix has to be a possible mass matrix.
    ///
    /// `A_33 A_55 ≥ A_35²`, which is Cauchy-Schwarz on the sectional
    /// distribution and therefore true for any hull whose sectional added mass
    /// is positive. A matrix that failed it would accelerate a boat in the wrong
    /// direction under some combination of heave and pitch. Checked on an
    /// asymmetric hull, where the coupling is large enough for the inequality to
    /// have something to say.
    #[test]
    fn the_added_mass_matrix_is_positive_definite() {
        for &omega in &[0.5, 1.0, 2.0] {
            let solved = solve(&pontoon(21, 0.0, 12.0), omega);
            assert!(
                solved.added_mass_coupling.abs() > 0.1 * solved.added_mass_heave,
                "this hull should couple strongly, or the test proves nothing"
            );
            let determinant = solved.added_mass_heave * solved.added_mass_pitch
                - solved.added_mass_coupling * solved.added_mass_coupling;
            assert!(
                determinant > 0.0,
                "added mass determinant went negative at omega = {omega}: {determinant}"
            );
            let damping_determinant = solved.damping_heave * solved.damping_pitch
                - solved.damping_coupling * solved.damping_coupling;
            assert!(
                damping_determinant > 0.0,
                "damping determinant went negative at omega = {omega}"
            );
        }
    }

    /// The moments of a constant section are analytic, so check against them.
    ///
    /// A pontoon of constant section over `[-6, 6]` has `A_33 = 12 a` and
    /// `A_55 = ∫x² dx · a = 144 a`, so the ratio `A_55 / A_33` is exactly 12 m²
    /// whatever the sectional coefficient turns out to be. That makes it a real
    /// target rather than a regression baseline.
    ///
    /// The trapezoidal rule is exact for the zeroth moment at any spacing and
    /// *not* exact for the second, which is worth having in a test rather than
    /// discovering later: on three stations it overestimates `∫x²` by exactly
    /// half — 216 against 144 — and the ratio comes out 18 instead of 12. The
    /// error is the textbook `h²/12 · (b - a) · f''`, which for `f = x²` is
    /// `h² (b - a) / 6`, and both grids are checked against it.
    #[test]
    fn the_moments_of_a_constant_section_are_the_analytic_ones() {
        let span = 12.0;
        for stations in [3_usize, 5, 41] {
            let solved = solve(&pontoon(stations, -6.0, 6.0), 1.0);
            let step = span / (stations - 1) as f64;
            // ∫x² over [-6, 6] is 144; the trapezoid overshoots by h²(b-a)/6.
            let expected = (144.0 + step * step * span / 6.0) / span;
            assert_relative_eq!(
                solved.added_mass_pitch / solved.added_mass_heave,
                expected,
                max_relative = 1e-9
            );
        }
        // The zeroth moment, by contrast, does not care about the spacing.
        assert_relative_eq!(
            solve(&pontoon(3, -6.0, 6.0), 1.0).added_mass_heave,
            solve(&pontoon(41, -6.0, 6.0), 1.0).added_mass_heave,
            max_relative = 1e-12
        );
    }

    /// Dry stations carry length without carrying added mass.
    ///
    /// Adding a dry station beyond the last wet one must lengthen the hull for
    /// the purpose of the taper and change nothing else, because the trapezoid
    /// it adds has zero height at one end. Dropping such stations instead — the
    /// obvious simplification — truncates the hull at its last wet section.
    #[test]
    fn a_dry_station_extends_the_hull_without_adding_mass() {
        let wet = pontoon(21, -6.0, 6.0);
        let mut extended = wet.clone();
        extended.push(Strip { x: 7.0, form: None });

        let bare = solve(&wet, 1.0);
        let tapered = solve(&extended, 1.0);
        // The added trapezoid runs from the last wet section down to nothing, so
        // it contributes half of the section's coefficient over its span.
        assert!(tapered.added_mass_heave > bare.added_mass_heave);
        assert!(tapered.slenderness > bare.slenderness);
    }

    /// Radiation damping falls away past its peak — but slowly, for a yacht.
    ///
    /// This test was first written asserting that damping had all but vanished
    /// by 4 rad/s, which is true of a ship and false of a boat. For this 3.2 m
    /// section the damping peak sits near 1.6 rad/s and 4 rad/s still returns
    /// 59 % of it.
    ///
    /// The reason is that the frequency which governs radiation is the reduced
    /// one, `ξ_b = ω² B / 2g`, and a narrow section reaches a given `ξ_b` only at
    /// a much higher `ω`. A yacht therefore radiates over a far wider band of
    /// encounter frequencies than ship experience suggests — which is a physical
    /// result about the boat, not a numerical one about the method, and the
    /// reason this test is written in `ξ_b` and not in `ω`.
    #[test]
    fn hull_damping_falls_away_past_its_peak() {
        let hull = pontoon(21, -6.0, 6.0);
        let beam = 3.2;
        // ω for a wanted reduced frequency ξ_b = ω² B / 2g.
        let omega_for = |reduced: f64| (reduced * 2.0 * GRAVITY / beam).sqrt();

        let peak = solve(&hull, omega_for(0.42));
        let far = solve(&hull, omega_for(6.0));
        assert!(
            far.damping_heave < 0.3 * peak.damping_heave,
            "damping at reduced frequency 6 was {} against a peak of {}",
            far.damping_heave,
            peak.damping_heave
        );

        // And it is monotone on the way out, which a resonance artefact
        // would not be.
        let mut previous = f64::INFINITY;
        for reduced in [1.0, 2.0, 3.0, 4.0, 6.0] {
            let damping = solve(&hull, omega_for(reduced)).damping_heave;
            assert!(
                damping < previous,
                "damping rose again at reduced frequency {reduced}"
            );
            previous = damping;
        }
    }

    /// A hull too short to integrate over is not a hull.
    #[test]
    fn one_station_is_not_a_hull() {
        let single = pontoon(3, -6.0, 6.0);
        assert!(heave_pitch_coefficients(
            &single[..1],
            1.0,
            WATER,
            GRAVITY,
            &SectionSolver::new(TasaiOptions::default())
        )
        .is_none());
        assert!(heave_pitch_coefficients(
            &single,
            0.0,
            WATER,
            GRAVITY,
            &SectionSolver::new(TasaiOptions::default())
        )
        .is_none());
    }

    /// The datum of the pontoon fixture: it floats at 0.6 m, so that is its `OG`.
    const PONTOON_DRAFT: f64 = 0.6;

    fn solve_lateral(strips: &[Strip], waterline_height: f64, omega: f64) -> LateralModes {
        lateral_coefficients(
            strips,
            waterline_height,
            omega,
            WATER,
            GRAVITY,
            &SectionSolver::new(TasaiOptions::default()),
        )
        .expect("a positive frequency over a length has a solution")
    }

    /// The roll added mass is exactly a parabola in the vertical datum.
    ///
    /// `A_44(h) = ∫M'_44 + 2h ∫M'_42 + h² A_22` — a quadratic whose leading
    /// coefficient is the sway added mass, and nothing else. Sampling it at three
    /// equally spaced `h` and taking the second difference kills the constant and
    /// the linear term and leaves `2 δ² A_22`, so the check needs no knowledge of
    /// the two raw integrals it cannot see.
    ///
    /// This is the strongest available check on the whole assembly step: it pins
    /// the `h²` coefficient to a quantity computed by a different code path, and
    /// it fails if either `h` term has the wrong sign, the wrong factor of two,
    /// or the wrong integral behind it.
    #[test]
    fn the_roll_added_mass_is_a_parabola_in_the_vertical_datum() {
        let hull = pontoon(21, -6.0, 6.0);
        let step = 0.5;
        for &omega in &[0.6, 1.2, 2.4] {
            let below = solve_lateral(&hull, PONTOON_DRAFT - step, omega);
            let here = solve_lateral(&hull, PONTOON_DRAFT, omega);
            let above = solve_lateral(&hull, PONTOON_DRAFT + step, omega);

            let second_difference =
                above.added_mass_roll - 2.0 * here.added_mass_roll + below.added_mass_roll;
            assert_relative_eq!(
                second_difference / (2.0 * step * step),
                here.added_mass_sway,
                max_relative = 1e-12
            );

            // The damping obeys the same shift, with `N'` for `M'`.
            let damping_second_difference =
                above.damping_roll - 2.0 * here.damping_roll + below.damping_roll;
            assert_relative_eq!(
                damping_second_difference / (2.0 * step * step),
                here.damping_sway,
                max_relative = 1e-12
            );

            // And the coupling is linear in `h` with slope `A_22`, which is the
            // same statement one derivative down.
            assert_relative_eq!(
                (above.added_mass_sway_roll - below.added_mass_sway_roll) / (2.0 * step),
                here.added_mass_sway,
                max_relative = 1e-12
            );
        }
    }

    /// Sway, yaw and the raw roll integral do not move with the datum.
    ///
    /// Only roll and the couplings into it have a vertical lever arm, so `A_22`
    /// and `A_66` must come out bit-identical at any `h`. The raw `∫M'_44` is not
    /// stored, but it is recoverable — `∫M'_42 = A_24 - h A_22` and then
    /// `∫M'_44 = A_44 - 2h ∫M'_42 - h² A_22` — and recovering it from three
    /// different data must give one answer, which says the shift is invertible
    /// and therefore that no information was destroyed by applying it.
    #[test]
    fn sway_yaw_and_the_raw_roll_integral_do_not_move_with_the_datum() {
        let hull = pontoon(21, -6.0, 6.0);
        let reference = solve_lateral(&hull, 0.0, 1.2);
        let unshift = |solved: &LateralModes, height: f64| {
            let coupling = solved.added_mass_sway_roll - height * solved.added_mass_sway;
            solved.added_mass_roll
                - 2.0 * height * coupling
                - height * height * solved.added_mass_sway
        };

        for height in [0.0, PONTOON_DRAFT, 2.5] {
            let solved = solve_lateral(&hull, height, 1.2);
            assert_relative_eq!(
                solved.added_mass_sway,
                reference.added_mass_sway,
                max_relative = 1e-12
            );
            assert_relative_eq!(
                solved.added_mass_yaw,
                reference.added_mass_yaw,
                max_relative = 1e-12
            );
            assert_relative_eq!(
                solved.damping_sway,
                reference.damping_sway,
                max_relative = 1e-12
            );
            assert_relative_eq!(
                solved.damping_yaw,
                reference.damping_yaw,
                max_relative = 1e-12
            );
            assert_relative_eq!(
                unshift(&solved, height),
                reference.added_mass_roll,
                max_relative = 1e-12
            );
        }
    }

    /// The lateral added mass matrix has to be a possible mass matrix.
    ///
    /// Sylvester's criterion on the three leading principal minors, plus
    /// Cauchy-Schwarz on each two-by-two sub-block — the same inequality the
    /// vertical test checks, three times over. It is not a formality: the hull
    /// matrix is `∫ Tᵀ S' T dx` with `T = [[1, h, x], [0, 1, 0]]` mapping hull
    /// sway, roll and yaw onto the section's own sway and roll, so it inherits
    /// positive semi-definiteness from the sections only if the shift is a
    /// congruence and not something else. Get a sign wrong in the `h` terms and
    /// `T` is no longer the rigid-body kinematics, and this fails.
    ///
    /// The three-by-three minor is checked against a scaled floor rather than
    /// zero. The sectional two-by-two of a shallow Lewis form is close to
    /// singular — a 3.2 m by 0.6 m section has very little roll inertia of its
    /// own next to `M'_42²/M'_22` — and a congruence of a nearly singular form
    /// is nearly singular, so the full determinant genuinely sits many orders
    /// below the product of the diagonal. Comparing it to zero would be
    /// comparing numerical noise to zero.
    #[test]
    fn the_lateral_added_mass_matrix_is_positive_definite() {
        for &omega in &[0.5, 1.0, 2.0] {
            // An asymmetric hull, so the sway-yaw coupling is not zero and the
            // inequalities have something to say.
            let solved = solve_lateral(&pontoon(21, 0.0, 12.0), PONTOON_DRAFT, omega);
            assert!(
                solved.added_mass_sway_yaw.abs() > 0.1 * solved.added_mass_sway,
                "this hull should couple sway to yaw, or the test proves nothing"
            );

            for (label, matrix) in [
                (
                    "added mass",
                    Matrix3::new(
                        solved.added_mass_sway,
                        solved.added_mass_sway_roll,
                        solved.added_mass_sway_yaw,
                        solved.added_mass_sway_roll,
                        solved.added_mass_roll,
                        solved.added_mass_roll_yaw,
                        solved.added_mass_sway_yaw,
                        solved.added_mass_roll_yaw,
                        solved.added_mass_yaw,
                    ),
                ),
                (
                    "damping",
                    Matrix3::new(
                        solved.damping_sway,
                        solved.damping_sway_roll,
                        solved.damping_sway_yaw,
                        solved.damping_sway_roll,
                        solved.damping_roll,
                        solved.damping_roll_yaw,
                        solved.damping_sway_yaw,
                        solved.damping_roll_yaw,
                        solved.damping_yaw,
                    ),
                ),
            ] {
                assert!(
                    matrix[(0, 0)] > 0.0,
                    "{label} sway went non-positive at omega = {omega}"
                );
                for (first, second) in [(0, 1), (0, 2), (1, 2)] {
                    let block = matrix[(first, first)] * matrix[(second, second)]
                        - matrix[(first, second)] * matrix[(first, second)];
                    assert!(
                        block >= 0.0,
                        "{label} Cauchy-Schwarz failed on the ({first}, {second}) block \
                         at omega = {omega}: {block}"
                    );
                }
                // The scale of a full three-by-three minor is the product of the
                // diagonal; anything smaller than a relative 1e-12 of that is
                // round-off in the quadrature, not a physical negative.
                let scale = matrix[(0, 0)] * matrix[(1, 1)] * matrix[(2, 2)];
                assert!(
                    matrix.determinant() >= -1e-12 * scale,
                    "{label} determinant went negative at omega = {omega}: {}",
                    matrix.determinant()
                );
            }
        }
    }

    /// A hull symmetric about the body origin has no sway-yaw coupling.
    ///
    /// `A_26` and `B_26` are the first moment of a sectional coefficient that is
    /// here constant, so they integrate to zero by symmetry. The lateral twin of
    /// the heave-pitch coupling test, and it catches the same mistake: measure
    /// the arm from an end rather than from the origin and it does not pass.
    /// Unlike pitch there is no minus sign to get wrong, which is precisely why
    /// the sign derivation is documented on [`lateral_coefficients`] rather than
    /// left to a test.
    #[test]
    fn a_hull_symmetric_about_the_body_origin_does_not_couple_sway_to_yaw() {
        let solved = solve_lateral(&pontoon(21, -6.0, 6.0), PONTOON_DRAFT, 1.0);
        assert!(solved.added_mass_sway > 0.0);
        assert!(solved.added_mass_yaw > 0.0);
        assert_relative_eq!(
            solved.added_mass_sway_yaw / solved.added_mass_sway,
            0.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            solved.damping_sway_yaw / solved.damping_sway,
            0.0,
            epsilon = 1e-12
        );
        // The roll-yaw coupling is the first moment of `M'_42` plus `h` times
        // the sway-yaw one, so it vanishes by the same symmetry.
        assert_relative_eq!(
            solved.added_mass_roll_yaw / solved.added_mass_sway,
            0.0,
            epsilon = 1e-12
        );
    }

    /// A hull too short to integrate over is not a lateral hull either.
    #[test]
    fn one_station_is_not_a_lateral_hull() {
        let single = pontoon(3, -6.0, 6.0);
        let solver = SectionSolver::new(TasaiOptions::default());
        assert!(
            lateral_coefficients(&single[..1], PONTOON_DRAFT, 1.0, WATER, GRAVITY, &solver)
                .is_none()
        );
        assert!(
            lateral_coefficients(&single, PONTOON_DRAFT, 0.0, WATER, GRAVITY, &solver).is_none(),
            "a non-positive frequency has no radiation problem"
        );
        assert!(
            lateral_coefficients(&single, PONTOON_DRAFT, -1.0, WATER, GRAVITY, &solver).is_none()
        );
    }

    /// The spectra are the pointwise sweep and nothing else.
    ///
    /// [`lateral_spectra`] exists only to share section solves across the six
    /// coefficients, so every sample it returns must equal what
    /// [`lateral_coefficients`] gives at that frequency, in the documented
    /// `[22, 24, 26, 44, 46, 66]` order. Pinned because the order is positional
    /// and a transposition of two entries would otherwise be silent.
    #[test]
    fn the_lateral_spectra_agree_with_the_pointwise_coefficients() {
        let hull = pontoon(11, -6.0, 6.0);
        let frequencies = [0.4, 0.8, 1.6, 3.2];
        let (spectra, _) = lateral_spectra(
            &hull,
            PONTOON_DRAFT,
            &frequencies,
            WATER,
            GRAVITY,
            &SectionSolver::new(TasaiOptions::default()),
        )
        .expect("a positive grid over a hull has spectra");

        for (index, &frequency) in frequencies.iter().enumerate() {
            let solved = solve_lateral(&hull, PONTOON_DRAFT, frequency);
            let expected_mass = [
                solved.added_mass_sway,
                solved.added_mass_sway_roll,
                solved.added_mass_sway_yaw,
                solved.added_mass_roll,
                solved.added_mass_roll_yaw,
                solved.added_mass_yaw,
            ];
            let expected_damping = [
                solved.damping_sway,
                solved.damping_sway_roll,
                solved.damping_sway_yaw,
                solved.damping_roll,
                solved.damping_roll_yaw,
                solved.damping_yaw,
            ];
            for (mode, spectrum) in spectra.iter().enumerate() {
                assert_relative_eq!(
                    spectrum.added_mass()[index],
                    expected_mass[mode],
                    max_relative = 1e-12
                );
                assert_relative_eq!(
                    spectrum.damping()[index],
                    expected_damping[mode],
                    max_relative = 1e-12
                );
            }
        }
    }
}
