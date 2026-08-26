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
//! The sectional loads `X'_h3` come from [`crate::tasai`]; this module is only
//! the integral, and the minus sign on pitch.
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

use crate::lewis::LewisForm;
use crate::tasai::{self, TasaiOptions};

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
#[must_use]
pub fn heave_pitch_coefficients(
    strips: &[Strip],
    omega: f64,
    density: f64,
    gravity: f64,
    options: TasaiOptions,
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
                let solved = tasai::heave_coefficients(form, omega, density, gravity, options)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lewis::SectionGeometry;
    use approx::assert_relative_eq;

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
        heave_pitch_coefficients(strips, omega, WATER, GRAVITY, TasaiOptions::default())
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
            TasaiOptions::default()
        )
        .is_none());
        assert!(
            heave_pitch_coefficients(&single, 0.0, WATER, GRAVITY, TasaiOptions::default())
                .is_none()
        );
    }
}
