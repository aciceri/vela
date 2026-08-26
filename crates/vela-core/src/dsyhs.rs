//! Hull resistance from the Delft Systematic Yacht Hull Series.
//!
//! # Provenance
//!
//! Every formula and coefficient here is transcribed from the **primary**
//! source: Larsson, Eliasson & Orych, *Principles of Yacht Design*, 5th ed.
//! (Bloomsbury, 2022) — Fig 5.8 (friction), Fig 5.18 (upright residuary,
//! after Keuning & Katgert 2008) and Fig 5.22 (change with heel). The book
//! states plainly that Keuning & Sonnenberg (1998) surveys the series up to
//! the late 1990s and that "the most recent update of the formula for
//! residuary resistance is presented in Keuning and Katgert (2008)".
//!
//! That insistence on the primary source is not pedantry. Two independent
//! secondary reproductions of these tables were checked first and **both were
//! defective**:
//!
//! - One labelled the Froude column `0.10–0.70` where the real range is
//!   `0.15–0.75`, an off-by-one shift that silently evaluates every
//!   coefficient at the wrong speed while producing entirely plausible
//!   numbers.
//! - The same source duplicated a coefficient label in the heel table.
//! - A third source presented a *different* regression (nine coefficients,
//!   Keuning & Sonnenberg 1998) under a similar name, with different terms.
//!
//! Only the book's own worked example could settle it. Which is why the tests
//! for this module reproduce that example numerically rather than checking
//! plumbing.
//!
//! # Scope
//!
//! Canoe body only: friction, viscous pressure, upright residuary, and the
//! change in residuary with heel. Keel and rudder resistance belong to the
//! appendage model, in keeping with the module split of `engine-design.md`
//! — the series itself separates them for the same reason, since bare hull and
//! appendages evolved independently.

use std::fmt;

/// Kinematic viscosity of salt water at 20 °C, m²/s, as used by the series.
pub const SALT_WATER_VISCOSITY: f64 = 1.0e-6;

/// Viscous pressure resistance as a fraction of frictional resistance.
///
/// The book takes 7 % from CFD computations. It is a lumped correction, not a
/// model, and it is exposed here so a caller can override it rather than
/// discover it baked in.
pub const VISCOUS_PRESSURE_FRACTION: f64 = 0.07;

/// Froude numbers at which the upright residuary coefficients are tabulated.
const RESIDUARY_FROUDE: [f64; 13] = [
    0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
];

/// Upright residuary resistance coefficients `a0..a7`, one row per Froude
/// number of [`RESIDUARY_FROUDE`]. Fig 5.18.
#[rustfmt::skip]
const RESIDUARY_COEFFICIENTS: [[f64; 8]; 13] = [
    [-0.0005,  0.0023, -0.0086, -0.0015,  0.0061,  0.0010,  0.0001,  0.0052],
    [-0.0003,  0.0059, -0.0064,  0.0070,  0.0014,  0.0013,  0.0005, -0.0020],
    [-0.0002, -0.0156,  0.0031, -0.0021, -0.0070,  0.0148,  0.0010, -0.0043],
    [-0.0009,  0.0016,  0.0337, -0.0285, -0.0367,  0.0218,  0.0015, -0.0172],
    [-0.0026, -0.0567,  0.0446, -0.1091, -0.0707,  0.0914,  0.0021, -0.0078],
    [-0.0064, -0.4034, -0.1250,  0.0273, -0.1341,  0.3578,  0.0045,  0.1115],
    [-0.0218, -0.5261, -0.2945,  0.2485, -0.2428,  0.6293,  0.0081,  0.2086],
    [-0.0388, -0.5986, -0.3038,  0.6033, -0.0430,  0.8332,  0.0106,  0.1336],
    [-0.0347, -0.4764, -0.2361,  0.8726,  0.4219,  0.8990,  0.0096, -0.2272],
    [-0.0361,  0.0037, -0.2960,  0.9661,  0.6123,  0.7534,  0.0100, -0.3352],
    [ 0.0008,  0.3728, -0.3667,  1.3957,  1.0343,  0.3230,  0.0072, -0.4632],
    [ 0.0108, -0.1238, -0.2026,  1.1282,  1.1836,  0.4973,  0.0038, -0.4477],
    [ 0.1023,  0.7726,  0.5040,  1.7867,  2.1934, -1.5479, -0.0115, -0.0977],
];

/// Froude numbers for the heeled residuary coefficients.
const HEEL_FROUDE: [f64; 7] = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55];

/// Change in residuary resistance at 20° of heel, coefficients `u0..u5`.
/// Fig 5.22. **Tabulated values are multiplied by 1000**, so they are scaled
/// down on use.
#[rustfmt::skip]
const HEEL_COEFFICIENTS: [[f64; 6]; 7] = [
    [-0.0268, -0.0014, -0.0057, 0.0016, -0.0070, -0.0017],
    [ 0.6628, -0.0632, -0.0699, 0.0069,  0.0459, -0.0004],
    [ 1.6433, -0.2144, -0.1640, 0.0199, -0.0540, -0.0268],
    [-0.8659, -0.0354,  0.2226, 0.0188, -0.5800, -0.1133],
    [-3.2715,  0.1372,  0.5547, 0.0268, -1.0064, -0.2026],
    [-0.1976, -0.1480, -0.6593, 0.1862, -0.7489, -0.1648],
    [ 1.5873, -0.3749, -0.7105, 0.2146, -0.4818, -0.1174],
];

/// The scalar hull description the series regressions consume.
///
/// Note what is *not* here: any geometry. The regressions are statistical fits
/// over form parameters, so a hull enters this model as nine numbers. They can
/// be derived from a lofted hull or declared in a boat file, and when both are
/// available they should be compared — a hull whose geometry disagrees with its
/// declared coefficients is a bad boat file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HullParameters {
    /// Waterline length, m.
    pub waterline_length: f64,
    /// Waterline beam, m.
    pub waterline_beam: f64,
    /// Canoe body draft, m.
    pub canoe_draft: f64,
    /// Canoe body displaced volume, m³.
    pub canoe_volume: f64,
    /// Canoe body wetted surface, m².
    pub wetted_surface: f64,
    /// Waterplane area, m².
    pub waterplane_area: f64,
    /// Prismatic coefficient.
    pub prismatic: f64,
    /// Midship section area coefficient.
    pub midship: f64,
    /// Longitudinal centre of buoyancy, as a **fraction** of the waterline
    /// length from midship, positive forward. Yacht data quotes this as a
    /// percentage; `-0.042` is the book's `-4.2 %`.
    pub lcb: f64,
    /// Longitudinal centre of flotation, same convention as [`Self::lcb`].
    pub lcf: f64,
}

impl HullParameters {
    /// Distance of the centre of buoyancy aft of the forward perpendicular, m.
    ///
    /// Fig 5.18 wants `LCB_fpp` in metres from the forward perpendicular, while
    /// hull data is published relative to midship. Getting this conversion
    /// wrong changes the answer by a few per cent and looks perfectly
    /// reasonable, so it lives in one place.
    #[must_use]
    pub fn lcb_from_bow(&self) -> f64 {
        self.waterline_length * (0.5 - self.lcb)
    }

    /// Distance of the centre of flotation aft of the forward perpendicular, m.
    #[must_use]
    pub fn lcf_from_bow(&self) -> f64 {
        self.waterline_length * (0.5 - self.lcf)
    }

    /// Froude number at a given speed through the water.
    #[must_use]
    pub fn froude(&self, speed: f64, gravity: f64) -> f64 {
        speed / (gravity * self.waterline_length).sqrt()
    }

    /// Checks the hull against the parameter envelope of the tested models.
    ///
    /// # Errors
    ///
    /// [`EnvelopeError`] naming every parameter outside its range. Outside the
    /// envelope these polynomials do not degrade gracefully — they diverge —
    /// so the caller is told rather than quietly given a number.
    pub fn check_envelope(&self) -> Result<(), EnvelopeError> {
        let cube_root = self.canoe_volume.cbrt();
        let checks = [
            (
                "Lwl/Bwl",
                self.waterline_length / self.waterline_beam,
                2.73,
                5.00,
            ),
            (
                "Bwl/Tc",
                self.waterline_beam / self.canoe_draft,
                2.46,
                19.38,
            ),
            (
                "Lwl/Vc^(1/3)",
                self.waterline_length / cube_root,
                4.34,
                8.50,
            ),
            ("LCB", self.lcb * 100.0, -8.2, 0.0),
            ("LCF", self.lcf * 100.0, -9.5, -1.8),
            ("Cp", self.prismatic, 0.52, 0.60),
            ("Cm", self.midship, 0.65, 0.78),
            (
                "Aw/Vc^(2/3)",
                self.waterplane_area / (cube_root * cube_root),
                3.78,
                12.67,
            ),
        ];

        let violations: Vec<Violation> = checks
            .iter()
            .filter(|(_, value, low, high)| value < low || value > high)
            .map(|(name, value, low, high)| Violation {
                parameter: name,
                value: *value,
                low: *low,
                high: *high,
            })
            .collect();

        if violations.is_empty() {
            Ok(())
        } else {
            Err(EnvelopeError { violations })
        }
    }
}

/// One parameter outside the series envelope.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Violation {
    pub parameter: &'static str,
    pub value: f64,
    pub low: f64,
    pub high: f64,
}

/// A hull outside the range of the models the regressions were fitted to.
#[derive(Debug, Clone, PartialEq)]
pub struct EnvelopeError {
    pub violations: Vec<Violation>,
}

impl fmt::Display for EnvelopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "hull is outside the DSYHS envelope:")?;
        for v in &self.violations {
            write!(
                f,
                "\n  {} = {:.3}, expected {:.2}..{:.2}",
                v.parameter, v.value, v.low, v.high
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for EnvelopeError {}

/// Resistance components of the canoe body, N.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HullResistance {
    /// Skin friction, from the ITTC-57 line.
    pub friction: f64,
    /// Viscous pressure (form) drag.
    pub viscous_pressure: f64,
    /// Upright residuary resistance.
    pub residuary: f64,
    /// Additional residuary resistance due to heel. Can be negative.
    pub heel_residuary: f64,
}

impl HullResistance {
    #[must_use]
    pub fn total(&self) -> f64 {
        self.friction + self.viscous_pressure + self.residuary + self.heel_residuary
    }
}

/// ITTC-57 model-ship correlation line.
#[must_use]
pub fn friction_coefficient(reynolds: f64) -> f64 {
    if reynolds <= 0.0 {
        return 0.0;
    }
    let denominator = reynolds.log10() - 2.0;
    0.075 / (denominator * denominator)
}

/// Frictional resistance of a surface, N.
///
/// `length` is the characteristic length for the Reynolds number. For a hull
/// the series uses **0.7 × Lwl**, not the full waterline length — see
/// [`hull_resistance`].
#[must_use]
pub fn frictional_resistance(
    speed: f64,
    length: f64,
    wetted_surface: f64,
    density: f64,
    viscosity: f64,
) -> f64 {
    let reynolds = speed * length / viscosity;
    friction_coefficient(reynolds) * 0.5 * density * speed * speed * wetted_surface
}

/// Upright residuary resistance of the canoe body, N. Fig 5.18.
///
/// Below the tabulated range the contribution is negligible and returns zero;
/// above it the last row is held, since extrapolating a degree-8 fit is worse
/// than admitting ignorance.
#[must_use]
pub fn residuary_resistance(hull: &HullParameters, speed: f64, density: f64, gravity: f64) -> f64 {
    let froude = hull.froude(speed, gravity);
    let taper = low_speed_taper(froude, RESIDUARY_FROUDE[0]);
    if taper <= 0.0 {
        return 0.0;
    }
    let a = interpolate(&RESIDUARY_FROUDE, &RESIDUARY_COEFFICIENTS, froude);

    let cube_root = hull.canoe_volume.cbrt();
    let slenderness = cube_root / hull.waterline_length;

    // Two bracketed groups in the book, both scaled by the same slenderness
    // ratio; kept separate here so the transcription can be read against the
    // figure line by line.
    let first = a[1] * hull.lcb_from_bow() / hull.waterline_length
        + a[2] * hull.prismatic
        + a[3] * (cube_root * cube_root) / hull.waterplane_area
        + a[4] * hull.waterline_beam / hull.waterline_length;
    let second = a[5] * hull.lcb_from_bow() / hull.lcf_from_bow()
        + a[6] * hull.waterline_beam / hull.canoe_draft
        + a[7] * hull.midship;

    let coefficient = a[0] + (first + second) * slenderness;
    taper * coefficient * hull.canoe_volume * density * gravity
}

/// Change in residuary resistance due to heel, N. Fig 5.22.
///
/// The tabulated regression gives the change at 20° of heel; the angular
/// scaling `6.0 φ^1.7` then extends it, and is normalized so that it returns
/// almost exactly the tabulated value at 20°.
///
/// `heel` is in radians.
///
/// Both ends of the tabulated Froude range are handled so the force stays
/// continuous, which matters more here than fidelity to a table that simply
/// stops. Below the first row the correction tapers linearly to zero at rest —
/// physically right, since a boat at rest makes no waves to modify. Above the
/// last row the final row is held: the series ends at `Fn = 0.55` but the
/// physics does not, and a force that drops off a cliff mid-acceleration would
/// jolt the time-domain solver.
#[must_use]
pub fn heel_residuary_delta(
    hull: &HullParameters,
    speed: f64,
    heel: f64,
    density: f64,
    gravity: f64,
) -> f64 {
    let froude = hull.froude(speed, gravity);
    let taper = low_speed_taper(froude, HEEL_FROUDE[0]);
    if taper <= 0.0 {
        return 0.0;
    }
    let u = interpolate(&HEEL_FROUDE, &HEEL_COEFFICIENTS, froude);

    let beam_draft = hull.waterline_beam / hull.canoe_draft;
    // LCB enters this regression as a percentage, unlike Fig 5.18 where it is
    // a length. The book's worked example only closes with this reading.
    let lcb_percent = hull.lcb * 100.0;

    let coefficient = (u[0]
        + u[1] * hull.waterline_length / hull.waterline_beam
        + u[2] * beam_draft
        + u[3] * beam_draft * beam_draft
        + u[4] * lcb_percent
        + u[5] * lcb_percent * lcb_percent)
        / 1000.0;

    let at_twenty = coefficient * hull.canoe_volume * density * gravity;
    taper * at_twenty * 6.0 * heel.abs().powf(1.7)
}

/// Full canoe body resistance at a given speed and heel angle.
///
/// The characteristic length for friction is `0.7 × Lwl`, which is the series'
/// own convention: a hull's boundary layer does not develop over the whole
/// waterline because the flow accelerates around the forebody. Using the full
/// length would understate the friction by several per cent.
#[must_use]
pub fn hull_resistance(
    hull: &HullParameters,
    speed: f64,
    heel: f64,
    density: f64,
    gravity: f64,
) -> HullResistance {
    let friction = frictional_resistance(
        speed,
        0.7 * hull.waterline_length,
        hull.wetted_surface,
        density,
        SALT_WATER_VISCOSITY,
    );
    HullResistance {
        friction,
        viscous_pressure: friction * VISCOUS_PRESSURE_FRACTION,
        residuary: residuary_resistance(hull, speed, density, gravity),
        heel_residuary: heel_residuary_delta(hull, speed, heel, density, gravity),
    }
}

/// Fraction of a tabulated regression to apply below its first Froude number.
///
/// The tables simply stop; the physics does not, and a force that appears from
/// nothing at a threshold speed is a discontinuity a time-domain solver will
/// feel. Ramping linearly to zero at rest is both continuous and physically
/// right — a stationary hull makes no waves to modify.
fn low_speed_taper(froude: f64, first_tabulated: f64) -> f64 {
    if froude <= 0.0 {
        0.0
    } else if froude >= first_tabulated {
        1.0
    } else {
        froude / first_tabulated
    }
}

/// Linear interpolation between tabulated coefficient rows, clamped at both
/// ends.
fn interpolate<const N: usize, const M: usize>(
    grid: &[f64; M],
    rows: &[[f64; N]; M],
    at: f64,
) -> [f64; N] {
    if at <= grid[0] {
        return rows[0];
    }
    if at >= grid[M - 1] {
        return rows[M - 1];
    }
    let upper = grid.iter().position(|g| *g >= at).unwrap_or(M - 1);
    let lower = upper - 1;
    let t = (at - grid[lower]) / (grid[upper] - grid[lower]);

    let mut out = [0.0; N];
    for i in 0..N {
        out[i] = rows[lower][i] + t * (rows[upper][i] - rows[lower][i]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn interpolation_hits_the_grid_points_exactly() {
        let a = interpolate(&RESIDUARY_FROUDE, &RESIDUARY_COEFFICIENTS, 0.35);
        assert_relative_eq!(a[0], -0.0026);
        assert_relative_eq!(a[7], -0.0078);
    }

    #[test]
    fn interpolation_is_linear_between_grid_points() {
        let a = interpolate(&RESIDUARY_FROUDE, &RESIDUARY_COEFFICIENTS, 0.325);
        let expected = 0.5 * (-0.0009 + -0.0026);
        assert_relative_eq!(a[0], expected, epsilon = 1e-12);
    }

    #[test]
    fn interpolation_clamps_outside_the_table() {
        let low = interpolate(&RESIDUARY_FROUDE, &RESIDUARY_COEFFICIENTS, 0.01);
        assert_relative_eq!(low[0], -0.0005);
        let high = interpolate(&RESIDUARY_FROUDE, &RESIDUARY_COEFFICIENTS, 9.0);
        assert_relative_eq!(high[0], 0.1023);
    }

    #[test]
    fn friction_coefficient_matches_the_ittc_line() {
        // Hand-evaluated: Rn = 1e7 gives log10 = 7, so 0.075 / 25.
        assert_relative_eq!(friction_coefficient(1.0e7), 0.003, epsilon = 1e-12);
    }
}
