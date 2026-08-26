//! Lewis conformal mapping of hull sections.
//!
//! # Provenance
//!
//! Transcribed from Journée, J.M.J. & Massie, W.W., *Offshore Hydromechanics*,
//! 1st ed., Delft University of Technology, 2001, §7.3 — equations 7.88 to
//! 7.95, which present the transformation of [[Lewis, 1929]], *The Inertia of
//! Water Surrounding a Vibrating Ship*, Transactions SNAME.
//!
//! A different source from the rest of this engine, and deliberately so:
//! Larsson, Eliasson & Orych state in their chapter on design evaluation that
//! added mass and damping are *"out of scope of the present book"*, so the
//! seakeeping side has to come from somewhere else. Journée & Massie is the
//! natural somewhere: Delft, like the hull series this engine already uses, and
//! freely published.
//!
//! # What this is for, and what it is not
//!
//! The mapping is the **geometric** half of strip theory. It replaces a real
//! section by the shape a two-parameter conformal map produces from a circle,
//! matching the section's beam, draft and area exactly — and because the flow
//! around a circle is known in closed form, the flow around the mapped shape
//! follows.
//!
//! It computes **no** hydrodynamic coefficient. The added mass and damping of a
//! mapped section come from Ursell's multipole expansion as extended by Tasai,
//! which is a numerical solution of the radiation problem and a substantial
//! piece of work in its own right. This module is its prerequisite, and it is
//! also useful on its own: a hull whose sections fall outside the Lewis
//! envelope is a hull strip theory will struggle with, and that is worth
//! knowing *before* the multipole expansion is written rather than after.
//!
//! # Frames
//!
//! Sections come from the boat file, so this module works in the **file frame**:
//! half-breadths `y ≥ 0` and heights `z` measured up from the baseline. It is
//! the one place downstream of the loader that legitimately does, because a
//! section is a file-frame object and the mapping is indifferent to which way
//! the world calls up.

use crate::boat::Station;
use std::f64::consts::PI;

/// Numerical guard on the half-beam-to-draft ratio, from the source: *"Numerical
/// problems with bulbous or shallow cross sections can be avoided by the
/// requirement `0.01 < H0 < 100.0`"*.
const RATIO_MIN: f64 = 0.01;
const RATIO_MAX: f64 = 100.0;

/// The three quantities a Lewis form is fitted to.
///
/// Beam, draft and area of the *immersed* part of a section. Nothing else about
/// the section survives the fit, which is the whole point and the whole
/// limitation: two sections with the same three numbers map to the same Lewis
/// form however differently they are actually shaped.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SectionGeometry {
    /// Breadth at the waterline, m — the full breadth, not the half.
    pub beam: f64,
    /// Draft of the section below the waterline, m.
    pub draft: f64,
    /// Immersed area of the section, m².
    pub area: f64,
}

impl SectionGeometry {
    /// Half-beam to draft ratio, `H0` in the source.
    #[must_use]
    pub fn ratio(&self) -> f64 {
        (0.5 * self.beam / self.draft).clamp(RATIO_MIN, RATIO_MAX)
    }

    /// Sectional area coefficient, `σ_s = A_s / (B_s D_s)`.
    #[must_use]
    pub fn area_coefficient(&self) -> f64 {
        self.area / (self.beam * self.draft)
    }
}

/// The bounds the source puts on the area coefficient, equation 7.95.
///
/// Outside them the transformation produces shapes that are re-entrant (they
/// fold back into themselves) or asymmetric, neither of which is a ship
/// section. Both branches of the lower bound are transcribed as printed; note
/// that they differ only in whether `H0` or its reciprocal appears, which makes
/// the pair symmetric about `H0 = 1` as it has to be.
#[must_use]
pub fn area_coefficient_bounds(ratio: f64) -> (f64, f64) {
    let lower = if ratio <= 1.0 {
        3.0 * PI / 32.0 * (2.0 - ratio)
    } else {
        3.0 * PI / 32.0 * (2.0 - 1.0 / ratio)
    };
    let upper = PI / 32.0 * (10.0 + ratio + 1.0 / ratio);
    (lower, upper)
}

/// A section replaced by its Lewis form.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LewisForm {
    /// Scale factor `M_s`, m.
    pub scale: f64,
    /// First mapping coefficient `a_1`.
    pub a1: f64,
    /// Second mapping coefficient `a_3`.
    pub a3: f64,
    /// The area coefficient actually used.
    ///
    /// Equal to the section's own unless it fell outside
    /// [`area_coefficient_bounds`], in which case the source prescribes using
    /// the nearest border: *"If a value of σ_s is outside this range, it has to
    /// be set to the value at the nearest border of this range, in order to
    /// calculate the (best possible) Lewis coefficients."* Clamping here is
    /// therefore the method, not a liberty taken with it.
    pub area_coefficient: f64,
    /// Whether that clamp was applied. Reported rather than hidden: a clamped
    /// fit no longer reproduces the section's area, so anything built on it
    /// inherits an error nobody should have to rediscover.
    pub clamped: bool,
}

impl LewisForm {
    /// Fits the two-parameter mapping to a section's beam, draft and area.
    ///
    /// Equations 7.90 to 7.94. The quadratic in `a_3` takes the root with the
    /// plus sign; the source discards the other because it produces forms that
    /// *"intersect themselves somewhere in the fourth quadrant"*.
    #[must_use]
    pub fn fit(section: &SectionGeometry) -> Self {
        let ratio = section.ratio();
        let (lower, upper) = area_coefficient_bounds(ratio);
        let requested = section.area_coefficient();
        let area_coefficient = requested.clamp(lower, upper);
        let clamped = area_coefficient != requested;

        let u = 4.0 * area_coefficient / PI;
        let asymmetry = (ratio - 1.0) / (ratio + 1.0);
        let c1 = 3.0 + u + (1.0 - u) * asymmetry * asymmetry;

        // `9 - 2 c1` is non-negative for every area coefficient the bounds
        // above admit — the upper bound is precisely where it would vanish —
        // but the guard costs nothing and turns a hypothetical NaN into a
        // degenerate-but-finite form.
        let discriminant = (9.0 - 2.0 * c1).max(0.0);
        let a3 = (-c1 + 3.0 + discriminant.sqrt()) / c1;
        let a1 = asymmetry * (a3 + 1.0);
        let scale = 0.5 * section.beam / (1.0 + a1 + a3);

        Self {
            scale,
            a1,
            a3,
            area_coefficient,
            clamped,
        }
    }

    /// A point on the mapped contour, equation 7.88.
    ///
    /// `theta` runs from 0 at the keel to `π/2` at the waterline. Returns
    /// `(half_breadth, depth_below_waterline)`.
    #[must_use]
    pub fn contour(&self, theta: f64) -> (f64, f64) {
        let half_breadth =
            self.scale * ((1.0 + self.a1) * theta.sin() - self.a3 * (3.0 * theta).sin());
        let depth = self.scale * ((1.0 - self.a1) * theta.cos() + self.a3 * (3.0 * theta).cos());
        (half_breadth, depth)
    }

    /// The beam the fitted form actually has, m.
    #[must_use]
    pub fn beam(&self) -> f64 {
        2.0 * self.scale * (1.0 + self.a1 + self.a3)
    }

    /// The draft the fitted form actually has, m.
    #[must_use]
    pub fn draft(&self) -> f64 {
        self.scale * (1.0 - self.a1 + self.a3)
    }

    /// The area coefficient the mapping coefficients imply, equation 7.91.
    ///
    /// Independent of [`LewisForm::area_coefficient`], which is what went in.
    /// The two agreeing is the check that the fit solved its own equations; the
    /// two differing means the area was clamped.
    #[must_use]
    pub fn implied_area_coefficient(&self) -> f64 {
        PI / 4.0 * (1.0 - self.a1 * self.a1 - 3.0 * self.a3 * self.a3)
            / ((1.0 + self.a3) * (1.0 + self.a3) - self.a1 * self.a1)
    }
}

/// The immersed beam, draft and area of one station of a boat file.
///
/// `waterline_height` is measured up from the baseline, in the file frame, so
/// for an upright hull it is the canoe body draft.
///
/// Returns `None` for a station with nothing immersed, or one whose immersed
/// part has no breadth — the forward and after ends of a hull, where a Lewis
/// form means nothing and a division by zero is waiting.
///
/// The area is the trapezoidal integral of the half-breadths, doubled. Offsets
/// are ordered from the keel upwards, which the loader has already validated,
/// so no sorting is needed here.
#[must_use]
pub fn station_geometry(station: &Station, waterline_height: f64) -> Option<SectionGeometry> {
    let points = &station.points;
    if points.len() < 2 {
        return None;
    }

    let keel = points.first()?.z;
    if keel >= waterline_height {
        return None;
    }

    let mut area = 0.0_f64;
    let mut beam_half = 0.0_f64;
    let mut previous = points[0];

    for point in &points[1..] {
        if previous.z >= waterline_height {
            break;
        }
        // Interpolate onto the waterline when this pair straddles it, so the
        // integral stops exactly there rather than at the nearest offset.
        let (y, z) = if point.z > waterline_height {
            let span = point.z - previous.z;
            let fraction = if span > 0.0 {
                (waterline_height - previous.z) / span
            } else {
                0.0
            };
            (
                previous.y + fraction * (point.y - previous.y),
                waterline_height,
            )
        } else {
            (point.y, point.z)
        };

        area += 0.5 * (previous.y + y) * (z - previous.z);
        beam_half = beam_half.max(y);
        previous = crate::boat::Offset { y, z };
    }

    let beam = 2.0 * beam_half;
    let draft = waterline_height - keel;
    if beam <= 0.0 || draft <= 0.0 || area <= 0.0 {
        return None;
    }

    Some(SectionGeometry {
        beam,
        draft,
        area: 2.0 * area,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boat::Offset;
    use approx::assert_relative_eq;

    /// The source's own worked special case, and the sharpest possible check on
    /// the transcription: *"It is obvious that a transformation of a half
    /// immersed circle with radius R will result in `M_s = R`, `a_1 = 0` and
    /// `a_3 = 0`."*
    ///
    /// A half-immersed circle of radius `R` has beam `2R`, draft `R` and area
    /// `πR²/2`, so `H0 = 1` and `σ_s = π/4`. Every one of the five transcribed
    /// equations has to be right for all three outputs to land.
    #[test]
    fn a_half_immersed_circle_maps_to_the_circle_itself() {
        let radius = 1.7;
        let section = SectionGeometry {
            beam: 2.0 * radius,
            draft: radius,
            area: 0.5 * PI * radius * radius,
        };
        assert_relative_eq!(section.ratio(), 1.0, epsilon = 1e-12);
        assert_relative_eq!(section.area_coefficient(), PI / 4.0, epsilon = 1e-12);

        let form = LewisForm::fit(&section);
        assert!(!form.clamped);
        assert_relative_eq!(form.a1, 0.0, epsilon = 1e-12);
        assert_relative_eq!(form.a3, 0.0, epsilon = 1e-12);
        assert_relative_eq!(form.scale, radius, epsilon = 1e-12);
    }

    /// The fit is defined by reproducing beam, draft and area, so it has to.
    ///
    /// Checked across a spread of shapes — beamy and shallow, narrow and deep,
    /// fine and full — because a fit that happens to work at `H0 = 1` and
    /// nowhere else would pass the circle test above.
    #[test]
    fn the_fitted_form_reproduces_the_section_it_was_given() {
        for &(beam, draft, coefficient) in &[
            (3.0, 0.5, 0.70),
            (3.0, 0.5, 0.80),
            (1.2, 1.4, 0.75),
            (4.0, 0.4, 0.85),
            (2.0, 2.0, 0.72),
        ] {
            let section = SectionGeometry {
                beam,
                draft,
                area: coefficient * beam * draft,
            };
            let form = LewisForm::fit(&section);
            assert!(
                !form.clamped,
                "case B={beam} D={draft} sigma={coefficient} should be inside the envelope"
            );

            assert_relative_eq!(form.beam(), beam, max_relative = 1e-9);
            assert_relative_eq!(form.draft(), draft, max_relative = 1e-9);
            assert_relative_eq!(
                form.implied_area_coefficient(),
                coefficient,
                max_relative = 1e-9
            );
        }
    }

    /// The envelope is symmetric about `H0 = 1`, which the two branches of the
    /// lower bound only satisfy if both were transcribed correctly.
    #[test]
    fn the_envelope_is_symmetric_in_the_beam_draft_ratio() {
        for ratio in [0.25, 0.5, 0.8] {
            let (low_narrow, high_narrow) = area_coefficient_bounds(ratio);
            let (low_wide, high_wide) = area_coefficient_bounds(1.0 / ratio);
            assert_relative_eq!(low_narrow, low_wide, max_relative = 1e-12);
            assert_relative_eq!(high_narrow, high_wide, max_relative = 1e-12);
        }
    }

    /// A section too *fine* for the mapping is clamped to the border, as the
    /// source prescribes, and says so.
    ///
    /// Fine rather than full, and that is the interesting part. The upper bound
    /// of equation 7.95 has its minimum at `H0 = 1`, where it evaluates to
    /// `12π/32 = 1.178` — above the largest area coefficient any section can
    /// physically have, since the immersed area cannot exceed its bounding
    /// rectangle. So the upper bound never bites, and the only clamp that ever
    /// fires is the lower one. It fires on wedge-like sections, which is to say
    /// on the ends of a hull: a fine bow is exactly the shape a two-parameter
    /// map cannot reach.
    #[test]
    fn a_section_too_fine_to_map_is_clamped_and_reports_it() {
        let section = SectionGeometry {
            beam: 3.0,
            draft: 0.5,
            // Far finer than a wedge, which is already at the boundary.
            area: 0.30 * 3.0 * 0.5,
        };
        let form = LewisForm::fit(&section);
        assert!(form.clamped, "a knife-fine section must not fit silently");

        let (lower, _) = area_coefficient_bounds(section.ratio());
        assert_relative_eq!(form.area_coefficient, lower, max_relative = 1e-12);
        // Beam and draft still match; it is the area that had to give.
        assert_relative_eq!(form.beam(), section.beam, max_relative = 1e-9);
        assert_relative_eq!(form.draft(), section.draft, max_relative = 1e-9);
    }

    /// The upper bound of the envelope is unreachable by a real section, which
    /// is worth pinning so that nobody adds a clamp-to-full path for a case
    /// that cannot arise.
    #[test]
    fn no_physical_section_can_be_too_full_for_the_mapping() {
        for ratio in [0.2, 0.5, 1.0, 2.0, 5.0] {
            let (_, upper) = area_coefficient_bounds(ratio);
            assert!(
                upper > 1.0,
                "at H0 = {ratio} the upper bound is {upper}, which a rectangle could exceed"
            );
        }
    }

    /// The contour runs from the keel on the centreline to the waterline at the
    /// maximum half-breadth, which is what the parameter range means.
    #[test]
    fn the_contour_runs_from_keel_to_waterline() {
        let section = SectionGeometry {
            beam: 3.0,
            draft: 0.6,
            area: 0.78 * 3.0 * 0.6,
        };
        let form = LewisForm::fit(&section);

        let (keel_y, keel_z) = form.contour(0.0);
        assert_relative_eq!(keel_y, 0.0, epsilon = 1e-12);
        assert_relative_eq!(keel_z, form.draft(), max_relative = 1e-9);

        let (deck_y, deck_z) = form.contour(std::f64::consts::FRAC_PI_2);
        assert_relative_eq!(deck_y, 0.5 * form.beam(), max_relative = 1e-9);
        assert_relative_eq!(deck_z, 0.0, epsilon = 1e-12);
    }

    fn wedge_station() -> Station {
        // A triangular section: half-breadth grows linearly with height.
        Station {
            x: 5.0,
            points: (0..=10)
                .map(|step| {
                    let z = f64::from(step) * 0.1;
                    Offset { y: z, z }
                })
                .collect(),
        }
    }

    /// The integration is checked against a shape whose immersed area is known
    /// in closed form, including a waterline that falls between two offsets.
    #[test]
    fn station_geometry_integrates_a_wedge_exactly() {
        let station = wedge_station();
        // Waterline at 0.55, deliberately between offsets at 0.5 and 0.6.
        let geometry = station_geometry(&station, 0.55).expect("the wedge is immersed");

        assert_relative_eq!(geometry.draft, 0.55, epsilon = 1e-12);
        assert_relative_eq!(geometry.beam, 2.0 * 0.55, max_relative = 1e-12);
        // Triangle of half-width 0.55 and depth 0.55, both sides.
        assert_relative_eq!(geometry.area, 0.55 * 0.55, max_relative = 1e-12);
        assert_relative_eq!(geometry.area_coefficient(), 0.5, max_relative = 1e-12);
    }

    #[test]
    fn a_dry_station_has_no_geometry() {
        assert!(station_geometry(&wedge_station(), 0.0).is_none());
        assert!(station_geometry(&wedge_station(), -1.0).is_none());
    }
}
