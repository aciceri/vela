//! Scalar hull form parameters, read off the geometry.
//!
//! # Why this module exists
//!
//! [`crate::dsyhs`] consumes a hull as ten numbers and never looks at a
//! surface; [`crate::loft`] and [`crate::hydrostatics`] produce surfaces and
//! integrals over them. This is the bridge. Given a lofted hull and a
//! waterplane it derives exactly the parameter set the resistance regressions
//! want, so a boat file that carries offsets needs no declared coefficients —
//! and a file that carries both can be checked against itself, which is what
//! [`compare`] is for.
//!
//! Everything here is an **upright, level** quantity. That is not a
//! simplification: the series' form parameters are defined on the design
//! waterline, and the effect of heel enters its regressions as a separate
//! fitted correction rather than as re-measured coefficients.
//!
//! # Section areas without a triangulator
//!
//! A cross-section area is a planar integral, and the obvious route to one is
//! to intersect the hull with a transverse plane, order the resulting segments
//! into a loop, and integrate around it. The ordering step is the fragile,
//! tolerance-ridden one — it has to decide which segment ends touch.
//!
//! None of it is needed. [`crate::clip`] already cuts a mesh against a signed
//! depth function and hands back the cut segments, and Green's theorem
//! (`½ ∮ r × dr · n̂`) turns an unordered *set* of boundary segments into an
//! area, which is how [`enclosed_area`] gets the waterplane. So a section area
//! is two clips: clip the hull against the waterplane to get the immersed
//! surface, then clip that surface against the transverse plane and take the
//! enclosed area of the second set of cuts. No second clipper, no ordering, no
//! triangulation.
//!
//! # Why the origin sits on the waterplane
//!
//! The catch is that the immersed surface is **open**: the clipper deliberately
//! never generates the waterplane lid (the argument is in
//! [`crate::hydrostatics`]). The transverse cut of an open surface is therefore
//! an open arc running from one waterline point to the other, not a closed
//! loop, and Green's theorem over an open path `p₀ … p_k` returns the area of
//! the polygon closed by the straight segment `p_k → p₀` *minus* that segment's
//! own contribution `½ (p_k × p₀) · x̂`.
//!
//! Both ends of the arc lie on the waterplane. Written out, the missing
//! contribution is `½ (y_k z₀ − z_k y₀)`, so if the origin lies on the
//! waterplane as well then `z₀ = z_k = 0` and the term vanishes identically.
//! The chord costs nothing.
//!
//! That is why every integral here is evaluated in a frame translated so the
//! waterplane passes through the origin: the one shift that makes the missing
//! lid free for the volume integral (spanning tetrahedra from a point in the
//! plane) makes it free for every section integral too. In that frame body `z`
//! still points down, so an immersed point has `z > 0`.
//!
//! A hull with several immersed lobes at one station — a catamaran, or a fin
//! sliced across — yields several arcs, each with both ends on the waterplane,
//! and each closing chord vanishes for the same reason. The arcs inherit their
//! direction from the outward triangle winding, so their contributions add
//! instead of cancelling.

use crate::clip::{enclosed_area, Clipped};
use crate::geometry::{Point, Tri, TriMesh};
use nalgebra::Vector3;
use std::fmt;

/// Where the water is, and how finely the hull is sliced longitudinally.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FormOptions {
    /// Body-frame `z` of the waterplane, m.
    ///
    /// Body `z` points down, so a hull lofted with its keel on the baseline —
    /// which is what [`crate::loft`] produces — floating at draft `T` has its
    /// waterplane at `-T`. [`FormOptions::at_draft`] exists so that sign has to
    /// be got right only once.
    pub waterline_z: f64,
    /// Number of stations at which the section area is evaluated. Values below
    /// one are raised to one.
    pub station_count: usize,
}

/// Stations per hull for [`FormOptions::new`].
///
/// A floor chosen for the one quantity that is sensitive to it — the midship
/// area, which is a maximum over the sampled stations. Near a smooth peak the
/// sampling error is quadratic in the spacing, and a displacement hull's area
/// curve is flat over most of its middle body, so fifty stations put `A_m`
/// well inside a tenth of a per cent. It is a resolution, not a physical
/// constant; raise it if the prismatic coefficient has to be better than that.
pub const DEFAULT_STATION_COUNT: usize = 50;

impl FormOptions {
    /// Options for a waterplane at body-frame height `waterline_z`.
    #[must_use]
    pub fn new(waterline_z: f64) -> Self {
        Self {
            waterline_z,
            station_count: DEFAULT_STATION_COUNT,
        }
    }

    /// Options for a hull immersed to `draft`, measured from its deepest point.
    ///
    /// This is the reading of "draft" that [`crate::hydrostatics`] reports and
    /// that a boat's documentation means: the immersion of the lowest point of
    /// the hull, not the sinkage of the body origin. Returns `None` for an
    /// empty mesh, which has no deepest point.
    #[must_use]
    pub fn at_draft(mesh: &TriMesh, draft: f64) -> Option<Self> {
        let (_, deepest) = mesh.bounds()?;
        Some(Self::new(deepest.z - draft))
    }
}

/// The immersed cross-section area at a series of longitudinal stations.
///
/// Stations are the **midpoints** of equal intervals spanning the immersed
/// length, and never its ends. That is deliberate. A hull with an immersed
/// transom — most modern yachts — has a genuinely discontinuous area curve
/// there, jumping from the full transom area to zero across the station, so a
/// sample taken exactly on the end is not merely inaccurate but ill-posed: its
/// value depends on which side of the plane the clipper counts as inside.
/// Midpoints never land on the discontinuity, and the midpoint rule integrates
/// a linear area curve exactly, which is all the trapezoid rule over the
/// endpoints would have bought.
#[derive(Debug, Clone, PartialEq)]
pub struct SectionalAreaCurve {
    stations: Vec<f64>,
    areas: Vec<f64>,
    spacing: f64,
}

impl SectionalAreaCurve {
    /// Body-frame `x` of each station, m, ascending.
    #[must_use]
    pub fn stations(&self) -> &[f64] {
        &self.stations
    }

    /// Immersed cross-section area at each station, m².
    #[must_use]
    pub fn areas(&self) -> &[f64] {
        &self.areas
    }

    /// Longitudinal spacing of the stations, m.
    #[must_use]
    pub fn spacing(&self) -> f64 {
        self.spacing
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.areas.is_empty()
    }

    /// Largest sampled section area, m², or zero for an empty curve.
    ///
    /// On a hull whose true maximum falls at the very end of the immersed
    /// length — an immersed transom again — the sampled maximum sits half a
    /// station inboard of it and is correspondingly small.
    #[must_use]
    pub fn midship_area(&self) -> f64 {
        self.areas
            .iter()
            .copied()
            .fold(0.0, |best, area| if area > best { area } else { best })
    }

    /// Station at which [`Self::midship_area`] occurs, m.
    #[must_use]
    pub fn midship_station(&self) -> Option<f64> {
        let (index, _) = self
            .areas
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))?;
        Some(self.stations[index])
    }

    /// Displaced volume by integrating the curve, m³.
    ///
    /// An independent second opinion on the volume: the divergence theorem over
    /// the immersed panels gives the same number by an unrelated route, and the
    /// two agreeing is a real check on the section areas at *every* station
    /// rather than only at the maximum. Exact for any hull whose area curve is
    /// piecewise linear over the station spacing.
    #[must_use]
    pub fn integrated_volume(&self) -> f64 {
        self.spacing * self.areas.iter().sum::<f64>()
    }
}

/// A hull's scalar form parameters, in the field set the resistance model
/// consumes.
///
/// Deliberately a separate type from [`crate::dsyhs::HullParameters`] rather
/// than the same one: this is a *measurement* of a mesh, that is an *input* to
/// a regression, and the two want to be comparable without one depending on the
/// other. The fields line up one for one, plus [`Self::midship_area`], which
/// the regressions do not use but which is what the midship coefficient is
/// normalized from and is worth reporting rather than throwing away.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HullForm {
    /// Waterline length, m.
    pub waterline_length: f64,
    /// Waterline beam, m.
    pub waterline_beam: f64,
    /// Canoe body draft, m: the immersion of the deepest immersed point.
    pub canoe_draft: f64,
    /// Immersed volume, m³.
    pub canoe_volume: f64,
    /// Wetted hull area, m². Excludes the waterplane, which is not hull.
    pub wetted_surface: f64,
    /// Waterplane area, m².
    pub waterplane_area: f64,
    /// Midship (maximum) section area `A_m`, m².
    pub midship_area: f64,
    /// Prismatic coefficient, `V / (A_m · L_wl)`.
    pub prismatic: f64,
    /// Midship section coefficient, `A_m / (B_wl · T_c)`.
    pub midship: f64,
    /// Longitudinal centre of buoyancy, as a **fraction** of the waterline
    /// length from midship, positive forward. Yacht tables quote this as a
    /// percentage, so `-0.042` is a published `-4.2 %`.
    pub lcb: f64,
    /// Longitudinal centre of flotation, same convention as [`Self::lcb`].
    pub lcf: f64,
}

/// Why a hull's form parameters could not be measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormError {
    /// Nothing is immersed: the waterplane is at or below the hull, or the
    /// mesh is empty.
    Dry,
    /// The hull crosses the waterplane nowhere, so it has no waterline length,
    /// beam or centre of flotation. A wholly submerged hull lands here, and
    /// correctly so — these are surface-piercing quantities and it has none.
    NoWaterplane,
    /// Every sampled section came out empty, so there is no midship area to
    /// normalize the coefficients by. A mesh that encloses volume without any
    /// station cutting it is not a hull.
    NoMidshipSection,
}

impl fmt::Display for FormError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dry => write!(f, "no part of the hull is immersed"),
            Self::NoWaterplane => write!(
                f,
                "the hull does not cross the waterplane, so it has no waterline"
            ),
            Self::NoMidshipSection => {
                write!(f, "no station yielded an immersed cross-section")
            }
        }
    }
}

impl std::error::Error for FormError {}

/// Immersed cross-section area at one longitudinal station, m².
///
/// `mesh` is in the body frame and `waterline_z` is the body-frame height of
/// the waterplane, as in [`FormOptions`]. Stations outside the immersed length
/// return zero.
///
/// This clips the whole hull against the waterplane on every call. Use
/// [`sectional_area_curve`] for a series of stations, which does that once.
#[must_use]
pub fn section_area(mesh: &TriMesh, waterline_z: f64, station: f64) -> f64 {
    let hull = submerged(mesh, waterline_z);
    let mut scratch = Clipped::default();
    section_area_at(&hull.triangles, station, &mut scratch)
}

/// The sectional area curve of a hull, over its immersed length.
#[must_use]
pub fn sectional_area_curve(mesh: &TriMesh, options: &FormOptions) -> SectionalAreaCurve {
    let hull = submerged(mesh, options.waterline_z);
    sample_curve(&hull.triangles, options.station_count)
}

/// Derives a hull's scalar form parameters from its geometry.
///
/// # Errors
///
/// [`FormError`] if the hull is dry, wholly submerged, or degenerate enough
/// that a coefficient would be a division by zero. The parameters are reported
/// or they are not; a plausible-looking number from a hull that does not float
/// the way the caller assumed is worse than an error, because it propagates
/// silently into a resistance curve.
pub fn hull_form(mesh: &TriMesh, options: &FormOptions) -> Result<HullForm, FormError> {
    let hull = submerged(mesh, options.waterline_z);

    // One pass over the immersed panels for everything the surface integrals
    // give: volume and its moment by the divergence theorem (exact, with the
    // origin in the waterplane covering for the absent lid), wetted area, and
    // the deepest immersion.
    let mut canoe_volume = 0.0;
    let mut volume_moment = Point::zeros();
    let mut wetted_surface = 0.0;
    let mut canoe_draft = 0.0_f64;
    for tri in &hull.triangles {
        let tetrahedron = tri.signed_tetrahedron_volume();
        canoe_volume += tetrahedron;
        volume_moment += tetrahedron * (tri.a + tri.b + tri.c) / 4.0;
        wetted_surface += tri.area();
        canoe_draft = canoe_draft.max(tri.a.z).max(tri.b.z).max(tri.c.z);
    }
    if canoe_volume <= f64::EPSILON || canoe_draft <= f64::EPSILON {
        return Err(FormError::Dry);
    }

    // Waterline length and beam are measured on the **waterplane**, not on the
    // bounding box of the immersed volume. The two differ on a hull with a bulb,
    // and every ratio the series is fitted against is a waterplane quantity.
    let (aft, forward) = cut_extent(&hull.cuts).ok_or(FormError::NoWaterplane)?;
    let waterline_length = forward.x - aft.x;
    let waterline_beam = forward.y - aft.y;
    let waterplane_area = enclosed_area(&hull.cuts, Vector3::z());
    if waterline_length <= f64::EPSILON
        || waterline_beam <= f64::EPSILON
        || waterplane_area <= f64::EPSILON
    {
        return Err(FormError::NoWaterplane);
    }

    let midship_x = 0.5 * (aft.x + forward.x);
    let flotation_x = waterplane_centroid_x(&hull.cuts).ok_or(FormError::NoWaterplane)?;

    let curve = sample_curve(&hull.triangles, options.station_count);
    let midship_area = curve.midship_area();
    if midship_area <= f64::EPSILON {
        return Err(FormError::NoMidshipSection);
    }

    Ok(HullForm {
        waterline_length,
        waterline_beam,
        canoe_draft,
        canoe_volume,
        wetted_surface,
        waterplane_area,
        midship_area,
        prismatic: canoe_volume / (midship_area * waterline_length),
        midship: midship_area / (waterline_beam * canoe_draft),
        // Body x is forward and both centroids are already in body x, so the
        // sign follows from the frame with nothing to invert.
        lcb: (volume_moment.x / canoe_volume - midship_x) / waterline_length,
        lcf: (flotation_x - midship_x) / waterline_length,
    })
}

/// One parameter on which two descriptions of the same hull disagree.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FormDifference {
    /// Short label, in the notation [`crate::dsyhs`] reports violations in.
    pub parameter: &'static str,
    /// The value measured from the geometry.
    pub derived: f64,
    /// The value the file declared.
    pub declared: f64,
    /// Size of the disagreement, as a fraction.
    ///
    /// For the positive-definite parameters this is the difference over the
    /// **larger** of the two magnitudes, which keeps the measure symmetric and
    /// finite even when one side is zero. For `LCB` and `LCF` it is the plain
    /// difference: those are already fractions of the waterline length, so the
    /// difference is a relative measure as it stands, and dividing by a
    /// quantity that legitimately passes through zero would report a trivial
    /// disagreement as an infinite one.
    pub relative: f64,
}

/// Compares measured form parameters against declared ones.
///
/// Returns one entry per parameter whose disagreement exceeds `tolerance`,
/// in field order. An empty result means the two descriptions agree.
///
/// This is the check that makes carrying both offsets and coefficients in a
/// boat file worth anything: they describe the same hull twice, so a file whose
/// declared prismatic coefficient contradicts its own stations is wrong, and
/// which of the two is wrong is a question a human has to answer — hence a
/// report rather than a silent correction.
#[must_use]
pub fn compare(derived: &HullForm, declared: &HullForm, tolerance: f64) -> Vec<FormDifference> {
    // Fractions of length last, and flagged, because they are scored
    // differently — see `FormDifference::relative`.
    let checks: [(&'static str, f64, f64, bool); 11] = [
        (
            "Lwl",
            derived.waterline_length,
            declared.waterline_length,
            false,
        ),
        (
            "Bwl",
            derived.waterline_beam,
            declared.waterline_beam,
            false,
        ),
        ("Tc", derived.canoe_draft, declared.canoe_draft, false),
        ("Vc", derived.canoe_volume, declared.canoe_volume, false),
        ("S", derived.wetted_surface, declared.wetted_surface, false),
        (
            "Aw",
            derived.waterplane_area,
            declared.waterplane_area,
            false,
        ),
        ("Am", derived.midship_area, declared.midship_area, false),
        ("Cp", derived.prismatic, declared.prismatic, false),
        ("Cm", derived.midship, declared.midship, false),
        ("LCB", derived.lcb, declared.lcb, true),
        ("LCF", derived.lcf, declared.lcf, true),
    ];

    checks
        .into_iter()
        .map(|(parameter, measured, stated, is_fraction_of_length)| {
            let difference = (measured - stated).abs();
            let scale = measured.abs().max(stated.abs());
            let relative = if is_fraction_of_length || scale <= f64::EPSILON {
                difference
            } else {
                difference / scale
            };
            FormDifference {
                parameter,
                derived: measured,
                declared: stated,
                relative,
            }
        })
        .filter(|difference| difference.relative > tolerance)
        .collect()
}

/// Clips a hull to its immersed part, in the frame whose origin lies on the
/// waterplane.
///
/// Only `z` is translated, so station coordinates and the longitudinal
/// centroids stay in body-frame `x` and need no shifting back.
fn submerged(mesh: &TriMesh, waterline_z: f64) -> Clipped {
    let mut clipped = Clipped::default();
    let lift = Vector3::new(0.0, 0.0, waterline_z);
    clipped.extend_from(
        mesh.triangles()
            .map(|tri| Tri::new(tri.a - lift, tri.b - lift, tri.c - lift)),
        &|p: Point| p.z,
    );
    clipped
}

/// Samples the section area over the immersed length.
fn sample_curve(immersed: &[Tri], station_count: usize) -> SectionalAreaCurve {
    let count = station_count.max(1);
    let Some((aft, forward)) = triangle_extent(immersed) else {
        return SectionalAreaCurve {
            stations: Vec::new(),
            areas: Vec::new(),
            spacing: 0.0,
        };
    };

    let spacing = (forward.x - aft.x) / count as f64;
    let mut scratch = Clipped::default();
    let stations: Vec<f64> = (0..count)
        .map(|i| aft.x + (i as f64 + 0.5) * spacing)
        .collect();
    let areas = stations
        .iter()
        .map(|station| section_area_at(immersed, *station, &mut scratch))
        .collect();

    SectionalAreaCurve {
        stations,
        areas,
        spacing,
    }
}

/// Area of the immersed section at one station, given the already-clipped
/// immersed panels and a scratch buffer to cut them into.
fn section_area_at(immersed: &[Tri], station: f64, scratch: &mut Clipped) -> f64 {
    scratch.clear();
    // Keep the part aft of the station; which side is kept cannot matter to the
    // cut segments, and this way the depth function reads as a distance.
    let transverse = |p: Point| station - p.x;
    scratch.extend_from(immersed.iter().copied(), &transverse);
    enclosed_area(&scratch.cuts, Vector3::x())
}

/// Longitudinal centroid of the region bounded by a set of coplanar cut
/// segments, by the same Green's theorem as [`enclosed_area`].
///
/// The first moment of a polygon about the `y` axis is
/// `⅙ Σ (xᵢ + xᵢ₊₁)(xᵢ yᵢ₊₁ − xᵢ₊₁ yᵢ)`, and the bracket is the very quantity
/// `enclosed_area` sums, so this is that computation carrying one extra factor.
/// Signed area and signed moment are divided by each other, so an inverted
/// traversal cancels and no absolute value is needed — nor wanted, since a
/// centroid has a sign.
fn waterplane_centroid_x(cuts: &[[Point; 2]]) -> Option<f64> {
    let mut twice_area = 0.0;
    let mut six_moment = 0.0;
    for [p, q] in cuts {
        let cross = p.x * q.y - p.y * q.x;
        twice_area += cross;
        six_moment += (p.x + q.x) * cross;
    }
    if twice_area.abs() <= f64::EPSILON {
        return None;
    }
    Some(six_moment / (3.0 * twice_area))
}

/// Componentwise bounds of the endpoints of a set of cut segments.
fn cut_extent(cuts: &[[Point; 2]]) -> Option<(Point, Point)> {
    extent(cuts.iter().flat_map(|[p, q]| [*p, *q]))
}

/// Componentwise bounds of the vertices of a set of triangles.
fn triangle_extent(triangles: &[Tri]) -> Option<(Point, Point)> {
    extent(triangles.iter().flat_map(|tri| [tri.a, tri.b, tri.c]))
}

fn extent<I: Iterator<Item = Point>>(mut points: I) -> Option<(Point, Point)> {
    let first = points.next()?;
    Some(points.fold((first, first), |(low, high), p| (low.inf(&p), high.sup(&p))))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit square in the plane `z = 0`, as unordered, inconsistently
    /// directed segments — the shape the clipper actually delivers.
    #[test]
    fn waterplane_centroid_is_the_polygon_centroid() {
        let corner = |x: f64, y: f64| Point::new(x, y, 0.0);
        let cuts = [
            [corner(2.0, 0.0), corner(4.0, 0.0)],
            [corner(4.0, 0.0), corner(4.0, 1.0)],
            [corner(4.0, 1.0), corner(2.0, 1.0)],
            [corner(2.0, 1.0), corner(2.0, 0.0)],
        ];
        assert!((waterplane_centroid_x(&cuts).unwrap() - 3.0).abs() < 1e-12);
        assert!((enclosed_area(&cuts, Vector3::z()) - 2.0).abs() < 1e-12);

        // Traversed the other way the area is unchanged and so is the centroid,
        // because the moment flips sign with the area it is divided by.
        let reversed: Vec<[Point; 2]> = cuts.iter().rev().map(|[p, q]| [*q, *p]).collect();
        assert!((waterplane_centroid_x(&reversed).unwrap() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn a_degenerate_waterplane_has_no_centroid() {
        let segment = [Point::new(0.0, 0.0, 0.0), Point::new(1.0, 0.0, 0.0)];
        assert_eq!(waterplane_centroid_x(&[segment]), None);
    }

    #[test]
    fn an_empty_hull_gives_an_empty_curve() {
        let curve = sample_curve(&[], 8);
        assert!(curve.is_empty());
        assert_eq!(curve.midship_area(), 0.0);
        assert_eq!(curve.midship_station(), None);
        assert_eq!(curve.integrated_volume(), 0.0);
    }
}
