//! Lofting station offsets into a closed physics mesh.
//!
//! # Why a mesh at all
//!
//! Sections are the right storage format, but buoyancy and wave loads are
//! surface integrals, and a surface integral needs a surface. This module builds
//! it once at load time, at a resolution chosen for physics rather than for
//! looks — the frontend's visual hull is a different, finer, unrelated mesh.
//!
//! # Construction
//!
//! Each station contour is subdivided to a common point count (original
//! offsets preserved verbatim, see [`refine`]) and turned into a **closed
//! ring**: up the starboard
//! side from keel to deck edge, across the deck, down the port side. Rings are
//! then lofted into a tube.
//!
//! The ends are closed by prepending and appending rings collapsed onto the
//! centroid of the first and last sections. That is the whole trick: caps come
//! out of the same loft loop as everything else, so their winding cannot
//! disagree with the sides. Cap orientation is a classic source of
//! silently-wrong volumes, and this construction deletes the possibility rather
//! than testing for it. The degenerate quads collapse into zero-area triangles,
//! which are dropped at the end.
//!
//! # Deliberate simplifications
//!
//! No longitudinal refinement: the mesh has exactly as many rings as the file
//! has stations, because interpolating between stations needs a longitudinal
//! spline and a wrong spline is worse than a coarse mesh. The deck is flat
//! between port and starboard deck edges — camber only matters at heel angles
//! where the deck edge is already immersed.

use crate::boat::{HullSpec, Offset, Station};
use crate::geometry::{Point, TriMesh};

/// Mesh resolution controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoftOptions {
    /// Points per half-contour after resampling, keel to deck edge inclusive.
    pub points_per_station: usize,
}

impl Default for LoftOptions {
    fn default() -> Self {
        // Enough to resolve a bilge radius on a yacht section without making
        // the per-frame clip expensive. Revisit against measured frame times,
        // not intuition.
        Self {
            points_per_station: 16,
        }
    }
}

/// Lofts a hull into a closed, outward-oriented mesh in the **body frame**.
///
/// Expects a hull that has already passed [`HullSpec::validate`]; with fewer
/// than two stations it returns an empty mesh rather than panicking.
#[must_use]
pub fn loft_hull(hull: &HullSpec, options: &LoftOptions) -> TriMesh {
    if hull.stations.len() < 2 {
        return TriMesh::default();
    }
    // The requested resolution is a floor, not a target: every station must
    // yield the same ring length, and no station may lose one of its own
    // offsets. The busiest section therefore sets the count for all of them.
    let busiest = hull
        .stations
        .iter()
        .map(|station| station.points.len())
        .max()
        .unwrap_or(2);
    let per_station = options.points_per_station.max(busiest).max(2);

    let rings: Vec<Vec<Point>> = hull
        .stations
        .iter()
        .map(|station| ring(station, per_station))
        .collect();

    let ring_len = rings[0].len();
    let mut vertices: Vec<Point> = Vec::with_capacity(ring_len * (rings.len() + 2));
    let mut indices: Vec<[u32; 3]> = Vec::new();

    // Degenerate opening ring: every vertex at the centroid of the first
    // section. Same for the closing ring at the other end.
    let push_collapsed = |vertices: &mut Vec<Point>, ring: &[Point]| {
        let centre = ring.iter().sum::<Point>() / ring.len() as f64;
        vertices.extend(std::iter::repeat_n(centre, ring.len()));
    };

    push_collapsed(&mut vertices, &rings[0]);
    for r in &rings {
        vertices.extend_from_slice(r);
    }
    push_collapsed(&mut vertices, &rings[rings.len() - 1]);

    let ring_count = rings.len() + 2;
    for i in 0..ring_count - 1 {
        let a = (i * ring_len) as u32;
        let b = ((i + 1) * ring_len) as u32;
        for j in 0..ring_len {
            let jn = ((j + 1) % ring_len) as u32;
            let j = j as u32;
            indices.push([a + j, a + jn, b + jn]);
            indices.push([a + j, b + jn, b + j]);
        }
    }

    let mut mesh = TriMesh::new(vertices, indices);
    mesh = drop_degenerate(&mesh);
    mesh.orient_outward();
    mesh
}

/// Builds one closed ring in the body frame from a station contour.
///
/// Starboard is `+y` in the body frame and the file stores non-negative
/// half-breadths, so the starboard side takes `+y` and the port side `-y`. File
/// heights are measured up from the baseline while body `z` points down, hence
/// the negation.
fn ring(station: &Station, per_station: usize) -> Vec<Point> {
    let contour = refine(&station.points, per_station);
    let mut ring = Vec::with_capacity(2 * per_station - 1);

    // Keel to deck edge on starboard.
    for offset in &contour {
        ring.push(Point::new(station.x, offset.y, -offset.z));
    }
    // Port deck edge back down to just above the keel. Only the keel point is
    // skipped: it sits on the centerline and both sides share it. Skipping the
    // port deck edge as well would cut the corner and leave the deck open.
    for offset in contour.iter().rev().take(per_station - 1) {
        ring.push(Point::new(station.x, -offset.y, -offset.z));
    }
    ring
}

/// Refines a contour to exactly `count` points by **subdividing** it, never by
/// resampling it.
///
/// Every offset the file specifies survives verbatim; the extra points are
/// distributed among the segments in proportion to their length, with the
/// leftovers going to the longest segments.
///
/// The distinction is not cosmetic. Even arc-length resampling drops any corner
/// that does not happen to land on a sample, so a flat bottom meeting a
/// vertical side, or any hard chine, gets quietly rounded off and the hull
/// loses displacement — measurably: a rectangular barge came out 4 % light at
/// 16 points per station, and the error only vanished as the resolution grew.
/// Subdividing reproduces the given polyline exactly at every resolution, so a
/// prismatic hull is exact even at the coarsest setting.
fn refine(points: &[Offset], count: usize) -> Vec<Offset> {
    debug_assert!(count >= points.len());
    debug_assert!(points.len() >= 2);

    let lengths: Vec<f64> = points
        .windows(2)
        .map(|pair| {
            let (a, b) = (pair[0], pair[1]);
            ((b.y - a.y).powi(2) + (b.z - a.z).powi(2)).sqrt()
        })
        .collect();
    let total: f64 = lengths.iter().sum();
    let mut extras = vec![0usize; lengths.len()];
    let budget = count - points.len();

    if budget > 0 {
        if total > 0.0 {
            // Proportional allocation, then largest-remainder for the rounding
            // leftovers so the counts sum to the budget exactly.
            let mut remainders: Vec<(f64, usize)> = Vec::with_capacity(lengths.len());
            let mut assigned = 0usize;
            for (i, length) in lengths.iter().enumerate() {
                let exact = budget as f64 * length / total;
                extras[i] = exact.floor() as usize;
                assigned += extras[i];
                remainders.push((exact - exact.floor(), i));
            }
            remainders.sort_by(|a, b| b.0.total_cmp(&a.0));
            for (_, i) in remainders.into_iter().take(budget - assigned) {
                extras[i] += 1;
            }
        } else {
            extras[0] = budget;
        }
    }

    let mut out = Vec::with_capacity(count);
    for (i, extra) in extras.iter().enumerate() {
        out.push(points[i]);
        let (a, b) = (points[i], points[i + 1]);
        for step in 1..=*extra {
            let t = step as f64 / (*extra + 1) as f64;
            out.push(Offset {
                y: a.y + t * (b.y - a.y),
                z: a.z + t * (b.z - a.z),
            });
        }
    }
    out.push(points[points.len() - 1]);
    out
}

/// Rebuilds the mesh without triangles of negligible area.
///
/// The collapsed end rings generate one degenerate triangle per quad. They
/// contribute exactly nothing to any integral, but they inflate the triangle
/// count that the per-frame clip walks.
fn drop_degenerate(mesh: &TriMesh) -> TriMesh {
    let kept: Vec<[u32; 3]> = mesh
        .indices()
        .iter()
        .enumerate()
        .filter(|(i, _)| mesh.triangle(*i).area() > 1e-12)
        .map(|(_, indices)| *indices)
        .collect();
    TriMesh::new(mesh.vertices().to_vec(), kept)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boat::Station;
    use approx::assert_relative_eq;

    /// A rectangular barge: constant rectangular section, so every hydrostatic
    /// quantity has a closed form.
    fn barge(length: f64, half_beam: f64, depth: f64) -> HullSpec {
        let section = vec![
            Offset { y: 0.0, z: 0.0 },
            Offset {
                y: half_beam,
                z: 0.0,
            },
            Offset {
                y: half_beam,
                z: depth,
            },
        ];
        HullSpec {
            stations: vec![
                Station {
                    x: 0.0,
                    points: section.clone(),
                },
                Station {
                    x: length,
                    points: section,
                },
            ],
        }
    }

    #[test]
    fn a_barge_lofts_to_its_analytic_volume() {
        let mesh = loft_hull(&barge(10.0, 1.5, 2.0), &LoftOptions::default());
        let (volume, centroid) = mesh.volume_centroid();
        assert_relative_eq!(volume, 10.0 * 3.0 * 2.0, epsilon = 1e-9);
        // Centroid amidships, on the centerline, half way up: body z is down,
        // so half the depth above the baseline is z = -1.
        assert_relative_eq!(centroid.x, 5.0, epsilon = 1e-9);
        assert_relative_eq!(centroid.y, 0.0, epsilon = 1e-9);
        assert_relative_eq!(centroid.z, -1.0, epsilon = 1e-9);
    }

    #[test]
    fn volume_does_not_depend_on_resolution() {
        // A prismatic hull is represented exactly at any resolution; if it is
        // not, the resampling is losing or duplicating contour length.
        let hull = barge(10.0, 1.5, 2.0);
        let coarse = loft_hull(
            &hull,
            &LoftOptions {
                points_per_station: 4,
            },
        );
        let fine = loft_hull(
            &hull,
            &LoftOptions {
                points_per_station: 64,
            },
        );
        assert_relative_eq!(coarse.signed_volume(), fine.signed_volume(), epsilon = 1e-9);
    }

    #[test]
    fn the_mesh_is_outward_oriented() {
        let mut mesh = loft_hull(&barge(6.0, 1.0, 1.0), &LoftOptions::default());
        assert!(mesh.signed_volume() > 0.0);
        // Already correct, so orienting again changes nothing.
        assert!(!mesh.orient_outward());
    }

    #[test]
    fn a_closed_mesh_has_no_net_area_normal() {
        // The strongest cheap watertightness check available: for any closed
        // surface the area-weighted normals cancel exactly. A hole, a missing
        // cap or a flipped triangle leaves a residual.
        let mesh = loft_hull(&barge(10.0, 1.5, 2.0), &LoftOptions::default());
        let residual: Point = mesh.triangles().map(|t| t.area_normal()).sum();
        assert_relative_eq!(residual, Point::zeros(), epsilon = 1e-9);
    }

    #[test]
    fn a_tapered_hull_is_also_closed() {
        // Tapering to a point at the bow exercises the collapsed-ring path in
        // the middle of the hull rather than only at the caps.
        let hull = HullSpec {
            stations: vec![
                Station {
                    x: 0.0,
                    points: vec![
                        Offset { y: 0.0, z: 0.0 },
                        Offset { y: 1.5, z: 0.2 },
                        Offset { y: 1.6, z: 1.5 },
                    ],
                },
                Station {
                    x: 5.0,
                    points: vec![
                        Offset { y: 0.0, z: 0.0 },
                        Offset { y: 1.8, z: 0.3 },
                        Offset { y: 1.9, z: 1.5 },
                    ],
                },
                Station {
                    x: 10.0,
                    points: vec![
                        Offset { y: 0.0, z: 0.4 },
                        Offset { y: 0.05, z: 0.9 },
                        Offset { y: 0.1, z: 1.5 },
                    ],
                },
            ],
        };
        let mesh = loft_hull(&hull, &LoftOptions::default());
        let residual: Point = mesh.triangles().map(|t| t.area_normal()).sum();
        assert_relative_eq!(residual, Point::zeros(), epsilon = 1e-9);
        assert!(mesh.signed_volume() > 0.0);
    }

    #[test]
    fn refining_preserves_every_original_offset() {
        // The regression this whole function exists for: a hard corner between
        // a flat bottom and a vertical side must survive refinement. Even
        // arc-length resampling chords across it and the hull loses volume.
        let points = vec![
            Offset { y: 0.0, z: 0.0 },
            Offset { y: 1.5, z: 0.0 },
            Offset { y: 1.5, z: 2.0 },
        ];
        let out = refine(&points, 16);
        assert_eq!(out.len(), 16);
        for original in &points {
            assert!(
                out.iter()
                    .any(|p| (p.y - original.y).abs() < 1e-12 && (p.z - original.z).abs() < 1e-12),
                "offset ({}, {}) was lost in refinement",
                original.y,
                original.z
            );
        }
    }

    #[test]
    fn refining_preserves_the_endpoints() {
        let points = vec![
            Offset { y: 0.0, z: 0.0 },
            Offset { y: 1.0, z: 0.1 },
            Offset { y: 1.4, z: 2.0 },
        ];
        let out = refine(&points, 9);
        assert_eq!(out.len(), 9);
        assert_relative_eq!(out[0].y, 0.0);
        assert_relative_eq!(out[0].z, 0.0);
        assert_relative_eq!(out[8].y, 1.4);
        assert_relative_eq!(out[8].z, 2.0);
        // Monotone in z, like the input.
        assert!(out.windows(2).all(|w| w[1].z >= w[0].z - 1e-12));
    }

    #[test]
    fn refining_spaces_points_evenly_along_a_straight_contour() {
        let points = vec![Offset { y: 0.0, z: 0.0 }, Offset { y: 0.0, z: 4.0 }];
        let out = refine(&points, 5);
        for (i, offset) in out.iter().enumerate() {
            assert_relative_eq!(offset.z, i as f64, epsilon = 1e-12);
        }
    }
}
