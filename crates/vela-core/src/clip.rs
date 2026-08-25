//! Clipping triangles against a free surface.
//!
//! The submerged part of the hull is found by cutting every triangle against
//! the water surface, given as a **signed depth** function: positive below the
//! surface, negative above, zero on it. That signature is the generalization
//! point — a flat waterplane and an FFT wave field differ only in the closure
//! passed here, so the hydrostatics of today and the wave loads of later share
//! this code path rather than duplicating it.
//!
//! Clipping is Sutherland-Hodgman against the half-space `depth ≥ 0`, which
//! handles all vertex sign combinations in one loop instead of an explicit case
//! analysis. A triangle yields a polygon of 0, 3 or 4 vertices; the latter is
//! fan-triangulated on the spot.

use crate::geometry::{Point, Tri};

/// The submerged remains of one triangle.
#[derive(Debug, Clone, Default)]
pub struct Clipped {
    /// Fully or partially submerged triangles, in the original winding.
    pub triangles: Vec<Tri>,
    /// The segments where triangles crossed the surface, oriented consistently
    /// with the triangle winding.
    ///
    /// Their union tiles the boundary of the waterplane, which is what makes
    /// the waterplane area computable by Green's theorem without ever ordering
    /// the loop or triangulating it.
    pub cuts: Vec<[Point; 2]>,
}

impl Clipped {
    /// Clips a whole mesh's worth of triangles, reusing one allocation.
    pub fn extend_from<I, F>(&mut self, triangles: I, depth: &F)
    where
        I: IntoIterator<Item = Tri>,
        F: Fn(Point) -> f64,
    {
        for tri in triangles {
            clip_triangle(&tri, depth, self);
        }
    }

    pub fn clear(&mut self) {
        self.triangles.clear();
        self.cuts.clear();
    }
}

/// Clips one triangle against `depth ≥ 0`, appending results to `out`.
pub fn clip_triangle<F>(tri: &Tri, depth: &F, out: &mut Clipped)
where
    F: Fn(Point) -> f64,
{
    let vertices = [tri.a, tri.b, tri.c];
    let depths = [depth(tri.a), depth(tri.b), depth(tri.c)];

    // Fast paths worth having: for a hull in still water most triangles are
    // wholly in or wholly out, and skipping the polygon machinery for them is
    // the difference between a cheap per-frame clip and a wasteful one.
    if depths.iter().all(|d| *d < 0.0) {
        return;
    }
    if depths.iter().all(|d| *d >= 0.0) {
        out.triangles.push(*tri);
        return;
    }

    // Sutherland-Hodgman. `is_new` marks vertices created on the surface.
    let mut polygon: Vec<Point> = Vec::with_capacity(4);
    let mut is_new: Vec<bool> = Vec::with_capacity(4);

    for i in 0..3 {
        let j = (i + 1) % 3;
        let (di, dj) = (depths[i], depths[j]);
        let (vi, vj) = (vertices[i], vertices[j]);

        if di >= 0.0 {
            polygon.push(vi);
            is_new.push(false);
        }
        // Sign change along this edge: insert the crossing point.
        if (di >= 0.0) != (dj >= 0.0) {
            let t = di / (di - dj);
            polygon.push(vi + t * (vj - vi));
            is_new.push(true);
        }
    }

    if polygon.len() < 3 {
        return;
    }

    for k in 1..polygon.len() - 1 {
        out.triangles
            .push(Tri::new(polygon[0], polygon[k], polygon[k + 1]));
    }

    // The two surface vertices are adjacent in the polygon by construction; the
    // edge between them, in traversal order, is the cut.
    let new_positions: Vec<usize> = (0..polygon.len()).filter(|i| is_new[*i]).collect();
    if let [first, second] = new_positions[..] {
        // Adjacent either directly or across the wrap.
        if second == first + 1 {
            out.cuts.push([polygon[first], polygon[second]]);
        } else {
            out.cuts.push([polygon[second], polygon[first]]);
        }
    }
}

/// Area enclosed by a set of cut segments lying in a plane with unit normal
/// `normal`.
///
/// Green's theorem: the boundary integral `½ ∮ r × dr` projected on the plane
/// normal gives the enclosed area, and because the sum is over all segments at
/// once it needs neither loop ordering nor a consistent global direction — the
/// magnitude is taken at the end.
#[must_use]
pub fn enclosed_area(cuts: &[[Point; 2]], normal: Point) -> f64 {
    let twice: f64 = cuts.iter().map(|[p, q]| p.cross(q).dot(&normal)).sum();
    0.5 * twice.abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Depth for a flat surface at `z = 0` with z pointing down: a point with
    /// positive z is submerged.
    fn flat(p: Point) -> f64 {
        p.z
    }

    #[test]
    fn a_fully_submerged_triangle_survives_unchanged() {
        let tri = Tri::new(
            Point::new(0.0, 0.0, 1.0),
            Point::new(1.0, 0.0, 1.0),
            Point::new(0.0, 1.0, 2.0),
        );
        let mut out = Clipped::default();
        clip_triangle(&tri, &flat, &mut out);
        assert_eq!(out.triangles, vec![tri]);
        assert!(out.cuts.is_empty());
    }

    #[test]
    fn a_dry_triangle_disappears() {
        let tri = Tri::new(
            Point::new(0.0, 0.0, -1.0),
            Point::new(1.0, 0.0, -1.0),
            Point::new(0.0, 1.0, -2.0),
        );
        let mut out = Clipped::default();
        clip_triangle(&tri, &flat, &mut out);
        assert!(out.triangles.is_empty());
        assert!(out.cuts.is_empty());
    }

    #[test]
    fn one_vertex_submerged_yields_one_triangle_of_the_right_area() {
        // Right triangle in the x-z plane, apex 1 below the surface, base 1
        // above: the submerged part is similar with half the linear scale.
        let tri = Tri::new(
            Point::new(0.0, 0.0, 1.0),
            Point::new(2.0, 0.0, -1.0),
            Point::new(0.0, 2.0, -1.0),
        );
        let mut out = Clipped::default();
        clip_triangle(&tri, &flat, &mut out);
        assert_eq!(out.triangles.len(), 1);
        assert_relative_eq!(out.triangles[0].area(), tri.area() * 0.25, epsilon = 1e-12);
        assert_eq!(out.cuts.len(), 1);
    }

    #[test]
    fn two_vertices_submerged_yield_a_quad_as_two_triangles() {
        let tri = Tri::new(
            Point::new(0.0, 0.0, 1.0),
            Point::new(2.0, 0.0, 1.0),
            Point::new(0.0, 0.0, -1.0),
        );
        let mut out = Clipped::default();
        clip_triangle(&tri, &flat, &mut out);
        assert_eq!(out.triangles.len(), 2);
        // The dry corner is the similar triangle of quarter area.
        let submerged: f64 = out.triangles.iter().map(Tri::area).sum();
        assert_relative_eq!(submerged, tri.area() * 0.75, epsilon = 1e-12);
    }

    #[test]
    fn clipping_preserves_winding() {
        let tri = Tri::new(
            Point::new(0.0, 0.0, 1.0),
            Point::new(2.0, 0.0, -1.0),
            Point::new(0.0, 2.0, -1.0),
        );
        let mut out = Clipped::default();
        clip_triangle(&tri, &flat, &mut out);
        // Normals of the parts must point the same way as the original, or the
        // pressure integral changes sign on partially wet panels.
        let original = tri.area_normal().normalize();
        for part in &out.triangles {
            assert_relative_eq!(part.area_normal().normalize(), original, epsilon = 1e-9);
        }
    }

    #[test]
    fn waterplane_area_of_a_clipped_box_matches_its_cross_section() {
        // A box from z = -1 (above water) to z = 1 (below), 2 by 3 in plan.
        // Cutting it at z = 0 must expose a 2 by 3 waterplane.
        let (lo, hi) = (Point::new(0.0, 0.0, -1.0), Point::new(2.0, 3.0, 1.0));
        let corners = |z: f64| {
            [
                Point::new(lo.x, lo.y, z),
                Point::new(hi.x, lo.y, z),
                Point::new(hi.x, hi.y, z),
                Point::new(lo.x, hi.y, z),
            ]
        };
        let bottom = corners(hi.z);
        let top = corners(lo.z);

        let mut mesh = Vec::new();
        // Four sides, each a quad split into two triangles, wound outward.
        for i in 0..4 {
            let j = (i + 1) % 4;
            mesh.push(Tri::new(top[i], bottom[i], bottom[j]));
            mesh.push(Tri::new(top[i], bottom[j], top[j]));
        }

        let mut out = Clipped::default();
        out.extend_from(mesh, &flat);
        assert_relative_eq!(
            enclosed_area(&out.cuts, Point::new(0.0, 0.0, 1.0)),
            6.0,
            epsilon = 1e-12
        );
    }
}
