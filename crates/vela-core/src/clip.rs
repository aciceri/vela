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
    /// Pressure head at each vertex of each triangle in [`Clipped::triangles`],
    /// when the caller asked for it.
    ///
    /// Empty, or exactly as long as `triangles`. Filled by
    /// [`Clipped::extend_from_indexed`] and left empty by
    /// [`Clipped::extend_from`], because a clip against a bare plane has no
    /// pressure to carry.
    pub heads: Vec<[f64; 3]>,
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

    /// Clips an indexed mesh whose vertex depths are already known.
    ///
    /// The reason to prefer this over [`Clipped::extend_from`] is arithmetic that
    /// is not done rather than arithmetic that is done faster. A hull mesh shares
    /// each vertex between five or six triangles, so a per-triangle closure asks
    /// the surface the same question six times over — and for a seaway that
    /// question is a sum over every wave component, which is the most expensive
    /// thing in the step. Same clip, same cut points, one evaluation per vertex.
    ///
    /// # Panics
    ///
    /// If `depths` or `heads` is shorter than `vertices`, or an index is out of
    /// range: both are caller bugs rather than conditions to handle.
    pub fn extend_from_indexed(
        &mut self,
        vertices: &[Point],
        indices: &[[u32; 3]],
        depths: &[f64],
        heads: &[f64],
    ) {
        assert!(
            depths.len() >= vertices.len() && heads.len() >= vertices.len(),
            "a depth and a head are needed for every vertex"
        );
        for triangle in indices {
            let [i, j, k] = triangle.map(|index| index as usize);
            let tri = Tri::new(vertices[i], vertices[j], vertices[k]);
            clip_triangle_loaded(
                &tri,
                [depths[i], depths[j], depths[k]],
                [heads[i], heads[j], heads[k]],
                self,
            );
        }
    }

    pub fn clear(&mut self) {
        self.triangles.clear();
        self.cuts.clear();
        self.heads.clear();
    }
}

/// Clips one triangle against `depth ≥ 0`, appending results to `out`.
pub fn clip_triangle<F>(tri: &Tri, depth: &F, out: &mut Clipped)
where
    F: Fn(Point) -> f64,
{
    clip_triangle_with_depths(tri, [depth(tri.a), depth(tri.b), depth(tri.c)], out);
}

/// Clips one triangle whose three vertex depths are already known.
pub fn clip_triangle_with_depths(tri: &Tri, depths: [f64; 3], out: &mut Clipped) {
    clip_one(tri, depths, None, out);
}

/// Clips one triangle, carrying a pressure head through with each vertex.
///
/// The heads land in [`Clipped::heads`], parallel to [`Clipped::triangles`], so
/// an integrator downstream never has to ask the surface a second time.
///
/// Vertices the surface *creates* — the cut points — are given a head of zero,
/// and that is exact rather than convenient: a cut point lies on the surface by
/// construction, and a free surface carries atmospheric pressure. See
/// [`crate::seaway::Seaway::pressure_head`], whose stretching exists to make
/// that identity hold to machine precision.
pub fn clip_triangle_loaded(tri: &Tri, depths: [f64; 3], heads: [f64; 3], out: &mut Clipped) {
    clip_one(tri, depths, Some(heads), out);
}

/// The clip itself. `heads` present means the caller wants them carried.
fn clip_one(tri: &Tri, depths: [f64; 3], heads: Option<[f64; 3]>, out: &mut Clipped) {
    let vertices = [tri.a, tri.b, tri.c];

    // Fast paths worth having: for a hull in still water most triangles are
    // wholly in or wholly out, and skipping the polygon machinery for them is
    // the difference between a cheap per-frame clip and a wasteful one.
    if depths.iter().all(|d| *d < 0.0) {
        return;
    }
    if depths.iter().all(|d| *d >= 0.0) {
        out.triangles.push(*tri);
        if let Some(heads) = heads {
            out.heads.push(heads);
        }
        return;
    }

    // Sutherland-Hodgman. A head of `None` marks a vertex the surface created.
    let mut polygon: Vec<Point> = Vec::with_capacity(4);
    let mut carried: Vec<Option<f64>> = Vec::with_capacity(4);

    for i in 0..3 {
        let j = (i + 1) % 3;
        let (di, dj) = (depths[i], depths[j]);
        let (vi, vj) = (vertices[i], vertices[j]);

        if di >= 0.0 {
            polygon.push(vi);
            carried.push(Some(heads.map_or(0.0, |it| it[i])));
        }
        // Sign change along this edge: insert the crossing point.
        if (di >= 0.0) != (dj >= 0.0) {
            let t = di / (di - dj);
            polygon.push(vi + t * (vj - vi));
            carried.push(None);
        }
    }

    if polygon.len() < 3 {
        return;
    }

    // A cut point sits on the surface, so its head is zero.
    let head_at = |k: usize| carried[k].unwrap_or(0.0);
    for k in 1..polygon.len() - 1 {
        out.triangles
            .push(Tri::new(polygon[0], polygon[k], polygon[k + 1]));
        if heads.is_some() {
            out.heads.push([head_at(0), head_at(k), head_at(k + 1)]);
        }
    }

    // The two surface vertices are adjacent in the polygon by construction; the
    // edge between them, in traversal order, is the cut.
    let new_positions: Vec<usize> = (0..polygon.len())
        .filter(|i| carried[*i].is_none())
        .collect();
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

    /// The indexed clip exists only to avoid repeated surface evaluations, so it
    /// has to agree with the closure clip triangle for triangle. Run over a
    /// tilted, shared-vertex mesh straddling the surface, because that is where
    /// the two differ if the index bookkeeping is wrong: a mesh whose triangles
    /// were all wholly in or wholly out would pass with the polygon code broken.
    #[test]
    fn the_indexed_clip_matches_the_closure_clip() {
        // A fan of triangles around a shared apex, tilted so that the rim crosses
        // z = 0 at several angles.
        let mut vertices = vec![Point::new(0.0, 0.0, 0.6)];
        let rim = 12;
        for i in 0..rim {
            let angle = std::f64::consts::TAU * f64::from(i) / f64::from(rim);
            vertices.push(Point::new(
                2.0 * angle.cos(),
                2.0 * angle.sin(),
                1.4 * angle.sin() - 0.3,
            ));
        }
        let indices: Vec<[u32; 3]> = (0..rim).map(|i| [0, 1 + i, 1 + (i + 1) % rim]).collect();

        let mut closure_clip = Clipped::default();
        closure_clip.extend_from(
            indices.iter().map(|[a, b, c]| {
                Tri::new(
                    vertices[*a as usize],
                    vertices[*b as usize],
                    vertices[*c as usize],
                )
            }),
            &flat,
        );

        let depths: Vec<f64> = vertices.iter().map(|it| flat(*it)).collect();
        let mut indexed = Clipped::default();
        indexed.extend_from_indexed(&vertices, &indices, &depths, &depths);

        assert_eq!(indexed.triangles, closure_clip.triangles);
        assert_eq!(indexed.cuts, closure_clip.cuts);
        // Something crossed, or the comparison above proved nothing.
        assert!(!indexed.cuts.is_empty(), "no triangle met the surface");

        // Heads come through parallel to the triangles, and the ones the surface
        // created read zero: for this flat surface the head *is* the depth, so a
        // cut point's zero is the physical answer and not a placeholder.
        assert_eq!(indexed.heads.len(), indexed.triangles.len());
        for (tri, heads) in indexed.triangles.iter().zip(&indexed.heads) {
            for (vertex, head) in [tri.a, tri.b, tri.c].iter().zip(heads) {
                assert_relative_eq!(*head, vertex.z, epsilon = 1e-12);
            }
        }
    }
}
