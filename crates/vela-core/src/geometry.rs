//! Triangle meshes and the integral quantities read off them.
//!
//! Volumes and centroids come from the divergence theorem, evaluated as a sum
//! of signed tetrahedra spanned from the origin. This is exact for a closed,
//! consistently oriented mesh — no approximation, no quadrature — which is why
//! the hydrostatics can be checked against closed-form shapes to machine
//! precision.
//!
//! The choice of origin matters and is exploited deliberately: see
//! [`crate::hydrostatics`].

use nalgebra::Vector3;

/// A point or vector in whichever frame the caller is working in. Meshes in
/// this engine are built and stored in the **body frame**.
pub type Point = Vector3<f64>;

/// A single triangle, carried by value.
///
/// Clipping produces triangles that do not exist in any mesh, so the primitive
/// has to stand alone rather than being an index triple.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tri {
    pub a: Point,
    pub b: Point,
    pub c: Point,
}

impl Tri {
    #[must_use]
    pub fn new(a: Point, b: Point, c: Point) -> Self {
        Self { a, b, c }
    }

    /// The area-weighted outward normal, `½ (b-a) × (c-a)`.
    ///
    /// Returned unnormalized because every integral here wants `n dA`, not `n`.
    /// Normalizing and re-multiplying would just add a square root and a
    /// division by zero on degenerate triangles.
    #[must_use]
    pub fn area_normal(&self) -> Point {
        0.5 * (self.b - self.a).cross(&(self.c - self.a))
    }

    #[must_use]
    pub fn area(&self) -> f64 {
        self.area_normal().norm()
    }

    #[must_use]
    pub fn centroid(&self) -> Point {
        (self.a + self.b + self.c) / 3.0
    }

    /// Signed volume of the tetrahedron spanned by the origin and this
    /// triangle. Positive when the triangle's winding faces away from the
    /// origin.
    #[must_use]
    pub fn signed_tetrahedron_volume(&self) -> f64 {
        self.a.dot(&self.b.cross(&self.c)) / 6.0
    }

    /// Reverses the winding, flipping the normal.
    #[must_use]
    pub fn flipped(&self) -> Self {
        Self {
            a: self.a,
            b: self.c,
            c: self.b,
        }
    }
}

/// An indexed triangle mesh.
///
/// Closed and outward-oriented by contract — [`TriMesh::orient_outward`]
/// enforces the second part, and the first is the responsibility of whatever
/// builds it (see [`crate::loft`]).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TriMesh {
    vertices: Vec<Point>,
    indices: Vec<[u32; 3]>,
}

impl TriMesh {
    #[must_use]
    pub fn new(vertices: Vec<Point>, indices: Vec<[u32; 3]>) -> Self {
        Self { vertices, indices }
    }

    #[must_use]
    pub fn vertices(&self) -> &[Point] {
        &self.vertices
    }

    #[must_use]
    pub fn indices(&self) -> &[[u32; 3]] {
        &self.indices
    }

    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }

    /// The `i`-th triangle. Panics if `i` is out of range, like any indexing
    /// operation.
    #[must_use]
    pub fn triangle(&self, i: usize) -> Tri {
        let [x, y, z] = self.indices[i];
        Tri::new(
            self.vertices[x as usize],
            self.vertices[y as usize],
            self.vertices[z as usize],
        )
    }

    pub fn triangles(&self) -> impl Iterator<Item = Tri> + '_ {
        (0..self.triangle_count()).map(|i| self.triangle(i))
    }

    /// Enclosed volume, negative if the mesh is wound inward.
    #[must_use]
    pub fn signed_volume(&self) -> f64 {
        self.triangles()
            .map(|t| t.signed_tetrahedron_volume())
            .sum()
    }

    /// Enclosed volume and the centroid of that volume.
    ///
    /// The centroid is the tetrahedron-volume-weighted mean of tetrahedron
    /// centroids, which is exact rather than an approximation by surface
    /// centroids. Returns a zero centroid for a degenerate (zero-volume) mesh
    /// rather than dividing by zero.
    #[must_use]
    pub fn volume_centroid(&self) -> (f64, Point) {
        let mut volume = 0.0;
        let mut moment = Point::zeros();
        for tri in self.triangles() {
            let v = tri.signed_tetrahedron_volume();
            volume += v;
            // Centroid of the tetrahedron (origin, a, b, c).
            moment += v * (tri.a + tri.b + tri.c) / 4.0;
        }
        if volume.abs() < f64::EPSILON {
            return (volume, Point::zeros());
        }
        (volume, moment / volume)
    }

    #[must_use]
    pub fn surface_area(&self) -> f64 {
        self.triangles().map(|t| t.area()).sum()
    }

    /// Flips every triangle if the mesh encloses a negative volume.
    ///
    /// Winding is a single global degree of freedom for a uniformly constructed
    /// mesh, so correcting its sign once is enough; a locally inconsistent mesh
    /// is a builder bug that this cannot and should not paper over. Returns
    /// whether a flip happened.
    pub fn orient_outward(&mut self) -> bool {
        if self.signed_volume() >= 0.0 {
            return false;
        }
        for tri in &mut self.indices {
            tri.swap(1, 2);
        }
        true
    }

    /// Axis-aligned bounds, or `None` for an empty mesh.
    #[must_use]
    pub fn bounds(&self) -> Option<(Point, Point)> {
        let mut iter = self.vertices.iter();
        let first = *iter.next()?;
        Some(iter.fold((first, first), |(lo, hi), v| (lo.inf(v), hi.sup(v))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// An axis-aligned box, outward oriented, spanning `lo..hi`.
    fn unit_box(lo: Point, hi: Point) -> TriMesh {
        let v = vec![
            Point::new(lo.x, lo.y, lo.z),
            Point::new(hi.x, lo.y, lo.z),
            Point::new(hi.x, hi.y, lo.z),
            Point::new(lo.x, hi.y, lo.z),
            Point::new(lo.x, lo.y, hi.z),
            Point::new(hi.x, lo.y, hi.z),
            Point::new(hi.x, hi.y, hi.z),
            Point::new(lo.x, hi.y, hi.z),
        ];
        let indices = vec![
            [0, 2, 1],
            [0, 3, 2], // z = lo
            [4, 5, 6],
            [4, 6, 7], // z = hi
            [0, 1, 5],
            [0, 5, 4], // y = lo
            [3, 7, 6],
            [3, 6, 2], // y = hi
            [0, 4, 7],
            [0, 7, 3], // x = lo
            [1, 2, 6],
            [1, 6, 5], // x = hi
        ];
        let mut mesh = TriMesh::new(v, indices);
        mesh.orient_outward();
        mesh
    }

    #[test]
    fn box_volume_and_centroid_are_exact() {
        let mesh = unit_box(Point::new(-1.0, -2.0, -3.0), Point::new(2.0, 1.0, 0.0));
        let (volume, centroid) = mesh.volume_centroid();
        assert_relative_eq!(volume, 3.0 * 3.0 * 3.0, epsilon = 1e-12);
        assert_relative_eq!(centroid, Point::new(0.5, -0.5, -1.5), epsilon = 1e-12);
    }

    #[test]
    fn volume_is_independent_of_the_origin() {
        // The divergence theorem guarantees this; a sign error in the tetra
        // formula would not.
        let a = unit_box(Point::zeros(), Point::new(1.0, 2.0, 3.0));
        let b = unit_box(
            Point::new(100.0, -50.0, 7.0),
            Point::new(101.0, -48.0, 10.0),
        );
        assert_relative_eq!(a.signed_volume(), b.signed_volume(), epsilon = 1e-9);
    }

    #[test]
    fn box_surface_area_is_exact() {
        let mesh = unit_box(Point::zeros(), Point::new(1.0, 2.0, 3.0));
        // 2(1*2 + 1*3 + 2*3)
        assert_relative_eq!(mesh.surface_area(), 22.0, epsilon = 1e-12);
    }

    #[test]
    fn orient_outward_fixes_an_inverted_mesh() {
        let mut mesh = unit_box(Point::zeros(), Point::new(1.0, 1.0, 1.0));
        for tri in mesh.indices.iter_mut() {
            tri.swap(1, 2);
        }
        assert!(mesh.signed_volume() < 0.0);
        assert!(mesh.orient_outward());
        assert_relative_eq!(mesh.signed_volume(), 1.0, epsilon = 1e-12);
        // Idempotent: a correctly oriented mesh is left alone.
        assert!(!mesh.orient_outward());
    }

    #[test]
    fn area_normal_points_along_the_winding() {
        let tri = Tri::new(
            Point::zeros(),
            Point::new(1.0, 0.0, 0.0),
            Point::new(0.0, 1.0, 0.0),
        );
        assert_relative_eq!(tri.area_normal(), Point::new(0.0, 0.0, 0.5));
        assert_relative_eq!(tri.area(), 0.5);
        assert_relative_eq!(tri.flipped().area_normal(), Point::new(0.0, 0.0, -0.5));
    }
}
