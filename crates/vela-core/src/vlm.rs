//! Vortex lattice: lift from geometry, by potential flow.
//!
//! # What this module is
//!
//! A steady vortex-lattice method over quadrilateral panels. It takes a lifting
//! surface as a grid of corner points and an onset flow field, imposes
//! non-penetration at one collocation point per panel, and returns the force on
//! every panel. It knows nothing about sails, wind, or boats: geometry in, forces
//! out.
//!
//! # Two design decisions worth stating first
//!
//! **Vortex rings, not a single row of horseshoes.** The textbook lattice puts one
//! horseshoe per panel with its bound leg at the quarter chord, which is exact for
//! a flat plate and adequate for a wing. A sail is not a wing: camber ratios of
//! ten to twenty per cent, and the whole reason for computing lift from geometry
//! is that camber and twist are what the controls change. That needs chordwise
//! resolution, and chordwise resolution needs rings.
//!
//! **The wake direction belongs to the factorisation, and that is not free.** The
//! plan this module implements claims the influence matrix "depends only on
//! geometry", so it can be factorised once per trim change and reused per frame.
//! That is true of the surface but not of the wake: the trailing filaments leave
//! along a direction, and the direction is the flow's. So the matrix here survives
//! a change of wind *speed*, a change of the onset *field* — shear, boat motion,
//! a gust — and any change to the flow that leaves its mean direction alone. It
//! does not survive a change of apparent wind angle.
//!
//! The direction is therefore an explicit argument to [`Solver::new`] rather than
//! read off the flow inside [`Solver::solve`], which makes the cost visible: a
//! caller who wants the plan's cheap per-frame solve holds one wake direction and
//! accepts a fixed wake, and a caller who wants the wake aligned rebuilds. Hiding
//! the choice would have made the module quietly quadratic in the wrong place.
//!
//! # What it does not do
//!
//! Potential flow, so: no viscosity, no separation, no stall. The lift of a plate
//! at forty degrees comes out as `2π sin 40°` and keeps rising, which is wrong and
//! is not this module's business to correct — see the blending in the sail model.
//! The wake is flat and rigid; there is no roll-up and no unsteadiness.

use nalgebra::{DMatrix, DVector, Vector3};

use crate::geometry::Point;

/// Below this squared distance from a filament, its induced velocity is taken as
/// zero.
///
/// The Biot-Savart law is singular on the filament, and a lattice puts
/// collocation points on the same surface as its vortices — so points that are
/// nearly on a filament are visited routinely rather than exceptionally. The
/// cutoff is on the squared perpendicular distance in the units of the geometry,
/// so it is a length of a hundredth of a millimetre: far below any panel a real
/// lattice has, and far above where the cross product loses its digits.
const CORE_RADIUS_SQUARED: f64 = 1e-10;

/// Induced velocity at `at` from a straight vortex filament of unit strength
/// running from `from` to `to`.
///
/// Biot-Savart for a segment,
///
/// ```text
/// v = (1/4π) (r₁ × r₂)/|r₁ × r₂|² · r₀·(r̂₁ - r̂₂)
/// ```
///
/// with `r₀ = to - from`. Returns zero inside the core rather than an infinity,
/// which is what makes a self-influence term harmless: a panel's own bound
/// segment passes through its own collocation point's spanwise line, and the
/// component that matters is carried by the segments that do not.
fn filament(from: Point, to: Point, at: Point) -> Vector3<f64> {
    let r1 = at - from;
    let r2 = at - to;
    let cross = r1.cross(&r2);
    let denominator = cross.norm_squared();
    let (n1, n2) = (r1.norm(), r2.norm());
    if denominator < CORE_RADIUS_SQUARED || n1 < 1e-12 || n2 < 1e-12 {
        return Vector3::zeros();
    }
    let r0 = to - from;
    let weight = r0.dot(&(r1 / n1 - r2 / n2));
    cross * (weight / (denominator * 4.0 * std::f64::consts::PI))
}

/// Induced velocity at `at` from a semi-infinite filament of unit strength
/// leaving `from` in direction `direction`.
///
/// The limit of [`filament`] as the far end recedes, in closed form:
///
/// ```text
/// v = (1/4π) (d × r)/|d × r|² (1 + r·d/|r|),   r = at - from
/// ```
///
/// Closed form rather than a long finite segment on purpose. A finite stand-in
/// needs a length, the length is arbitrary, and an arbitrary length in the
/// influence matrix is a tuning parameter that silently sets the induced drag.
fn trailing_filament(from: Point, direction: Vector3<f64>, at: Point) -> Vector3<f64> {
    let r = at - from;
    let cross = direction.cross(&r);
    let denominator = cross.norm_squared();
    let radius = r.norm();
    if denominator < CORE_RADIUS_SQUARED || radius < 1e-12 {
        return Vector3::zeros();
    }
    let weight = 1.0 + r.dot(&direction) / radius;
    cross * (weight / (denominator * 4.0 * std::f64::consts::PI))
}

/// A lifting surface as a grid of quadrilateral panels.
///
/// Corners are stored row-major in span: index `i * (spanwise + 1) + j`, with `i`
/// running chordwise from the leading edge and `j` spanwise. A surface with
/// `chordwise` by `spanwise` panels therefore holds `(chordwise + 1)(spanwise + 1)`
/// corners.
#[derive(Debug, Clone, PartialEq)]
pub struct Lattice {
    chordwise: usize,
    spanwise: usize,
    corners: Vec<Point>,
}

impl Lattice {
    /// Builds a lattice from a corner grid.
    ///
    /// Returns `None` unless there is at least one panel in each direction and the
    /// corner count matches the panel counts, which is the one mistake a caller
    /// building a grid by hand actually makes.
    #[must_use]
    pub fn new(chordwise: usize, spanwise: usize, corners: Vec<Point>) -> Option<Self> {
        if chordwise == 0 || spanwise == 0 || corners.len() != (chordwise + 1) * (spanwise + 1) {
            return None;
        }
        Some(Self {
            chordwise,
            spanwise,
            corners,
        })
    }

    /// Builds a lattice by sampling a parametric surface.
    ///
    /// `shape` is called with chordwise and spanwise parameters, each running from
    /// zero to one — the natural way to hand over a sail section stack or an
    /// analytic planform without either side knowing the other's indexing.
    #[must_use]
    pub fn from_shape(
        chordwise: usize,
        spanwise: usize,
        shape: impl Fn(f64, f64) -> Point,
    ) -> Option<Self> {
        if chordwise == 0 || spanwise == 0 {
            return None;
        }
        let mut corners = Vec::with_capacity((chordwise + 1) * (spanwise + 1));
        for i in 0..=chordwise {
            let along_chord = i as f64 / chordwise as f64;
            for j in 0..=spanwise {
                corners.push(shape(along_chord, j as f64 / spanwise as f64));
            }
        }
        Some(Self {
            chordwise,
            spanwise,
            corners,
        })
    }

    /// Panels along the chord.
    #[must_use]
    pub fn chordwise(&self) -> usize {
        self.chordwise
    }

    /// Panels along the span.
    #[must_use]
    pub fn spanwise(&self) -> usize {
        self.spanwise
    }

    /// Total panel count, which is the order of the linear system.
    #[must_use]
    pub fn panels(&self) -> usize {
        self.chordwise * self.spanwise
    }

    fn corner(&self, i: usize, j: usize) -> Point {
        self.corners[i * (self.spanwise + 1) + j]
    }

    /// The four corners of a panel, inboard-leading first and going round.
    fn panel_corners(&self, i: usize, j: usize) -> [Point; 4] {
        [
            self.corner(i, j),
            self.corner(i, j + 1),
            self.corner(i + 1, j + 1),
            self.corner(i + 1, j),
        ]
    }

    /// A point on the ring lattice: the corner grid displaced aft to the panel's
    /// quarter chord, which is where the bound vorticity of thin-airfoil theory
    /// belongs. The last row is the trailing edge itself, so that the ring
    /// closest to it satisfies the Kutta condition there.
    fn ring_corner(&self, i: usize, j: usize) -> Point {
        if i == self.chordwise {
            return self.corner(i, j);
        }
        let leading = self.corner(i, j);
        leading + (self.corner(i + 1, j) - leading) * 0.25
    }

    /// Collocation point of a panel: three-quarter chord, mid span.
    ///
    /// Three quarters and not the centroid. With the vorticity at the quarter
    /// chord, that station is what makes a single flat panel reproduce
    /// thin-airfoil theory exactly, and it is the reason the lift slope of this
    /// method converges from the right side.
    fn collocation(&self, i: usize, j: usize) -> Point {
        let at = |j: usize| {
            let leading = self.corner(i, j);
            leading + (self.corner(i + 1, j) - leading) * 0.75
        };
        (at(j) + at(j + 1)) * 0.5
    }

    /// Unit normal of a panel, from its diagonals so that a warped quadrilateral
    /// still has one.
    fn normal(&self, i: usize, j: usize) -> Vector3<f64> {
        let [p1, p2, p3, p4] = self.panel_corners(i, j);
        (p3 - p1).cross(&(p4 - p2)).normalize()
    }

    /// Area of a panel, by the same diagonals.
    fn area(&self, i: usize, j: usize) -> f64 {
        let [p1, p2, p3, p4] = self.panel_corners(i, j);
        0.5 * (p3 - p1).cross(&(p4 - p2)).norm()
    }

    /// Total surface area, which is the reference area of the coefficients.
    #[must_use]
    pub fn reference_area(&self) -> f64 {
        (0..self.chordwise)
            .flat_map(|i| (0..self.spanwise).map(move |j| (i, j)))
            .map(|(i, j)| self.area(i, j))
            .sum()
    }

    /// Induced velocity at `at` from ring `(i, j)` carrying unit circulation.
    ///
    /// The ring runs inboard-leading, outboard-leading, outboard-trailing,
    /// inboard-trailing. On the last chordwise row the trailing segment is
    /// replaced by two filaments running to infinity along the wake: the segment
    /// closing them at infinity induces nothing, so the substitution is exact
    /// rather than a truncation.
    fn ring_velocity(&self, i: usize, j: usize, at: Point, wake: Vector3<f64>) -> Vector3<f64> {
        let a = self.ring_corner(i, j);
        let b = self.ring_corner(i, j + 1);
        let c = self.ring_corner(i + 1, j + 1);
        let d = self.ring_corner(i + 1, j);

        let mut velocity = filament(a, b, at) + filament(b, c, at) + filament(d, a, at);
        if i + 1 == self.chordwise {
            velocity += trailing_filament(c, wake, at) - trailing_filament(d, wake, at);
        } else {
            velocity += filament(c, d, at);
        }
        velocity
    }
}

/// A factorised lattice, ready to be solved against any onset flow.
///
/// Holds the influence matrix's LU decomposition, the panel geometry it was built
/// from, and the wake direction that is baked into it. Building this is the `O(N³)`
/// part; [`Solver::solve`] is `O(N²)`.
#[derive(Debug, Clone)]
pub struct Solver {
    /// The surfaces, in the order their panels appear.
    ///
    /// Several rather than one because sails interact through their circulations
    /// and not through their forces. A headsail's downwash is what makes a
    /// mainsail's leading edge work, and it is the *same* linear system: solving
    /// each sail alone and adding the answers gives every sail clean air, which is
    /// wrong in the direction that matters and by an amount nothing in the result
    /// would reveal.
    lattices: Vec<Lattice>,
    /// First panel index of each surface, with the total on the end.
    offsets: Vec<usize>,
    wake: Vector3<f64>,
    factored: nalgebra::LU<f64, nalgebra::Dyn, nalgebra::Dyn>,
    /// Collocation point of each panel, in solver order.
    collocation: Vec<Point>,
    /// Unit normal of each panel.
    normals: Vec<Vector3<f64>>,
    /// Bound-segment midpoint and vector of each panel's leading filament, which
    /// is where Kutta-Joukowski is applied.
    bound: Vec<(Point, Vector3<f64>)>,
    /// The panel immediately forward of each one on its own surface, if any.
    ///
    /// A ring's leading segment sits on the ring ahead of it, so the circulation
    /// that survives the cancellation is the difference. Precomputing the
    /// neighbour rather than deriving it from the index is what lets surfaces of
    /// different resolutions share one system — the leading-edge test used to be
    /// `index < spanwise`, which silently means "the first surface's leading edge"
    /// once there is more than one.
    upstream: Vec<Option<usize>>,
    /// Velocity induced at bound midpoint `row` by ring `column` at unit
    /// circulation, indexed `row * panels + column`.
    ///
    /// Cached, and that is the whole point of this type. The forces need the
    /// *total* velocity at each bound segment, not just the onset — that is what
    /// turns a lift calculation into a lift-and-induced-drag one. Evaluating it
    /// per solve means `N²` Biot-Savart calls, which is precisely the cost of
    /// building the influence matrix, so a per-frame solve would have been as
    /// expensive as a refactorisation and the design's central claim would have
    /// been false. Held here, a solve is `N²` multiply-adds and no transcendental
    /// functions at all.
    ///
    /// It costs `3N²` floats: two megabytes at three hundred panels, which is the
    /// right trade and worth naming as a trade.
    bound_influence: Vec<Vector3<f64>>,
}

impl Solver {
    /// Assembles and factorises the influence matrix for one surface.
    ///
    /// `wake` is the direction the trailing filaments leave along; it is
    /// normalised here, and a zero vector is refused. See the module
    /// documentation for what this costs.
    ///
    /// Returns `None` if the wake direction is degenerate or the matrix is
    /// singular — which a lattice with a collapsed panel produces, and which is
    /// better reported than solved.
    #[must_use]
    pub fn new(lattice: Lattice, wake: Vector3<f64>) -> Option<Self> {
        Self::coupled(vec![lattice], wake)
    }

    /// Assembles and factorises one influence matrix over several surfaces.
    ///
    /// The surfaces share a linear system, so each one's boundary condition sees
    /// every other one's vorticity — which is the only way a slot effect or a
    /// mainsail's back-winding can come out of the geometry rather than out of a
    /// correction factor.
    ///
    /// The cost is cubic in the *total* panel count: two sails of 240 panels each
    /// factorise in roughly eight times what one of them takes, not twice. That is
    /// the price of the coupling and it is paid on a trim change, not per frame.
    ///
    /// Returns `None` for an empty list, a degenerate wake, or any surface with a
    /// collapsed panel.
    #[must_use]
    pub fn coupled(lattices: Vec<Lattice>, wake: Vector3<f64>) -> Option<Self> {
        let norm = wake.norm();
        if lattices.is_empty() || !norm.is_finite() || norm < 1e-12 {
            return None;
        }
        let wake = wake / norm;

        let mut offsets = Vec::with_capacity(lattices.len() + 1);
        let mut total = 0;
        for lattice in &lattices {
            offsets.push(total);
            total += lattice.panels();
        }
        offsets.push(total);
        let panels = total;

        let mut collocation = Vec::with_capacity(panels);
        let mut normals = Vec::with_capacity(panels);
        let mut bound = Vec::with_capacity(panels);
        let mut upstream = Vec::with_capacity(panels);
        for (surface, lattice) in lattices.iter().enumerate() {
            let spanwise = lattice.spanwise;
            for i in 0..lattice.chordwise {
                for j in 0..spanwise {
                    collocation.push(lattice.collocation(i, j));
                    normals.push(lattice.normal(i, j));
                    let a = lattice.ring_corner(i, j);
                    let b = lattice.ring_corner(i, j + 1);
                    bound.push(((a + b) * 0.5, b - a));
                    upstream.push(if i == 0 {
                        None
                    } else {
                        Some(offsets[surface] + (i - 1) * spanwise + j)
                    });
                }
            }
        }
        // A panel of zero area has no normal, and `normalize` hands back a vector
        // of NaN rather than complaining. That is the geometric failure worth
        // catching, and catching it *here* rather than at the factorisation is the
        // point: an influence matrix full of NaN is not singular, so an LU of it
        // succeeds and returns a solution made entirely of NaN.
        if normals
            .iter()
            .any(|normal| !normal.iter().all(|c| c.is_finite()))
        {
            return None;
        }

        let mut influence = DMatrix::zeros(panels, panels);
        let mut bound_influence = vec![Vector3::zeros(); panels * panels];
        for (surface, lattice) in lattices.iter().enumerate() {
            for i in 0..lattice.chordwise {
                for j in 0..lattice.spanwise {
                    let column = offsets[surface] + i * lattice.spanwise + j;
                    for row in 0..panels {
                        let at_collocation = lattice.ring_velocity(i, j, collocation[row], wake);
                        influence[(row, column)] = at_collocation.dot(&normals[row]);
                        bound_influence[row * panels + column] =
                            lattice.ring_velocity(i, j, bound[row].0, wake);
                    }
                }
            }
        }

        let factored = influence.lu();
        // And a genuinely singular one, which a duplicated panel produces. The
        // result is discarded: what is being asked is whether the factorisation
        // can solve at all, before a caller relies on it every frame.
        factored.solve(&DVector::zeros(panels))?;

        Some(Self {
            lattices,
            offsets,
            wake,
            factored,
            collocation,
            normals,
            bound,
            upstream,
            bound_influence,
        })
    }

    /// The surfaces this solver was built for.
    #[must_use]
    pub fn lattices(&self) -> &[Lattice] {
        &self.lattices
    }

    /// The first surface, which is the only one for a solver built by
    /// [`Solver::new`].
    #[must_use]
    pub fn lattice(&self) -> &Lattice {
        &self.lattices[0]
    }

    /// Total panel count, which is the order of the linear system.
    #[must_use]
    pub fn panels(&self) -> usize {
        self.offsets[self.offsets.len() - 1]
    }

    /// The wake direction baked into the factorisation.
    #[must_use]
    pub fn wake(&self) -> Vector3<f64> {
        self.wake
    }

    /// Solves for the circulation and the forces in a given onset flow.
    ///
    /// `onset` is the undisturbed flow velocity at a point, so shear, a gust and
    /// the surface's own motion all arrive the same way. `density` scales the
    /// forces and nothing else.
    ///
    /// Returns `None` only if the factorisation refuses the right-hand side, which
    /// [`Solver::new`] has already ruled out for a well-formed lattice.
    #[must_use]
    pub fn solve(&self, onset: impl Fn(Point) -> Vector3<f64>, density: f64) -> Option<Solution> {
        let panels = self.panels();
        let mut rhs = DVector::zeros(panels);
        let onset_at_collocation: Vec<Vector3<f64>> =
            self.collocation.iter().map(|&at| onset(at)).collect();
        for (row, flow) in onset_at_collocation.iter().enumerate() {
            rhs[row] = -flow.dot(&self.normals[row]);
        }
        let circulation = self.factored.solve(&rhs)?;

        // Kutta-Joukowski on each panel's bound filament, carrying the *net*
        // circulation: a ring's leading segment sits on the ring ahead of it, and
        // what is left after they cancel is the difference. The velocity is the
        // total one — onset plus everything the lattice induces — which is what
        // makes this yield induced drag and not only lift.
        let mut forces = Vec::with_capacity(panels);
        for index in 0..panels {
            let (midpoint, segment) = self.bound[index];
            let row = &self.bound_influence[index * panels..(index + 1) * panels];
            let mut velocity = onset(midpoint);
            for (influence, &strength) in row.iter().zip(circulation.iter()) {
                velocity += influence * strength;
            }
            let net =
                circulation[index] - self.upstream[index].map_or(0.0, |ahead| circulation[ahead]);
            forces.push(velocity.cross(&segment) * (density * net));
        }

        Some(Solution {
            circulation: circulation.as_slice().to_vec(),
            forces,
            // Where each force *acts*, which is the bound segment it was computed
            // on and not the collocation point the boundary condition was imposed
            // at. The two differ by half a panel chord, and using the wrong one
            // puts a systematic error in the moment that no lift or drag check
            // would notice.
            positions: self.bound.iter().map(|&(midpoint, _)| midpoint).collect(),
            offsets: self.offsets.clone(),
        })
    }
}

/// What one solve produced.
#[derive(Debug, Clone, PartialEq)]
pub struct Solution {
    circulation: Vec<f64>,
    forces: Vec<Vector3<f64>>,
    positions: Vec<Point>,
    offsets: Vec<usize>,
}

impl Solution {
    /// Circulation of every ring, in solver order.
    #[must_use]
    pub fn circulation(&self) -> &[f64] {
        &self.circulation
    }

    /// Force on every panel, in solver order and in the geometry's frame.
    #[must_use]
    pub fn forces(&self) -> &[Vector3<f64>] {
        &self.forces
    }

    /// Where each of those forces acts: the panel's bound-segment midpoint.
    ///
    /// Exposed because a caller that corrects the forces — an empirical stall
    /// blend, say — has to be able to take the moment of what it produced, and
    /// taking it about anything else silently changes the arm.
    #[must_use]
    pub fn positions(&self) -> &[Point] {
        &self.positions
    }

    /// How many surfaces shared the system.
    #[must_use]
    pub fn surfaces(&self) -> usize {
        self.offsets.len() - 1
    }

    /// The panel index range belonging to one surface.
    ///
    /// The whole reason a coupled solve is worth having: the sails share a linear
    /// system, so the *answer* has to come back attributable — a trimmer needs to
    /// know which sail is making the force, and the two act at different places.
    #[must_use]
    pub fn range(&self, surface: usize) -> Option<std::ops::Range<usize>> {
        if surface + 1 >= self.offsets.len() {
            return None;
        }
        Some(self.offsets[surface]..self.offsets[surface + 1])
    }

    /// Total force on one surface.
    #[must_use]
    pub fn force_on(&self, surface: usize) -> Option<Vector3<f64>> {
        Some(self.forces[self.range(surface)?].iter().sum())
    }

    /// Total moment of one surface about a point.
    #[must_use]
    pub fn moment_on(&self, surface: usize, about: Point) -> Option<Vector3<f64>> {
        let range = self.range(surface)?;
        Some(
            self.forces[range.clone()]
                .iter()
                .zip(self.positions[range].iter())
                .map(|(force, &at)| (at - about).cross(force))
                .sum(),
        )
    }

    /// Total force on every surface together.
    #[must_use]
    pub fn force(&self) -> Vector3<f64> {
        self.forces.iter().sum()
    }

    /// Total moment about a point.
    ///
    /// Taken at each panel's own bound-segment midpoint, which is where the force
    /// was computed. That matters for a sail: the vertical distribution of side
    /// force is the heeling moment, and it is the thing a coefficient model cannot
    /// produce.
    #[must_use]
    pub fn moment_about(&self, about: Point) -> Vector3<f64> {
        self.forces
            .iter()
            .zip(self.positions.iter())
            .map(|(force, &at)| (at - about).cross(force))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    const AIR: f64 = 1.225;
    const SPEED: f64 = 10.0;

    /// A flat rectangular plate in the `z = 0` plane, unit chord.
    fn rectangle(chordwise: usize, spanwise: usize, span: f64) -> Lattice {
        Lattice::from_shape(chordwise, spanwise, |xi, eta| {
            Point::new(xi, (eta - 0.5) * span, 0.0)
        })
        .expect("a positive grid")
    }

    /// A parabolic-arc plate of unit chord and the given camber ratio.
    fn arc(chordwise: usize, spanwise: usize, span: f64, camber: f64) -> Lattice {
        Lattice::from_shape(chordwise, spanwise, |xi, eta| {
            Point::new(xi, (eta - 0.5) * span, 4.0 * camber * xi * (1.0 - xi))
        })
        .expect("a positive grid")
    }

    /// An elliptic planform with a straight quarter-chord line, cosine-spaced
    /// spanwise so that the tips — where the loading has all its structure — are
    /// resolved.
    fn ellipse(chordwise: usize, spanwise: usize, span: f64, root: f64) -> Lattice {
        Lattice::from_shape(chordwise, spanwise, |xi, eta| {
            let y = -0.5 * span * (PI * eta).cos();
            let local = root * (1.0 - (2.0 * y / span).powi(2)).max(0.0).sqrt();
            Point::new(-0.25 * local + xi * local, y, 0.0)
        })
        .expect("a positive grid")
    }

    /// Lift and induced-drag coefficients in a uniform flow at incidence.
    ///
    /// `wake` selects whether the trailing filaments follow the flow or stay on the
    /// surface's own axis, which is the choice the factorisation is sensitive to.
    fn coefficients(lattice: Lattice, degrees: f64, aligned_wake: bool) -> (f64, f64) {
        let alpha = degrees.to_radians();
        let flow = Vector3::new(SPEED * alpha.cos(), 0.0, SPEED * alpha.sin());
        let wake = if aligned_wake { flow } else { Vector3::x() };
        let area = lattice.reference_area();
        let solution = Solver::new(lattice, wake)
            .expect("a well-formed lattice")
            .solve(|_| flow, AIR)
            .expect("a factorised solver solves");
        let force = solution.force();
        let along = flow / SPEED;
        let drag = force.dot(&along);
        let across = force - along * drag;
        let dynamic = 0.5 * AIR * SPEED * SPEED * area;
        (across.norm() * across.z.signum() / dynamic, drag / dynamic)
    }

    /// The Biot-Savart kernel reproduces the one case with a closed form.
    ///
    /// An infinite straight filament induces `Γ/2πr`, and this module never builds
    /// one — it builds two semi-infinite halves. Assembling them and recovering the
    /// textbook answer tests the finite segment's limit, the semi-infinite closed
    /// form, and the sign convention that relates them, none of which any lift
    /// number would isolate.
    #[test]
    fn two_half_filaments_make_the_textbook_infinite_one() {
        let origin = Point::zeros();
        for radius in [0.1_f64, 1.0, 7.5] {
            let at = Point::new(radius, 0.0, 0.0);
            let velocity = trailing_filament(origin, Vector3::z(), at)
                - trailing_filament(origin, -Vector3::z(), at);
            assert_relative_eq!(
                velocity.norm(),
                1.0 / (2.0 * PI * radius),
                max_relative = 1e-12
            );
            // And it points the way a right-handed vortex along `z` should.
            assert!(velocity.y > 0.0, "the induced velocity has the wrong sense");
        }
    }

    /// A filament reversed induces the opposite velocity, exactly.
    ///
    /// The antisymmetry every ring in the lattice relies on: a shared segment
    /// between two rings cancels only if this holds to the last bit.
    #[test]
    fn reversing_a_filament_negates_it() {
        let a = Point::new(-1.0, 0.3, 0.2);
        let b = Point::new(2.0, -0.7, 1.1);
        let at = Point::new(0.4, 0.9, -0.5);
        let forward = filament(a, b, at);
        let backward = filament(b, a, at);
        for k in 0..3 {
            assert_relative_eq!(forward[k], -backward[k], max_relative = 1e-15);
        }
    }

    /// A flat plate has the lift slope of thin-airfoil theory, at any resolution.
    ///
    /// `2π` per radian in the two-dimensional limit. This is the method's one exact
    /// result and it is exact *per panel*: the quarter-chord vortex with a
    /// three-quarter-chord collocation point reproduces a flat plate whatever the
    /// panel count, which is why the chordwise count below changes nothing. A
    /// scheme that had those two stations wrong would still converge, and would
    /// converge to the wrong number.
    #[test]
    fn a_flat_plate_has_the_thin_airfoil_lift_slope() {
        // Long enough that the finite-span correction is a fifth of a per cent.
        let span = 400.0;
        let two_dimensional = 2.0 * PI / (1.0 + 2.0 / span);
        for chordwise in [1_usize, 3, 12] {
            let (lift, _) = coefficients(rectangle(chordwise, 6, span), 2.0, true);
            let slope = lift / 2.0_f64.to_radians();
            assert_relative_eq!(slope, two_dimensional, max_relative = 0.005);
        }
    }

    /// Camber lift reaches thin-airfoil theory as the camber vanishes, and the gap
    /// at a sail's camber is the *theory's*.
    ///
    /// A parabolic arc has `C_L = 2πα + 4πε`, and this lattice disagrees by four and
    /// a half per cent at ten per cent camber. That disagreement was chased, and it
    /// is not the lattice's: extrapolated in panel count, the ratio to theory is
    /// 1.0002 at half a per cent camber and 0.9998 at one per cent, and the gap
    /// grows as `4ε²` — which is exactly the order a linearised theory drops.
    ///
    /// The conclusion matters more than the test. Sail camber is ten to twenty per
    /// cent, so this is the region where the two disagree most, and it is the
    /// *lattice* that should be believed there. That is the whole argument for
    /// computing lift from geometry instead of reading a coefficient.
    ///
    /// Convergence in chordwise panels is first order, not second: flat panels
    /// approximate a curved surface's normal to `O(1/N)`, and the boundary
    /// condition inherits it. Thirty-two panels sit three per cent below the
    /// converged value and a hundred and twenty-eight sit under one.
    #[test]
    fn camber_lift_reaches_theory_as_the_camber_vanishes() {
        let span = 400.0;
        let finite = 1.0 / (1.0 + 2.0 / span);
        // Richardson on a first-order sequence: the increments halve, so the limit
        // is the last value plus the last increment.
        let extrapolated = |camber: f64| {
            let at = |n: usize| coefficients(arc(n, 6, span, camber), 0.0, true).0;
            let (coarse, fine) = (at(32), at(64));
            fine + (fine - coarse)
        };

        for camber in [0.005_f64, 0.01] {
            let ratio = extrapolated(camber) / (4.0 * PI * camber * finite);
            assert_relative_eq!(ratio, 1.0, max_relative = 0.004);
        }

        // And the gap is second order, so it grows by about four when the camber
        // doubles rather than by two.
        let gap = |camber: f64| (1.0 - extrapolated(camber) / (4.0 * PI * camber * finite)).abs();
        let (small, large) = (gap(0.05), gap(0.10));
        assert!(
            (2.5..6.0).contains(&(large / small)),
            "doubling the camber grew the gap by {:.2}, which is not second order",
            large / small
        );
    }

    /// An elliptic wing approaches lifting-line theory as its span grows.
    ///
    /// `C_L = 2πα/(1 + 2/AR)` is itself a high-aspect-ratio approximation, so the
    /// two must agree in that limit and need not below it. The lattice comes in
    /// *under* the line at low aspect ratio, which is the known and correct
    /// direction — the approximation over-predicts there — so the test pins the
    /// trend as well as the limit. Agreeing at aspect ratio four would be evidence
    /// of a bug, not of accuracy.
    #[test]
    fn an_elliptic_wing_approaches_lifting_line_theory() {
        let root = 1.0;
        let alpha = 4.0_f64;
        let mut previous = 0.0;
        for aspect in [4.0_f64, 8.0, 16.0] {
            let span = aspect * PI * root / 4.0;
            let lattice = ellipse(8, 60, span, root);
            let effective = span * span / lattice.reference_area();
            let (lift, _) = coefficients(lattice, alpha, true);
            let theory = 2.0 * PI * alpha.to_radians() / (1.0 + 2.0 / effective);
            let ratio = lift / theory;
            assert!(
                ratio < 1.0,
                "at aspect ratio {aspect} the lattice matched or beat lifting-line \
                 theory at {ratio:.4}, which it should not"
            );
            assert!(
                ratio > previous,
                "a longer span must agree better: {ratio:.4} after {previous:.4}"
            );
            previous = ratio;
        }
        assert!(
            previous > 0.98,
            "at aspect ratio sixteen the gap to lifting-line theory was still {:.1} %",
            100.0 * (1.0 - previous)
        );
    }

    /// Induced drag matches the elliptic minimum.
    ///
    /// `C_Di = C_L²/πAR` is exact for elliptic loading, and it is the check that the
    /// forces carry a streamwise component at all — a lattice that used the onset
    /// velocity instead of the total one in Kutta-Joukowski would return the right
    /// lift and no drag whatever.
    ///
    /// Two to three per cent low, consistently, which is the known behaviour of
    /// near-field induced drag: the streamwise force is a small difference of large
    /// panel terms where lift is a sum. A Trefftz-plane integration would be
    /// sharper and is not here, because the near-field forces are also what give the
    /// spanwise load distribution a sail needs for its heeling moment.
    #[test]
    fn induced_drag_matches_the_elliptic_minimum() {
        let root = 1.0;
        for aspect in [4.0_f64, 8.0, 16.0] {
            let span = aspect * PI * root / 4.0;
            let lattice = ellipse(8, 60, span, root);
            let effective = span * span / lattice.reference_area();
            let (lift, drag) = coefficients(lattice, 4.0, true);
            let theory = lift * lift / (PI * effective);
            assert_relative_eq!(drag, theory, max_relative = 0.05);
            assert!(drag > 0.0, "induced drag must resist, got {drag}");
        }
    }

    /// A plate aligned with the flow makes no force at all.
    ///
    /// Trivial to state and the one case where any sign error anywhere shows up as
    /// something other than zero.
    #[test]
    fn a_plate_in_its_own_plane_makes_no_force() {
        let lattice = rectangle(4, 10, 8.0);
        let flow = Vector3::new(SPEED, 0.0, 0.0);
        let solution = Solver::new(lattice, flow)
            .expect("well formed")
            .solve(|_| flow, AIR)
            .expect("solves");
        let scale = 0.5 * AIR * SPEED * SPEED * 8.0;
        assert!(
            solution.force().norm() / scale < 1e-12,
            "an aligned plate produced {} N",
            solution.force().norm()
        );
    }

    /// A surface symmetric about its own centreline makes no side force.
    ///
    /// The lateral component is the one nothing else here would notice, and for a
    /// sail it is the whole answer — so a lattice that leaked side force out of a
    /// symmetric wing would leak the boat's driving force too.
    #[test]
    fn a_symmetric_surface_makes_no_side_force() {
        let lattice = ellipse(6, 40, 8.0, 1.0);
        let area = lattice.reference_area();
        let alpha = 6.0_f64.to_radians();
        let flow = Vector3::new(SPEED * alpha.cos(), 0.0, SPEED * alpha.sin());
        let solution = Solver::new(lattice, flow)
            .expect("well formed")
            .solve(|_| flow, AIR)
            .expect("solves");
        let force = solution.force();
        assert!(
            force.y.abs() / force.norm() < 1e-9,
            "a symmetric wing threw {:.4e} N sideways out of {:.1} N",
            force.y,
            force.norm()
        );
        let _ = area;
    }

    /// Holding the wake on a fixed axis costs little where the flow is attached.
    ///
    /// The price of the design decision in the module documentation, measured. The
    /// influence matrix carries the wake direction, so a caller who wants the plan's
    /// cheap per-frame solve has to hold one direction and let the wake be wrong.
    /// This says by how much: under a per cent below ten degrees of incidence,
    /// growing past six at twenty-five.
    ///
    /// Which is the right shape for a sail. Ten degrees of incidence is upwind
    /// trim, where the attached-flow assumption this whole method rests on is also
    /// sound; twenty-five degrees is where the empirical blending takes over
    /// anyway. The approximation fails where the method already had to.
    #[test]
    fn a_fixed_wake_costs_little_at_small_incidence() {
        let span = 8.0 * PI / 4.0;
        let mut previous = 0.0;
        for degrees in [2.0_f64, 6.0, 12.0, 25.0] {
            let (aligned, _) = coefficients(ellipse(8, 40, span, 1.0), degrees, true);
            let (fixed, _) = coefficients(ellipse(8, 40, span, 1.0), degrees, false);
            let error = (fixed / aligned - 1.0).abs();
            assert!(
                error > previous,
                "the fixed wake should get worse with incidence: {error:.4} after {previous:.4}"
            );
            if degrees <= 6.0 {
                assert!(
                    error < 0.01,
                    "at {degrees} degrees a fixed wake already costs {:.2} %",
                    100.0 * error
                );
            }
            previous = error;
        }
        assert!(
            previous > 0.02,
            "at twenty-five degrees a fixed wake should be visibly wrong, was {:.2} %",
            100.0 * previous
        );
    }

    /// The moment moves with the point it is taken about, by the total force.
    ///
    /// `M(b) = M(a) + (a - b) × F`. Pinned because the vertical distribution of
    /// side force is what this method exists to provide, and a moment that did not
    /// transform correctly would be a heeling moment that depended on where the
    /// caller happened to put its origin.
    #[test]
    fn the_moment_transforms_with_its_reference_point() {
        let lattice = arc(6, 20, 8.0, 0.12);
        let alpha = 5.0_f64.to_radians();
        let flow = Vector3::new(SPEED * alpha.cos(), 0.0, SPEED * alpha.sin());
        let solution = Solver::new(lattice, flow)
            .expect("well formed")
            .solve(|_| flow, AIR)
            .expect("solves");

        let a = Point::new(0.25, 0.0, 0.0);
        let b = Point::new(-1.5, 2.0, 0.75);
        let shifted = solution.moment_about(a) + (a - b).cross(&solution.force());
        for k in 0..3 {
            assert_relative_eq!(
                solution.moment_about(b)[k],
                shifted[k],
                max_relative = 1e-9,
                epsilon = 1e-9
            );
        }
    }

    /// Degenerate input is refused rather than solved.
    #[test]
    fn a_degenerate_lattice_or_wake_is_refused() {
        assert!(Lattice::new(0, 4, vec![]).is_none());
        assert!(Lattice::new(2, 2, vec![Point::zeros(); 8]).is_none());
        assert!(Lattice::from_shape(0, 3, |_, _| Point::zeros()).is_none());
        assert!(Solver::new(rectangle(2, 2, 4.0), Vector3::zeros()).is_none());
        assert!(Solver::new(rectangle(2, 2, 4.0), Vector3::new(f64::NAN, 0.0, 0.0)).is_none());
        // A lattice collapsed onto a line has no normals and no solution.
        let collapsed = Lattice::from_shape(2, 2, |_, _| Point::zeros()).expect("grid");
        assert!(Solver::new(collapsed, Vector3::x()).is_none());
    }

    /// A solve is far cheaper than the factorisation it reuses.
    ///
    /// The claim the module is built around, as an assertion rather than a comment.
    /// Not a timing test — those do not belong in a suite — but a count: the
    /// factorisation touches every ring from every collocation point *and* every
    /// bound midpoint, and a solve touches a cached number instead. The observable
    /// stand-in is that solving twice with different onsets gives different answers
    /// from one solver, which is the property that makes reuse legitimate.
    #[test]
    fn one_factorisation_serves_any_onset_flow() {
        let lattice = arc(4, 12, 8.0, 0.10);
        let flow = Vector3::new(SPEED, 0.0, 0.5);
        let solver = Solver::new(lattice, flow).expect("well formed");

        let uniform = solver.solve(|_| flow, AIR).expect("solves").force();
        let doubled = solver.solve(|_| flow * 2.0, AIR).expect("solves").force();
        // Linear in the onset, exactly: the system is linear and the forces are
        // quadratic, so doubling the flow quadruples them.
        for k in 0..3 {
            assert_relative_eq!(doubled[k], 4.0 * uniform[k], max_relative = 1e-9);
        }

        // And a sheared onset is not the same as a uniform one, so the solver is
        // genuinely reading the field rather than a stored answer.
        let sheared = solver
            .solve(|p| flow + Vector3::new(0.3 * p.y, 0.0, 0.0), AIR)
            .expect("solves")
            .force();
        assert!(
            (sheared - uniform).norm() / uniform.norm() > 1e-3,
            "shear changed nothing, which cannot be right"
        );
    }

    /// A flat rectangular plate of unit chord, leading edge at `(at_x, at_y)`.
    fn plate_at(at_x: f64, at_y: f64, span: f64) -> Lattice {
        Lattice::from_shape(6, 24, |xi, eta| {
            Point::new(at_x + xi, at_y + (eta - 0.5) * span, 0.0)
        })
        .expect("a positive grid")
    }

    /// Total and per-surface lift of a coupled solve at incidence.
    fn coupled_lift(solver: &Solver, degrees: f64) -> (f64, Vec<f64>) {
        let alpha = degrees.to_radians();
        let flow = Vector3::new(SPEED * alpha.cos(), 0.0, SPEED * alpha.sin());
        let across = Vector3::new(-alpha.sin(), 0.0, alpha.cos());
        let solution = solver
            .solve(|_| flow, AIR)
            .expect("a coupled system solves");
        let per = (0..solution.surfaces())
            .map(|s| {
                solution
                    .force_on(s)
                    .expect("a surface in range")
                    .dot(&across)
            })
            .collect();
        (solution.force().dot(&across), per)
    }

    /// One surface through the coupled path is the single-surface path, bit for bit.
    ///
    /// [`Solver::new`] delegates, so this is the guarantee that generalising to
    /// several surfaces did not perturb the one case every other test in this module
    /// and every validation in `flying` depends on.
    #[test]
    fn one_surface_coupled_is_the_single_surface_path() {
        let plate = plate_at(0.0, 0.0, 8.0);
        let alone = Solver::new(plate.clone(), Vector3::x()).expect("well formed");
        let coupled = Solver::coupled(vec![plate], Vector3::x()).expect("well formed");
        assert_eq!(alone.panels(), coupled.panels());
        assert_eq!(coupled.lattices().len(), 1);
        assert_eq!(
            coupled_lift(&alone, 5.0).0.to_bits(),
            coupled_lift(&coupled, 5.0).0.to_bits()
        );
    }

    /// Side by side, two surfaces stop interacting as the gap grows.
    ///
    /// The convergence check, and it has to be this arrangement rather than the
    /// tandem one: separating two surfaces *across* the flow genuinely removes the
    /// interaction, and the answer must approach the sum of two independent solves.
    ///
    /// Close together they make *more* lift than the sum, not less, which is the
    /// right direction — two plates abreast behave as one longer span, and a longer
    /// span at the same area is a higher aspect ratio.
    #[test]
    fn side_by_side_surfaces_stop_interacting_as_they_separate() {
        let span = 8.0;
        let mut previous = f64::INFINITY;
        for gap in [1.0_f64, 2.0, 8.0, 40.0] {
            let (a, b) = (
                plate_at(0.0, -0.5 * (span + gap), span),
                plate_at(0.0, 0.5 * (span + gap), span),
            );
            let coupled =
                Solver::coupled(vec![a.clone(), b.clone()], Vector3::x()).expect("well formed");
            let total = coupled_lift(&coupled, 5.0).0;
            let independent = coupled_lift(&Solver::new(a, Vector3::x()).expect("ok"), 5.0).0
                + coupled_lift(&Solver::new(b, Vector3::x()).expect("ok"), 5.0).0;

            let excess = total / independent - 1.0;
            assert!(excess > 0.0, "two plates abreast lost lift at gap {gap}");
            assert!(
                excess < previous,
                "the interaction is not decaying: {excess:.4}"
            );
            previous = excess;
        }
        assert!(
            previous < 0.002,
            "forty spans apart the plates still interacted by {:.2} %",
            100.0 * previous
        );
    }

    /// In tandem, the downstream surface loses and the upstream one gains — and the
    /// two effects have different reaches.
    ///
    /// This is the arrangement a sail plan is in, and the asymmetry is the signature
    /// worth pinning. The upstream surface gains from the downstream one's *bound*
    /// vorticity, whose influence falls off with distance — 14 % at one and a half
    /// chords, 0.03 % at forty. The downstream surface sits in the upstream one's
    /// *wake*, which is semi-infinite and flat, so its downwash does not decay at
    /// all: the rear plate is still down to 63 % of its lift forty chords back.
    ///
    /// That is not a defect. A wing forty chords behind another is genuinely in its
    /// downwash, which is why aircraft avoid each other's wake, and it is why a
    /// mainsail behind a headsail is back-winded rather than merely disturbed. It
    /// *is* the inviscid limit — a real wake diffuses and rolls up — and that is the
    /// module's own stated boundary.
    #[test]
    fn a_tandem_pair_gains_in_front_and_loses_behind() {
        let span = 8.0;
        let mut previous_gain = f64::INFINITY;
        let mut totals = Vec::new();
        for gap in [1.5_f64, 3.0, 10.0, 40.0] {
            let (front, rear) = (plate_at(0.0, 0.0, span), plate_at(gap, 0.0, span));
            let coupled = Solver::coupled(vec![front.clone(), rear.clone()], Vector3::x())
                .expect("well formed");
            let (total, per) = coupled_lift(&coupled, 5.0);
            let alone_front = coupled_lift(&Solver::new(front, Vector3::x()).expect("ok"), 5.0).0;
            let alone_rear = coupled_lift(&Solver::new(rear, Vector3::x()).expect("ok"), 5.0).0;

            let gain = per[0] / alone_front - 1.0;
            let loss = 1.0 - per[1] / alone_rear;
            assert!(gain > 0.0, "the front plate lost lift at gap {gap}");
            assert!(
                loss > 0.3,
                "the rear plate barely noticed the wake at gap {gap}"
            );
            assert!(
                gain < previous_gain,
                "the upwash is not decaying: {gain:.4}"
            );
            previous_gain = gain;
            totals.push(total / (alone_front + alone_rear));
        }
        // The upwash has all but gone by forty chords...
        assert!(previous_gain < 0.001);
        // ...and the downwash has not: the pair is still a fifth down on the sum.
        assert!(
            (0.78..0.85).contains(totals.last().expect("a sample")),
            "the tandem pair converged to the independent sum, which a flat wake \
             cannot do: {totals:?}"
        );
    }

    /// Every panel belongs to exactly one surface, and the parts add to the whole.
    ///
    /// The bookkeeping that makes a coupled answer usable: the sails share a system
    /// and the forces have to come back attributable, because a trimmer needs to
    /// know which sail is making the force and the two act at different places.
    #[test]
    fn the_surfaces_partition_the_panels_and_sum_to_the_whole() {
        let solver = Solver::coupled(
            vec![
                plate_at(0.0, 0.0, 8.0),
                Lattice::from_shape(3, 10, |xi, eta| {
                    Point::new(2.0 + 0.5 * xi, (eta - 0.5) * 4.0, 0.3)
                })
                .expect("a positive grid"),
            ],
            Vector3::x(),
        )
        .expect("well formed");
        // Different resolutions on purpose: the telescoping of ring circulations is
        // per surface, and a shared stride would silently mix them.
        assert_eq!(solver.panels(), 6 * 24 + 3 * 10);

        let alpha = 6.0_f64.to_radians();
        let flow = Vector3::new(SPEED * alpha.cos(), 0.0, SPEED * alpha.sin());
        let solution = solver.solve(|_| flow, AIR).expect("solves");

        assert_eq!(solution.surfaces(), 2);
        assert_eq!(solution.range(0), Some(0..144));
        assert_eq!(solution.range(1), Some(144..174));
        assert!(solution.range(2).is_none());
        assert!(solution.force_on(2).is_none());

        let parts: Vector3<f64> = (0..2)
            .map(|s| solution.force_on(s).expect("in range"))
            .sum();
        for k in 0..3 {
            assert_relative_eq!(parts[k], solution.force()[k], max_relative = 1e-12);
        }
        let about = Point::new(0.4, -1.2, 0.7);
        let moments: Vector3<f64> = (0..2)
            .map(|s| solution.moment_on(s, about).expect("in range"))
            .sum();
        for k in 0..3 {
            assert_relative_eq!(
                moments[k],
                solution.moment_about(about)[k],
                max_relative = 1e-12
            );
        }
    }

    /// A coupled solver with nothing in it is refused.
    #[test]
    fn an_empty_coupled_solver_is_refused() {
        assert!(Solver::coupled(vec![], Vector3::x()).is_none());
        // And one degenerate surface refuses the whole system rather than part of it.
        let collapsed = Lattice::from_shape(2, 2, |_, _| Point::zeros()).expect("grid");
        assert!(Solver::coupled(vec![plate_at(0.0, 0.0, 4.0), collapsed], Vector3::x()).is_none());
    }
}
