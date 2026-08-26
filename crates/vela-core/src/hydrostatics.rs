//! Buoyancy, hydrostatic properties, and the flotation equilibrium.
//!
//! # Exactness, and why it is available for free
//!
//! Two independent computations are done side by side:
//!
//! - **Volume and centre of buoyancy** by the divergence theorem over the
//!   clipped hull.
//! - **Force and moment** by integrating hydrostatic pressure over the same
//!   panels.
//!
//! Both are *exact*, not quadratures, and that is worth the small effort it
//! costs. Hydrostatic pressure `p = ρgz` is linear in position, so its integral
//! over a triangle has a closed form:
//!
//! ```text
//! ∫ p dA   = A (p₁+p₂+p₃)/3
//! ∫ p r dA = (A/12) [ Σ pᵢrᵢ + (Σ pᵢ)(Σ rⱼ) ]
//! ```
//!
//! Using the panel centroid instead would be second-order accurate and would
//! leave the resultant slightly off the centre of buoyancy — turning an exact
//! cross-check between two independent methods into a fuzzy one. With the
//! closed form, `‖F‖ = ρgV` and the line of action passes through the centre of
//! buoyancy to machine precision, which is the single sharpest test available
//! for this whole module.
//!
//! # The missing waterplane cap
//!
//! Clipping the hull leaves an open surface: the waterplane lid is not
//! generated. That is deliberate, and correct on both counts:
//!
//! - **Pressure**: the lid lies at `z = 0`, where `p = 0`. It contributes no
//!   force and no moment.
//! - **Volume**: the tetrahedra of the divergence theorem are spanned from the
//!   world origin, which lies *on* the waterplane. Any triangle in that plane is
//!   coplanar with the origin, so its tetrahedron has zero volume.
//!
//! So the lid would contribute exactly nothing to anything computed here, and
//! triangulating the cut polygon — the fiddliest part of a clipper — is simply
//! skipped. This shortcut is specific to a flat free surface; wave loads will
//! need the general treatment.

use crate::clip::{enclosed_area, Clipped};
use crate::geometry::{Point, Tri, TriMesh};
use crate::rigid_body::RigidBody;
use crate::state::BodyState;
use crate::wrench::Wrench;
use crate::SEA_WATER_DENSITY;
use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use std::fmt;

/// Water properties.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Water {
    /// Mass density, kg/m³.
    pub density: f64,
}

impl Default for Water {
    fn default() -> Self {
        Self {
            density: SEA_WATER_DENSITY,
        }
    }
}

/// Hydrostatic state of a hull at one pose.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hydrostatics {
    /// Displaced volume, m³.
    pub volume: f64,
    /// Centre of buoyancy in the **body frame**, m.
    pub centre_of_buoyancy: Point,
    /// Wetted hull area, m². Excludes the waterplane, which is not hull.
    pub wetted_area: f64,
    /// Area of the waterplane, m². The heave stiffness `ρ g A_wp` and the
    /// metacentric radius both come from this.
    pub waterplane_area: f64,
    /// Buoyancy as a body-frame wrench about the body origin.
    pub buoyancy: Wrench,
}

impl Hydrostatics {
    /// Displaced mass, kg.
    #[must_use]
    pub fn displacement(&self, water: &Water) -> f64 {
        water.density * self.volume
    }

    /// Vertical stiffness in heave, N/m.
    #[must_use]
    pub fn heave_stiffness(&self, water: &Water, gravity: f64) -> f64 {
        water.density * gravity * self.waterplane_area
    }
}

/// Computes the hydrostatics of a hull mesh at a given pose, in still water
/// whose surface is the world plane `z = 0`.
///
/// `mesh` is in the body frame; the returned wrench is too.
#[must_use]
pub fn hydrostatics(
    mesh: &TriMesh,
    state: &BodyState,
    water: &Water,
    gravity: f64,
) -> Hydrostatics {
    hydrostatics_on(mesh, state, &FlatWater, water, gravity)
}

/// The free surface a hull is floating in, as the two questions a pressure
/// integral asks of it.
///
/// Two and not one, because they are different questions and a wave answers them
/// differently. Which triangles are wet is geometry: how far is this point below
/// the surface. What they carry is dynamics: a wave's pressure decays with depth,
/// so a deeply immersed keel feels less of a passing crest than its submergence
/// suggests. Collapsing the two — integrating `ρ g d` below a wavy datum — is the
/// usual shortcut and it over-drives anything deep in short waves.
///
/// Both are in world coordinates, where `z` is down.
pub trait FreeSurface {
    /// Signed depth below the surface, m, positive below.
    fn depth(&self, at: Point) -> f64;

    /// Pressure divided by `ρ g`, m. Defaults to the depth, which is exactly
    /// right for still water and is why [`FlatWater`] needs no body.
    fn pressure_head(&self, at: Point) -> f64 {
        self.depth(at)
    }
}

/// Still water with its surface on the world plane `z = 0`.
#[derive(Debug, Clone, Copy, Default)]
pub struct FlatWater;

impl FreeSurface for FlatWater {
    fn depth(&self, at: Point) -> f64 {
        at.z
    }
}

/// Integrates pressure over the wetted hull against an arbitrary free surface.
///
/// The general form of [`hydrostatics`]. Clipping uses
/// [`FreeSurface::depth`] and the load uses [`FreeSurface::pressure_head`], and
/// keeping them separate is what lets a wave excite a hull correctly rather than
/// merely wet it.
#[must_use]
pub fn hydrostatics_on(
    mesh: &TriMesh,
    state: &BodyState,
    surface: &dyn FreeSurface,
    water: &Water,
    gravity: f64,
) -> Hydrostatics {
    let mut clipped = Clipped::default();
    let rotation = state.attitude.to_rotation_matrix();
    let origin = state.position;

    // Work in world coordinates: that is where the surface is defined, and where
    // the origin-on-the-plane volume trick applies.
    let to_world = |p: Point| origin + rotation * p;
    clipped.extend_from(
        mesh.triangles()
            .map(|t| Tri::new(to_world(t.a), to_world(t.b), to_world(t.c))),
        &|p: Point| surface.depth(p),
    );

    let mut volume = 0.0;
    let mut volume_moment = Point::zeros();
    let mut wetted_area = 0.0;
    let mut force = Point::zeros();
    let mut moment = Point::zeros();

    let unit_weight = water.density * gravity;

    for tri in &clipped.triangles {
        let tetra = tri.signed_tetrahedron_volume();
        volume += tetra;
        volume_moment += tetra * (tri.a + tri.b + tri.c) / 4.0;

        let area_normal = tri.area_normal();
        let area = area_normal.norm();
        if area <= 0.0 {
            continue;
        }
        wetted_area += area;
        let normal = area_normal / area;

        // Pressure at the vertices, from the surface's own head. For still water
        // the head is the depth and this is `ρ g z`; for a seaway it carries the
        // wave's dynamic part with its decay already in it.
        let pressures = [
            unit_weight * surface.pressure_head(tri.a),
            unit_weight * surface.pressure_head(tri.b),
            unit_weight * surface.pressure_head(tri.c),
        ];
        let pressure_sum = pressures[0] + pressures[1] + pressures[2];

        // Exact integrals of a linear pressure field over the triangle.
        let integral_p = area * pressure_sum / 3.0;
        let integral_pr = (area / 12.0)
            * (pressures[0] * tri.a
                + pressures[1] * tri.b
                + pressures[2] * tri.c
                + pressure_sum * (tri.a + tri.b + tri.c));

        // Pressure pushes inward, against the outward normal.
        force -= integral_p * normal;
        moment -= (integral_pr - origin * integral_p).cross(&normal);
    }

    let centre_of_buoyancy = if volume.abs() > f64::EPSILON {
        // Back into the body frame: the caller thinks in hull coordinates.
        rotation.inverse() * (volume_moment / volume - origin)
    } else {
        Point::zeros()
    };

    Hydrostatics {
        volume,
        centre_of_buoyancy,
        wetted_area,
        waterplane_area: enclosed_area(&clipped.cuts, Vector3::z()),
        buoyancy: Wrench::new(rotation.inverse() * force, rotation.inverse() * moment),
    }
}

/// A solved floating equilibrium.
#[derive(Debug, Clone)]
pub struct Flotation {
    /// Pose at which the hull floats. Surge, sway and yaw are left at zero:
    /// nothing in a hydrostatic problem constrains them.
    pub state: BodyState,
    pub hydrostatics: Hydrostatics,
    /// Immersion of the deepest point of the hull, m.
    pub draft: f64,
    /// Heel angle, rad. Positive is starboard down.
    pub heel: f64,
    /// Trim angle, rad. Positive is bow **up**: a rotation about the body `y`
    /// axis carries `x` towards negative `z`, and negative `z` is upward.
    pub trim: f64,
    pub iterations: usize,
}

/// Why a hull would not float.
#[derive(Debug, Clone, PartialEq)]
pub enum FlotationError {
    /// The hull cannot displace its own mass even fully submerged. Either the
    /// mass is wrong or the hull is.
    InsufficientBuoyancy {
        /// Displacement available at full submersion, kg.
        available: f64,
        /// Displacement required, kg.
        required: f64,
    },
    /// The attitude solve failed to settle.
    NotConverged { residual: f64, iterations: usize },
    /// The mesh is empty or has no vertical extent.
    DegenerateHull,
}

impl fmt::Display for FlotationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientBuoyancy {
                available,
                required,
            } => write!(
                f,
                "hull displaces at most {available:.1} kg fully submerged but must carry {required:.1} kg"
            ),
            Self::NotConverged {
                residual,
                iterations,
            } => write!(
                f,
                "flotation did not converge in {iterations} iterations (residual {residual:.3e})"
            ),
            Self::DegenerateHull => write!(f, "hull mesh has no vertical extent"),
        }
    }
}

impl std::error::Error for FlotationError {}

/// Iteration limits for [`solve_flotation`].
#[derive(Debug, Clone, Copy)]
pub struct FlotationOptions {
    pub max_iterations: usize,
    /// Convergence threshold on the dimensionless residual.
    pub tolerance: f64,
}

impl Default for FlotationOptions {
    fn default() -> Self {
        Self {
            max_iterations: 60,
            tolerance: 1e-10,
        }
    }
}

/// Finds the pose at which buoyancy balances weight in both force and moment.
///
/// Three unknowns — sinkage, heel, trim — against three equations: the vertical
/// force balance and the two horizontal moment components. The remaining three
/// degrees of freedom are unconstrained in still water, since buoyancy and
/// weight are both vertical and therefore exert no yaw moment and no horizontal
/// force.
///
/// Sinkage is bracketed by bisection before Newton starts. Vertical force is
/// monotone in sinkage, which makes bisection unconditionally reliable, whereas
/// a Newton step from a dry hull has a zero derivative and goes nowhere. Newton
/// then handles the coupled attitude problem, with step halving when a step
/// fails to reduce the residual.
///
/// # Errors
///
/// [`FlotationError`] if the hull cannot carry the mass, the mesh is
/// degenerate, or the attitude solve does not settle.
pub fn solve_flotation(
    mesh: &TriMesh,
    body: &RigidBody,
    water: &Water,
    options: &FlotationOptions,
) -> Result<Flotation, FlotationError> {
    let Some((low, high)) = mesh.bounds() else {
        return Err(FlotationError::DegenerateHull);
    };
    // Body z points down: the keel is at the largest z, the deck at the
    // smallest. Sinking the origin by the hull's full height submerges it.
    let hull_height = high.z - low.z;
    if !hull_height.is_finite() || hull_height <= 0.0 {
        return Err(FlotationError::DegenerateHull);
    }

    let gravity = body.gravity();
    let weight = body.mass_properties().mass() * gravity;
    let reference_length = (high.x - low.x).max(hull_height);

    let evaluate = |sinkage: f64, heel: f64, trim: f64| -> (Hydrostatics, Vector3<f64>) {
        let state = pose(sinkage, heel, trim);
        let hydro = hydrostatics(mesh, &state, water, gravity);
        let total = hydro.buoyancy + body.gravity_wrench(&state);
        // Residuals in the world frame, made dimensionless so that a force and
        // two moments can share one convergence threshold.
        let force_world = state.to_world(total.force);
        let moment_world = state.to_world(total.moment);
        (
            hydro,
            Vector3::new(
                force_world.z / weight,
                moment_world.x / (weight * reference_length),
                moment_world.y / (weight * reference_length),
            ),
        )
    };

    // Can it float at all?
    let (_, fully_submerged) = evaluate(hull_height * 2.0, 0.0, 0.0);
    // The residual's first component is (buoyancy + weight) projected on world
    // z, and world z points down: it is +1 for a hull in mid air and goes
    // negative once buoyancy wins. So a hull that still reads positive when
    // fully submerged can never carry its mass.
    if fully_submerged[0] > 0.0 {
        let available = hydrostatics(mesh, &pose(hull_height * 2.0, 0.0, 0.0), water, gravity)
            .displacement(water);
        return Err(FlotationError::InsufficientBuoyancy {
            available,
            required: body.mass_properties().mass(),
        });
    }

    // Bracket sinkage on the vertical equation alone, which is monotone in it.
    let mut lo = 0.0;
    let mut hi = hull_height * 2.0;
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if evaluate(mid, 0.0, 0.0).1[0] > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let mut unknowns = Vector3::new(0.5 * (lo + hi), 0.0, 0.0);
    let (mut hydro, mut residual) = evaluate(unknowns[0], unknowns[1], unknowns[2]);

    let steps = Vector3::new(1e-4 * hull_height.max(1e-3), 1e-5, 1e-5);
    let mut lambda = 1e-9;

    for iteration in 0..options.max_iterations {
        if residual.norm() < options.tolerance {
            return Ok(finish(mesh, unknowns, hydro, iteration));
        }

        // Central-differenced Jacobian: six extra clips per iteration, which is
        // irrelevant at load time and buys robustness on hulls whose waterplane
        // changes shape sharply with heel.
        let mut jacobian = Matrix3::zeros();
        for column in 0..3 {
            let mut forward = unknowns;
            let mut backward = unknowns;
            forward[column] += steps[column];
            backward[column] -= steps[column];
            let high = evaluate(forward[0], forward[1], forward[2]).1;
            let low = evaluate(backward[0], backward[1], backward[2]).1;
            jacobian.set_column(column, &((high - low) / (2.0 * steps[column])));
        }

        // Levenberg-Marquardt rather than plain Newton, because rank deficiency
        // here is *physical*, not numerical: a circular-sectioned hull with its
        // CoG on the centerline is neutrally stable in heel, so that column of
        // the Jacobian genuinely vanishes and every heel angle is an
        // equilibrium. Newton would take an unbounded step along the flat
        // direction and stall; the damped normal equations pick the
        // minimum-norm move instead and converge on the directions that are
        // actually determined.
        let transpose = jacobian.transpose();
        let normal = transpose * jacobian;
        let gradient = -(transpose * residual);
        let scale = normal.diagonal().max().max(f64::MIN_POSITIVE);

        let mut improved = false;
        for _ in 0..24 {
            let damped = normal + Matrix3::identity() * (lambda * scale);
            if let Some(delta) = damped.lu().solve(&gradient) {
                let trial = unknowns + delta;
                let (trial_hydro, trial_residual) = evaluate(trial[0], trial[1], trial[2]);
                if trial_residual.norm() < residual.norm() {
                    unknowns = trial;
                    hydro = trial_hydro;
                    residual = trial_residual;
                    lambda = (lambda * 0.2).max(1e-14);
                    improved = true;
                    break;
                }
            }
            lambda *= 8.0;
            if lambda > 1e12 {
                break;
            }
        }

        if !improved {
            // No direction improves things: this is the best available point.
            return accept_or_fail(mesh, unknowns, hydro, residual, iteration, options);
        }
    }

    accept_or_fail(
        mesh,
        unknowns,
        hydro,
        residual,
        options.max_iterations,
        options,
    )
}

/// Accepts a stalled solve if it is good enough to be an equilibrium, and
/// reports failure otherwise.
///
/// The looser threshold exists because a stalled LM iteration on a
/// neutrally-stable hull can sit at a residual far below any physical
/// significance while still being above a machine-precision target.
fn accept_or_fail(
    mesh: &TriMesh,
    unknowns: Vector3<f64>,
    hydrostatics: Hydrostatics,
    residual: Vector3<f64>,
    iterations: usize,
    options: &FlotationOptions,
) -> Result<Flotation, FlotationError> {
    if residual.norm() < options.tolerance.max(1e-8) {
        Ok(finish(mesh, unknowns, hydrostatics, iterations))
    } else {
        Err(FlotationError::NotConverged {
            residual: residual.norm(),
            iterations,
        })
    }
}

/// Builds the pose for a given sinkage, heel and trim, leaving the horizontal
/// degrees of freedom at zero.
fn pose(sinkage: f64, heel: f64, trim: f64) -> BodyState {
    BodyState {
        position: Vector3::new(0.0, 0.0, sinkage),
        attitude: UnitQuaternion::from_euler_angles(heel, trim, 0.0),
        ..BodyState::default()
    }
}

/// Assembles the result, measuring the draft from the geometry rather than from
/// the solver's sinkage variable: the two coincide only for an upright hull
/// whose keel sits exactly on the baseline. The draft is the immersion of the
/// deepest point of the hull, which is what a boat's documentation means by the
/// word.
fn finish(
    mesh: &TriMesh,
    unknowns: Vector3<f64>,
    hydrostatics: Hydrostatics,
    iterations: usize,
) -> Flotation {
    let state = pose(unknowns[0], unknowns[1], unknowns[2]);
    let rotation = state.attitude.to_rotation_matrix();
    let draft = mesh
        .vertices()
        .iter()
        .map(|v| (state.position + rotation * v).z)
        .fold(f64::NEG_INFINITY, f64::max);

    Flotation {
        state,
        hydrostatics,
        draft,
        heel: unknowns[1],
        trim: unknowns[2],
        iterations,
    }
}
