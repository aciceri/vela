//! Solving for steady sailing.
//!
//! # Why a solver and not just a long run
//!
//! The engine integrates in time, so the obvious way to find steady sailing is
//! to start the boat and wait. Both routes now exist, and they are kept for
//! different reasons rather than one being a stand-in for the other.
//!
//! A solver is **cheap and unconditional**. It reaches the answer in a handful
//! of force evaluations instead of tens of thousands, it needs no damping to get
//! there, and it cannot be defeated by a mode that rings for longer than the run
//! — which matters most for the thing this is used for, a polar, where the
//! answer is wanted at a hundred conditions rather than one. It is also what a
//! velocity prediction program has always been, which is what makes the
//! published polars of §10 comparable at all.
//!
//! A long run is the **independent check**. Nothing makes a Newton iteration on
//! four residuals agree with a six-degree-of-freedom body stepped at two hundred
//! hertz with the water's memory in it, so agreement is a real closure over the
//! whole stack. It is asserted in `tests/sailing.rs`, and it took two fixes to
//! get: the restraint had to move into the mass matrix
//! ([`crate::rigid_body::RigidBody::restrain`]) and the trial state here had to
//! stop giving the boat a vertical velocity. Until both, the
//! run walked away from a solution it had been released at.
//!
//! The reason the check could not run before is worth keeping: without radiation
//! there is no hydrodynamic damping in heave, and buoyancy supplies restoring
//! while nothing removes energy. What damping there is comes from the appendages
//! — a heaving keel is a foil at incidence — and it is enough to settle a boat in
//! a calm but not enough to make the settling quick. Radiation is what removes
//! the energy in the real ship, and [`crate::assembly::sailing_sim`] is the
//! assembly that mounts it.
//!
//! # The system
//!
//! Four unknowns against four equations, matching the degrees of freedom that
//! [`crate::sim::Captive::velocity_prediction`] leaves free:
//!
//! | unknown | balances |
//! |---|---|
//! | surge speed | driving force against resistance |
//! | sway speed | sail side force against the lateral plane |
//! | sinkage | buoyancy against weight |
//! | heel | heeling moment against righting moment |
//!
//! Leeway is not an unknown of its own: it is the ratio of sway to surge, and
//! carrying both the ratio and its numerator would make the system singular.

use crate::sim::Sim;
use crate::state::BodyState;
use nalgebra::{Matrix4, UnitQuaternion, Vector3, Vector4};
use std::fmt;

/// Nominal step handed to the force modules while solving.
///
/// None of the present modules is rate dependent, so the value cannot change
/// the answer; it exists because the module contract takes a step, and a
/// simulation-shaped number is less surprising to a reader than a zero.
const EVALUATION_STEP: f64 = 1.0 / 120.0;

/// Finite-difference perturbations, one per unknown, in SI units.
///
/// Sized by the *mesh*, not by floating-point precision, and the difference
/// matters: the buoyancy module integrates pressure over a clipped triangle
/// mesh, so its derivative is exact between waterline crossings but the
/// crossings themselves move in discrete jumps as triangles enter and leave the
/// water. A perturbation that moves the waterline by far less than a triangle's
/// own height differentiates the discretization rather than the hull, and
/// returns a derivative made of quantisation noise.
///
/// One milliradian of heel moves the waterline by roughly 1.5 mm on a yacht of
/// this beam, which is the same order as a lofted triangle and is where a real
/// signal starts. An earlier version used 1e-5 rad — fifteen microns of
/// waterline movement — and the resulting Jacobian was noise: the line search
/// collapsed after two iterations in every strong-wind case while converging
/// happily in light air, which is the signature of a derivative that is fine
/// when the step is large and meaningless when it is small.
const PERTURBATION: [f64; 4] = [1e-3, 1e-3, 1e-3, 1e-3];

/// Largest excursion allowed in one iteration, per unknown, in SI units:
/// surge, sway, sinkage, heel.
///
/// Chosen as "a change a boat could plausibly make while still being described
/// by the same linearisation" — half a knot of speed, a centimetre of
/// immersion, three degrees of heel — rather than by tuning against a test.
const MAX_STEP: [f64; 4] = [0.25, 0.25, 0.01, 0.05];

/// Heel the iteration is seeded with, radians, when the caller gives none.
///
/// Five degrees or so: far enough off the corner at upright that a one-sided
/// difference is taken on a smooth branch, small enough that it is not a guess
/// about the answer.
const HEEL_SEED: f64 = 0.09;

/// Sway the iteration is seeded with, m/s, when the caller gives no heel.
///
/// Sized to put the leeway near a degree at a working boat speed: clear of the
/// square-root singularity in the downwash term, and small enough not to be a
/// guess about the answer.
const SWAY_SEED: f64 = 0.05;

/// A solved sailing condition.
#[derive(Debug, Clone)]
pub struct Equilibrium {
    /// The state the boat sails in.
    pub state: BodyState,
    /// Speed through the water, m/s.
    pub speed: f64,
    /// Leeway angle, radians, positive when the lateral plane lifts to
    /// starboard.
    pub leeway: f64,
    /// Heel angle, radians, positive starboard-down.
    pub heel: f64,
    /// Immersion of the body origin below the still-water plane, m.
    pub sinkage: f64,
    /// Norm of the converged residual, non-dimensional. See
    /// [`Equilibrium::residual`] for the scaling.
    pub residual: f64,
    /// Newton iterations taken.
    pub iterations: usize,
}

/// Why a sailing condition could not be found.
#[derive(Debug, Clone, PartialEq)]
pub enum EquilibriumError {
    /// The iteration ran out of steps while still above tolerance.
    ///
    /// Carries the residual reached, because a run that stopped at 1e-7 is a
    /// tolerance to relax and one that stopped at 1e-1 is a condition the boat
    /// cannot sail — and the caller cannot tell those apart from a bare
    /// failure.
    NoConvergence { residual: f64, iterations: usize },
    /// The Jacobian could not be factorized, which means two unknowns stopped
    /// being independent — typically a boat that has come to a stop, where
    /// sway and heel no longer influence anything.
    Singular { residual: f64, iterations: usize },
}

impl fmt::Display for EquilibriumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoConvergence {
                residual,
                iterations,
            } => write!(
                f,
                "no sailing equilibrium after {iterations} iterations; \
                 residual still {residual:.3e}"
            ),
            Self::Singular {
                residual,
                iterations,
            } => write!(
                f,
                "the force balance became singular after {iterations} iterations \
                 at residual {residual:.3e}; the boat is probably not sailing"
            ),
        }
    }
}

impl std::error::Error for EquilibriumError {}

/// Where the iteration starts and when it stops.
#[derive(Debug, Clone, Copy)]
pub struct EquilibriumOptions {
    /// Initial guess for speed through the water, m/s.
    pub initial_speed: f64,
    /// Initial guess for the immersion of the body origin, m. The canoe body
    /// draft is the natural choice: the hull is described from its baseline up,
    /// so this is roughly where the waterplane sits.
    pub initial_sinkage: f64,
    /// Initial guess for heel, radians.
    pub initial_heel: f64,
    pub max_iterations: usize,
    /// Convergence threshold on the non-dimensional residual.
    pub tolerance: f64,
}

impl Default for EquilibriumOptions {
    fn default() -> Self {
        Self {
            initial_speed: 3.0,
            initial_sinkage: 0.4,
            initial_heel: 0.0,
            max_iterations: 60,
            tolerance: 1e-9,
        }
    }
}

impl Equilibrium {
    /// The residual scaling: forces by the boat's weight, moments by weight
    /// times waterline length.
    ///
    /// Non-dimensionalising matters here rather than being tidy: a newton and a
    /// newton-metre are not comparable, so an unscaled norm would let a large
    /// moment hide behind a converged force or the reverse, depending only on
    /// the size of the boat.
    fn scale(weight: f64, length: f64) -> Vector4<f64> {
        Vector4::new(weight, weight, weight, weight * length)
    }
}

/// Solves for the steady sailing condition of an assembled simulation.
///
/// The simulation's controls and environment are used as they stand, so a
/// caller sweeps a polar by changing the wind and solving again. The state is
/// left at the solution.
///
/// # Errors
///
/// [`EquilibriumError`] if the iteration does not reach the tolerance.
pub fn solve(
    sim: &mut Sim,
    reference_length: f64,
    options: &EquilibriumOptions,
) -> Result<Equilibrium, EquilibriumError> {
    let weight = sim.body().mass_properties().mass() * sim.body().gravity();
    let scale = Equilibrium::scale(weight, reference_length);

    let mut unknowns = Vector4::new(
        options.initial_speed,
        0.0,
        options.initial_sinkage,
        options.initial_heel,
    );
    let mut residual = evaluate(sim, &unknowns, &scale);

    // Move off two non-differentiable points before differentiating anything.
    //
    // The force models have exactly two places where a derivative does not
    // exist, and an iteration started from rest sits on both of them:
    //
    // * **Zero leeway.** The keel's downwash on the rudder goes as
    //   `sqrt(|C_L|)` with the sign of the leeway restored afterwards, so its
    //   derivative with respect to leeway is *infinite* at zero. A finite
    //   difference taken across it measures the square root, not a slope: the
    //   observed `d(Fy)/d(sway)` came out half its true value, and the entry
    //   then doubled on the next iteration a thousandth of a metre per second
    //   away. That is the noise that made one tack converge and its mirror
    //   stall.
    // * **Zero heel.** The appendage heel factors are written in `|phi|` —
    //   heeling to port must cost what heeling to starboard costs — so upright
    //   is a corner, and a one-sided difference there measures the wrong branch
    //   on one of the two tacks.
    //
    // The athwartships force at the starting guess says which way the boat is
    // about to lie down and which way it is about to slip, and both take its
    // sign: the sails push to leeward, and the lateral plane must answer with a
    // leeway of the opposite sense. Seeding a degree of leeway and a few of
    // heel puts the first Jacobian on a smooth branch. It is a starting point
    // and not a constraint — the solver is free to come back through zero if
    // that is where the balance turns out to be.
    let tack = residual[1].signum();
    if options.initial_heel == 0.0 && residual[1] != 0.0 {
        unknowns[1] = SWAY_SEED * tack;
        unknowns[3] = HEEL_SEED * tack;
        residual = evaluate(sim, &unknowns, &scale);
    }

    let mut norm = residual.norm();

    for iteration in 1..=options.max_iterations {
        if norm < options.tolerance {
            return Ok(finish(sim, &unknowns, norm, iteration - 1));
        }

        let jacobian = jacobian(sim, &unknowns, &residual, &scale);
        let Some(step) = jacobian.lu().solve(&(-residual)) else {
            return Err(EquilibriumError::Singular {
                residual: norm,
                iterations: iteration,
            });
        };

        // Limit the step to a physically sized excursion before the line
        // search sees it, preserving its direction. A full Newton step taken
        // from an upright guess at a condition that heels twenty degrees can
        // land with the deck under water or the boat stopped, where the force
        // models are so nonlinear that halving never recovers — the line search
        // then collapses on the first iteration and reports a failure that is
        // an artefact of the starting point rather than a boat that cannot
        // sail. Capping the excursion is a trust region whose radius is stated
        // in metres, metres per second and radians instead of in norms.
        let step = limited(step);

        // Halve the step until it actually reduces the residual.
        let mut fraction = 1.0;
        loop {
            let trial = unknowns + step * fraction;
            let trial_residual = evaluate(sim, &trial, &scale);
            let trial_norm = trial_residual.norm();
            if trial_norm < norm {
                unknowns = trial;
                residual = trial_residual;
                norm = trial_norm;
                break;
            }
            fraction *= 0.5;
            if fraction < 1e-4 {
                // The direction is no longer productive. Report where it stalled
                // rather than iterating on a step that changes nothing.
                return Err(EquilibriumError::NoConvergence {
                    residual: norm,
                    iterations: iteration,
                });
            }
        }
    }

    if norm < options.tolerance {
        Ok(finish(sim, &unknowns, norm, options.max_iterations))
    } else {
        Err(EquilibriumError::NoConvergence {
            residual: norm,
            iterations: options.max_iterations,
        })
    }
}

/// Builds the state described by an unknown vector.
///
/// Pitch and yaw are held at zero: they are the restrained modes, and letting
/// the solver move them would be solving for an attitude no equation
/// constrains.
///
/// # Why the heave velocity is not zero
///
/// The unknowns are the boat's **body-frame** surge and sway, and a body frame
/// at `φ` of heel is tilted. Leaving the body heave velocity at zero therefore
/// gives the boat a *world* vertical velocity of `v sin φ` — sixteen centimetres
/// a second of sinking at twenty degrees and half a knot of leeway — and a boat
/// that is sinking is not in steady sailing however well its forces balance.
///
/// It is not harmless, because the horizontal speed the resistance regressions
/// are given is measured in the world: at zero body heave the athwartships part
/// of it is `v cos φ`, and at true steady state it is `v / cos φ`. The two
/// differ by `tan²φ`, which is four per cent of the sway term at twenty degrees
/// and moved the solved speed by nearly two per cent against a time-domain run
/// of the same boat.
///
/// So the heave velocity is whatever makes the world vertical velocity vanish:
/// `w = -(R₃₁u + R₃₂v) / R₃₃`. With pitch and yaw held at zero the attitude is a
/// pure roll and that is `-v tan φ`, but it is written against the rotation so
/// that it stays right if the restrained attitude ever stops being one.
///
/// `R₃₃` is `cos φ` here, so it only vanishes with the mast in the water. That is
/// a knockdown rather than a sailing condition and no heave velocity can hold a
/// boat level in it, so the degenerate case takes zero rather than an infinity
/// that would poison every residual downstream.
fn state_of(unknowns: &Vector4<f64>) -> BodyState {
    let attitude = UnitQuaternion::from_euler_angles(unknowns[3], 0.0, 0.0);
    let horizontal = attitude * Vector3::new(unknowns[0], unknowns[1], 0.0);
    let vertical = (attitude * Vector3::z()).z;
    let heave = if vertical.abs() < 1e-6 {
        0.0
    } else {
        -horizontal.z / vertical
    };
    BodyState {
        position: Vector3::new(0.0, 0.0, unknowns[2]),
        attitude,
        velocity: Vector3::new(unknowns[0], unknowns[1], heave),
        angular_velocity: Vector3::zeros(),
    }
}

/// The non-dimensional residual of the four free degrees of freedom.
fn evaluate(sim: &mut Sim, unknowns: &Vector4<f64>, scale: &Vector4<f64>) -> Vector4<f64> {
    sim.set_state(state_of(unknowns));
    let wrench = sim.applied_wrench(EVALUATION_STEP);
    Vector4::new(
        wrench.force.x / scale[0],
        wrench.force.y / scale[1],
        wrench.force.z / scale[2],
        wrench.moment.x / scale[3],
    )
}

/// Forward-difference Jacobian.
///
/// Forward rather than central differences: four extra force evaluations per
/// iteration instead of eight, and the residual at the base point is already in
/// hand. The buoyancy module's mesh clip is the expensive part of an
/// evaluation, so halving their number is worth the loss of one order in the
/// derivative — which the step-halving line search absorbs.
fn jacobian(
    sim: &mut Sim,
    unknowns: &Vector4<f64>,
    residual: &Vector4<f64>,
    scale: &Vector4<f64>,
) -> Matrix4<f64> {
    let mut jacobian = Matrix4::zeros();
    for column in 0..4 {
        let mut perturbed = *unknowns;
        perturbed[column] += PERTURBATION[column];
        let shifted = evaluate(sim, &perturbed, scale);
        let derivative = (shifted - residual) / PERTURBATION[column];
        jacobian.set_column(column, &derivative);
    }
    jacobian
}

fn finish(sim: &mut Sim, unknowns: &Vector4<f64>, residual: f64, iterations: usize) -> Equilibrium {
    // Leave the simulation at the solution, and recompute its telemetry there,
    // so a caller can read the force breakdown of the condition it just solved.
    let state = state_of(unknowns);
    sim.set_state(state.clone());
    sim.applied_wrench(EVALUATION_STEP);

    // Speed is read off the state, not rebuilt from the unknowns, because
    // "speed through the water" has one definition in this engine and it is
    // [`crate::sim::StepCtx::speed_through_water`]: the horizontal velocity in
    // the *world*. At heel that is not the norm of the body-frame surge and
    // sway, and reporting a second answer to the same question is how a polar
    // and the resistance it was computed from come to disagree.
    let sway = unknowns[1];
    let speed = state.world_velocity().xy().norm();
    Equilibrium {
        state,
        speed,
        leeway: (-sway).atan2(unknowns[0]),
        heel: unknowns[3],
        sinkage: unknowns[2],
        residual,
        iterations,
    }
}

/// Scales a Newton step down until every component is within [`MAX_STEP`],
/// keeping its direction.
///
/// Scaling the whole vector rather than clamping each component matters: a
/// per-component clamp would bend the step away from the Newton direction and
/// can point it somewhere the residual does not decrease at all, which is worse
/// than a short step along the right line.
fn limited(step: Vector4<f64>) -> Vector4<f64> {
    let mut fraction: f64 = 1.0;
    for index in 0..4 {
        let magnitude = step[index].abs();
        if magnitude > MAX_STEP[index] {
            fraction = fraction.min(MAX_STEP[index] / magnitude);
        }
    }
    step * fraction
}

/// Largest helm the solver will ask for, radians.
///
/// Thirty degrees. Past there a real rudder has stalled and this engine's
/// appendage model has not — its lift is linear in angle with no limiting term —
/// so a solution beyond the bound would be a number the model is not entitled
/// to. Reaching it is reported rather than clamped silently: a boat that needs
/// more than thirty degrees of rudder to hold its course is telling you
/// something about its balance, not about its helm.
const MAX_HELM: f64 = 0.52;

/// Rudder angle the secant is opened with, radians.
///
/// Two degrees: enough that the yawing moment moves measurably, small enough to
/// stay in the linear part of the rudder's lift.
const HELM_PROBE: f64 = 0.035;

/// A sailing condition with the helm balanced as well as the rest.
#[derive(Debug, Clone)]
pub struct Helm {
    /// The four-degree-of-freedom balance at the solved rudder angle.
    pub equilibrium: Equilibrium,
    /// Rudder angle that holds the course, radians, in the sign convention of
    /// [`crate::appendages`].
    pub rudder_angle: f64,
    /// Yawing moment left over, non-dimensional on weight times length.
    ///
    /// The honest measure of whether the helm was found: the outer iteration
    /// drives this to zero, and reporting it means a caller need not trust that
    /// it did.
    pub yaw_residual: f64,
    /// Whether [`MAX_HELM`] was reached, in which case the condition is *not*
    /// balanced and the rudder angle is the bound rather than a solution.
    pub helm_saturated: bool,
}

/// Solves the steady sailing condition *and* the rudder angle that holds it.
///
/// Adds yaw balance to what [`solve`] does. It is an outer iteration around that
/// solver rather than a fifth unknown inside it, and the reason is that the
/// four-degree-of-freedom Newton is validated against the source's published
/// polar: widening it would put that validation at risk for a coupling that is
/// weak in the first place. The rudder carries a few per cent of the lateral
/// plane's side force, so changing its angle barely moves the balance the inner
/// solver found, and a secant on the yawing moment converges in a handful of
/// steps.
///
/// The claim that the coupling is weak is not assumed:
/// [`Helm::yaw_residual`] is the residual of the *fifth* equation at the
/// solution, so if the alternation had failed to converge it would say so.
///
/// A boat whose file declares no layout has no longitudinal arms, so its yawing
/// moment is identically zero at every rudder angle. That is not an error — the
/// moment is already balanced — and this returns immediately with a rudder angle
/// of zero.
///
/// # Errors
///
/// [`EquilibriumError`] if the inner solve fails at any trial angle.
pub fn solve_with_helm(
    sim: &mut Sim,
    reference_length: f64,
    options: &EquilibriumOptions,
) -> Result<Helm, EquilibriumError> {
    let weight = sim.body().mass_properties().mass() * sim.body().gravity();
    let yaw_scale = weight * reference_length;
    let base = *sim.controls();

    // One evaluation to find out whether there is anything to balance.
    let attempt = |sim: &mut Sim, angle: f64| -> Result<(Equilibrium, f64), EquilibriumError> {
        sim.set_controls(base.with_rudder(angle));
        let solved = solve(sim, reference_length, options)?;
        let moment = sim.applied_wrench(EVALUATION_STEP).moment.z / yaw_scale;
        Ok((solved, moment))
    };

    let (mut equilibrium, mut moment) = attempt(sim, 0.0)?;
    if moment.abs() < options.tolerance {
        return Ok(Helm {
            equilibrium,
            rudder_angle: 0.0,
            yaw_residual: moment,
            helm_saturated: false,
        });
    }

    // Secant on the yawing moment. Two points to start, then the usual update,
    // with the step limited so that a nearly flat secant cannot throw the angle
    // past the bound in one jump.
    let mut previous_angle = 0.0;
    let mut previous_moment = moment;
    let mut angle = HELM_PROBE;
    let mut saturated = false;

    for _ in 0..options.max_iterations {
        let (solved, current) = attempt(sim, angle)?;
        equilibrium = solved;
        moment = current;
        if moment.abs() < options.tolerance {
            break;
        }
        let slope = (moment - previous_moment) / (angle - previous_angle);
        if !slope.is_finite() || slope == 0.0 {
            break;
        }
        let step = (-moment / slope).clamp(-0.2, 0.2);
        previous_angle = angle;
        previous_moment = moment;
        angle = (angle + step).clamp(-MAX_HELM, MAX_HELM);
        if angle.abs() >= MAX_HELM {
            saturated = true;
            let (solved, current) = attempt(sim, angle)?;
            equilibrium = solved;
            moment = current;
            break;
        }
    }

    Ok(Helm {
        equilibrium,
        rudder_angle: angle,
        yaw_residual: moment,
        helm_saturated: saturated,
    })
}
