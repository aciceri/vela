//! Integration tests for the 6-DOF rigid-body dynamics.
//!
//! Each test checks a property with an independent ground truth — an analytic
//! solution, a conservation law, or an exact algebraic consequence of the
//! equations of motion — rather than a previously recorded output.

use approx::assert_relative_eq;
use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use vela_core::{BodyState, MassProperties, RigidBody, Wrench, STANDARD_GRAVITY};

/// Integrates for `duration` with no external forces, returning the final state.
fn free_flight(body: &RigidBody, mut state: BodyState, duration: f64, dt: f64) -> BodyState {
    let steps = (duration / dt).round() as u64;
    for _ in 0..steps {
        body.step(&mut state, Wrench::zero(), dt);
    }
    state
}

fn centred_body(gravity: f64) -> RigidBody {
    let mass = MassProperties::from_gyradii(1000.0, Vector3::zeros(), Vector3::new(1.0, 2.0, 2.5))
        .expect("valid mass properties");
    RigidBody::with_gravity(mass, gravity).expect("valid rigid body")
}

/// Ballistic flight: with gravity the only force, the trajectory is the
/// textbook parabola. Semi-implicit Euler is first-order, so the tolerance is
/// tied to the step size rather than being an arbitrary epsilon.
#[test]
fn ballistic_trajectory_matches_the_analytic_parabola() {
    let body = centred_body(STANDARD_GRAVITY);
    let initial_velocity = Vector3::new(6.0, 0.0, -3.0);
    let state = BodyState {
        velocity: initial_velocity,
        ..BodyState::at_rest(Vector3::zeros())
    };

    let duration = 2.0;
    let dt = 1.0 / 240.0;
    let final_state = free_flight(&body, state, duration, dt);

    let expected = initial_velocity * duration
        + 0.5 * Vector3::new(0.0, 0.0, STANDARD_GRAVITY) * duration * duration;

    // Leading error of the scheme for constant acceleration is g·T·dt/2.
    let bound = STANDARD_GRAVITY * duration * dt;
    assert!(
        (final_state.position - expected).norm() < bound,
        "position {:?} deviates from analytic {:?} by more than {bound}",
        final_state.position,
        expected
    );
    // Velocity under constant acceleration is integrated exactly.
    assert_relative_eq!(
        final_state.velocity,
        initial_velocity + Vector3::new(0.0, 0.0, STANDARD_GRAVITY) * duration,
        epsilon = 1e-9
    );
}

/// Halving the step must halve the position error: the observable signature of
/// a first-order scheme. Catches an integrator silently degrading to zeroth
/// order, which a single-tolerance test would not.
#[test]
fn position_error_scales_linearly_with_the_step() {
    let body = centred_body(STANDARD_GRAVITY);
    let duration = 2.0;

    let error_at = |dt: f64| {
        let final_state = free_flight(&body, BodyState::default(), duration, dt);
        let expected = 0.5 * Vector3::new(0.0, 0.0, STANDARD_GRAVITY) * duration * duration;
        (final_state.position - expected).norm()
    };

    let coarse = error_at(1.0 / 120.0);
    let fine = error_at(1.0 / 240.0);
    assert_relative_eq!(coarse / fine, 2.0, epsilon = 0.05);
}

/// A body in free fall must not start spinning, however far the CoG sits from
/// the body origin. Gravity produces a moment about the origin, and the
/// translation-rotation coupling of the mass matrix must cancel it exactly.
/// This is the test that fails if a coupling block carries the wrong sign.
#[test]
fn free_fall_does_not_spin_a_body_with_an_offset_centre_of_gravity() {
    let mass = MassProperties::from_gyradii(
        3800.0,
        Vector3::new(0.35, -0.1, 0.9),
        Vector3::new(1.1, 2.6, 2.7),
    )
    .expect("valid mass properties");
    let body = RigidBody::with_gravity(mass, STANDARD_GRAVITY).expect("valid rigid body");

    // Arbitrary attitude: the cancellation must not depend on being upright.
    let state = BodyState {
        attitude: UnitQuaternion::from_euler_angles(0.4, -0.25, 1.3),
        ..BodyState::default()
    };

    let final_state = free_flight(&body, state, 2.0, 1.0 / 240.0);

    assert_relative_eq!(
        final_state.angular_velocity,
        Vector3::zeros(),
        epsilon = 1e-12
    );
}

/// Torque-free rotation of an asymmetric body: angular momentum about the CoG
/// is conserved in the world frame while the body-frame components wander
/// (Euler precession).
///
/// The bound is not arbitrary. Evaluating the inertial terms at the midpoint
/// makes the scheme conserve quadratic invariants, and the measured drift here
/// is 1.6e-8 at this step; 1e-6 leaves room for platform floating-point
/// differences while still failing by orders of magnitude if the midpoint
/// treatment is ever dropped (explicit evaluation drifts by ~6e-3).
#[test]
fn torque_free_rotation_conserves_angular_momentum() {
    let body = centred_body(0.0);
    // Tumbling about no principal axis, so all three Euler equations are active.
    let state = BodyState {
        angular_velocity: Vector3::new(0.6, 1.4, 0.3),
        ..BodyState::default()
    };

    let initial = body.angular_momentum_world(&state);
    let final_state = free_flight(&body, state.clone(), 10.0, 1.0 / 480.0);
    let final_momentum = body.angular_momentum_world(&final_state);

    let magnitude_drift = (final_momentum.norm() - initial.norm()).abs() / initial.norm();
    assert!(
        magnitude_drift < 1e-6,
        "angular momentum magnitude drifted by {magnitude_drift}"
    );

    // The body-frame angular velocity must actually have moved, otherwise the
    // conservation check above is vacuous.
    assert!(
        (final_state.angular_velocity - state.angular_velocity).norm() > 0.1,
        "precession did not occur; the test proves nothing"
    );
}

/// The integrator must not *inject* energy into a torque-free body: that is an
/// instability that grows without bound, and an explicit treatment of the
/// gyroscopic term does exactly that (+1.6 % over 60 s at 120 Hz).
///
/// The property checked is the absence of secular growth, not monotonic decay.
/// A conservative scheme oscillates around the initial value by a few parts in
/// `1e9`, so demanding a monotonic decrease would test floating-point noise
/// rather than dynamics. The band below is tight enough that an explicit
/// treatment overshoots it within the first second.
#[test]
fn torque_free_energy_stays_bounded() {
    let body = centred_body(0.0);
    let state = BodyState {
        angular_velocity: Vector3::new(0.6, 1.4, 0.3),
        ..BodyState::default()
    };

    let initial = body.kinetic_energy(&state);
    let mut current = state;
    let dt = 1.0 / 240.0;
    let mut worst: f64 = 0.0;

    for _ in 0..(240 * 30) {
        body.step(&mut current, Wrench::zero(), dt);
        let deviation = (body.kinetic_energy(&current) - initial) / initial;
        worst = worst.max(deviation.abs());
        assert!(
            deviation < 1e-6,
            "kinetic energy grew by {deviation} — the scheme is injecting energy"
        );
    }

    // Guard against the test passing because nothing happened at all.
    assert!(worst > 0.0, "energy never moved; the test proves nothing");
}

/// The inertial-term treatment is third-order accurate in the step: halving the
/// step must cut the conservation error by about eight. Pins the midpoint rule
/// specifically — a first- or second-order variant would show a factor of two
/// or four and fail here.
#[test]
fn conservation_error_is_third_order_in_the_step() {
    let body = centred_body(0.0);

    let drift_at = |dt: f64| {
        let state = BodyState {
            angular_velocity: Vector3::new(0.6, 1.4, 0.3),
            ..BodyState::default()
        };
        let initial = body.angular_momentum_world(&state).norm();
        let final_state = free_flight(&body, state, 10.0, dt);
        (body.angular_momentum_world(&final_state).norm() - initial).abs() / initial
    };

    let ratio = drift_at(1.0 / 120.0) / drift_at(1.0 / 240.0);
    assert!(
        (6.0..10.0).contains(&ratio),
        "expected third-order convergence (ratio near 8), measured {ratio}"
    );
}

/// A constant angular velocity about a single axis integrates as a pure
/// rotation by `ω t`, exactly: the exponential map carries no small-angle
/// error. Guards the attitude update against a linearized shortcut.
#[test]
fn constant_yaw_rate_integrates_to_the_exact_angle() {
    let body = centred_body(0.0);
    let rate = 0.5;
    let state = BodyState {
        angular_velocity: Vector3::new(0.0, 0.0, rate),
        ..BodyState::default()
    };

    let duration = 4.0;
    let final_state = free_flight(&body, state, duration, 1.0 / 240.0);
    let (roll, pitch, yaw) = final_state.euler_angles();

    assert_relative_eq!(yaw, rate * duration, epsilon = 1e-9);
    assert_relative_eq!(roll, 0.0, epsilon = 1e-12);
    assert_relative_eq!(pitch, 0.0, epsilon = 1e-12);
    // Rotation about a principal axis of a body at rest generates no other
    // motion.
    assert_relative_eq!(
        final_state.angular_velocity,
        Vector3::new(0.0, 0.0, rate),
        epsilon = 1e-12
    );
}

/// Ballast below the body origin must produce a restoring roll moment: the
/// mechanism behind a yacht's righting moment, and a direct check that the body
/// frame really is z-down.
#[test]
fn ballast_below_the_origin_produces_a_restoring_roll_moment() {
    let depth = 1.2;
    let mass = MassProperties::from_gyradii(
        3800.0,
        Vector3::new(0.0, 0.0, depth),
        Vector3::new(1.1, 2.6, 2.7),
    )
    .expect("valid mass properties");
    let body = RigidBody::with_gravity(mass, STANDARD_GRAVITY).expect("valid rigid body");

    let heel = 0.25;
    let state = BodyState {
        attitude: UnitQuaternion::from_euler_angles(heel, 0.0, 0.0),
        ..BodyState::default()
    };

    let wrench = body.gravity_wrench(&state);
    let expected = -depth * 3800.0 * STANDARD_GRAVITY * heel.sin();

    assert_relative_eq!(wrench.moment.x, expected, epsilon = 1e-9);
    assert!(
        wrench.moment.x < 0.0,
        "heeling to starboard must produce a moment back to port"
    );
    // Weight is unchanged by attitude.
    assert_relative_eq!(
        wrench.force.norm(),
        3800.0 * STANDARD_GRAVITY,
        epsilon = 1e-9
    );
}

/// Upright, the whole weight acts downward in the body frame with no moment
/// when the CoG is on the origin.
#[test]
fn upright_gravity_acts_purely_downward() {
    let body = centred_body(STANDARD_GRAVITY);
    let wrench = body.gravity_wrench(&BodyState::default());

    assert_relative_eq!(
        wrench.force,
        Vector3::new(0.0, 0.0, 1000.0 * STANDARD_GRAVITY),
        epsilon = 1e-9
    );
    assert_relative_eq!(wrench.moment, Vector3::zeros(), epsilon = 1e-9);
}

/// An applied pure moment produces the angular acceleration predicted by the
/// inertia tensor, and no linear acceleration when the CoG is on the origin.
#[test]
fn applied_moment_produces_the_expected_angular_acceleration() {
    let body = centred_body(0.0);
    let moment = Vector3::new(0.0, 0.0, 500.0);
    let acceleration = body.acceleration(&BodyState::default(), Wrench::from_moment(moment));

    let inertia: Matrix3<f64> = body.mass_properties().inertia_cog();
    assert_relative_eq!(
        acceleration.angular,
        inertia.try_inverse().expect("invertible") * moment,
        epsilon = 1e-9
    );
    assert_relative_eq!(acceleration.linear, Vector3::zeros(), epsilon = 1e-12);
}

/// Torque-free motion conserves kinetic energy too. Tracked separately from
/// angular momentum because the integrator can conserve one while drifting in
/// the other.
#[test]
fn torque_free_rotation_conserves_kinetic_energy() {
    let body = centred_body(0.0);
    let state = BodyState {
        angular_velocity: Vector3::new(0.6, 1.4, 0.3),
        ..BodyState::default()
    };

    let initial = body.kinetic_energy(&state);
    let final_state = free_flight(&body, state, 10.0, 1.0 / 480.0);
    let drift = (body.kinetic_energy(&final_state) - initial).abs() / initial;

    // Measured 3.3e-8 at this step; see the angular-momentum test for why the
    // bound sits where it does.
    assert!(drift < 1e-6, "kinetic energy drifted by {drift}");
}

/// A body whose mass matrix couples heave to pitch, the way added mass does.
///
/// The off-diagonal term is the whole subject of the two tests below, so it is
/// stated here rather than buried: `A₃₅ = A₅₃ = -40000 kg·m` is the order a
/// yacht's heave-pitch coupling actually reaches, and the diagonal entries keep
/// the matrix positive definite.
fn coupled_body() -> RigidBody {
    let mass = MassProperties::from_gyradii(6000.0, Vector3::zeros(), Vector3::new(1.5, 3.5, 3.5))
        .expect("valid mass properties");
    let mut body = RigidBody::with_gravity(mass, 0.0).expect("valid rigid body");
    let mut added = nalgebra::Matrix6::zeros();
    added[(2, 2)] = 26000.0;
    added[(4, 4)] = 900_000.0;
    added[(2, 4)] = -40_000.0;
    added[(4, 2)] = -40_000.0;
    body.add_added_mass(added).expect("positive definite");
    body
}

/// The coupling is real, and this is the test that says so.
///
/// A pure pitch moment on a free body accelerates it in heave, because the mass
/// matrix is not diagonal. Without this the restraint test below would pass on a
/// body that had no coupling to suppress, and would be checking nothing.
#[test]
fn a_pitch_moment_alone_accelerates_an_unrestrained_body_in_heave() {
    let body = coupled_body();
    let state = BodyState::at_rest(Vector3::zeros());
    let moment = Wrench {
        force: Vector3::zeros(),
        moment: Vector3::new(0.0, 50_000.0, 0.0),
    };

    let acceleration = body.acceleration(&state, moment);
    assert!(
        acceleration.linear.z.abs() > 0.05,
        "a pitch moment must reach heave through the coupling, got {} m/s²",
        acceleration.linear.z
    );
}

/// Restraining pitch must stop that force reaching heave at all.
///
/// This is the failure the restraint exists to prevent, and it is not a
/// numerical nicety. A yacht held in pitch — the condition a velocity
/// prediction is posed in — carries an unbalanced trim moment, because no
/// equation balanced it. Cancelling the pitch *velocity* after the step leaves
/// the heave acceleration that moment induced, so the boat rises until it has
/// bought an equal and opposite buoyancy error, and settles carrying a fifth of
/// its weight in vertical force that the telemetry reports and the motion
/// denies.
///
/// Both parts are asserted: the held mode gets exactly zero, and so does the
/// mode it would otherwise have driven.
#[test]
fn a_restrained_mode_cannot_drive_a_free_one_through_the_mass_matrix() {
    let mut body = coupled_body();
    // Surge, sway, heave, roll, pitch, yaw — pitch alone.
    body.restrain([false, false, false, false, true, false]);
    let state = BodyState::at_rest(Vector3::zeros());
    let moment = Wrench {
        force: Vector3::zeros(),
        moment: Vector3::new(0.0, 50_000.0, 0.0),
    };

    let acceleration = body.acceleration(&state, moment);
    assert_relative_eq!(acceleration.angular.y, 0.0, epsilon = 1e-12);
    assert_relative_eq!(acceleration.linear.z, 0.0, epsilon = 1e-12);
}

/// The free block must be untouched by the restraint.
///
/// A restraint that also changed the answer for the modes it does not hold would
/// be a different boat, not a captive one. Heave alone is loaded, and the heave
/// acceleration must be what `M₃₃` says whether or not pitch is held — which it
/// is only because the held row and column are removed rather than merely
/// zeroed on one side.
#[test]
fn restraining_one_mode_leaves_the_others_on_their_own_mass() {
    let mut body = coupled_body();
    let state = BodyState::at_rest(Vector3::zeros());
    let heave = Wrench {
        force: Vector3::new(0.0, 0.0, 32_000.0),
        moment: Vector3::zeros(),
    };

    body.restrain([false, false, false, false, true, false]);
    let held = body.acceleration(&state, heave);
    let expected = heave.force.z / body.mass_matrix()[(2, 2)];

    assert_relative_eq!(held.linear.z, expected, max_relative = 1e-12);
    assert_relative_eq!(held.angular.y, 0.0, epsilon = 1e-12);
}
