//! Manual controls and an explicitly engaged helmsman.
//!
//! # What a control is here
//!
//! `vela_core::Controls` is what the crew can change while sailing, and every
//! field of it is either an angle in radians or a line position normalised from
//! 0 eased to 1 hard. This module maps keyboard and cockpit input onto those
//! controls. It computes no forces and never touches the simulation's pose.
//!
//! # Rates, not steps
//!
//! A key held down moves a control at a rate rather than by an increment per
//! frame, so the feel does not depend on the frame rate. The rates are chosen
//! from what the thing being moved actually takes: a helm goes from hard over to
//! hard over in about two seconds on a boat this size, and a mainsheet through
//! its full travel in about three.
//!
//! # The helmsman
//!
//! Course hold starts engaged and captures the boat's heading on its first
//! frame. It steers about the balanced rudder angle with proportional, rate,
//! and slow integral corrections, limited to the rate a wheel turns.
//!
//! Manual rudder input disengages course hold. Releasing a key or slider keeps
//! that rudder angle: assistance resumes only through [`set_course_hold`],
//! capturing the current heading on the next steering frame.

use bevy::{input_focus::InputFocus, prelude::*, ui_widgets::Slider};
use vela_core::equilibrium::MAX_HELM;

use crate::sim::Engine;

/// Radians of rudder per second, held.
///
/// Hard over in a little under two seconds, which is about what a wheel takes.
const HELM_RATE: f64 = 0.30;

/// Rudder per radian of heading error, rad/rad.
///
/// Three degrees of helm per degree off course. One was tried first and was
/// not enough: in half a metre of sea the boat crept seven degrees to windward
/// in ten seconds against a helmsman who never got past a third of the rudder
/// he had, because the wave-driven yaw rate was eating the rest (below).
const COURSE_GAIN: f64 = 3.0;

/// Rudder per radian per second of yaw rate, s.
///
/// A second and a half's worth. This is what keeps the course from hunting,
/// and the boat's yaw damping was measured (§5.4a) small enough that it is
/// needed; but in a seaway the yaw rate is mostly the waves' — a twentieth of
/// a radian a second either way at a six second period — and three seconds'
/// worth of it, which was the first value, had the helmsman chasing the swell
/// with nine degrees of rudder instead of holding the course.
const RATE_GAIN: f64 = 1.5;

/// Rudder per radian-second of accumulated heading error, rad/(rad s).
///
/// The slow part of a helmsman: the standing offset between the balanced helm
/// the solve found — trim held, flat water — and the helm the free boat in a
/// seaway actually needs is learned rather than known, at a rate that takes
/// tens of seconds to absorb a few degrees and cannot wind up past
/// [`BIAS_LIMIT`].
const BIAS_GAIN: f64 = 0.15;

/// The most the learned offset can be, rad: fifteen degrees, which is more
/// than the difference between the two balances has ever measured.
const BIAS_LIMIT: f64 = 0.26;

/// Fraction of a line's travel per second, held.
const SHEET_RATE: f64 = 0.35;

/// Sets a persistent manual rudder angle in physical radians.
///
/// Positive is port, negative starboard. Taking the wheel disengages course
/// hold and forgets its course and learned bias without changing sail controls.
pub fn set_rudder(engine: &mut Engine, angle: f64) {
    set_course_hold(engine, false);
    let mut controls = *engine.sim.controls();
    controls.rudder_angle = angle.clamp(-MAX_HELM, MAX_HELM);
    engine.sim.set_controls(controls);
}

/// Engages or disengages course hold without moving the rudder.
///
/// Engaging captures the current heading on the next steering frame. Either
/// transition forgets the previous course and learned bias.
pub fn set_course_hold(engine: &mut Engine, enabled: bool) {
    engine.hold_course = enabled;
    engine.course = None;
    engine.helm_bias = 0.0;
}

/// Applies manual keyboard controls and, while engaged, course hold.
///
/// A focused slider owns keyboard input; course hold continues independently.
pub fn steer(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    focus: Res<InputFocus>,
    sliders: Query<(), With<Slider>>,
    mut engine: ResMut<Engine>,
) {
    let dt = time.delta_secs() as f64;
    let mut controls = *engine.sim.controls();
    let slider_focused = focus.get().is_some_and(|entity| sliders.contains(entity));
    let pressed = |key| !slider_focused && keys.pressed(key);

    // Left and right are the boat's, not the screen's. A positive rudder angle
    // adds angle of attack in the same sense as positive leeway — see
    // `vela_core::controls` — which turns the bow to port, so "steer to
    // starboard" is a negative angle.
    let port = pressed(KeyCode::ArrowLeft) || pressed(KeyCode::KeyA);
    let starboard = pressed(KeyCode::ArrowRight) || pressed(KeyCode::KeyD);

    if port || starboard {
        // Even opposing keys take the wheel; releasing never re-engages hold.
        set_course_hold(&mut engine, false);
        let steering = f64::from(u8::from(port)) - f64::from(u8::from(starboard));
        controls.rudder_angle =
            (controls.rudder_angle + steering * HELM_RATE * dt).clamp(-MAX_HELM, MAX_HELM);
    } else if engine.hold_course {
        let state = engine.sim.state();
        let heading = state.attitude.euler_angles().2;
        let yaw_rate = state.angular_velocity.z;
        // Capture only when assistance is engaged, including the first frame.
        let course = *engine.course.get_or_insert(heading);
        // Wrapped, so that a course across north is a small error and not a
        // full turn the wrong way.
        let error = (course - heading + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
            - std::f64::consts::PI;
        engine.helm_bias =
            (engine.helm_bias + BIAS_GAIN * error * dt).clamp(-BIAS_LIMIT, BIAS_LIMIT);
        // Yaw is positive bow-to-starboard and a positive rudder turns the bow
        // to port, so the heading error and its accumulation enter with a
        // minus and the rate with a plus: all three oppose the way the bow is
        // going.
        let wanted = (engine.balanced_helm - COURSE_GAIN * error - engine.helm_bias
            + RATE_GAIN * yaw_rate)
            .clamp(-MAX_HELM, MAX_HELM);
        // At the rate a wheel turns, not instantly.
        let step = HELM_RATE * dt;
        let gap = wanted - controls.rudder_angle;
        controls.rudder_angle = if gap.abs() <= step {
            wanted
        } else {
            controls.rudder_angle + step * gap.signum()
        };
    }

    // Sheet and traveller, the two the boom angle is the product of. Trimming
    // both with one pair of keys would hide the reason the boat has both.
    let mut sheet = 0.0;
    if pressed(KeyCode::ArrowUp) || pressed(KeyCode::KeyW) {
        sheet += 1.0;
    }
    if pressed(KeyCode::ArrowDown) || pressed(KeyCode::KeyS) {
        sheet -= 1.0;
    }
    if sheet != 0.0 {
        controls.shape.sheet = (controls.shape.sheet + sheet * SHEET_RATE * dt).clamp(0.0, 1.0);
    }

    let mut traveller = 0.0;
    if pressed(KeyCode::KeyE) {
        traveller += 1.0;
    }
    if pressed(KeyCode::KeyQ) {
        traveller -= 1.0;
    }
    if traveller != 0.0 {
        controls.shape.traveller =
            (controls.shape.traveller + traveller * SHEET_RATE * dt).clamp(0.0, 1.0);
    }

    // Flattening, which is what the *tabular* model reads. The reference boat
    // sails on that model, so this is the key that actually depowers it — and
    // the fact that it is a different control from the sheet is §4.5's point,
    // not an oversight.
    let mut flat = 0.0;
    if pressed(KeyCode::KeyR) {
        flat += 1.0;
    }
    if pressed(KeyCode::KeyF) {
        flat -= 1.0;
    }
    if flat != 0.0 {
        let target = (controls.trim.flat + flat * SHEET_RATE * dt).clamp(0.4, 1.0);
        controls.trim = controls.trim.with_flat(target);
    }

    engine.sim.set_controls(controls);
}
