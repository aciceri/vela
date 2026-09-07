//! The keyboard, as a crew.
//!
//! # What a control is here
//!
//! `vela_core::Controls` is what the crew can change while sailing, and every
//! field of it is either an angle in radians or a line position normalised from
//! 0 eased to 1 hard. This module maps keys onto those and does nothing else: it
//! computes no forces, holds no state of its own beyond a rate, and never
//! touches the simulation's pose.
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
//! When neither steering key is held, a helmsman steers the boat to a
//! **course** — the heading it was on when the keys were last released — with
//! a proportional-plus-rate law on the rudder about the balanced angle. Not
//! back to the balanced angle itself, which is what this file did second, and
//! not to amidships, which is what it did first.
//!
//! Both earlier versions failed for the reason §5.5a states: a boat with a
//! fixed rudder has no course stability. Amidships releases a trimmed boat with
//! the helm in the wrong place and it luffs up in seconds. The balanced angle
//! is the equilibrium solve's, found with trim held and in flat water; released
//! with trim free into a seaway the yaw balance moves at once, and the boat
//! luffed head to wind inside fifteen seconds, stopped, then bore away to a
//! broad reach and back — a viewer read a boat that "does not move against the
//! sea", and was right. Every real boat has someone on the wheel doing what
//! this does: holding the course with small corrections, at the rate a wheel
//! turns. The gains are a helmsman's, not a tuned controller's — a degree of
//! rudder per degree off course, and three seconds' worth of yaw rate to keep
//! it from hunting — and the balanced angle is the trim the corrections sit
//! on, so in flat water the rudder rests exactly where the solve put it.
//!
//! Steering by hand takes over completely and sets a new course on release.

use bevy::prelude::*;
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

/// Applies the keyboard to the engine's controls, and the helmsman when the
/// keyboard is not steering.
pub fn steer(keys: Res<ButtonInput<KeyCode>>, time: Res<Time>, mut engine: ResMut<Engine>) {
    let dt = time.delta_secs() as f64;
    let mut controls = *engine.sim.controls();

    // Left and right are the boat's, not the screen's. A positive rudder angle
    // adds angle of attack in the same sense as positive leeway — see
    // `vela_core::controls` — which turns the bow to port, so "steer to
    // starboard" is a negative angle.
    let mut steering = 0.0;
    if keys.pressed(KeyCode::ArrowLeft) || keys.pressed(KeyCode::KeyA) {
        steering += 1.0;
    }
    if keys.pressed(KeyCode::ArrowRight) || keys.pressed(KeyCode::KeyD) {
        steering -= 1.0;
    }

    let state = engine.sim.state();
    let heading = state.attitude.euler_angles().2;
    let yaw_rate = state.angular_velocity.z;

    if steering == 0.0 {
        // The course is the heading the keys were last released on; the first
        // frame sets it to the heading the boat was released on.
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
    } else {
        engine.course = None;
        engine.helm_bias = 0.0;
        controls.rudder_angle =
            (controls.rudder_angle + steering * HELM_RATE * dt).clamp(-MAX_HELM, MAX_HELM);
    }

    // Sheet and traveller, the two the boom angle is the product of. Trimming
    // both with one pair of keys would hide the reason the boat has both.
    let mut sheet = 0.0;
    if keys.pressed(KeyCode::ArrowUp) || keys.pressed(KeyCode::KeyW) {
        sheet += 1.0;
    }
    if keys.pressed(KeyCode::ArrowDown) || keys.pressed(KeyCode::KeyS) {
        sheet -= 1.0;
    }
    if sheet != 0.0 {
        controls.shape.sheet = (controls.shape.sheet + sheet * SHEET_RATE * dt).clamp(0.0, 1.0);
    }

    let mut traveller = 0.0;
    if keys.pressed(KeyCode::KeyE) {
        traveller += 1.0;
    }
    if keys.pressed(KeyCode::KeyQ) {
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
    if keys.pressed(KeyCode::KeyR) {
        flat += 1.0;
    }
    if keys.pressed(KeyCode::KeyF) {
        flat -= 1.0;
    }
    if flat != 0.0 {
        let target = (controls.trim.flat + flat * SHEET_RATE * dt).clamp(0.4, 1.0);
        controls.trim = controls.trim.with_flat(target);
    }

    engine.sim.set_controls(controls);
}
