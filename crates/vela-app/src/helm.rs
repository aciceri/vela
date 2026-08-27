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
//! The rudder centres itself when neither key is held. That is a lie about a
//! tiller and a truth about a wheel, and it is here for a reason the engine
//! makes unavoidable: §5.5a records that a boat with a fixed rudder has no course
//! stability at all, so a helm that stayed where it was left would send the boat
//! into a slow uncommanded turn the moment the player stopped steering. Centring
//! is the smallest honest stand-in for a helmsman.

use bevy::prelude::*;
use vela_core::equilibrium::MAX_HELM;

use crate::sim::Engine;

/// Radians of rudder per second, held.
///
/// Hard over in a little under two seconds, which is about what a wheel takes.
const HELM_RATE: f64 = 0.30;

/// How fast the helm returns to centre when nobody is steering.
///
/// Slower than the steering rate, so that a correction is not immediately undone.
const CENTRING_RATE: f64 = 0.12;

/// Fraction of a line's travel per second, held.
const SHEET_RATE: f64 = 0.35;

/// Applies the keyboard to the engine's controls.
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

    if steering == 0.0 {
        // Towards centre, never past it: a decay would leave a residual angle
        // forever and a step could overshoot into the other tack.
        let step = CENTRING_RATE * dt;
        controls.rudder_angle = if controls.rudder_angle.abs() <= step {
            0.0
        } else {
            controls.rudder_angle - step * controls.rudder_angle.signum()
        };
    } else {
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
