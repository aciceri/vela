//! Vela's frontend: a browser and desktop window onto the physics engine.
//!
//! # What this crate is allowed to be
//!
//! `vela-core` forbids itself Bevy, wgpu, winit and anything else that windows,
//! renders or plays sound. This crate is where all of that lives, and the
//! boundary is enforced structurally by being two crates rather than by
//! convention. The engine is embeddable, headless-testable and deterministic
//! because it never learns that a frame exists; this file is the only place that
//! knows about one.
//!
//! Three joins carry the whole design, and each has its own module because each
//! is a place a mistake would be invisible:
//!
//! - **[`frame`]** — the naval frame the physics works in against the y-up frame
//!   the renderer wants. One rotation, one home, tested against handedness.
//! - **[`ocean`]** — the sea, shared with the GPU as a *realisation* rather than
//!   as a height field. The shader recomputes the same closed form the physics
//!   samples, so the two agree by construction and nothing is transferred.
//! - **[`sim`]** — the fixed-step accumulator. The engine steps when it is told
//!   to, and the schedule is what tells it.
//!
//! # What is drawn and what is not
//!
//! The hull is the physics hull. The rig is drawn from the boat file's published
//! IOR dimensions. The sails are flat triangles at the trimmed angle, and their
//! *shape* is not modelled — the reference boat sails on the tabular
//! aerodynamic model, which has no shape, only areas and centres of effort. That
//! limitation is stated in [`boat`] rather than papered over with a plausible
//! curve.

mod boat;
mod frame;
mod helm;
mod hud;
mod ocean;
mod sim;
mod sky;
mod view;

use bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "vela".into(),
                resolution: (1280, 720).into(),
                // The canvas takes its size from the page rather than
                // from a hard-coded pixel count, which is what makes the
                // wasm build usable at all.
                fit_canvas_to_parent: true,
                // Leave the browser its own shortcuts: swallowing F5 and
                // Ctrl-R in a physics demo is hostile.
                prevent_default_event_handling: false,
                ..default()
            }),
            ..default()
        }))
        // The ocean's shader is embedded in the binary rather than loaded from
        // an assets directory, so there is no asset path to configure and no
        // directory to ship. See `ocean`.
        .add_plugins(ocean::OceanPlugin)
        // The sky owns the sun, and registers the atmosphere library that it and
        // the ocean both import. Before the ocean would work too; after it is
        // where a reader looks for "what is the water reflecting".
        .add_plugins(sky::SkyPlugin)
        .add_plugins(sim::EnginePlugin)
        .init_resource::<view::Orbit>()
        .init_resource::<hud::FrameRate>()
        .add_systems(Startup, (boat::spawn, view::spawn, hud::spawn))
        // Input before the fixed step, so a key held during a frame that runs
        // several physics steps is applied to all of them rather than to the
        // next frame's.
        .add_systems(Update, helm::steer)
        // Everything that reads the engine runs after it has stepped. Ordering
        // is stated rather than left to insertion order: a HUD that read the
        // pose before the step would be one frame stale, which is invisible and
        // wrong.
        .add_systems(
            Update,
            (
                boat::follow,
                boat::trim,
                view::advance_sea,
                view::follow_sea,
                hud::update,
            )
                .after(helm::steer),
        )
        // The camera follows the boat's drawn pose, so it must run after it.
        .add_systems(
            Update,
            (
                view::orbit,
                view::orbit_with_mouse,
                view::chase.after(boat::follow),
            ),
        )
        .run();
}
