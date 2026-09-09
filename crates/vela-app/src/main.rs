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
//! IOR dimensions. The sails are the engine's own flying shape — camber, twist
//! and sheeting angle from the controls the player holds, rebuilt as they move
//! — while the *forces* come from the tabular aerodynamic model the reference
//! boat sails on, which has no shape, only areas and centres of effort. That
//! split is stated in [`boat`] rather than papered over.

mod boat;
mod frame;
mod helm;
mod hud;
mod ocean;
mod reflection;
mod sim;
mod sky;
mod spray;
mod view;

use bevy::prelude::*;

/// The window's starting size, and on the web its scale.
///
/// On a desktop the compositor sets both and this is a hint. In a browser the
/// canvas takes the page's size, and its *scale* is the display's pixel ratio
/// - 1.25, 2, 3 on a phone - which is overridden to one here, for two reasons.
///
/// The first is a defect: with a ratio other than one, Bevy 0.19's UI hit test
/// on the web compares the cursor in CSS pixels against node rectangles in
/// physical ones, and the sea button answers a click a quarter of the screen
/// away from where it is drawn and none where it is. Measured in headless
/// Chromium at 1.25 and at 1; the override makes the two coordinate systems
/// the same one. The second is cost: the sea is a full-screen fragment
/// shader, and a display with a pixel ratio of two asks for four times the
/// fragments to draw the same picture a little sharper. At CSS resolution
/// the water is what a web game draws, and the text is what a browser draws
/// at that ratio anyway.
fn resolution() -> bevy::window::WindowResolution {
    let resolution = bevy::window::WindowResolution::new(1280, 720);
    if cfg!(target_arch = "wasm32") {
        resolution.with_scale_factor_override(1.0)
    } else {
        resolution
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "vela".into(),
                resolution: resolution(),
                // The canvas takes its size from the page rather than
                // from a hard-coded pixel count, which is what makes the
                // wasm build usable at all.
                fit_canvas_to_parent: true,
                // Leave the browser its own shortcuts: swallowing F5 and
                // Ctrl-R in a physics demo is hostile.
                prevent_default_event_handling: false,
                // Not vsync. On this frontend the GPU frame is the ocean shader, and
                // under Fifo a frame that misses one vblank by a millisecond waits
                // for the next: a 20 ms frame was drawn at 33 ms and a 28 ms one at
                // 50, which is where "17 fps" came from on a frame the GPU finished
                // in 28. Mailbox where the platform has it, immediate otherwise;
                // Wayland composites either way, so nothing tears.
                present_mode: bevy::window::PresentMode::AutoNoVsync,
                ..default()
            }),
            ..default()
        }))
        // The ocean's shader is embedded in the binary rather than loaded from
        // an assets directory, so there is no asset path to configure and no
        // directory to ship. See `ocean`. The boat's model is embedded the same
        // way, by `boat::BoatPlugin`.
        .add_plugins(ocean::OceanPlugin)
        // The sky owns the sun, and registers the atmosphere library that it and
        // the ocean both import. Before the ocean would work too; after it is
        // where a reader looks for "what is the water reflecting".
        .add_plugins(sky::SkyPlugin)
        .add_plugins(boat::BoatPlugin)
        .add_plugins(sim::EnginePlugin)
        .init_resource::<view::Orbit>()
        .init_resource::<hud::FrameRate>()
        // The mirror attaches to the ocean material the view creates, so after it.
        .add_systems(
            Startup,
            (
                boat::spawn,
                view::spawn,
                hud::spawn,
                reflection::spawn.after(view::spawn),
                spray::spawn.after(boat::spawn),
            ),
        )
        // Input before the fixed step. `Update` runs *after* `RunFixedMainLoop`
        // in Bevy's main schedule, so a `helm::steer` placed there would write
        // the controls after every physics step of the frame had already read
        // them: a key pressed in this frame would reach the engine in the next.
        // `BeforeFixedMainLoop` is the slot Bevy provides for exactly this — a
        // variable-rate system whose output the fixed step consumes — and it
        // puts a key held during a frame that runs several steps into all of
        // them rather than into none.
        .add_systems(
            RunFixedMainLoop,
            helm::steer.in_set(RunFixedMainLoopSystems::BeforeFixedMainLoop),
        )
        // Everything that reads the engine runs after it has stepped, which
        // `Update` guarantees by its place in the main schedule: no ordering
        // edge is needed against the step or the helm. What is stated is the
        // one edge that *is* an ordering within `Update` — the camera follows
        // the boat's drawn pose, below.
        .add_systems(
            Update,
            (
                boat::follow,
                boat::trim,
                view::advance_sea,
                view::follow_sea,
                hud::update,
                hud::sea_button,
                reflection::resize,
                spray::emit_and_advance,
            ),
        )
        // The camera follows the boat's drawn pose, so it must run after it;
        // the mirror is that camera reflected, so after it in turn.
        .add_systems(
            Update,
            (
                view::orbit,
                view::orbit_with_mouse,
                view::chase.after(boat::follow),
                reflection::follow.after(view::chase),
            ),
        )
        .run();
}
