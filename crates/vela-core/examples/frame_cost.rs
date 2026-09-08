//! What one physics step costs, and which knob pays for it.
//!
//! The frontend runs the engine at 200 Hz inside a 60 Hz frame loop, which is
//! 3.3 steps a frame. That is only affordable if a step is well under a
//! millisecond, and "well under" is not something to assume: the buoyancy module
//! clips a thousand-triangle hull against a sea built from sixty-four wave
//! components every step, and both of those numbers are configuration rather
//! than physics.
//!
//! Run with `cargo run --release -p vela-core --example frame_cost`.
//!
//! Reported per step and as a fraction of a 60 Hz frame, because the second
//! number is the one that decides whether the simulator is watchable.

use std::time::Instant;
use vela_core::aero::SailSet;
use vela_core::assembly::{sailing_sim, RadiationOptions};
use vela_core::seaway::SeaState;
use vela_core::{BoatSpec, Controls, LoftOptions, Seaway2D, UniformWind};

/// The step the frontend and `vela-cli` both use.
const STEP: f64 = 1.0 / 200.0;

/// Steps per rendered frame at 60 Hz.
const STEPS_PER_FRAME: f64 = (1.0 / 60.0) / STEP;

const SPEC: &str = include_str!("../../../boats/yd41-form-study.ron");

/// Builds a sim and times a run of steps, returning microseconds per step.
fn cost(
    points_per_station: usize,
    components: usize,
    choppiness: f64,
) -> Result<(f64, usize), String> {
    let spec = BoatSpec::parse_ron(SPEC).map_err(|error| error.to_string())?;
    let loft = LoftOptions { points_per_station };
    let triangles = vela_core::loft_hull(
        spec.hull.as_ref().ok_or("the boat file has no hull")?,
        &loft,
    )
    .triangle_count();
    let wind = UniformWind::uniform(5.0, 40.0_f64.to_radians());
    let sea = Seaway2D::new(
        wind,
        SeaState {
            significant_height: 1.0,
            peak_period: 5.0,
            components,
            choppiness,
            ..SeaState::default()
        },
    );
    let mut sim = sailing_sim(
        &spec,
        Box::new(sea),
        Controls::close_hauled(SailSet::upwind()),
        &loft,
        RadiationOptions::default(),
    )
    .map_err(|error| error.to_string())?;

    // Warm the memory states and let the transient start before timing: a step
    // taken from rest touches fewer radiation states than a step under way.
    for _ in 0..400 {
        sim.step(STEP);
    }

    let reps = 2000;
    let start = Instant::now();
    for _ in 0..reps {
        sim.step(STEP);
    }
    let each = start.elapsed().as_secs_f64() / f64::from(reps);
    Ok((each * 1e6, triangles))
}

fn main() {
    let default_points = LoftOptions::default().points_per_station;
    let default_components = SeaState::default().components;
    println!(
        "one step at {:.0} Hz, and {STEPS_PER_FRAME:.1} of them per 60 Hz frame",
        1.0 / STEP
    );
    println!("defaults: {default_points} sections, {default_components} wave components\n");

    println!("    points  waves  chop  triangles     us/step   frame at 60 Hz");
    for (points, components, choppiness) in [
        (default_points, default_components, 0.0),
        (default_points, default_components, 0.8),
        (default_points, 16, 0.0),
        (default_points, 4, 0.0),
        (13, default_components, 0.0),
        (13, 16, 0.0),
        (9, 16, 0.0),
    ] {
        match cost(points, components, choppiness) {
            Ok((micros, triangles)) => {
                let frame = micros * STEPS_PER_FRAME / 16_666.7 * 100.0;
                println!(
                    "  {points:8}  {components:5}  {choppiness:4}  {triangles:9}  {micros:10.1}  {frame:8.0} %"
                );
            }
            Err(error) => println!("  {points:8}  {components:5}  {choppiness:4}  failed: {error}"),
        }
    }
}
