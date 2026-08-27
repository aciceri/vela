//! What the boat is doing, in numbers.
//!
//! # Everything here is read, nothing is computed
//!
//! Every line comes from `Sim::telemetry` or from the body state. That is a rule
//! and not a convenience: a HUD that computed its own boat speed would be a
//! second answer to a question the engine already answers, and the two would
//! diverge at exactly the moment someone was using the display to debug the
//! first. The one exception is unit conversion, and knots are marked as a
//! display unit in `vela-cli` for the same reason.
//!
//! The keys are the same ones `vela-cli sail` prints, so a number on the screen
//! and a number in a terminal are the same number.

use bevy::prelude::*;

use crate::sim::Engine;

/// Knots per metre per second. Display only; nothing in the engine knows what a
/// knot is.
const KNOTS: f64 = 1.943_844_492_440_605;

/// Marks the readout.
#[derive(Component)]
pub struct Readout;

/// Spawns the readout and the key legend.
pub fn spawn(mut commands: Commands) {
    commands.spawn((
        Text::new(""),
        TextFont {
            font_size: 15.0.into(),
            ..default()
        },
        TextColor(Color::srgb(0.92, 0.95, 0.98)),
        Node {
            position_type: PositionType::Absolute,
            top: px(12),
            left: px(14),
            ..default()
        },
        Readout,
    ));

    commands.spawn((
        Text::new(
            "left/right  helm        up/down  mainsheet\n\
             q/e  traveller          f/r  flatten\n\
             j/l  orbit              i/k  raise      u/o  zoom\n\
             right-drag orbit   wheel zoom",
        ),
        TextFont {
            font_size: 13.0.into(),
            ..default()
        },
        TextColor(Color::srgb(0.70, 0.76, 0.82)),
        Node {
            position_type: PositionType::Absolute,
            bottom: px(12),
            left: px(14),
            ..default()
        },
    ));
}

/// A smoothed frame rate, kept here because nothing else needs it.
///
/// Displayed rather than trusted to feel: "it seems slow" is not a number, and
/// the physics step and the render frame are two different rates that can each
/// be the one at fault. The engine's own step rate is fixed and known, so what
/// this adds is the other half.
#[derive(Resource, Default)]
pub struct FrameRate {
    /// Exponentially smoothed frame time, seconds.
    smoothed: f64,
}

/// Rewrites the readout from the engine's telemetry.
pub fn update(
    engine: Res<Engine>,
    time: Res<Time>,
    mut rate: ResMut<FrameRate>,
    mut readouts: Query<&mut Text, With<Readout>>,
) {
    let sim = &engine.sim;
    // Smoothed over about half a second: an unsmoothed frame time flickers too
    // fast to read, and the question being asked is about the run rather than
    // about this frame.
    let frame = f64::from(time.delta_secs());
    rate.smoothed = if rate.smoothed <= 0.0 {
        frame
    } else {
        rate.smoothed + 0.06 * (frame - rate.smoothed)
    };

    let state = sim.state();
    let telemetry = sim.telemetry();
    let (heel, trim, _) = state.attitude.euler_angles();
    let speed = state.world_velocity().xy().norm();

    let read = |key: &str| telemetry.get(key);
    let wind_speed = read("aero.apparent_wind.speed");
    let wind_angle = read("aero.apparent_wind.angle");

    // A key that is absent is shown as absent. A HUD that printed zero for a
    // quantity the engine had not computed would be inventing a measurement,
    // which is the one thing this project refuses everywhere else.
    let number = |value: Option<f64>, scale: f64| match value {
        Some(it) => format!("{:>8.2}", it * scale),
        None => format!("{:>8}", "—"),
    };

    let text = format!(
        "speed        {:>8.2} kn\n\
         heel         {:>8.1} deg\n\
         trim         {:>8.2} deg\n\
         sinkage      {:>8.3} m\n\
         \n\
         apparent     {} kn  at {} deg\n\
         drive        {} N\n\
         heeling      {} N\n\
         hull drag    {} N\n\
         keel C_L     {}\n\
         \n\
         helm         {:>8.1} deg\n\
         sheet        {:>8.2}   traveller {:>5.2}\n\
         flat         {:>8.2}\n\
         \n\
         t            {:>8.1} s\n\
         frame        {:>8.1} fps  {:>5.1} ms\n\
         steps/frame  {:>8.1}",
        speed * KNOTS,
        heel.to_degrees(),
        trim.to_degrees(),
        state.position.z,
        number(wind_speed, KNOTS),
        number(wind_angle, 180.0 / std::f64::consts::PI),
        number(read("aero.driving_force"), 1.0),
        number(read("aero.heeling_force"), 1.0),
        number(read("hull.total"), 1.0),
        number(read("lateral.keel.lift_coefficient"), 1.0),
        sim.controls().rudder_angle.to_degrees(),
        sim.controls().shape.sheet,
        sim.controls().shape.traveller,
        sim.controls().trim.flat,
        sim.time(),
        1.0 / rate.smoothed.max(1e-6),
        rate.smoothed * 1e3,
        rate.smoothed / crate::sim::STEP,
    );

    for mut readout in &mut readouts {
        **readout = text.clone();
    }
}
