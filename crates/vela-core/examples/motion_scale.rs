//! Does the boat move like a forty-footer or like a model of one?
//!
//! This exists because of a complaint that cannot be answered by reading code:
//! the boat "moves too much and too fast, like a little wooden model in the sea".
//! That is a statement about *scale*, and scale in ship motions is carried almost
//! entirely by the natural periods. A hull whose heave and pitch periods are too
//! short looks like a toy however correct its geometry, because the eye reads
//! frequency against size and a real forty-footer cannot bob at two hertz.
//!
//! Run with `cargo run --release -p vela-core --example motion_scale`.
//!
//! Reported against published expectations rather than against nothing: for a
//! twelve metre waterline yacht, heave and pitch natural periods are both
//! typically in the two to three second range, and roll — being stiffness-limited
//! by a ballasted keel — in the three to five second range. Anything much below
//! those and the model is too stiff or too light in added inertia.

use vela_core::aero::SailSet;
use vela_core::assembly::{sailing_sim, RadiationOptions};
use vela_core::hydrostatics::{hydrostatics_on, FlatWater, Water};
use vela_core::seaway::SeaState;
use vela_core::state::BodyState;
use vela_core::{BoatSpec, Controls, LoftOptions, Seaway2D, UniformWind};

const SPEC: &str = include_str!("../../../boats/yd41-form-study.ron");

/// The period of an undamped oscillator with the given mass and stiffness, s.
fn period(mass: f64, stiffness: f64) -> f64 {
    if mass <= 0.0 || stiffness <= 0.0 {
        return f64::NAN;
    }
    2.0 * std::f64::consts::PI * (mass / stiffness).sqrt()
}

fn main() {
    let spec = BoatSpec::parse_ron(SPEC).expect("the reference boat parses");
    let loft = LoftOptions::default();
    let parameters = spec.hull_parameters().expect("parameters");
    let hull = vela_core::loft_hull(spec.hull.as_ref().expect("a hull"), &loft);
    let wind = UniformWind::uniform(5.0, 40.0_f64.to_radians());

    // Assembled exactly as the frontend assembles it, so the mass matrix reported
    // is the one the simulation integrates and not a fresh guess at it.
    let mut sim = sailing_sim(
        &spec,
        Box::new(Seaway2D::new(
            wind,
            SeaState {
                significant_height: 1.5,
                peak_period: 6.0,
                ..SeaState::default()
            },
        )),
        Controls::close_hauled(SailSet::upwind()),
        &loft,
        RadiationOptions::default(),
    )
    .expect("the boat assembles");

    let mass = sim.body().mass_matrix();
    let displacement = mass[(0, 0)];
    println!("displacement            {displacement:8.0} kg");
    println!(
        "heave generalised mass  {:8.0} kg   (added mass is the difference)",
        mass[(2, 2)]
    );
    println!("pitch generalised inertia {:6.0} kg m^2", mass[(4, 4)]);
    println!("roll generalised inertia  {:6.0} kg m^2", mass[(3, 3)]);

    // Hydrostatic stiffness by finite difference on the buoyancy wrench, which is
    // the same integral the simulation restores with. Measured rather than taken
    // from a waterplane formula so that it includes whatever the real sections do.
    let water = Water::default();
    let gravity = 9.81;
    let mut floating = BodyState::default();
    floating.position.z = parameters.canoe_draft;

    // Bring it to its own flotation first: a stiffness measured about the wrong
    // sinkage is a stiffness of the wrong waterplane.
    for _ in 0..80 {
        let hydro = hydrostatics_on(&hull, &floating, &FlatWater, &water, gravity);
        let net = hydro.buoyancy.force.z + displacement * gravity;
        // Buoyancy is negative-up in this frame, weight positive-down.
        if net.abs() < 1.0 {
            break;
        }
        floating.position.z += 0.02 * net.signum();
    }

    let heave_step = 0.02;
    let mut raised = floating.clone();
    raised.position.z -= heave_step;
    let mut lowered = floating.clone();
    lowered.position.z += heave_step;
    let force = |state: &BodyState| {
        hydrostatics_on(&hull, state, &FlatWater, &water, gravity)
            .buoyancy
            .force
            .z
    };
    let heave_stiffness = (force(&lowered) - force(&raised)) / (2.0 * heave_step);

    // Rotational stiffness has to include the *weight*, not only the buoyancy.
    //
    // Both wrenches are taken about the body origin, and heeling a hull moves the
    // centre of buoyancy and the centre of gravity in different directions: the
    // restoring couple is the difference, and the classical `ρg∇·GM` is exactly
    // that difference written out. Differencing the buoyancy moment alone leaves
    // out the `−ρg∇·KG` term, which for a ballasted keel is most of it — it
    // reported a metre and a half too much GM here, and so a roll period a
    // factor of the square root of two short, which would have been read as a
    // physics fault in the engine rather than an omission in this probe.
    let body = sim.body();
    let restoring = |attitude: nalgebra::UnitQuaternion<f64>| {
        let mut state = floating.clone();
        state.attitude = attitude;
        let hydro = hydrostatics_on(&hull, &state, &FlatWater, &water, gravity);
        let weight = body.gravity_wrench(&state);
        hydro.buoyancy.moment + weight.moment
    };

    let pitch_step = 0.5_f64.to_radians();
    let pitch =
        |angle: f64| restoring(nalgebra::UnitQuaternion::from_euler_angles(0.0, angle, 0.0)).y;
    let pitch_stiffness = (pitch(-pitch_step) - pitch(pitch_step)) / (2.0 * pitch_step);

    let heel_step = 2.0_f64.to_radians();
    let roll =
        |angle: f64| restoring(nalgebra::UnitQuaternion::from_euler_angles(angle, 0.0, 0.0)).x;
    let roll_stiffness = (roll(-heel_step) - roll(heel_step)) / (2.0 * heel_step);
    println!(
        "\nGM implied by the roll stiffness {:5.2} m   (a forty-footer carries 1.2 to 1.8)",
        roll_stiffness.abs() / (displacement * gravity)
    );

    println!(
        "\nstiffness about the flotation at sinkage {:.3} m:",
        floating.position.z
    );
    println!("  heave  {heave_stiffness:12.0} N/m");
    println!("  pitch  {pitch_stiffness:12.0} N m/rad");
    println!("  roll   {roll_stiffness:12.0} N m/rad");

    println!("\nnatural periods, and what a twelve metre yacht should show:");
    println!(
        "  heave  {:5.2} s   expected 2.0 to 3.0",
        period(mass[(2, 2)], heave_stiffness.abs())
    );
    println!(
        "  pitch  {:5.2} s   expected 2.0 to 3.0",
        period(mass[(4, 4)], pitch_stiffness.abs())
    );
    println!(
        "  roll   {:5.2} s   expected 3.0 to 5.0",
        period(mass[(3, 3)], roll_stiffness.abs())
    );

    // And what the assembled simulation actually does when released in that sea:
    // the periods above are linear and undamped, and the motion a viewer objects
    // to is the damped, forced one.
    let start = sim.state().clone();
    sim.set_state(start);
    let dt = 1.0 / 200.0;
    let mut sinkage = (f64::INFINITY, f64::NEG_INFINITY);
    let mut trim = (f64::INFINITY, f64::NEG_INFINITY);
    let mut zero_crossings = 0usize;
    let mut previous = 0.0;
    let mut mean = 0.0;
    let steps = (120.0 / dt) as usize;
    for step in 0..steps {
        sim.step(dt);
        if step * 2 < steps {
            continue;
        }
        let state = sim.state();
        let (_, pitch, _) = state.attitude.euler_angles();
        sinkage = (
            sinkage.0.min(state.position.z),
            sinkage.1.max(state.position.z),
        );
        trim = (trim.0.min(pitch), trim.1.max(pitch));
        mean += state.position.z;
        let relative = pitch;
        if previous < 0.0 && relative >= 0.0 {
            zero_crossings += 1;
        }
        previous = relative;
    }
    let sampled = steps - steps / 2;
    mean /= sampled as f64;
    let sampled_seconds = sampled as f64 * dt;
    println!("\nreleased in Hs 1.5 m, Tp 6 s, over {sampled_seconds:.0} s of sailing:");
    println!(
        "  sinkage  mean {mean:6.3} m  swing {:6.3} m",
        sinkage.1 - sinkage.0
    );
    println!(
        "  trim     swing {:6.2} deg",
        (trim.1 - trim.0).to_degrees()
    );
    if zero_crossings > 1 {
        println!(
            "  pitch    apparent period {:5.2} s  from {zero_crossings} upward crossings",
            sampled_seconds / zero_crossings as f64
        );
    }
}
