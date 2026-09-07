//! End-to-end lateral motion: a real boat file in, a rolling hull out.
//!
//! The antisymmetric twin of `vertical_motion`, and the last thing the seakeeping
//! phase was missing. These exercise the whole lateral chain — station offsets,
//! Lewis mapping, the *antisymmetric* Ursell-Tasai sectional solve, strip
//! integration with its vertical datum shift, Ogilvie's relation, the rational
//! fit, the state-space realisation, the added-mass matrix and the integrator.
//!
//! What they defend is the property the phase exists for: **a heeled hull comes
//! back, and it comes back at the rate its own coefficients predict**. Before
//! this chain a boat released from heel oscillated on hydrostatic stiffness
//! forever in roll, exactly as it used to in heave.
//!
//! Note what is *not* here: viscous roll damping. The plan named Ikeda's
//! semi-empirical method as the source for it, and that method was measured
//! against this hull and found outside its stated envelope — its own text warns
//! that "for ships with a very large breadth to draft ratio the method is not
//! always accurate sufficiently", and this canoe body's ratio is 7.8. So the roll
//! damping here is wave-making alone, which for a hull this beamy is the larger
//! part rather than the smaller one, and the same beam that makes it large is why
//! the viscous correction could not be computed.

use nalgebra::{UnitQuaternion, Vector3};
use vela_core::assembly::{seakeeping_sim, RadiationOptions};
use vela_core::env::Seaway2D;
use vela_core::seaway::SeaState;
use vela_core::{BoatSpec, LoftOptions, Sim, StillWater, UniformWind};

fn boat() -> BoatSpec {
    let text = include_str!("../../../boats/yd41-form-study.ron");
    BoatSpec::parse_ron(text).expect("the shipped boat file parses")
}

/// Coarser than the default so that a test suite stays a test suite, matching
/// `vertical_motion::quick` exactly so the two files resolve the same physics.
fn quick() -> RadiationOptions {
    RadiationOptions {
        samples: 60,
        tasai: vela_core::tasai::TasaiOptions {
            multipoles: 8,
            quadrature: 48,
        },
        transform: vela_core::cummins::TransformOptions {
            horizon: 60.0,
            steps: 3000,
        },
        ..RadiationOptions::default()
    }
}

fn in_water(env: impl vela_core::env::Environment + 'static, options: RadiationOptions) -> Sim {
    seakeeping_sim(&boat(), Box::new(env), &LoftOptions::default(), options)
        .expect("the shipped boat admits a seakeeping simulation")
}

/// Lets the hull find its own flotation before anything is asked of it, and
/// returns the pose it found. The assembled state is upright at the origin, which
/// is not where the boat floats.
fn settled(options: RadiationOptions) -> (Sim, f64, f64) {
    let mut sim = in_water(StillWater::new(UniformWind::uniform(0.0, 0.0)), options);
    let dt = 0.005;
    for _ in 0..(40.0 / dt) as usize {
        sim.step(dt);
    }
    let (roll, pitch, _) = sim.state().attitude.euler_angles();
    (sim, roll, pitch)
}

/// Releases the hull from an angle of heel and returns the successive roll peak
/// amplitudes in degrees, with the time of each.
fn roll_decay(sim: &mut Sim, rest_roll: f64, offset: f64, seconds: f64) -> (Vec<f64>, Vec<f64>) {
    let mut state = sim.state().clone();
    let (_, pitch, yaw) = state.attitude.euler_angles();
    state.attitude = UnitQuaternion::from_euler_angles(rest_roll + offset, pitch, yaw);
    state.velocity = Vector3::zeros();
    state.angular_velocity = Vector3::zeros();
    sim.set_state(state);

    let dt = 0.005;
    let start = sim.time();
    let mut peaks = Vec::new();
    let mut times = Vec::new();
    let mut previous = offset.to_degrees();
    let mut rising = false;
    for _ in 0..(seconds / dt) as usize {
        sim.step(dt);
        let now = (sim.state().attitude.euler_angles().0 - rest_roll).to_degrees();
        if now < previous && rising {
            peaks.push(previous);
            times.push(sim.time() - start);
        }
        rising = now > previous;
        previous = now;
    }
    (peaks, times)
}

/// The lateral chain assembles at all, and mounts memory in all three modes.
///
/// Worth its own test because the assembly is where it first failed: the lateral
/// spectra would not fit at the order the vertical ones use, and the reason was a
/// high-frequency guard that never engaged for them. See
/// `cummins::MemoryOptions::fit_ceiling`.
#[test]
fn a_boat_carries_memory_in_sway_roll_and_yaw() {
    let (sim, _, _) = settled(quick());
    let telemetry = sim.telemetry();
    for key in [
        "radiation.sway.memory_force",
        "radiation.roll.memory_moment",
        "radiation.yaw.memory_moment",
        "radiation.heave.memory_force",
        "radiation.pitch.memory_moment",
    ] {
        assert!(
            telemetry.get(key).is_some(),
            "the seakeeping rig published no {key}"
        );
    }
    let error = telemetry
        .get("radiation.worst_fit_error")
        .expect("a fit error");
    assert!(
        error < 0.02,
        "the worst of nine memory fits was off by {:.2} %",
        100.0 * error
    );
    let pole = telemetry.get("radiation.slowest_pole").expect("a pole");
    assert!(
        pole < 0.0,
        "a memory model with a pole at {pole:.4} is unstable"
    );
}

/// What the load-time chain found out reaches the telemetry, per set.
///
/// The clamp count, the energy and reciprocity residuals and the passivity of
/// the fit were all computed at every load and read by nothing but a CLI
/// report; an assembly that had clamped the bow said nothing. Now they are
/// published beside the fit they fed, so a test — or a HUD — can ask whether
/// the water the boat is moving in was built on coefficients the theory owned.
/// The values asserted are what this hull gives, and the first thing the
/// diagnostic found is worth stating: no station is clamped and the vertical
/// identities hold to 1e-3 over the fitted band, but the lateral band reaches
/// about 20 rad/s — the yaw damping peaks late — and up there the multipole
/// series is truncated enough that the energy identity is off by 2.4 %. That
/// is the accuracy of the coefficients the lateral fit was actually fed, which
/// nobody could see before; the bound is set at twice it so that a hull or a
/// grid that made it worse is caught, not so that it passes.
#[test]
fn the_radiation_fit_publishes_what_fed_it() {
    let (sim, _, _) = settled(quick());
    let telemetry = sim.telemetry();
    for set in ["vertical", "lateral"] {
        let read = |name: &str| {
            telemetry
                .get(&format!("radiation.{set}.{name}"))
                .unwrap_or_else(|| panic!("the {set} set published no {name}"))
        };
        assert_eq!(
            read("clamped_stations"),
            0.0,
            "{set}: a station was clamped"
        );
        let energy = read("worst_energy_residual");
        let bound = if set == "vertical" { 0.01 } else { 0.05 };
        assert!(
            energy < bound,
            "{set} energy identity off by {energy:.3e} within the fitted band"
        );
        // Not zero: the lateral set's worst diagonal dips 1e-5 negative
        // somewhere on the grid, which is the rational fit's ripple and not a
        // model that drives the boat. A real violation is orders larger.
        let passivity = read("passivity_violation");
        assert!(
            passivity < 1e-4,
            "{set}: a diagonal memory would drive the boat by {passivity:.3e}"
        );
        let fit = read("worst_fit_error");
        assert!(fit < 0.02, "{set} fit off by {fit:.3e}");
    }
    assert!(
        telemetry.get("radiation.vertical.slenderness").is_some(),
        "the vertical sweep publishes its slenderness"
    );
}

/// A hull released from heel comes back and stops.
///
/// The property the whole lateral chain exists for. Hydrostatic stiffness alone
/// would return the boat to upright and then keep it swinging; only radiation
/// takes the energy out.
#[test]
fn a_heeled_hull_settles() {
    let (mut sim, rest_roll, _) = settled(quick());
    let (peaks, _) = roll_decay(&mut sim, rest_roll, 10.0_f64.to_radians(), 60.0);

    assert!(
        peaks.len() >= 4,
        "expected at least four roll peaks to measure, got {}",
        peaks.len()
    );
    for pair in peaks.windows(2) {
        assert!(
            pair[1] < pair[0],
            "the roll grew: {} then {}",
            pair[0],
            pair[1]
        );
    }
    // And it actually stops, rather than merely trending down.
    assert!(
        peaks[peaks.len() - 1] < 0.05 * peaks[0],
        "after {} peaks the roll was still {:.3} deg of {:.3}",
        peaks.len(),
        peaks[peaks.len() - 1],
        peaks[0]
    );
}

/// The roll mode has a yacht's period and a hull's decrement, at any amplitude.
///
/// What this test does *not* do is compare the decrement against
/// `ζ = B₄₄ / 2(I₄₄ + A₄₄)ω`, and the reason is worth recording because that
/// formula is the obvious thing to reach for and it is not an oracle here.
///
/// Measured against it, this hull decays at 0.179 with sway and yaw held, where
/// the formula says 0.144 — twenty-four per cent apart, and constant. Three
/// explanations were tested and killed. **Amplitude**: releasing from ten degrees
/// and from half a degree give 0.1785 and 0.1793, so the response is linear and
/// the mesh-clipped restoring is not the culprit. **The roll axis**: the centre of
/// gravity sits 0.2 m off the origin, worth 249 kg·m² against an inertia of 9969,
/// which is 2.5 per cent and not 24. **The module**: driven at a fixed frequency
/// it returns `K̂₄₄(jω)` to half a per cent, damping and added mass both, which
/// `radiation::the_module_reproduces_every_lateral_coefficient` now pins.
///
/// What is left is that the substitution itself does not apply. `A₄₄` falls from
/// 14 700 to 6 800 between 1.5 and 5 rad/s — it halves across the bandwidth of a
/// mode this damped — and the memory's slowest pole has a time constant of 0.75 s
/// against a roll period of 2.3 s. The rolling mode is therefore a genuinely
/// coupled rigid-body-and-memory mode, not a mass-spring-damper with coefficients
/// evaluated at one frequency. The sharp check belongs in the frequency domain,
/// where the frequency is imposed rather than emergent, and that is where it is.
///
/// So this test asserts what a free-decay experiment can honestly assert: the
/// period and the decrement are a yacht's, and neither depends on how hard the
/// boat was pushed.
#[test]
fn the_roll_mode_is_a_yacht_s_and_does_not_depend_on_amplitude() {
    let mut measurements = Vec::new();
    for offset in [8.0_f64, 1.0] {
        let (mut sim, rest_roll, _) = settled(quick());
        let (peaks, times) = roll_decay(&mut sim, rest_roll, offset.to_radians(), 60.0);
        assert!(
            peaks.len() >= 3,
            "a {offset} degree release gave only {} peaks",
            peaks.len()
        );

        // Skip the first interval: the release is a step, and its first half cycle
        // carries the transient of a hull at rest in a pose it did not choose.
        let period = times[2] - times[1];
        let decrement = (peaks[1] / peaks[2]).ln() / (2.0 * std::f64::consts::PI);

        // A yacht of this size rolls in a few seconds, and this canoe body's beam
        // to draft ratio of about eight puts it at the quick end.
        assert!(
            (1.5..4.0).contains(&period),
            "a roll period of {period:.2} s is not a yacht's"
        );
        assert!(
            (0.02..0.30).contains(&decrement),
            "a roll damping ratio of {decrement:.3} is not a hull's"
        );
        measurements.push((period, decrement));
    }

    let (long, small) = (measurements[0], measurements[1]);
    assert!(
        (long.0 - small.0).abs() < 0.05 * small.0,
        "the period moved from {:.3} s to {:.3} s with amplitude",
        small.0,
        long.0
    );
    assert!(
        (long.1 - small.1).abs() < 0.05 * small.1,
        "the decrement moved from {:.4} to {:.4} with amplitude — the roll mode \
         is not behaving linearly",
        small.1,
        long.1
    );
}

/// A beam sea rolls the boat, and calm water does not.
///
/// The pair, because either alone proves less: a rolling boat in waves could be
/// an unstable integrator, and a still boat in calm could be a module that
/// returns zero. Together they say the excitation arrives through the wave field
/// and only through it.
#[test]
fn a_beam_sea_rolls_the_boat_and_a_calm_does_not() {
    // A regular 3 s wave, 14 m long, 0.8 m high: near the roll period, and
    // longer than the beam. It was a 1.4 s, 3.1 m wave — shorter than the
    // beam — while `components: 1` put the component mid-band instead of at
    // the peak, and the test passed on it because 0.8 m of anything abeam
    // rolls a boat more than a degree.
    let sea = SeaState {
        significant_height: 0.8,
        peak_period: 3.0,
        components: 1,
        // From abeam, which is what makes this a roll test rather than a pitch one.
        heading: std::f64::consts::FRAC_PI_2,
        seed: 11,
        // Long-crested, so that "from abeam" means exactly abeam. Spreading
        // would only add roll, so this is also the harder condition to pass.
        spreading: 0.0,
    };
    let dt = 0.01;
    let swing = |mut sim: Sim| {
        for _ in 0..(30.0 / dt) as usize {
            sim.step(dt);
        }
        let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
        for _ in 0..(30.0 / dt) as usize {
            sim.step(dt);
            let roll = sim.state().attitude.euler_angles().0.to_degrees();
            low = low.min(roll);
            high = high.max(roll);
        }
        high - low
    };

    let calm = swing(in_water(
        StillWater::new(UniformWind::uniform(0.0, 0.0)),
        quick(),
    ));
    let wavy = swing(in_water(
        Seaway2D::new(UniformWind::uniform(0.0, 0.0), sea),
        quick(),
    ));

    assert!(
        calm < 0.05,
        "a hull in calm water swung {calm:.4} deg of roll"
    );
    assert!(
        wavy > 1.0,
        "a beam sea of 0.8 m only rolled the boat {wavy:.4} deg"
    );
}
