//! End-to-end vertical motion: a real boat file in, a settling hull out.
//!
//! These tests exercise the whole seakeeping chain at once — station offsets,
//! Lewis mapping, the Ursell-Tasai sectional solve, strip integration, Ogilvie's
//! relation, the rational fit, the state-space realisation, the added-mass matrix
//! and the integrator. Nothing is stubbed and nothing is a stored answer.
//!
//! What they defend is the property the whole phase exists for: **a disturbed
//! hull settles**. Before this chain there was nothing in the equations to
//! remove energy from heave or pitch, so a boat pushed down oscillated on
//! hydrostatic stiffness forever. Whether it settles, how fast, and whether the
//! rate is the one the frequency domain predicts are the three questions here.

use vela_core::assembly::{vertical_motion_sim, RadiationOptions};
use vela_core::env::Seaway2D;
use vela_core::seaway::{SeaState, Seaway};
use vela_core::{BoatSpec, LoftOptions, Sim, StillWater, UniformWind};

/// The boat shipped with hull offsets.
fn boat() -> BoatSpec {
    let text = include_str!("../../../boats/yd41-form-study.ron");
    BoatSpec::parse_ron(text).expect("the shipped boat file parses")
}

/// Cheaper than the default so that a test suite stays a test suite: a coarser
/// frequency grid and fewer multipoles. The physics does not change, only how
/// finely it is resolved — and `the_pipeline_is_insensitive_to_its_own_grid`
/// pins that claim rather than leaving it as an assertion.
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

fn assembled(options: RadiationOptions) -> Sim {
    in_water(StillWater::new(UniformWind::uniform(0.0, 0.0)), options)
}

/// The same boat in whatever water is handed to it.
fn in_water(env: impl vela_core::env::Environment + 'static, options: RadiationOptions) -> Sim {
    vertical_motion_sim(&boat(), Box::new(env), &LoftOptions::default(), options)
        .expect("the shipped boat admits a vertical-motion simulation")
}

/// Lets the hull find its own flotation before anything is asked of it.
///
/// The assembled state is upright at the origin, which is not where the boat
/// floats: it sinks to its waterline over the first few seconds. Measuring a
/// decay against the origin instead measures that sinkage, and an earlier
/// version of these tests did exactly that — a 0.1 m push looked like 0.4 m of
/// leftover motion, and the decrement over consecutive peaks came out ten times
/// too small because both peaks were riding a drift.
fn settled(options: RadiationOptions) -> (Sim, f64) {
    let mut sim = assembled(options);
    let dt = 0.01;
    for _ in 0..(12.0 / dt) as usize {
        sim.step(dt);
    }
    let waterline = sim.state().position.z;
    (sim, waterline)
}

/// Displaces the hull in heave and returns the successive peak amplitudes, m,
/// measured from the settled waterline.
fn heave_decay(sim: &mut Sim, waterline: f64, offset: f64, seconds: f64) -> Vec<f64> {
    let mut state = sim.state().clone();
    state.position.z = waterline + offset;
    state.velocity.z = 0.0;
    sim.set_state(state);

    let dt = 0.005;
    let mut peaks = Vec::new();
    let mut previous = offset;
    let mut rising = false;
    for _ in 0..(seconds / dt) as usize {
        sim.step(dt);
        let now = sim.state().position.z - waterline;
        if now < previous && rising && previous > 0.0 {
            peaks.push(previous);
        }
        rising = now > previous;
        previous = now;
    }
    peaks
}

/// The point of the entire phase: a pushed hull comes back and stops.
///
/// Without radiation the amplitude after six seconds would equal the amplitude
/// at the start, because hydrostatic stiffness is conservative and nothing else
/// in the vertical modes removes energy.
#[test]
fn a_hull_pushed_down_settles() {
    let (mut sim, waterline) = settled(quick());
    let offset = 0.10;
    let mut state = sim.state().clone();
    state.position.z = waterline + offset;
    state.velocity.z = 0.0;
    sim.set_state(state);

    let dt = 0.005;
    let mut worst_late = 0.0_f64;
    for step in 1..=(10.0 / dt) as usize {
        sim.step(dt);
        if step as f64 * dt > 6.0 {
            worst_late = worst_late.max((sim.state().position.z - waterline).abs());
        }
    }
    assert!(
        worst_late < 0.2 * offset,
        "six seconds after a {offset} m push the hull was still {worst_late} m from its waterline"
    );
}

/// And it settles at a rate the frequency domain agrees with.
///
/// The logarithmic decrement between consecutive peaks is `2πζ` directly — no
/// period and no natural frequency needed, because consecutive peaks are one
/// cycle apart by definition. The comparison is against the damping ratio
/// `vela-cli radiation` reports from `B₃₃` and the hydrostatic stiffness, 0.33
/// for this hull. The band is wide because a decrement over one cycle of a
/// heavily damped mode is not a precision instrument, and because heave here is
/// coupled to a free pitch rather than isolated.
#[test]
fn the_settling_rate_is_the_one_the_spectrum_predicted() {
    let (mut sim, waterline) = settled(quick());
    let offset = 0.10;
    let peaks = heave_decay(&mut sim, waterline, offset, 8.0);
    assert!(
        !peaks.is_empty(),
        "expected at least one peak after the release to measure against"
    );

    // The release point is itself a peak, so the first recorded peak is one
    // cycle later. That is the whole measurement: this mode loses most of its
    // amplitude per cycle, so a second recorded peak is already down in the
    // resolution of the mesh-clipped buoyancy.
    let (first, second) = (offset, peaks[0]);
    assert!(
        second < first,
        "the oscillation grew: {first} then {second}"
    );
    let measured = (first / second).ln() / (2.0 * std::f64::consts::PI);
    assert!(
        (0.15..0.60).contains(&measured),
        "damping ratio came out {measured}, against about 0.33 from the spectrum"
    );
}

/// Added mass is in the mass matrix, and it is not a rounding error.
///
/// For this hull the heave added mass is several times the boat's own mass — a
/// shallow, wide canoe body moves far more water than it displaces — so the
/// combined matrix should be dominated by it in that mode. If a refactor ever
/// dropped the added mass, the mass matrix would fall by a factor of four here
/// and every vertical period would come out short.
#[test]
fn the_mass_matrix_carries_the_added_mass() {
    let sim = assembled(quick());
    let matrix = sim.body().mass_matrix();
    let bare = sim.body().mass_properties().mass();

    assert!(
        matrix[(2, 2)] > 3.0 * bare,
        "heave mass {} is not much more than the boat's {bare}",
        matrix[(2, 2)]
    );
    // Symmetric, and coupling present: heaving this hull must pitch it.
    assert!(matrix[(2, 4)].abs() > 0.0);
    approx::assert_relative_eq!(matrix[(2, 4)], matrix[(4, 2)], max_relative = 1e-12);
}

/// The answer must not depend on how finely the pipeline was run.
///
/// The cheap options above exist so the suite stays fast, and this is what
/// entitles them to: the settling behaviour under the coarse grid has to match
/// the default one. It is also the test that would catch the grid being too
/// coarse to resolve the memory, which is a failure mode that otherwise shows up
/// only as a slightly wrong period much later.
#[test]
fn the_pipeline_is_insensitive_to_its_own_grid() {
    let coarse = heave_decay(&mut assembled(quick()), 0.10, 0.002, 8.0);
    let fine = heave_decay(
        &mut assembled(RadiationOptions::default()),
        0.10,
        0.002,
        8.0,
    );

    assert_eq!(
        coarse.len(),
        fine.len(),
        "the two grids disagreed on how many oscillations there were"
    );
    for (index, (rough, exact)) in coarse.iter().zip(fine.iter()).enumerate() {
        approx::assert_relative_eq!(rough, exact, max_relative = 0.05, epsilon = 1e-4);
        let _ = index;
    }
}

/// A boat in a long wave rides it.
///
/// The check that makes the whole seaway chain mean something, and the one every
/// part of it has to survive: the wave surface, the pressure head with its decay,
/// the clipping against a moving surface, the added mass in the mass matrix and
/// the fluid memory.
///
/// In a wave much longer than the hull the pressure field is nearly uniform along
/// the boat, so the hull is simply lifted and lowered by it — a floating body
/// follows a long wave with unit amplitude ratio and no phase lag. Anything else
/// means a term is wrong somewhere: too little response and the excitation is
/// being lost, too much and it is being double-counted.
///
/// 150 m is thirteen times this waterline, which for a linear wave is a fourteen
/// second period, and it is deliberately long: the interesting regime — waves of
/// the hull's own length, where the response peaks and then dies — is where the
/// model is *predicting* rather than obeying a limit, and a limit is what makes a
/// test.
#[test]
fn a_hull_rides_a_wave_much_longer_than_itself() {
    // A 600 m wave: thirteen times this waterline is not enough, because pitch
    // has its own natural period of 1.75 s and heave 2.0 s, and the quasi-static
    // limit wants to be far below both. Six hundred metres is a 19.6 s period.
    let wavelength = 600.0_f64;
    let period = (2.0 * std::f64::consts::PI * wavelength / 9.81).sqrt();
    let state = SeaState {
        significant_height: 0.638,
        peak_period: period,
        components: 1,
        heading: 0.0,
        seed: 7,
        // Long-crested. A response amplitude operator is defined for a
        // long-crested wave, and a single component with a direction drawn off
        // the heading would not be the wave this test is about.
        spreading: 0.0,
    };
    let sea = Seaway::new(state, 9.81);
    // One component of variance a^2/2 gives H_s = 4a/sqrt(2).
    let amplitude = sea.realised_height() * 2.0_f64.sqrt() / 4.0;

    let mut sim = in_water(
        Seaway2D::new(UniformWind::uniform(0.0, 0.0), state),
        quick(),
    );
    let dt = 0.01;
    for _ in 0..(60.0 / dt) as usize {
        sim.step(dt);
    }

    // Track the vertical motion of a point AMIDSHIPS against the surface under
    // it. Not the body origin: that sits at the aft perpendicular, six metres
    // from midships, where a fraction of a degree of pitch swamps the heave. An
    // earlier version of this test measured there and found a response half
    // again too large, which was entirely the arm.
    let mut lowest = (f64::INFINITY, 0.0);
    let mut highest = (f64::NEG_INFINITY, 0.0);
    for _ in 0..(1.5 * period / dt) as usize {
        sim.step(dt);
        let state = sim.state();
        let mid = state.position
            + state.attitude.to_rotation_matrix() * nalgebra::Vector3::new(5.95, 0.0, 0.0);
        let surface = sea.elevation(mid.x, mid.y, sim.time());
        if mid.z < lowest.0 {
            lowest = (mid.z, surface);
        }
        if mid.z > highest.0 {
            highest = (mid.z, surface);
        }
    }

    // Body z is down and elevation is up, so the two swings are opposite in sign
    // and equal in size: the hull rides the wave.
    let hull_swing = highest.0 - lowest.0;
    let wave_swing = lowest.1 - highest.1;
    assert!(
        wave_swing > 1.2 * amplitude,
        "the sampling window missed the wave: only {wave_swing:.3} m of a {amplitude:.3} m amplitude"
    );
    let ratio = hull_swing / wave_swing;
    assert!(
        (0.9..1.15).contains(&ratio),
        "in a {wavelength} m wave the hull followed the surface with ratio {ratio:.3}"
    );
}

/// A wave excites the boat at all, and a calm does not.
///
/// Cheap, and it guards the thing a subtle plumbing mistake would break silently:
/// if the environment's surface never reached the pressure integral, the boat in
/// a seaway would sit as still as the boat in a calm and every wave test that
/// measured a *ratio* would still pass.
#[test]
fn a_seaway_moves_the_boat_and_a_calm_does_not() {
    let excited = {
        let state = SeaState {
            significant_height: 1.5,
            peak_period: 6.0,
            components: 30,
            ..SeaState::default()
        };
        let mut sim = in_water(
            Seaway2D::new(UniformWind::uniform(0.0, 0.0), state),
            quick(),
        );
        let dt = 0.01;
        for _ in 0..(40.0 / dt) as usize {
            sim.step(dt);
        }
        let mut lowest = f64::INFINITY;
        let mut highest = f64::NEG_INFINITY;
        for _ in 0..(30.0 / dt) as usize {
            sim.step(dt);
            lowest = lowest.min(sim.state().position.z);
            highest = highest.max(sim.state().position.z);
        }
        highest - lowest
    };

    let calm = {
        let (mut sim, waterline) = settled(quick());
        let dt = 0.01;
        let mut worst = 0.0_f64;
        for _ in 0..(30.0 / dt) as usize {
            sim.step(dt);
            worst = worst.max((sim.state().position.z - waterline).abs());
        }
        2.0 * worst
    };

    assert!(
        excited > 0.5,
        "a 1.5 m sea moved the hull through only {excited:.3} m"
    );
    assert!(
        calm < 0.02,
        "a calm moved the hull through {calm:.3} m, so the sea is not the cause"
    );
}

/// The wave-following pose becomes an equilibrium as the wave lengthens.
///
/// The right way to ask whether a hull rides a wave in *both* vertical modes, and
/// the one that replaced a badly posed question.
///
/// The tempting test is to compare pitch against the wave slope and expect unity.
/// It fails, by a factor of about 4.6, and the factor is neither a bug nor
/// amplitude nonlinearity — it survives shrinking the wave to two millimetres and
/// it survives freeing surge. It is arithmetic. From the moment balance,
///
/// ```text
/// θ = s - (C₅₃/C₅₅)(h - ζ)
/// ```
///
/// and for this hull `C₅₃/C₅₅` is 0.17 per metre because the body origin sits
/// 5.5 m from the centre of flotation. The slope `s = k a` vanishes as the wave
/// lengthens while the heave residual stays proportional to `a`, so the *ratio*
/// diverges even as both terms go to zero: an eight-tenths of a per cent heave
/// residual produces 4.2 times a vanishing slope. Measured 4.22 against 4.24
/// predicted.
///
/// So the ratio is not a testable quantity. This is: hold the hull at the pose
/// the wave implies — sunk by the elevation, trimmed by the slope — and the net
/// wrench must go to zero as the wave lengthens. No ratio of vanishing
/// quantities, and it tests the thing that matters, which is that the excitation
/// and the restoring are the same integral seen from two sides.
#[test]
fn the_wave_following_pose_becomes_an_equilibrium_in_long_waves() {
    let (mut calm, _) = settled(quick());
    let rest = calm.state().clone();
    let rest_pitch = rest.attitude.euler_angles().1;
    // Scale force against a kilonewton and moment against the same over a
    // waterline length, so the two are comparable in one number.
    let norm = |w: &vela_core::wrench::Wrench| (w.force.z / 1e3).hypot(w.moment.y / 1.19e4);

    let mut previous = f64::INFINITY;
    for wavelength in [600.0_f64, 2400.0] {
        let period = (2.0 * std::f64::consts::PI * wavelength / 9.81).sqrt();
        let state = SeaState {
            significant_height: 0.4,
            peak_period: period,
            components: 1,
            heading: 0.0,
            seed: 7,
            // Long-crested: see the note in the quasi-static test above.
            spreading: 0.0,
        };
        let sea = Seaway::new(state, 9.81);
        let mut sim = in_water(
            Seaway2D::new(UniformWind::uniform(0.0, 0.0), state),
            quick(),
        );

        // A quarter period off the crest, where the slope is largest.
        let time = 0.25 * period;
        let hold = |sim: &mut Sim, at: &vela_core::state::BodyState| {
            while sim.time() < time {
                sim.step(0.05);
                sim.set_state(at.clone());
            }
        };

        let amidships = 5.95;
        let elevation = sea.elevation(amidships, 0.0, time);
        let slope = (sea.elevation(amidships + 1.0, 0.0, time)
            - sea.elevation(amidships - 1.0, 0.0, time))
            / 2.0;

        let mut at_rest = rest.clone();
        at_rest.velocity = nalgebra::Vector3::zeros();
        at_rest.angular_velocity = nalgebra::Vector3::zeros();
        sim.set_state(at_rest.clone());
        hold(&mut sim, &at_rest);
        let unbalanced = norm(&sim.applied_wrench(0.05));

        let mut following = at_rest.clone();
        following.position.z = rest.position.z - elevation;
        following.attitude =
            nalgebra::UnitQuaternion::from_euler_angles(0.0, rest_pitch + slope, 0.0);
        sim.set_state(following.clone());
        hold(&mut sim, &following);
        let balanced = norm(&sim.applied_wrench(0.05));

        let reduction = unbalanced / balanced.max(1e-12);
        assert!(
            reduction > 10.0,
            "at {wavelength} m the wave-following pose left {balanced:.3} of \
             {unbalanced:.3}, a reduction of only {reduction:.1}x"
        );
        assert!(
            balanced < previous,
            "a longer wave must leave less unbalanced, got {balanced:.4} after {previous:.4}"
        );
        previous = balanced;
    }
    let _ = &mut calm;
}
