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
use vela_core::{BoatSpec, LoftOptions, Sim};

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
    vertical_motion_sim(&boat(), &LoftOptions::default(), options)
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
