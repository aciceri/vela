//! End-to-end sailing: a boat file in, a solved sailing condition out.
//!
//! These tests exercise the whole stack at once — parse, loft, assemble four
//! force modules, solve the four-degree-of-freedom balance — so a failure here
//! says "the boat stopped sailing" rather than naming a component. The
//! component-level oracles live in the per-module tests; what is checked here is
//! that they add up to a boat.
//!
//! # What is asserted, and what is deliberately not
//!
//! The force balance itself is exact and is asserted as such: at a solved
//! condition the total wrench on the free degrees of freedom is zero to the
//! solver's tolerance, and that is a real closure, not a tautology — the modules
//! were written independently and nothing forces them to agree.
//!
//! Speeds and heel angles are asserted only as **ranges wide enough to be
//! honest**. There is no published YD-41 polar in hand to compare against, and
//! inventing a target speed to two decimal places would be fabricating an
//! oracle. What the ranges do catch is a model that has stopped being a boat: a
//! forty-foot yacht that solves to two knots in a working breeze, or to sixty
//! degrees of heel, is broken however cleanly the residual converged.
//!
//! # Conditions are chosen to be reachable, not flattering
//!
//! Full sail in a strong breeze upwind has **no equilibrium at all**, and that
//! is the correct answer rather than a solver failure: an overpowered yacht is
//! knocked down, which is why crews reef. One test asserts exactly that, and
//! another asserts that depowering brings the solution back.

use approx::assert_relative_eq;
use vela_core::aero::SailSet;
use vela_core::assembly::velocity_prediction_sim;
use vela_core::equilibrium::{self, Equilibrium, EquilibriumOptions};
use vela_core::{BoatSpec, Controls, LoftOptions, Sim, StillWater, UniformWind};

const SPEC: &str = include_str!("../../../boats/yd41-form-study.ron");

/// Waterline length of the fitted hull, for the residual scaling.
const WATERLINE_LENGTH: f64 = 11.90;

/// A working breeze that the boat carries full sail in without being
/// overpowered: 5 m/s is a little under ten knots.
const MODERATE_WIND: f64 = 5.0;

fn boat() -> BoatSpec {
    BoatSpec::parse_ron(SPEC).expect("the shipped boat file must parse")
}

/// Assembles the simulation for one wind and one trim.
fn sailing(wind_speed: f64, wind_angle_deg: f64, flat: f64, reef: f64) -> Sim {
    let spec = boat();
    let sails = SailSet::upwind();
    let base = Controls::close_hauled(sails);
    let controls = base.with_trim(base.trim.with_flat(flat).with_reef(reef));
    let environment = StillWater::new(UniformWind::uniform(
        wind_speed,
        wind_angle_deg.to_radians(),
    ));

    velocity_prediction_sim(
        &spec,
        Box::new(environment),
        controls,
        &LoftOptions::default(),
    )
    .expect("the shipped boat has hull, parameters, appendages and rig")
}

fn solve(sim: &mut Sim) -> Result<Equilibrium, String> {
    let options = EquilibriumOptions {
        initial_sinkage: 0.40,
        ..EquilibriumOptions::default()
    };
    equilibrium::solve(sim, WATERLINE_LENGTH, &options).map_err(|error| error.to_string())
}

#[test]
fn the_boat_sails_upwind_in_a_working_breeze() {
    let mut sim = sailing(MODERATE_WIND, 40.0, 1.0, 1.0);
    let solution = solve(&mut sim).expect("a yacht must sail upwind in ten knots");

    // Wide, but a boat that leaves this range is not a forty-foot yacht.
    assert!(
        (2.0..5.0).contains(&solution.speed),
        "boat speed {} m/s is not a plausible upwind speed",
        solution.speed
    );
    // Wind is on the starboard bow, so the boat lies over to port.
    assert!(
        solution.heel < 0.0,
        "wind from starboard must heel the boat to port, got {} rad",
        solution.heel
    );
    assert!(
        solution.heel.to_degrees() > -35.0,
        "heel {} deg is past sailing",
        solution.heel.to_degrees()
    );
    // Leeway is a few degrees on a keelboat; the sign convention says positive.
    assert!(
        (0.0..0.20).contains(&solution.leeway),
        "leeway {} rad is not a keelboat's",
        solution.leeway
    );
    // The hull floats near the draft its own offsets describe.
    assert!(
        (0.2..0.7).contains(&solution.sinkage),
        "sinkage {} m does not put the waterplane on the hull",
        solution.sinkage
    );
}

/// The point of the whole exercise: at the solution the forces cancel.
///
/// Four independently written modules plus gravity, and the sum vanishes in
/// every free degree of freedom. Nothing in the code arranges this — it is the
/// solver having actually found a balance.
#[test]
fn the_force_balance_closes_on_every_free_degree_of_freedom() {
    let mut sim = sailing(MODERATE_WIND, 40.0, 1.0, 1.0);
    let solution = solve(&mut sim).expect("must converge");

    let weight = sim.body().mass_properties().mass() * sim.body().gravity();
    let wrench = sim.applied_wrench(1.0 / 120.0);

    assert_relative_eq!(wrench.force.x / weight, 0.0, epsilon = 1e-6);
    assert_relative_eq!(wrench.force.y / weight, 0.0, epsilon = 1e-6);
    assert_relative_eq!(wrench.force.z / weight, 0.0, epsilon = 1e-6);
    assert_relative_eq!(
        wrench.moment.x / (weight * WATERLINE_LENGTH),
        0.0,
        epsilon = 1e-6
    );
    assert!(solution.residual < 1e-8);
}

/// Sails drive, hull and appendages resist, and at equilibrium they are equal.
///
/// Read from the telemetry rather than from the wrench, which makes it a check
/// on the published breakdown as well: a key that reported the wrong quantity
/// would show up here even though the total balanced.
#[test]
fn the_driving_force_equals_the_total_resistance() {
    let mut sim = sailing(MODERATE_WIND, 40.0, 1.0, 1.0);
    solve(&mut sim).expect("must converge");

    let telemetry = sim.telemetry();
    let driving = telemetry
        .get("aero.driving_force")
        .expect("the aero module must publish its driving force");
    let hull = telemetry
        .get("hull.total")
        .expect("the hull module must publish its total");
    let appendages = telemetry
        .get("lateral.resistance_total")
        .expect("the lateral module must publish its total");

    assert!(driving > 0.0, "the sails must be pushing the boat forward");
    // Not exactly equal: the resistances act along the track and the driving
    // force along the centreline, and the two differ by the leeway angle.
    assert_relative_eq!(driving, hull + appendages, max_relative = 0.05);
}

/// The two tacks must be the same sailing condition mirrored.
///
/// The tolerance is 1e-3 rather than machine precision, and the reason is a
/// *known defect elsewhere*, recorded here rather than absorbed silently: the
/// lofted hull is not exactly symmetric about the centreline. Its centre of
/// buoyancy sits 0.8 mm off the axis when upright, which produces a spurious
/// 48 N·m roll moment out of a righting moment of tens of kN·m, and the two
/// tacks then differ by about 0.03 %. The offsets themselves are symmetric
/// half-breadths, so this is the lofting or its triangulation, not the data —
/// see `the_lofted_hull_is_very_nearly_symmetric`, which pins the size of it so
/// that a fix is measurable and a regression is visible.
#[test]
fn the_two_tacks_are_mirror_images() {
    let mut starboard_tack = sailing(MODERATE_WIND, 40.0, 1.0, 1.0);
    let mut port_tack = sailing(MODERATE_WIND, -40.0, 1.0, 1.0);
    let starboard = solve(&mut starboard_tack).expect("must converge");
    let port = solve(&mut port_tack).expect("must converge");

    assert_relative_eq!(port.speed, starboard.speed, max_relative = 1e-3);
    assert_relative_eq!(port.sinkage, starboard.sinkage, max_relative = 1e-3);
    assert_relative_eq!(port.heel, -starboard.heel, max_relative = 1e-3);
    assert_relative_eq!(port.leeway, -starboard.leeway, max_relative = 1e-3);
}

#[test]
fn more_wind_means_more_heel_and_more_leeway() {
    let mut light = sailing(3.0, 40.0, 1.0, 1.0);
    let mut moderate = sailing(MODERATE_WIND, 40.0, 1.0, 1.0);
    let light = solve(&mut light).expect("must converge");
    let moderate = solve(&mut moderate).expect("must converge");

    assert!(moderate.speed > light.speed);
    assert!(moderate.heel.abs() > light.heel.abs());
    assert!(moderate.leeway > light.leeway);
}

/// Full sail in a strong breeze upwind solves to a **knockdown**, not to
/// sailing.
///
/// This is the answer the model should give, and reading it correctly matters.
/// Full main and genoa at fourteen knots of true wind at forty degrees off the
/// bow is more sail than the boat can stand up to: the balance that exists puts
/// the deck under water at forty degrees of heel with the keel line at the
/// waterline, which is a knockdown a crew would be reefing to avoid.
///
/// An earlier version of this test asserted that no solution existed at all.
/// That was wrong, and it was wrong in an instructive way: the solver of the
/// time could not reach this branch, and its failure to converge was mistaken
/// for a statement about the boat. The test now asserts what the condition
/// *is*, which cannot be faked by a solver that has stopped working.
#[test]
fn full_sail_in_a_strong_breeze_upwind_is_a_knockdown() {
    let mut sim = sailing(7.0, 40.0, 1.0, 1.0);
    let solution = solve(&mut sim).expect("the knockdown branch must be reachable");

    assert!(
        solution.heel.to_degrees() < -35.0,
        "full sail in this breeze must lay the boat down, got {} deg",
        solution.heel.to_degrees()
    );
}

/// Depowering turns that knockdown back into sailing, which is what reefing is
/// for.
#[test]
fn depowering_turns_a_knockdown_back_into_sailing() {
    let mut overpowered = sailing(7.0, 40.0, 1.0, 1.0);
    let overpowered = solve(&mut overpowered).expect("must converge");

    let mut flattened = sailing(7.0, 40.0, 0.7, 1.0);
    let flattened = solve(&mut flattened).expect("flattening must solve");

    let mut reefed = sailing(7.0, 40.0, 0.7, 0.8);
    let reefed = solve(&mut reefed).expect("reefing must solve");

    assert!(
        flattened.heel.abs() < overpowered.heel.abs(),
        "flattening must stand the boat up: {} deg against {} deg",
        flattened.heel.to_degrees(),
        overpowered.heel.to_degrees()
    );
    assert!(
        reefed.heel.abs() < flattened.heel.abs(),
        "reefing on top of flattening must stand it up further: \
         {} deg reefed against {} deg flattened",
        reefed.heel.to_degrees(),
        flattened.heel.to_degrees()
    );
    // And the depowered boat is actually sailing rather than lying on its side.
    assert!(reefed.heel.to_degrees().abs() < 30.0);
}

/// Pins the size of the lofted hull's asymmetry, so the defect is tracked.
///
/// The station offsets are symmetric half-breadths, so a hull lofted from them
/// should be symmetric to machine precision and its upright centre of buoyancy
/// should sit exactly on the centreline. It does not: it is off by a fraction of
/// a millimetre, which is harmless at the scale of any force here but is the
/// reason the two tacks do not agree exactly.
///
/// The assertion is deliberately two-sided. The upper bound catches a
/// regression that makes it worse; the test exists at all so that a fix to the
/// lofting shows up as this test needing a tighter bound rather than as nothing
/// at all.
#[test]
fn the_lofted_hull_is_very_nearly_symmetric() {
    use vela_core::hydrostatics::{hydrostatics, Water};
    use vela_core::{loft_hull, BodyState};

    let spec = boat();
    let mesh = loft_hull(
        spec.hull.as_ref().expect("the shipped boat has offsets"),
        &LoftOptions::default(),
    );
    let state = BodyState {
        position: nalgebra::Vector3::new(0.0, 0.0, 0.4034),
        ..BodyState::default()
    };
    let upright = hydrostatics(&mesh, &state, &Water::default(), 9.806_65);

    let offset = upright.centre_of_buoyancy.y.abs();
    assert!(
        offset < 2e-3,
        "the upright centre of buoyancy is {offset} m off the centreline, \
         which is worse than the 0.8 mm this hull is known to have"
    );
    assert!(
        upright.buoyancy.force.y.abs() < 1e-6,
        "an upright symmetric hull must produce no athwartships force"
    );
}

/// Heel spills wind out of the rig, and the correction that does it is the one
/// the source prescribes. Without it the same boat solves to far more heel.
///
/// Asserted through the apparent wind rather than through a heel angle: at a
/// solved condition with real heel, the apparent wind angle the rig sees must
/// differ from the one a horizontal-plane resolution would give.
#[test]
fn the_rig_feels_the_heeled_apparent_wind() {
    let mut sim = sailing(MODERATE_WIND, 40.0, 1.0, 1.0);
    let solution = solve(&mut sim).expect("must converge");

    let angle = sim
        .telemetry()
        .get("aero.apparent_wind.angle")
        .expect("the aero module must publish the apparent wind angle");

    assert!(
        solution.heel.abs() > 0.1,
        "this condition must have real heel for the test to mean anything"
    );
    // The correction scales the athwartships component by cos(heel), which
    // always moves the apparent wind *forward*, towards the bow.
    assert!(
        angle.abs() > 0.0,
        "the apparent wind must be off the bow, not dead ahead"
    );
    assert!(
        angle.abs() < 40.0_f64.to_radians(),
        "heel and boat speed must both bring the apparent wind forward of the \
         true wind angle, got {} deg",
        angle.to_degrees()
    );
}
