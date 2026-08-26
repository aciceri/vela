//! Validation of the Hazen sail aerodynamic model against the values the book
//! tabulates for its design yacht, plus the model properties that a plausible
//! bug would break.
//!
//! The external oracles are:
//!
//! - the YD-41 sail areas of Appendix 1 (`SAM = 46.8 m²`, `SAF = 41.3 m²`,
//!   `SA = 88.1 m²`), which pin the area formulas of Fig 8.19 against numbers
//!   the book computed itself, and
//! - Table 8.1 in full, transposed on the way in so that a coefficient here is
//!   a second reading of the page rather than a copy of the table in the module.
//!
//! Where the book gives no number the tests assert *relations* — an exact
//! scaling, a conservation, an ordering — never a recorded output of this code.
//!
//! # A note on the rig dimensions
//!
//! Appendix 1 publishes only `I`, `J`, `P`, `E`, `BMAX` and the spinnaker leech
//! for the YD-41. The Hazen model also wants `LPG`, `BAD`, `FA`, `EHM` and
//! `EMDC`, and inventing five dimensions and then asserting against them would
//! be self-referential. So: `LPG` is *derived* from a stated choice of headsail
//! (a 100 % jib, i.e. one whose area equals the foretriangle's), and the other
//! four are declared assumptions, used only by tests that check relations which
//! hold for any positive rig. No published value is ever compared against a
//! quantity that depends on an assumed dimension.

use approx::assert_relative_eq;
use vela_core::aero::{
    lift_coefficient, viscous_drag_coefficient, ApparentWind, EffectiveSpan, MizzenDimensions,
    RigDimensions, Sail, SailPlan, SailPlanError, SailSet, Trim,
};
use vela_core::AIR_DENSITY;

/// Apparent wind speed for the force tests, m/s. Arbitrary: every force test
/// here is a ratio or a sign, so the dynamic pressure cancels.
const WIND_SPEED: f64 = 8.0;

/// YD-41 rig dimensions.
///
/// Published in Appendix 1: `I`, `J`, `P`, `E`, `BMAX`, and the spinnaker leech
/// `ASL = 18.0 m`. The rest are marked below.
fn yd41_rig() -> RigDimensions {
    RigDimensions {
        main_hoist: 16.7,
        main_foot: 5.6,
        foretriangle_height: 16.2,
        foretriangle_base: 5.1,
        // Derived, not assumed: the perpendicular of a 100 % jib, the one whose
        // area equals the foretriangle's. Setting
        // `LPG = I·J/√(I² + J²) = 4.865 m` makes `A_J = A_F` identically, which
        // `a_hundred_percent_jib_has_exactly_the_foretriangle_area` then uses
        // as an independent check on the jib area formula.
        jib_perpendicular: 16.2 * 5.1 / 16.2_f64.hypot(5.1),
        spinnaker_leech: 18.0,
        // Assumed, not published. Only relations are asserted on these.
        boom_above_sheer: 1.0,
        max_beam: 4.20,
        average_freeboard: 1.2,
        mast_height_above_sheer: 17.7,
        mast_diameter: 0.15,
        mizzen: None,
    }
}

fn upwind_plan() -> SailPlan {
    SailPlan::new(&yd41_rig(), SailSet::upwind()).expect("the YD-41 is a sloop with a jib")
}

fn downwind_plan() -> SailPlan {
    SailPlan::new(&yd41_rig(), SailSet::downwind()).expect("the YD-41 carries a spinnaker")
}

fn wind_at(degrees: f64) -> ApparentWind {
    ApparentWind {
        speed: WIND_SPEED,
        angle: degrees.to_radians(),
    }
}

// ---------------------------------------------------------------------------
// The published cross-checks: areas against the book's own numbers.
// ---------------------------------------------------------------------------

/// `A_M = 0.5·P·E = 0.5 · 16.7 · 5.6 = 46.76 m²` against the published
/// `SAM = 46.8 m²`. The published figure is rounded to three significant
/// figures, so the tolerance is one part in a thousand — measured discrepancy
/// 8.5e-4, i.e. the rounding and nothing else.
#[test]
fn mainsail_area_matches_the_published_sail_area() {
    let area = yd41_rig()
        .area(Sail::Main)
        .expect("a sloop carries a mainsail");
    assert_relative_eq!(area, 46.8, max_relative = 1e-3);
}

/// `A_F = 0.5·I·J = 0.5 · 16.2 · 5.1 = 41.31 m²` against the published
/// `SAF = 41.3 m²`. Measured discrepancy 2.4e-4.
#[test]
fn foretriangle_area_matches_the_published_sail_area() {
    assert_relative_eq!(yd41_rig().foretriangle_area(), 41.3, max_relative = 1e-3);
}

/// The nominal area is the sum of the two, and the book's `SA = 88.1 m²` is
/// exactly that sum — which also confirms that `A_N` for a sloop is
/// `A_F + A_M` with no third term. Measured `A_N = 88.07 m²`.
#[test]
fn nominal_area_matches_the_published_total_sail_area() {
    assert_relative_eq!(yd41_rig().nominal_area(), 88.1, max_relative = 1e-3);
}

/// The jib area formula runs on the luff length `√(I² + J²)`, not on `I`. With
/// `LPG` set to make the jib exactly 100 % of the foretriangle the two areas
/// must coincide identically — an algebraic identity, so the tolerance is
/// machine epsilon. Using `I` in place of the luff would miss by 4.6 %.
#[test]
fn a_hundred_percent_jib_has_exactly_the_foretriangle_area() {
    let rig = yd41_rig();
    let jib = rig.area(Sail::Jib).expect("the rig carries a jib");
    assert_relative_eq!(jib, rig.foretriangle_area(), max_relative = 1e-12);
}

/// The nominal area is a property of the rig, not of what is hoisted: it is the
/// reference for every coefficient, so a sail set must not be able to move it.
/// A spinnaker is bigger than the whole foretriangle it replaces, so if the set
/// leaked into `A_N` this would be off by tens of per cent.
#[test]
fn the_nominal_area_is_the_same_whatever_is_set() {
    let rig = yd41_rig();
    let reference = upwind_plan().nominal_area();
    assert_relative_eq!(
        downwind_plan().nominal_area(),
        reference,
        max_relative = 1e-12
    );
    let bare = SailPlan::new(&rig, SailSet::none()).expect("bare poles are a legal sail set");
    assert_relative_eq!(bare.nominal_area(), reference, max_relative = 1e-12);
    // ... and it is not the hoisted area, which the spinnaker does change.
    assert!(downwind_plan().set_area() > upwind_plan().set_area());
}

// ---------------------------------------------------------------------------
// Table 8.1, read a second time and transposed.
// ---------------------------------------------------------------------------

/// Table 8.1, one row per sail, in the angle order 27°, 50°, 80°, 100°, 180°.
/// The module stores it one row per angle; transposing it here means a
/// transcription error on either side shows up as a mismatch instead of being
/// copied through.
#[rustfmt::skip]
const TABULATED: [(Sail, [f64; 5], [f64; 5]); 5] = [
    //  sail                     lift                          viscous drag
    (Sail::Main,           [1.5, 1.5,  0.95, 0.85, 0.0], [0.02, 0.15, 0.8,  1.0, 0.9 ]),
    (Sail::Jib,            [1.5, 0.5,  0.3,  0.0,  0.0], [0.02, 0.25, 0.15, 0.0, 0.0 ]),
    (Sail::Spinnaker,      [0.0, 1.5,  1.0,  0.85, 0.0], [0.0,  0.25, 0.9,  1.2, 0.66]),
    (Sail::Mizzen,         [1.3, 1.4,  1.0,  0.8,  0.0], [0.02, 0.15, 0.75, 1.0, 0.8 ]),
    (Sail::MizzenStaysail, [0.0, 0.75, 1.0,  0.8,  0.0], [0.0,  0.1,  0.75, 1.0, 0.0 ]),
];

const ANGLES: [f64; 5] = [27.0, 50.0, 80.0, 100.0, 180.0];

#[test]
fn every_coefficient_at_every_tabulated_angle_is_the_published_value() {
    for (sail, lift, drag) in TABULATED {
        for (index, degrees) in ANGLES.into_iter().enumerate() {
            let radians = degrees.to_radians();
            assert_relative_eq!(
                lift_coefficient(sail, radians),
                lift[index],
                epsilon = 1e-12
            );
            assert_relative_eq!(
                viscous_drag_coefficient(sail, radians),
                drag[index],
                epsilon = 1e-12
            );
        }
    }
}

/// Below 27° the curves are held, which is the book's modelling choice and not
/// a clamp of convenience. Extrapolating the first two rows instead would give
/// the jib `C_L = 2.24` at 10° — more lift than any sail in the table makes
/// anywhere — and the main `C_DP = -0.076`, a negative drag. Which is how you
/// know the choice matters.
#[test]
fn coefficients_are_held_below_twenty_seven_degrees() {
    for (sail, lift, drag) in TABULATED {
        for degrees in [26.9_f64, 20.0, 10.0, 0.0] {
            let radians = degrees.to_radians();
            assert_relative_eq!(lift_coefficient(sail, radians), lift[0], epsilon = 1e-12);
            assert_relative_eq!(
                viscous_drag_coefficient(sail, radians),
                drag[0],
                epsilon = 1e-12
            );
        }
    }
}

/// The model has no port/starboard asymmetry, and an angle beyond a half turn
/// is the same angle on the other tack. A heading integrated over many tacks
/// must not be able to walk off the end of the table.
#[test]
fn the_coefficients_are_symmetric_about_the_centreline() {
    let jib = |degrees: f64| lift_coefficient(Sail::Jib, degrees.to_radians());
    assert_relative_eq!(jib(-50.0), jib(50.0), epsilon = 1e-12);
    assert_relative_eq!(jib(310.0), jib(50.0), epsilon = 1e-12);
    assert_relative_eq!(jib(-310.0), jib(50.0), epsilon = 1e-12);

    // Mirroring the wind is a bit-exact operation, not an approximate one: the
    // fold takes the magnitude of the angle before anything else touches it.
    let plan = upwind_plan();
    let trim = Trim::full(EffectiveSpan::CloseHauled);
    let port = plan.forces(wind_at(-35.0), trim, AIR_DENSITY);
    let starboard = plan.forces(wind_at(35.0), trim, AIR_DENSITY);
    assert_eq!(port.driving_force, starboard.driving_force);
    assert_eq!(port.heeling_force, starboard.heeling_force);
    assert_eq!(
        port.heeling_moment_about_waterline,
        starboard.heeling_moment_about_waterline
    );
}

// ---------------------------------------------------------------------------
// Induced drag.
// ---------------------------------------------------------------------------

/// `C_DI = C_L²·(1/(π·AR) + 0.005)`: quadratic in lift, and falling with
/// aspect ratio. Both halves matter — a linear induced drag would still look
/// plausible on a polar plot.
#[test]
fn induced_drag_grows_as_the_square_of_lift_and_falls_with_aspect_ratio() {
    let plan = upwind_plan();
    let span = EffectiveSpan::Eased;

    let single = plan.induced_drag_coefficient(1.0, span);
    assert_relative_eq!(
        plan.induced_drag_coefficient(2.0, span),
        4.0 * single,
        max_relative = 1e-12
    );
    assert_relative_eq!(
        plan.induced_drag_coefficient(0.5, span),
        0.25 * single,
        max_relative = 1e-12
    );
    // Sign-blind: the sail plan does not care which way the lift points.
    assert_relative_eq!(
        plan.induced_drag_coefficient(-1.0, span),
        single,
        max_relative = 1e-12
    );

    // A taller mast on the same sail area is a higher aspect ratio and less
    // induced drag for the same lift. Measured: AR 4.30 vs 17.22 for a mast
    // twice as tall, C_DI 0.0790 vs 0.0235.
    let mut taller = yd41_rig();
    taller.mast_height_above_sheer *= 2.0;
    let taller = SailPlan::new(&taller, SailSet::upwind()).expect("a taller rig is still a rig");
    assert!(taller.aspect_ratio(span) > plan.aspect_ratio(span));
    assert!(taller.induced_drag_coefficient(1.0, span) < plan.induced_drag_coefficient(1.0, span));
}

/// The two aspect-ratio definitions differ by the freeboard, and the
/// close-hauled one — where the jib closes the gap to the deck — must be the
/// larger and so the cheaper in induced drag. Measured: AR 4.91 close-hauled
/// against 4.30 eased.
#[test]
fn the_close_hauled_span_reaches_further_than_the_eased_one() {
    let plan = upwind_plan();
    let close_hauled = plan.aspect_ratio(EffectiveSpan::CloseHauled);
    let eased = plan.aspect_ratio(EffectiveSpan::Eased);
    assert!(close_hauled > eased, "{close_hauled} should exceed {eased}");
    assert!(
        plan.induced_drag_coefficient(1.0, EffectiveSpan::CloseHauled)
            < plan.induced_drag_coefficient(1.0, EffectiveSpan::Eased)
    );

    // The whole difference is the freeboard: with no freeboard the two
    // definitions coincide, which pins which dimension each one uses.
    let mut flush = yd41_rig();
    flush.average_freeboard = 0.0;
    let flush = SailPlan::new(&flush, SailSet::upwind()).expect("a flush-decked rig is a rig");
    assert_relative_eq!(
        flush.aspect_ratio(EffectiveSpan::CloseHauled),
        flush.aspect_ratio(EffectiveSpan::Eased),
        max_relative = 1e-12
    );
}

// ---------------------------------------------------------------------------
// Flat and reef: the asymmetry.
// ---------------------------------------------------------------------------

/// The exact invariant behind "flattening rotates the resultant forward": lift
/// falls as `F`, induced drag as `F²`. Everything else about flattening follows
/// from these two, so they are asserted to machine precision.
#[test]
fn flattening_scales_lift_linearly_and_induced_drag_quadratically() {
    let plan = upwind_plan();
    let full = Trim::full(EffectiveSpan::CloseHauled);
    let wind = wind_at(27.0);

    let reference = plan.forces(wind, full, AIR_DENSITY);
    for flat in [0.9, 0.8, 0.5] {
        let flattened = plan.forces(wind, full.with_flat(flat), AIR_DENSITY);
        assert_relative_eq!(flattened.lift, flat * reference.lift, max_relative = 1e-12);
        assert_relative_eq!(
            flattened.induced_drag_coefficient,
            flat * flat * reference.induced_drag_coefficient,
            max_relative = 1e-12
        );
        // Flattening takes camber out, not cloth: the viscous drag of the same
        // area of sail in the same flow is unchanged, and so is the arm.
        assert_relative_eq!(
            flattened.viscous_drag_coefficient,
            reference.viscous_drag_coefficient,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            flattened.centre_of_effort_height,
            reference.centre_of_effort_height,
            max_relative = 1e-12
        );
    }
}

/// The emergent consequence, and the reason a crew flattens before reefing:
/// close-hauled, a modest flattening buys driving force per unit heeling force.
///
/// Measured for this rig at 27°, close-hauled: `driving/heeling` = 0.2977 at
/// `F = 1`, 0.2994 at `F = 0.9`, 0.2989 at `F = 0.8`. The gain is small and it
/// is *not* monotone — it peaks near `F ≈ 0.87` and has reversed by `F = 0.5`
/// (0.2698), because the viscous and parasitic drag do not flatten and
/// eventually dominate what is left. That is the model being honest about a
/// real limit, not a bug: past a point you are throwing away lift for nothing,
/// which is when a crew reefs instead.
#[test]
fn modest_flattening_improves_driving_force_per_unit_heeling_force() {
    let plan = upwind_plan();
    let full = Trim::full(EffectiveSpan::CloseHauled);
    let wind = wind_at(27.0);

    let ratio = |flat: f64| {
        let forces = plan.forces(wind, full.with_flat(flat), AIR_DENSITY);
        forces.driving_force / forces.heeling_force
    };
    assert!(ratio(0.9) > ratio(1.0), "{} vs {}", ratio(0.9), ratio(1.0));
    assert!(ratio(0.8) > ratio(1.0), "{} vs {}", ratio(0.8), ratio(1.0));

    // Drag falls proportionally more than lift, which is the same statement
    // about the resultant rotating forward. Measured at F = 0.9: drag ratio
    // 0.8918 against a lift ratio of 0.9.
    let reference = plan.forces(wind, full, AIR_DENSITY);
    let flattened = plan.forces(wind, full.with_flat(0.9), AIR_DENSITY);
    assert!(flattened.drag / reference.drag < flattened.lift / reference.lift);
}

/// Reefing scales the coefficients by `R²` and the arm by `R`, so the moment
/// falls faster than the force. That is the point of reefing, and it is what
/// distinguishes it from flattening, which leaves the arm alone.
///
/// Measured at 27° close-hauled with `R = 0.7`: the heeling force falls to
/// 0.493 of full sail and the heeling moment to 0.367.
#[test]
fn reefing_reduces_the_heeling_moment_faster_than_the_heeling_force() {
    let plan = upwind_plan();
    let full = Trim::full(EffectiveSpan::CloseHauled);
    let wind = wind_at(27.0);

    let reference = plan.forces(wind, full, AIR_DENSITY);
    let reefed = plan.forces(wind, full.with_reef(0.7), AIR_DENSITY);

    let force_fraction = reefed.heeling_force / reference.heeling_force;
    let moment_fraction =
        reefed.heeling_moment_about_waterline / reference.heeling_moment_about_waterline;
    assert!(force_fraction < 1.0);
    assert!(
        moment_fraction < force_fraction,
        "moment fell to {moment_fraction} but force only to {force_fraction}"
    );

    // The extra reduction is exactly the arm coming down, and the arm comes
    // down by R on the sail plan only — the freeboard does not reef.
    let rig = yd41_rig();
    let expected_arm = 0.7 * plan.centre_of_effort_height() + rig.average_freeboard;
    assert_relative_eq!(
        reefed.centre_of_effort_height,
        expected_arm,
        max_relative = 1e-12
    );
    assert_relative_eq!(
        moment_fraction,
        force_fraction * expected_arm / reference.centre_of_effort_height,
        max_relative = 1e-12
    );

    // Flattening, by contrast, cannot buy this: same lift reduction, same arm.
    let flattened = plan.forces(wind, full.with_flat(0.49), AIR_DENSITY);
    assert_relative_eq!(
        flattened.centre_of_effort_height,
        reference.centre_of_effort_height,
        max_relative = 1e-12
    );
}

/// Reefing does not touch the mast and the topsides, because reefing does not
/// take them down. Struck sails leave exactly the windage behind, and that is
/// the drag a yacht is left with lying to a gale.
#[test]
fn reefing_never_reduces_the_drag_below_the_rigs_windage() {
    let plan = upwind_plan();
    let wind = wind_at(27.0);
    let struck = plan.forces(
        wind,
        Trim::full(EffectiveSpan::Eased).with_reef(0.0),
        AIR_DENSITY,
    );

    let dynamic_pressure = 0.5 * AIR_DENSITY * WIND_SPEED * WIND_SPEED;
    let windage = dynamic_pressure * plan.nominal_area() * plan.parasitic_drag_coefficient();
    assert_relative_eq!(struck.lift, 0.0, epsilon = 1e-12);
    assert_relative_eq!(struck.drag, windage, max_relative = 1e-12);
    assert_relative_eq!(
        struck.parasitic_drag_coefficient,
        plan.parasitic_drag_coefficient(),
        max_relative = 1e-12
    );

    // Bare poles must give the same thing: striking the sails and reefing them
    // to nothing are the same aerodynamic state.
    let bare = SailPlan::new(&yd41_rig(), SailSet::none()).expect("bare poles are a legal set");
    let bare = bare.forces(wind, Trim::full(EffectiveSpan::Eased), AIR_DENSITY);
    assert_relative_eq!(bare.drag, windage, max_relative = 1e-12);
}

// ---------------------------------------------------------------------------
// Forces and their resolution.
// ---------------------------------------------------------------------------

/// Dead downwind every lift coefficient in Table 8.1 is zero, so the whole
/// aerodynamic force is drag: it all drives, and none of it heels. A model that
/// leaked any lift into 180° would show up here as a heeling force on a run.
#[test]
fn dead_downwind_the_force_is_all_drag_and_all_of_it_drives() {
    for plan in [upwind_plan(), downwind_plan()] {
        let forces = plan.forces(
            wind_at(180.0),
            Trim::full(EffectiveSpan::Eased),
            AIR_DENSITY,
        );
        assert_relative_eq!(forces.lift_coefficient, 0.0, epsilon = 1e-12);
        assert_relative_eq!(forces.lift, 0.0, epsilon = 1e-12);
        assert_relative_eq!(forces.induced_drag_coefficient, 0.0, epsilon = 1e-12);

        assert!(forces.drag > 0.0, "the rig still has windage on a run");
        assert_relative_eq!(forces.driving_force, forces.drag, max_relative = 1e-12);
        assert_relative_eq!(forces.heeling_force, 0.0, epsilon = 1e-9);
        assert_relative_eq!(forces.heeling_moment_about_waterline, 0.0, epsilon = 1e-9);
    }
}

/// Pinched up inside the close-hauled angle, `D·cos β` beats `L·sin β` and the
/// driving force goes negative — the sails stop the boat. The sign convention
/// is load-bearing here, so it is pinned rather than clamped away. Measured at
/// 10°: driving coefficient −0.011 against +0.435 at 27°.
#[test]
fn pinching_takes_the_driving_force_negative() {
    let plan = upwind_plan();
    let trim = Trim::full(EffectiveSpan::CloseHauled);
    let close_hauled = plan.forces(wind_at(27.0), trim, AIR_DENSITY);
    let pinched = plan.forces(wind_at(10.0), trim, AIR_DENSITY);

    assert!(close_hauled.driving_force > 0.0);
    assert!(
        pinched.driving_force < 0.0,
        "pinching gave {} N of drive",
        pinched.driving_force
    );
    // The coefficients are held below 27°, so it is the *resolution* that
    // changed sign, not the table.
    assert_relative_eq!(
        pinched.lift_coefficient,
        close_hauled.lift_coefficient,
        max_relative = 1e-12
    );
}

/// Forces scale with the dynamic pressure and with nothing else: doubling the
/// apparent wind speed quadruples every force at a fixed angle and trim.
#[test]
fn forces_scale_with_dynamic_pressure() {
    let plan = upwind_plan();
    let trim = Trim::full(EffectiveSpan::CloseHauled);
    let slow = plan.forces(wind_at(35.0), trim, AIR_DENSITY);
    let fast = plan.forces(
        ApparentWind {
            speed: 2.0 * WIND_SPEED,
            angle: 35.0_f64.to_radians(),
        },
        trim,
        AIR_DENSITY,
    );
    assert_relative_eq!(
        fast.driving_force,
        4.0 * slow.driving_force,
        max_relative = 1e-12
    );
    assert_relative_eq!(
        fast.heeling_force,
        4.0 * slow.heeling_force,
        max_relative = 1e-12
    );
    // The coefficients are Reynolds-independent in this model, so they must not
    // move with speed at all.
    assert_relative_eq!(
        fast.lift_coefficient,
        slow.lift_coefficient,
        max_relative = 1e-12
    );
}

/// A trim control outside its physical range is a caller bug, but a solver
/// searching for an equilibrium will overshoot its bracket. Clamping keeps the
/// sail plan physical: no factor may invent lift the sail cannot make.
#[test]
fn trim_factors_outside_their_range_are_clamped() {
    let plan = upwind_plan();
    let full = Trim::full(EffectiveSpan::CloseHauled);
    let wind = wind_at(30.0);
    let reference = plan.forces(wind, full, AIR_DENSITY);

    for silly in [full.with_flat(1.5), full.with_reef(2.0)] {
        let clamped = plan.forces(wind, silly, AIR_DENSITY);
        assert_relative_eq!(clamped.lift, reference.lift, max_relative = 1e-12);
    }
    let negative = plan.forces(wind, full.with_reef(-0.5), AIR_DENSITY);
    assert_relative_eq!(negative.lift, 0.0, epsilon = 1e-12);
}

// ---------------------------------------------------------------------------
// The sail set is a choice, and the choice has consequences.
// ---------------------------------------------------------------------------

/// The whole reason a sail set is modelled explicitly: the right choice depends
/// on the apparent wind angle. A spinnaker is worth far more than a jib on a
/// broad reach and worth nothing at all close-hauled, where Table 8.1 gives it
/// `C_L = C_DP = 0` — a spinnaker set upwind is a mainsail alone.
///
/// Measured driving coefficients: at 120°, 1.95 under spinnaker against 0.61
/// under jib; at 27°, 0.225 against 0.435.
#[test]
fn a_spinnaker_wins_off_the_wind_and_loses_close_hauled() {
    let jib = upwind_plan();
    let spinnaker = downwind_plan();

    let broad = |plan: &SailPlan| {
        plan.forces(
            wind_at(120.0),
            Trim::full(EffectiveSpan::Eased),
            AIR_DENSITY,
        )
        .driving_force
    };
    assert!(
        broad(&spinnaker) > broad(&jib),
        "spinnaker gave {} N against the jib's {} N at 120°",
        broad(&spinnaker),
        broad(&jib)
    );

    let close = |plan: &SailPlan| {
        plan.forces(
            wind_at(27.0),
            Trim::full(EffectiveSpan::CloseHauled),
            AIR_DENSITY,
        )
        .driving_force
    };
    assert!(
        close(&spinnaker) < close(&jib),
        "spinnaker gave {} N against the jib's {} N at 27°",
        close(&spinnaker),
        close(&jib)
    );

    // And it loses for the specific reason the table gives: at 27° the
    // spinnaker contributes nothing, so the set is the mainsail alone.
    let main_only = SailPlan::new(&yd41_rig(), SailSet::none().with(Sail::Main))
        .expect("a mainsail alone is a legal set");
    assert_relative_eq!(close(&spinnaker), close(&main_only), max_relative = 1e-12);
}

/// A spinnaker's centre of effort is up in the head — `0.59·I` against the
/// jib's `0.39·I` — so a spinnaker set heels a yacht through a longer arm than
/// its area alone would suggest.
#[test]
fn the_spinnaker_centre_of_effort_is_higher_than_the_headsails() {
    let rig = yd41_rig();
    let spinnaker = rig
        .centre_of_effort_height(Sail::Spinnaker)
        .expect("the rig carries a spinnaker");
    let jib = rig
        .centre_of_effort_height(Sail::Jib)
        .expect("the rig carries a jib");
    assert_relative_eq!(
        spinnaker,
        0.59 * rig.foretriangle_height,
        max_relative = 1e-12
    );
    assert_relative_eq!(jib, 0.39 * rig.foretriangle_height, max_relative = 1e-12);
    assert!(downwind_plan().centre_of_effort_height() > upwind_plan().centre_of_effort_height());
}

/// The plan's centre of effort is the area-weighted mean of the set sails'.
/// Checked against the arithmetic done by hand, which is the only way to catch
/// a weighting that quietly used the nominal area instead of the hoisted one.
#[test]
fn the_centre_of_effort_is_the_area_weighted_mean_of_the_set_sails() {
    let rig = yd41_rig();
    let main_area = rig.area(Sail::Main).unwrap();
    let jib_area = rig.area(Sail::Jib).unwrap();
    let main_centre = rig.centre_of_effort_height(Sail::Main).unwrap();
    let jib_centre = rig.centre_of_effort_height(Sail::Jib).unwrap();

    let expected = (main_area * main_centre + jib_area * jib_centre) / (main_area + jib_area);
    assert_relative_eq!(
        upwind_plan().centre_of_effort_height(),
        expected,
        max_relative = 1e-12
    );
    // Weighting by the nominal area instead would be a different number, so
    // the test above actually discriminates.
    assert!((main_area + jib_area - rig.nominal_area()).abs() < 1e-9);
    assert_relative_eq!(main_centre, 0.39 * rig.main_hoist + rig.boom_above_sheer);
}

// ---------------------------------------------------------------------------
// Rigs the model cannot describe say so.
// ---------------------------------------------------------------------------

/// A sloop has no mizzen, and setting one must be an error rather than a sail
/// of zero area quietly diluting the coefficient sums.
#[test]
fn a_sail_the_rig_does_not_carry_cannot_be_set() {
    let sloop = yd41_rig();
    let error = SailPlan::new(&sloop, SailSet::upwind().with(Sail::Mizzen))
        .expect_err("a sloop has no mizzen");
    assert_eq!(error, SailPlanError::SailNotCarried(Sail::Mizzen));
    assert!(sloop.area(Sail::Mizzen).is_none());
}

/// The mizzen staysail's coefficients are transcribed but its area is not part
/// of this model, so it cannot be set. The gap is reported, not papered over.
#[test]
fn the_mizzen_staysail_has_coefficients_but_no_area() {
    let ketch = ketch_rig();
    assert!(ketch.area(Sail::MizzenStaysail).is_none());
    // Its centre of effort *is* defined — the asymmetry is real, and it is the
    // area formula alone that is missing.
    assert!(ketch
        .centre_of_effort_height(Sail::MizzenStaysail)
        .is_some());
    assert_relative_eq!(
        lift_coefficient(Sail::MizzenStaysail, 80.0_f64.to_radians()),
        1.0
    );

    let error = SailPlan::new(&ketch, SailSet::upwind().with(Sail::MizzenStaysail))
        .expect_err("the staysail area is unavailable");
    assert_eq!(error, SailPlanError::AreaUnavailable(Sail::MizzenStaysail));
}

/// Dimensions that would divide by zero are refused at construction rather than
/// producing an infinity several call frames later.
#[test]
fn a_rig_that_would_divide_by_zero_is_refused() {
    let mut no_mast = yd41_rig();
    no_mast.mast_height_above_sheer = 0.0;
    assert!(matches!(
        SailPlan::new(&no_mast, SailSet::upwind()),
        Err(SailPlanError::Dimension { .. })
    ));

    let mut no_foretriangle = yd41_rig();
    no_foretriangle.foretriangle_base = 0.0;
    assert!(matches!(
        SailPlan::new(&no_foretriangle, SailSet::upwind()),
        Err(SailPlanError::Dimension { .. })
    ));

    // A spinnaker leech is only required when a spinnaker is set.
    let mut no_spinnaker = yd41_rig();
    no_spinnaker.spinnaker_leech = 0.0;
    assert!(SailPlan::new(&no_spinnaker, SailSet::upwind()).is_ok());
    assert!(SailPlan::new(&no_spinnaker, SailSet::downwind()).is_err());
}

/// A yawl or ketch: the mizzen is `0.5·PY·EY` and it enters the nominal area,
/// so every coefficient on the boat is referenced to the larger plan.
#[test]
fn a_mizzen_enters_the_nominal_area() {
    let ketch = ketch_rig();
    let mizzen_area = ketch.area(Sail::Mizzen).expect("a ketch has a mizzen");
    assert_relative_eq!(mizzen_area, 0.5 * 8.0 * 3.0, max_relative = 1e-12);
    assert_relative_eq!(
        ketch.nominal_area(),
        yd41_rig().nominal_area() + mizzen_area,
        max_relative = 1e-12
    );

    let plan = SailPlan::new(&ketch, SailSet::upwind().with(Sail::Mizzen))
        .expect("a ketch may set its mizzen");
    // Adding a mizzen to the same hull dilutes the windage coefficient,
    // because C_D0 is referenced to the bigger nominal area while the mast and
    // topsides are unchanged.
    assert!(plan.parasitic_drag_coefficient() < upwind_plan().parasitic_drag_coefficient());
}

/// Not a published rig: a mizzen bolted onto the YD-41's dimensions so the
/// ketch code paths are exercised. Only relations are asserted on it.
fn ketch_rig() -> RigDimensions {
    RigDimensions {
        mizzen: Some(MizzenDimensions {
            mizzen_hoist: 8.0,
            mizzen_foot: 3.0,
            mizzen_boom_above_sheer: 1.0,
        }),
        ..yd41_rig()
    }
}
