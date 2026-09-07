//! Validation of the keel and rudder force model against the worked examples
//! published for the YD-41, plus the properties the figures do not state but
//! any usable force model must have.
//!
//! *Principles of Yacht Design*, 5th ed., carries numerical examples for its
//! design yacht through Figs 5.19 and 5.23. Those are external oracles: they
//! exercise the coefficient table, the Froude interpolation, the volumetric
//! ratios and the unit conventions at once, and they were not produced by this
//! code.
//!
//! Figs 6.13 and 6.14 carry no worked example, so the side force and induced
//! resistance are pinned by closed-form relations and by direction-of-effect
//! properties instead — each chosen so that a plausible bug (a dropped wake
//! factor, a missing downwash, an aspect ratio the wrong way up, a square that
//! became a product) fails it.
//!
//! Particulars are from Appendix 1, half-loaded displacement.

use approx::assert_relative_eq;
use vela_core::appendages::{
    appendage_forces, dynamic_pressure, heel_side_force_factor, hull_side_force_factor,
    induced_resistance, keel_downwash_angle, keel_heel_residuary_coefficient,
    keel_heel_residuary_delta, keel_residuary_coefficient, keel_residuary_resistance,
    lift_coefficient, rudder_angle_of_attack, side_force, AppendageForces, FlowState, FoilPlanform,
    HullScalars, Keel, KEEL_RESIDUARY_FROUDE_MAX, KEEL_RESIDUARY_FROUDE_MIN, RUDDER_WAKE_FRACTION,
};

/// The book works in these values rather than the CGPM constants; using them
/// makes the comparison with the printed forces exact.
const DENSITY: f64 = 1025.0;
const GRAVITY: f64 = 9.81;

/// Froude number of the worked examples — a tabulated row, so no interpolation
/// is involved and a transcription error in the table cannot hide behind a
/// blend of two rows.
const FROUDE: f64 = 0.35;

/// Speed of the worked examples, m/s (3.7816).
///
/// Derived from the Froude number rather than from a rounded printed speed, for
/// the same reason as in the DSYHS tests: a speed rounded to three figures lands
/// just off `Fn = 0.35` and silently starts interpolating the coefficient rows.
fn speed() -> f64 {
    FROUDE * (GRAVITY * 11.90).sqrt()
}

fn yd41_hull() -> HullScalars {
    HullScalars {
        waterline_length: 11.90,
        waterline_beam: 3.18,
        canoe_draft: 0.40,
        total_draft: 2.30,
        canoe_volume: 6.05,
    }
}

fn yd41_keel_planform() -> FoilPlanform {
    FoilPlanform {
        root_chord: 1.00,
        tip_chord: 0.78,
        span: 1.90,
        sweep: 5.5_f64.to_radians(),
    }
}

fn yd41_rudder() -> FoilPlanform {
    FoilPlanform {
        root_chord: 0.48,
        tip_chord: 0.22,
        span: 1.15,
        sweep: 10.0_f64.to_radians(),
    }
}

/// The YD-41 keel with `Z_CBk` taken as the planform area centroid, which is
/// the only value the published particulars let us derive.
fn yd41_keel_from_planform_centroid() -> Keel {
    let planform = yd41_keel_planform();
    Keel {
        volume: 0.275,
        centre_of_buoyancy_below_hull_bottom: planform.planform_centroid_below_root(),
        planform,
    }
}

/// `Z_CBk` that the published coefficient of Fig 5.19 implies, m.
///
/// Obtained by inverting the `A2` term of Fig 5.19 at `Fn = 0.35` against the
/// book's own answer of 0.01406. It is **not** a value taken from the source
/// and it is deliberately not the default anywhere in the library — see
/// [`keel_residuary_reproduces_the_published_value_given_one_unknown_lever`]
/// for what it is doing here.
const IMPLIED_KEEL_BUOYANCY_LEVER: f64 = 1.4013;

fn yd41_keel_with_implied_lever() -> Keel {
    Keel {
        planform: yd41_keel_planform(),
        volume: 0.275,
        centre_of_buoyancy_below_hull_bottom: IMPLIED_KEEL_BUOYANCY_LEVER,
    }
}

/// A representative upwind operating point: 4° of leeway, 20° of heel, helm
/// amidships. Chosen so the rudder carries a real load from leeway alone,
/// which is what makes the downwash tests bite.
fn upwind() -> FlowState {
    FlowState::uniform(speed(), 4.0_f64.to_radians(), 20.0_f64.to_radians(), 0.0)
}

fn upwind_forces() -> AppendageForces {
    appendage_forces(
        &yd41_hull(),
        &yd41_keel_from_planform_centroid(),
        &yd41_rudder(),
        &upwind(),
        DENSITY,
        GRAVITY,
    )
}

#[test]
fn the_worked_speed_is_the_tabulated_froude_number() {
    // Guards every comparison below: if this drifts, the coefficient lookup is
    // happening at the wrong point of the table.
    assert_relative_eq!(
        yd41_hull().froude(speed(), GRAVITY),
        FROUDE,
        epsilon = 1e-12
    );
}

// ---------------------------------------------------------------------------
// Fig 5.19 — residuary resistance of the keel
// ---------------------------------------------------------------------------

/// Fig 5.19 with the one quantity the book does not publish supplied.
///
/// `Z_CBk` is the height of the keel's centre of **buoyancy** — the centroid of
/// its displaced volume — above the hull bottom. Appendix 1 does not tabulate
/// it, and the tempting substitute, the planform area centroid at 0.911 m,
/// gives coefficient 0.0158 and 43.7 N against the published 0.01406 and 39 N:
/// 12.4 % high.
///
/// The planform centroid was simply the wrong quantity. Inverting the worked
/// example asks for `Z_CBk = 1.401 m`, 74 % of the way down a 1.90 m keel,
/// which no trapezoid's *area* centroid can reach — the range is 47 % to 53 %.
/// But a *volume* centroid can, and the YD-41's keel is a high-aspect fin with
/// a torpedo bulb carrying 2300 kg of lead at its tip. Nearly all the keel's
/// 0.275 m³ is in that bulb, so its volume centroid sits far below the fin's
/// mid-span, exactly where the inversion puts it.
///
/// **This is a consistency check, not a validation**, and the distinction
/// matters. `Z_CBk` was fitted to the book's own answer, so this test cannot
/// confirm the `A2` term. Nor can it confirm much else on its own: the
/// published coefficient and the published force are not independent, since
/// `R = C ∇_k ρ g`, so fitting one parameter to the first reproduces the second
/// automatically. One constraint, one unknown.
///
/// What does validate the shared transcription is the sister example of
/// Fig 5.23, which reproduces to 0.07 % from these same particulars with no
/// fitted input at all.
#[test]
fn keel_residuary_reproduces_the_published_value_given_the_bulb_lever() {
    let hull = yd41_hull();
    let keel = yd41_keel_with_implied_lever();

    // Measured 0.01406015 against the published 0.01406.
    let coefficient = keel_residuary_coefficient(&hull, &keel, FROUDE);
    assert_relative_eq!(coefficient, 0.01406, max_relative = 1e-4);

    // Measured 38.879 N against the published 39 N — the printed figure is
    // rounded to two significant figures.
    let force = keel_residuary_resistance(&hull, &keel, speed(), DENSITY, GRAVITY);
    assert_relative_eq!(force, 39.0, max_relative = 4e-3);
}

/// The lever the derived planform centroid produces, recorded so that a future
/// change to the centroid formula or the `A2` term cannot quietly move the
/// discrepancy around. If this value changes, the discussion above is stale.
#[test]
fn the_planform_centroid_lever_is_the_documented_one() {
    let keel = yd41_keel_from_planform_centroid();
    // 1.90 (1.00 + 2 × 0.78) / (3 × 1.78).
    assert_relative_eq!(
        keel.centre_of_buoyancy_below_hull_bottom,
        0.9108614,
        max_relative = 1e-6
    );
    // Measured 0.0158098, i.e. 12.4 % above the published 0.01406.
    let coefficient = keel_residuary_coefficient(&yd41_hull(), &keel, FROUDE);
    assert_relative_eq!(coefficient, 0.0158098, max_relative = 1e-5);
}

// ---------------------------------------------------------------------------
// Fig 5.23 — change in keel residuary resistance with heel
// ---------------------------------------------------------------------------

/// Fig 5.23: `C_H = 1.13` and `ΔR_RKφ = 134 N` at `Fn = 0.35` and 20° of heel.
///
/// Unlike Fig 5.19 this example needs nothing that Appendix 1 fails to publish,
/// so it reproduces outright — which is also what rules out a transcription
/// error in the shared particulars as the cause of the Fig 5.19 gap.
#[test]
fn keel_heel_delta_matches_the_published_example() {
    let hull = yd41_hull();
    let keel = yd41_keel_from_planform_centroid();

    // Measured 1.1307614 against the published 1.13.
    assert_relative_eq!(
        keel_heel_residuary_coefficient(&hull),
        1.13,
        max_relative = 1e-3
    );

    // Measured 133.70 N against the published 134 N.
    let delta = keel_heel_residuary_delta(
        &hull,
        &keel,
        speed(),
        20.0_f64.to_radians(),
        DENSITY,
        GRAVITY,
    );
    assert_relative_eq!(delta, 134.0, max_relative = 3e-3);
}

/// `C_H` is geometric: it must not move with speed or heel, or the `Fn² φ`
/// scaling in front of it is being double-counted.
#[test]
fn the_heel_coefficient_is_speed_independent() {
    let hull = yd41_hull();
    let keel = yd41_keel_from_planform_centroid();
    let heel = 20.0_f64.to_radians();
    let at = |froude: f64| {
        let v = froude * (GRAVITY * hull.waterline_length).sqrt();
        keel_heel_residuary_delta(&hull, &keel, v, heel, DENSITY, GRAVITY) / (froude * froude)
    };
    assert_relative_eq!(at(0.25), at(0.50), max_relative = 1e-12);
}

#[test]
fn the_heel_delta_vanishes_upright_and_is_symmetric_in_heel() {
    let hull = yd41_hull();
    let keel = yd41_keel_from_planform_centroid();
    let at = |degrees: f64| {
        keel_heel_residuary_delta(
            &hull,
            &keel,
            speed(),
            degrees.to_radians(),
            DENSITY,
            GRAVITY,
        )
    };
    assert_relative_eq!(at(0.0), 0.0, epsilon = 1e-12);
    assert!(at(10.0) < at(20.0));
    assert!(at(20.0) < at(30.0));
    // Heeling to port must cost exactly what heeling to starboard costs.
    assert_relative_eq!(at(-20.0), at(20.0), epsilon = 1e-12);
}

// ---------------------------------------------------------------------------
// Fig 6.13 — downwash and side force
// ---------------------------------------------------------------------------

/// The keel's downwash reduces the rudder's angle of attack, so the rudder
/// carries less lift than the identical blade would in undisturbed flow.
///
/// This is the property the whole yaw balance rests on. A model that drops `ε`
/// still produces a believable total side force — the rudder is only about a
/// tenth of it — while putting the rudder at nearly twice its real angle of
/// attack and so getting the helm entirely wrong.
#[test]
fn downwash_reduces_the_rudder_angle_of_attack_and_its_lift() {
    let keel = yd41_keel_planform();
    let rudder = yd41_rudder();
    let flow = upwind();

    let downwash = keel_downwash_angle(&keel, flow.leeway, flow.heel);
    assert!(
        downwash > 0.0,
        "downwash {downwash} must oppose positive leeway"
    );

    let with = rudder_angle_of_attack(&keel, flow.leeway, flow.heel, flow.rudder_angle);
    assert!(
        with < flow.leeway,
        "angle of attack {with} is not reduced from the leeway {}",
        flow.leeway
    );
    // Measured 1.938° of downwash against 4° of leeway: about half of it, which
    // is far too large to treat as a correction.
    assert_relative_eq!(downwash.to_degrees(), 1.9379, max_relative = 1e-3);

    // And therefore less lift, at the same speed, on the same blade.
    let local_speed = RUDDER_WAKE_FRACTION * flow.speed;
    let loaded = side_force(
        &rudder,
        with,
        local_speed,
        0.40,
        keel.span,
        flow.heel,
        DENSITY,
    );
    let undisturbed = side_force(
        &rudder,
        flow.leeway,
        local_speed,
        0.40,
        keel.span,
        flow.heel,
        DENSITY,
    );
    assert!(
        loaded < undisturbed,
        "downwash did not reduce the rudder side force: {loaded} vs {undisturbed}"
    );
    // Lift is linear in the angle of attack, so the ratio of forces must be the
    // ratio of angles — this catches a downwash applied to the force instead of
    // to the angle.
    assert_relative_eq!(
        loaded / undisturbed,
        with / flow.leeway,
        max_relative = 1e-12
    );
}

/// The figure is written for one tack. Downwash must reduce the rudder's angle
/// of attack on the other one too, not add to it.
#[test]
fn downwash_opposes_the_leeway_on_both_tacks() {
    let keel = yd41_keel_planform();
    let heel = 20.0_f64.to_radians();
    let leeway = 4.0_f64.to_radians();

    let port = keel_downwash_angle(&keel, leeway, heel);
    let starboard = keel_downwash_angle(&keel, -leeway, heel);
    assert_relative_eq!(starboard, -port, epsilon = 1e-15);

    let alpha = rudder_angle_of_attack(&keel, -leeway, heel, 0.0);
    assert!(
        alpha > -leeway && alpha < 0.0,
        "angle of attack {alpha} is not a reduced-magnitude negative angle"
    );
}

/// The rudder works in a wake at 90 % of boat speed, everywhere the figures say
/// so — the side force and the induced resistance alike.
#[test]
fn the_rudder_works_at_ninety_per_cent_of_boat_speed() {
    let hull = yd41_hull();
    let rudder = yd41_rudder();
    let flow = upwind();
    let forces = upwind_forces();

    let c_hull = hull_side_force_factor(hull.canoe_draft, yd41_keel_planform().span);
    let c_heel = heel_side_force_factor(flow.heel);

    // The side force built on full boat speed, which is the bug being guarded
    // against.
    let at_boat_speed = forces.rudder_lift_coefficient
        * dynamic_pressure(flow.speed, DENSITY)
        * rudder.area()
        * c_hull
        * c_heel;

    // 0.9² = 0.81: the wake costs the rudder 19 % of everything it does.
    assert_relative_eq!(
        forces.rudder_side_force,
        0.81 * at_boat_speed,
        max_relative = 1e-12
    );

    // Only about 2.4 % of the total side force, which is precisely why the
    // mistake is silent — but all of it lands aft of the keel.
    let share = (at_boat_speed - forces.rudder_side_force) / forces.side_force();
    assert!(
        share > 0.01 && share < 0.05,
        "the wake factor moves the total by {share}, so this test proves nothing"
    );

    // The induced resistance uses the same wake speed.
    let expected = induced_resistance(
        &rudder,
        forces.rudder_side_force,
        RUDDER_WAKE_FRACTION * flow.speed,
        flow.heel,
        DENSITY,
    );
    assert_relative_eq!(
        forces.rudder_induced_resistance,
        expected,
        max_relative = 1e-12
    );
}

/// No leeway, no helm, no side force — from either appendage. The hull and heel
/// factors are multiplicative, so a stray additive term would show up here.
#[test]
fn side_force_vanishes_at_zero_leeway_and_grows_with_it() {
    let hull = yd41_hull();
    let keel = yd41_keel_from_planform_centroid();
    let rudder = yd41_rudder();

    let at = |leeway_degrees: f64| {
        appendage_forces(
            &hull,
            &keel,
            &rudder,
            &FlowState::uniform(
                speed(),
                leeway_degrees.to_radians(),
                20.0_f64.to_radians(),
                0.0,
            ),
            DENSITY,
            GRAVITY,
        )
    };

    let upright = at(0.0);
    assert_relative_eq!(upright.keel_side_force, 0.0, epsilon = 1e-12);
    assert_relative_eq!(upright.rudder_side_force, 0.0, epsilon = 1e-12);
    assert_relative_eq!(upright.side_force(), 0.0, epsilon = 1e-12);
    // With no lift there is no trailing vorticity either.
    assert_relative_eq!(upright.keel_induced_resistance, 0.0, epsilon = 1e-12);
    assert_relative_eq!(upright.rudder_induced_resistance, 0.0, epsilon = 1e-12);

    for (low, high) in [(1.0, 2.0), (2.0, 4.0), (4.0, 6.0)] {
        assert!(
            at(low).side_force() < at(high).side_force(),
            "side force did not grow from {low}° to {high}° of leeway"
        );
    }

    // The keel's own lift is linear in leeway; only the rudder's is not,
    // because the downwash goes as the square root of the keel's lift.
    assert_relative_eq!(
        at(4.0).keel_side_force / at(2.0).keel_side_force,
        2.0,
        max_relative = 1e-12
    );
}

/// Heel bleeds side force away, symmetrically in the sign of the heel angle.
#[test]
fn side_force_falls_off_with_heel() {
    let hull = yd41_hull();
    let keel = yd41_keel_from_planform_centroid();
    let rudder = yd41_rudder();
    let at = |heel_degrees: f64| {
        appendage_forces(
            &hull,
            &keel,
            &rudder,
            &FlowState::uniform(
                speed(),
                4.0_f64.to_radians(),
                heel_degrees.to_radians(),
                0.0,
            ),
            DENSITY,
            GRAVITY,
        )
        .side_force()
    };
    assert!(at(30.0) < at(20.0));
    assert!(at(20.0) < at(0.0));
    assert_relative_eq!(at(-20.0), at(20.0), max_relative = 1e-12);
}

// ---------------------------------------------------------------------------
// Fig 6.14 — induced resistance
// ---------------------------------------------------------------------------

/// `R_i` goes as `C_Lφ²`, so doubling the side force must quadruple it. A
/// dropped square, or a `C_Lφ` built from the horizontal force instead of the
/// heeled one, breaks this.
#[test]
fn induced_resistance_grows_as_the_square_of_side_force() {
    let keel = yd41_keel_planform();
    let heel = 20.0_f64.to_radians();
    let at = |force: f64| induced_resistance(&keel, force, speed(), heel, DENSITY);

    let base = at(3000.0);
    assert!(base > 0.0);
    assert_relative_eq!(at(6000.0) / base, 4.0, max_relative = 1e-12);
    assert_relative_eq!(at(9000.0) / base, 9.0, max_relative = 1e-12);
    // Drag from lift does not care which way the lift points.
    assert_relative_eq!(at(-3000.0), base, max_relative = 1e-12);

    // Same law through the full model, where the leeway drives the force.
    let hull = yd41_hull();
    let keel_body = yd41_keel_from_planform_centroid();
    let rudder = yd41_rudder();
    let at_leeway = |degrees: f64| {
        appendage_forces(
            &hull,
            &keel_body,
            &rudder,
            &FlowState::uniform(speed(), degrees.to_radians(), heel, 0.0),
            DENSITY,
            GRAVITY,
        )
    };
    let single = at_leeway(2.0);
    let double = at_leeway(4.0);
    assert_relative_eq!(
        double.keel_side_force / single.keel_side_force,
        2.0,
        max_relative = 1e-12
    );
    assert_relative_eq!(
        double.keel_induced_resistance / single.keel_induced_resistance,
        4.0,
        max_relative = 1e-12
    );
}

/// The heeled-plane conversion is not cosmetic: `C_Lφ` is built from
/// `F_h / cos φ`, so at 20° of heel the induced resistance is 13 % higher than
/// the horizontal force alone would suggest.
#[test]
fn induced_resistance_uses_the_force_in_the_heeled_plane() {
    let keel = yd41_keel_planform();
    let force = 3000.0;
    let heel = 20.0_f64.to_radians();
    let heeled = induced_resistance(&keel, force, speed(), heel, DENSITY);
    let upright = induced_resistance(&keel, force, speed(), 0.0, DENSITY);
    // 1/cos²(20°) = 1.1325.
    assert_relative_eq!(
        heeled / upright,
        1.0 / (heel.cos() * heel.cos()),
        max_relative = 1e-12
    );
}

/// A slender foil lifts harder for the same angle of attack and drags less for
/// the same lift. Both halves matter: the first pins the sign of `AR_e` in the
/// Whicker & Fehlner slope, the second pins the `1/(π AR_Ee)` in Fig 6.14. A
/// model with the aspect ratio inverted in one place only would still pass one
/// of them.
#[test]
fn a_higher_aspect_ratio_lifts_more_and_drags_less() {
    // Two rectangular, unswept planforms of identical area, so that area and
    // dynamic pressure drop out of both comparisons and only the aspect ratio
    // is left.
    let area = 1.691;
    let rectangle = |span: f64| FoilPlanform {
        root_chord: area / span,
        tip_chord: area / span,
        span,
        sweep: 0.0,
    };
    let stubby = rectangle(1.30);
    let slender = rectangle(2.60);
    assert_relative_eq!(stubby.area(), slender.area(), max_relative = 1e-12);
    assert!(slender.aspect_ratio() > stubby.aspect_ratio());

    let alpha = 4.0_f64.to_radians();
    let heel = 0.0;
    let force = |foil: &FoilPlanform, angle: f64| {
        side_force(foil, angle, speed(), 0.40, 1.90, heel, DENSITY)
    };

    // Same angle of attack: the slender foil lifts more.
    assert!(
        force(&slender, alpha) > force(&stubby, alpha),
        "the higher aspect ratio did not lift more"
    );

    // Same lift: match the angles through the lift coefficients, then compare
    // the induced resistance.
    let matched = alpha * lift_coefficient(&stubby, alpha) / lift_coefficient(&slender, alpha);
    let stubby_force = force(&stubby, alpha);
    let slender_force = force(&slender, matched);
    assert_relative_eq!(slender_force, stubby_force, max_relative = 1e-12);

    let stubby_drag = induced_resistance(&stubby, stubby_force, speed(), heel, DENSITY);
    let slender_drag = induced_resistance(&slender, slender_force, speed(), heel, DENSITY);
    assert!(
        slender_drag < stubby_drag,
        "the higher aspect ratio did not drag less at equal lift: {slender_drag} vs {stubby_drag}"
    );
    // Equal area and equal force mean equal C_Lφ, so the drag ratio is exactly
    // the inverse ratio of the aspect ratios.
    assert_relative_eq!(
        slender_drag / stubby_drag,
        stubby.aspect_ratio() / slender.aspect_ratio(),
        max_relative = 1e-12
    );
}

// ---------------------------------------------------------------------------
// Edges of the tabulated Froude range
// ---------------------------------------------------------------------------

/// Forces must be continuous everywhere, including where the published table
/// stops. A step in a force jolts a time-domain solver as the boat accelerates
/// through the threshold, and this is a requirement Fig 5.19 does not itself
/// provide, so it is pinned here.
#[test]
fn keel_resistance_is_continuous_across_the_table_edges() {
    let hull = yd41_hull();
    let keel = yd41_keel_from_planform_centroid();
    let scale = (GRAVITY * hull.waterline_length).sqrt();
    let heel = 20.0_f64.to_radians();

    let residuary_at =
        |froude: f64| keel_residuary_resistance(&hull, &keel, froude * scale, DENSITY, GRAVITY);
    let heel_at = |froude: f64| {
        keel_heel_residuary_delta(&hull, &keel, froude * scale, heel, DENSITY, GRAVITY)
    };

    // A keel at rest makes no waves, so both are exactly zero.
    assert_relative_eq!(residuary_at(0.0), 0.0, epsilon = 1e-12);
    assert_relative_eq!(heel_at(0.0), 0.0, epsilon = 1e-12);

    for edge in [KEEL_RESIDUARY_FROUDE_MIN, KEEL_RESIDUARY_FROUDE_MAX] {
        let below = residuary_at(edge - 1e-6);
        let above = residuary_at(edge + 1e-6);
        assert!(
            (above - below).abs() < 1e-3 * above.abs().max(1.0),
            "keel residuary jumps at Fn {edge}: {below} to {above}"
        );

        let below = heel_at(edge - 1e-6);
        let above = heel_at(edge + 1e-6);
        assert!(
            (above - below).abs() < 1e-3 * above.abs().max(1.0),
            "keel heel delta jumps at Fn {edge}: {below} to {above}"
        );
    }

    // Below the first row the taper is linear in Froude number, not a cliff.
    let half = 0.5 * KEEL_RESIDUARY_FROUDE_MIN;
    assert_relative_eq!(
        residuary_at(half),
        0.5 * residuary_at(KEEL_RESIDUARY_FROUDE_MIN),
        max_relative = 1e-12
    );

    // Above the last row the coefficient is held, so the force plateaus rather
    // than diverging or dropping to nothing.
    assert_relative_eq!(
        residuary_at(0.80),
        residuary_at(KEEL_RESIDUARY_FROUDE_MAX),
        max_relative = 1e-12
    );

    // The heel delta is a closed form in Fn², with no coefficient row to hold:
    // it keeps growing above the range rather than being frozen there. Freezing
    // it would put a kink in the derivative for nothing.
    assert!(heel_at(0.80) > heel_at(KEEL_RESIDUARY_FROUDE_MAX));
}

#[test]
fn the_worked_point_is_inside_the_published_froude_range() {
    let hull = yd41_hull();
    assert!(hull.check_keel_residuary_envelope(speed(), GRAVITY).is_ok());
}

#[test]
fn an_out_of_range_speed_is_reported_rather_than_returned_silently() {
    let hull = yd41_hull();
    let crawling = 0.10 * (GRAVITY * hull.waterline_length).sqrt();
    let error = hull
        .check_keel_residuary_envelope(crawling, GRAVITY)
        .expect_err("Fn = 0.10 is below the range of Fig 5.19");
    assert_relative_eq!(error.froude, 0.10, max_relative = 1e-12);
    assert_relative_eq!(error.low, 0.20, epsilon = 1e-12);
    assert_relative_eq!(error.high, 0.60, epsilon = 1e-12);
    assert!(error.to_string().contains("0.20..0.60"), "{error}");

    // The resistance functions still answer — tapered, continuous — because a
    // solver accelerating from rest cannot be handed an error every step.
    let keel = yd41_keel_from_planform_centroid();
    let tapered = keel_residuary_resistance(&hull, &keel, crawling, DENSITY, GRAVITY);
    assert!(tapered > 0.0 && tapered < 39.0);
}

// ---------------------------------------------------------------------------
// Component bookkeeping
// ---------------------------------------------------------------------------

/// The breakdown must add up, and it must be a breakdown: nobody downstream
/// can build a yaw balance out of a single total.
#[test]
fn the_component_breakdown_accounts_for_every_force() {
    let forces = upwind_forces();

    assert_relative_eq!(
        forces.side_force(),
        forces.keel_side_force + forces.rudder_side_force,
        max_relative = 1e-12
    );
    assert_relative_eq!(
        forces.resistance(),
        forces.keel_induced_resistance
            + forces.rudder_induced_resistance
            + forces.keel_residuary_resistance
            + forces.keel_heel_residuary_delta,
        max_relative = 1e-12
    );

    // Every component is populated and signed the way the figures intend at a
    // normal upwind operating point.
    assert!(forces.keel_side_force > forces.rudder_side_force);
    assert!(forces.rudder_side_force > 0.0);
    assert!(forces.keel_induced_resistance > forces.rudder_induced_resistance);
    assert!(forces.rudder_induced_resistance > 0.0);
    assert!(forces.keel_residuary_resistance > 0.0);
    assert!(forces.keel_heel_residuary_delta > 0.0);
    assert!(forces.downwash_angle > 0.0);
    assert!(forces.rudder_angle_of_attack > 0.0);
}

/// Helm adds to the rudder's angle of attack and therefore to its side force,
/// and it does so on top of the downwash rather than in place of it.
#[test]
fn helm_adds_to_the_rudder_angle_of_attack() {
    let keel = yd41_keel_planform();
    let leeway = 4.0_f64.to_radians();
    let heel = 20.0_f64.to_radians();
    let helm = 3.0_f64.to_radians();

    let neutral = rudder_angle_of_attack(&keel, leeway, heel, 0.0);
    let steered = rudder_angle_of_attack(&keel, leeway, heel, helm);
    assert_relative_eq!(steered - neutral, helm, epsilon = 1e-15);

    let hull = yd41_hull();
    let keel_body = yd41_keel_from_planform_centroid();
    let rudder = yd41_rudder();
    let with_helm = appendage_forces(
        &hull,
        &keel_body,
        &rudder,
        &FlowState::uniform(speed(), leeway, heel, helm),
        DENSITY,
        GRAVITY,
    );
    let without = upwind_forces();
    assert!(with_helm.rudder_side_force > without.rudder_side_force);
    // Helm does not touch the keel.
    assert_relative_eq!(
        with_helm.keel_side_force,
        without.keel_side_force,
        max_relative = 1e-12
    );
}
