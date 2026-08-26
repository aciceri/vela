//! Validation of the DSYHS hull resistance model against the worked example
//! published for the YD-41.
//!
//! *Principles of Yacht Design*, 5th ed., carries a fully worked numerical
//! example through Chapter 5 for its design yacht, at one speed. Reproducing
//! those numbers exercises the coefficient tables, the polynomial structure,
//! the Froude interpolation, the reference-length convention for friction and
//! the two different LCB conventions all at once — and it is an *external*
//! oracle, not a recorded output of this code.
//!
//! Hull particulars are from Appendix 1 (half-loaded displacement); the
//! resistance figures are from Figs 5.8, 5.18 and 5.22.

use approx::assert_relative_eq;
use vela_core::dsyhs::{
    heel_residuary_delta, hull_resistance, residuary_resistance, HullParameters,
};

/// The book works in these values rather than the CGPM constants.
const DENSITY: f64 = 1025.0;
const GRAVITY: f64 = 9.81;

/// Froude number of the worked example — a tabulated row, so no interpolation.
const FROUDE: f64 = 0.35;

/// Speed of the worked example, m/s.
///
/// Derived from the Froude number rather than taken as the `3.78 m/s` printed
/// in Fig 5.8, which is rounded to three figures. That rounding puts the speed
/// at `Fn = 0.34985`, which makes the coefficient lookup interpolate between
/// the 0.30 and 0.35 rows and shifts the residuary coefficient by about one
/// part in a thousand — enough to matter at the precision this test is
/// checking, and a useful reminder that the tables are coarse in `Fn`.
fn speed() -> f64 {
    FROUDE * (GRAVITY * 11.90).sqrt()
}

/// YD-41, half-loaded displacement.
fn yd41() -> HullParameters {
    HullParameters {
        waterline_length: 11.90,
        waterline_beam: 3.18,
        canoe_draft: 0.40,
        canoe_volume: 6.05,
        wetted_surface: 28.20,
        waterplane_area: 26.75,
        prismatic: 0.56,
        midship: 0.715,
        lcb: -0.042,
        lcf: -0.073,
    }
}

#[test]
fn the_worked_speed_is_the_tabulated_froude_number() {
    // Guards the whole test file: if this drifts, every comparison below is
    // being made at the wrong point of the coefficient table.
    assert_relative_eq!(yd41().froude(speed(), GRAVITY), FROUDE, epsilon = 1e-12);
}

/// Fig 5.18: the residuary resistance coefficient is 0.00649 and the force
/// 394.9 N.
#[test]
fn residuary_resistance_matches_the_published_example() {
    let hull = yd41();
    let force = residuary_resistance(&hull, speed(), DENSITY, GRAVITY);

    let coefficient = force / (hull.canoe_volume * DENSITY * GRAVITY);
    assert_relative_eq!(coefficient, 0.00649, max_relative = 1e-3);
    assert_relative_eq!(force, 394.9, max_relative = 2e-3);
}

/// Fig 5.8: hull friction is 512.7 N, with the Reynolds number built on
/// 0.7 × Lwl rather than the full waterline length.
#[test]
fn frictional_resistance_matches_the_published_example() {
    let resistance = hull_resistance(&yd41(), speed(), 0.0, DENSITY, GRAVITY);
    assert_relative_eq!(resistance.friction, 512.7, max_relative = 2e-3);
}

/// The characteristic length convention is not cosmetic: using the full
/// waterline length instead of 0.7 of it moves the friction by several per
/// cent, away from the published value.
#[test]
fn using_the_full_waterline_length_would_miss_the_published_friction() {
    use vela_core::dsyhs::{frictional_resistance, SALT_WATER_VISCOSITY};
    let hull = yd41();
    let wrong = frictional_resistance(
        speed(),
        hull.waterline_length,
        hull.wetted_surface,
        DENSITY,
        SALT_WATER_VISCOSITY,
    );
    let error = (wrong - 512.7).abs() / 512.7;
    assert!(
        error > 0.02,
        "the two conventions differ by only {error}, so this test proves nothing"
    );
}

/// Fig 5.22: the increase in hull residuary resistance at 20° of heel is 33 N.
#[test]
fn heel_residuary_delta_matches_the_published_example() {
    let delta = heel_residuary_delta(&yd41(), speed(), 20.0_f64.to_radians(), DENSITY, GRAVITY);
    assert_relative_eq!(delta, 33.0, max_relative = 2e-2);
}

/// The angular scaling is normalized so that 20° reproduces the tabulated
/// regression almost exactly. A different exponent or factor would break this
/// while still passing a single-point check at some other angle.
#[test]
fn the_heel_scaling_is_unity_at_twenty_degrees() {
    let factor = 6.0 * 20.0_f64.to_radians().powf(1.7);
    assert_relative_eq!(factor, 1.0, max_relative = 5e-3);
}

#[test]
fn heel_correction_grows_with_angle_and_vanishes_upright() {
    let hull = yd41();
    let at =
        |degrees: f64| heel_residuary_delta(&hull, speed(), degrees.to_radians(), DENSITY, GRAVITY);
    assert_relative_eq!(at(0.0), 0.0, epsilon = 1e-12);
    assert!(at(10.0) < at(20.0));
    assert!(at(20.0) < at(30.0));
    // Symmetric: heeling to port costs the same as to starboard.
    assert_relative_eq!(at(-20.0), at(20.0), epsilon = 1e-12);
}

/// Forces must be continuous everywhere, including at the edges of the
/// tabulated Froude range. The tables stop; the physics does not, and a
/// discontinuity would jolt a time-domain solver as the boat accelerates
/// through the threshold. This is a requirement the published regression does
/// not itself provide, so it is worth pinning.
#[test]
fn resistance_is_continuous_across_the_table_edges() {
    let hull = yd41();
    let scale = (GRAVITY * hull.waterline_length).sqrt();
    let residuary_at = |froude: f64| residuary_resistance(&hull, froude * scale, DENSITY, GRAVITY);
    let heel_at = |froude: f64| {
        heel_residuary_delta(
            &hull,
            froude * scale,
            20.0_f64.to_radians(),
            DENSITY,
            GRAVITY,
        )
    };

    // At rest there is no wave making at all, so both must be exactly zero.
    assert_relative_eq!(residuary_at(0.0), 0.0, epsilon = 1e-12);
    assert_relative_eq!(heel_at(0.0), 0.0, epsilon = 1e-12);

    // No step at the first tabulated point of either table (0.15 and 0.25).
    for edge in [0.15, 0.25, 0.55] {
        let below = residuary_at(edge - 1e-6);
        let above = residuary_at(edge + 1e-6);
        assert!(
            (above - below).abs() < 1e-3 * above.abs().max(1.0),
            "residuary jumps at Fn {edge}: {below} to {above}"
        );

        let below = heel_at(edge - 1e-6);
        let above = heel_at(edge + 1e-6);
        assert!(
            (above - below).abs() < 1e-3 * above.abs().max(1.0),
            "heel correction jumps at Fn {edge}: {below} to {above}"
        );
    }

    // And above the last tabulated point the correction is held, not dropped.
    assert!(heel_at(0.70) > 0.5 * heel_at(0.55));
}

#[test]
fn resistance_rises_monotonically_through_the_displacement_range() {
    let hull = yd41();
    // Fn 0.20 to 0.45: the hump region, where residuary resistance rises
    // steeply. Non-monotonic total resistance here would mean a transcription
    // error in a coefficient row.
    let mut previous = 0.0;
    for step in 0..=10 {
        let froude = 0.20 + 0.025 * f64::from(step);
        let speed = froude * (GRAVITY * hull.waterline_length).sqrt();
        let total = hull_resistance(&hull, speed, 0.0, DENSITY, GRAVITY).total();
        assert!(
            total > previous,
            "total resistance fell at Fn = {froude}: {total} after {previous}"
        );
        previous = total;
    }
}

#[test]
fn the_yd41_sits_inside_the_series_envelope() {
    yd41()
        .check_envelope()
        .expect("the design yacht of the series' own textbook must be in range");
}

#[test]
fn an_out_of_range_hull_is_reported_parameter_by_parameter() {
    let barge = HullParameters {
        waterline_beam: 8.0, // Lwl/Bwl far too low, Bwl/Tc far too high
        prismatic: 0.85,     // barge-like, well outside 0.52..0.60
        ..yd41()
    };
    let error = barge.check_envelope().expect_err("must be rejected");
    let names: Vec<&str> = error.violations.iter().map(|v| v.parameter).collect();
    assert!(names.contains(&"Lwl/Bwl"), "got {names:?}");
    assert!(names.contains(&"Cp"), "got {names:?}");
    // The message names the offending values, so a boat file author can fix it.
    let text = error.to_string();
    assert!(text.contains("Cp"));
    assert!(text.contains("0.52"));
}

#[test]
fn lcb_conventions_convert_as_the_figure_requires() {
    let hull = yd41();
    // LCB 4.2% aft of midship on an 11.90 m waterline: midship is 5.95 m from
    // the bow, and the centre of buoyancy is half a metre further aft.
    assert_relative_eq!(hull.lcb_from_bow(), 5.95 + 0.4998, epsilon = 1e-9);
    assert_relative_eq!(hull.lcf_from_bow(), 5.95 + 0.8687, epsilon = 1e-9);
}

/// Doubling the displacement at a fixed Froude number should roughly double
/// the residuary resistance: the book states the resistance is approximately
/// proportional to displacement in the displacement speed range.
#[test]
fn residuary_resistance_scales_roughly_with_displacement() {
    let light = yd41();
    let heavy = HullParameters {
        canoe_volume: light.canoe_volume * 2.0,
        ..light
    };
    let ratio = residuary_resistance(&heavy, speed(), DENSITY, GRAVITY)
        / residuary_resistance(&light, speed(), DENSITY, GRAVITY);
    assert!(
        (1.5..2.6).contains(&ratio),
        "expected roughly proportional, got {ratio}"
    );
}

/// The shipped YD-41 file must parse, declare parameters, and sit inside the
/// series envelope. Pins the transcription from Appendix 1 against the light
/// displacement column.
#[test]
fn the_shipped_yd41_file_is_consistent() {
    const SPEC: &str = include_str!("../../../boats/yd41.ron");
    let spec = vela_core::BoatSpec::parse_ron(SPEC).expect("shipped file must be valid");

    assert!(
        spec.hull.is_none(),
        "the book publishes no offset table, so this file must carry no geometry"
    );
    let hull = spec
        .hull_parameters()
        .expect("must declare hull parameters");

    assert_relative_eq!(hull.waterline_length, 11.62);
    assert_relative_eq!(hull.canoe_volume, 5.46);
    assert_relative_eq!(hull.prismatic, 0.56);
    // Percentages in the file become fractions in the engine.
    assert_relative_eq!(hull.lcb, -0.045);
    assert_relative_eq!(hull.lcf, -0.073);

    hull.check_envelope()
        .expect("the series' own design yacht must be in range");
}

/// A parameters-only boat cannot be lofted, and the loader must say so rather
/// than produce an empty hull that silently displaces nothing.
#[test]
fn a_parameters_only_boat_has_no_geometry_to_loft() {
    const SPEC: &str = include_str!("../../../boats/yd41.ron");
    let spec = vela_core::BoatSpec::parse_ron(SPEC).expect("valid");
    assert!(spec.hull.is_none());
}
