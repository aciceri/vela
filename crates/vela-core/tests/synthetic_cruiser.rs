//! End-to-end test on the boat file shipped in `boats/`.
//!
//! The analytic shapes elsewhere check exactness on geometry with closed forms.
//! This one checks that the whole path — RON parsing, validation, frame
//! conversion, lofting, clipping, flotation — works on a hull with rocker, a
//! fine bow, a broad transom, flare and a section shape that varies along the
//! length. It also pins the shipped file: if an edit there makes the boat stop
//! floating level, this fails.

use approx::assert_relative_eq;
use vela_core::hydrostatics::{solve_flotation, FlotationOptions};
use vela_core::{loft_hull, BoatSpec, LoftOptions, RigidBody, Water};

const SPEC: &str = include_str!("../../../boats/synthetic-cruiser.ron");

#[test]
fn the_shipped_boat_parses_and_floats_level() {
    let spec = BoatSpec::parse_ron(SPEC).expect("shipped boat file must be valid");
    let mesh = loft_hull(&spec.hull, &LoftOptions::default());
    let body = RigidBody::new(spec.mass_properties().expect("valid mass")).expect("valid body");
    let water = Water::default();

    let flotation =
        solve_flotation(&mesh, &body, &water, &FlotationOptions::default()).expect("must float");

    // Displacement must equal the mass: this is the equation the solver solves,
    // so it is a check on convergence rather than on physics.
    assert_relative_eq!(
        flotation.hydrostatics.displacement(&water),
        body.mass_properties().mass(),
        max_relative = 1e-8
    );

    // The file's CoG was placed at the longitudinal centre of buoyancy, so the
    // boat is designed to float level.
    assert!(
        flotation.trim.abs() < 1e-3,
        "trim {} rad is not level",
        flotation.trim
    );
    assert!(
        flotation.heel.abs() < 1e-3,
        "heel {} rad is not upright",
        flotation.heel
    );

    // Canoe-body draft, from the file's design waterline of 0.62 m.
    assert!(
        (0.55..0.70).contains(&flotation.draft),
        "draft {} m is outside the design range",
        flotation.draft
    );
    // The deck edge must stay clear of the water, or the hull is being tested
    // outside the range its offsets describe.
    assert!(flotation.draft < 1.30);
}

/// Watertightness of a lofted hull with genuinely varying sections. A hole, a
/// missing cap or an inconsistently wound triangle leaves a residual in the
/// summed area-weighted normals, and every surface integral downstream would be
/// quietly wrong.
#[test]
fn the_lofted_hull_is_watertight() {
    let spec = BoatSpec::parse_ron(SPEC).expect("valid spec");
    let mesh = loft_hull(&spec.hull, &LoftOptions::default());

    let residual: vela_core::geometry::Point = mesh.triangles().map(|t| t.area_normal()).sum();
    // Scaled against the hull's own surface area so the bound means something.
    assert!(
        residual.norm() / mesh.surface_area() < 1e-12,
        "closure residual {} m^2 over {} m^2 of surface",
        residual.norm(),
        mesh.surface_area()
    );
}

/// Displacement must not depend on mesh resolution beyond the discretization of
/// the sections themselves. A large drift would mean the lofting is losing
/// geometry — which is exactly what arc-length resampling used to do at corners.
#[test]
fn draft_is_stable_across_resolutions() {
    let spec = BoatSpec::parse_ron(SPEC).expect("valid spec");
    let body = RigidBody::new(spec.mass_properties().expect("valid mass")).expect("valid body");
    let water = Water::default();

    let draft_at = |points: usize| {
        let mesh = loft_hull(
            &spec.hull,
            &LoftOptions {
                points_per_station: points,
            },
        );
        solve_flotation(&mesh, &body, &water, &FlotationOptions::default())
            .expect("must float")
            .draft
    };

    assert_relative_eq!(draft_at(13), draft_at(64), max_relative = 1e-3);
}
