//! End-to-end tests on `boats/yd41-form-study.ron`, the one shipped boat with
//! geometry.
//!
//! Two different things are checked here and they should not be confused.
//!
//! The first is that the whole geometry path works: RON parsing, validation,
//! frame conversion, lofting, clipping, flotation and the sectional area curve,
//! on a hull with rocker, a fine bow, a broad transom and flare. The analytic
//! shapes elsewhere prove exactness on prismatic solids; only this one exercises
//! a fair hull.
//!
//! The second is **how close the fitted geometry comes to the real YD-41's
//! published form coefficients**. That is an external oracle — Appendix 1 of
//! *Principles of Yacht Design*, 5th ed. — and it is the reason this hull was
//! fitted rather than drawn freehand. The tolerances below are wide enough to
//! be honest about discretization and tight enough that a broken fit fails.

use approx::assert_relative_eq;
use vela_core::hydrostatics::{solve_flotation, FlotationOptions};
use vela_core::sections::{hull_form, FormOptions};
use vela_core::{loft_hull, BoatSpec, LoftOptions, RigidBody, TriMesh, Water};

const SPEC: &str = include_str!("../../../boats/yd41-form-study.ron");

/// YD-41 half-loaded displacement, Appendix 1. The fit targets.
mod published {
    pub const WATERLINE_LENGTH: f64 = 11.90;
    pub const WATERLINE_BEAM: f64 = 3.18;
    pub const CANOE_DRAFT: f64 = 0.40;
    pub const CANOE_VOLUME: f64 = 6.05;
    pub const WETTED_SURFACE: f64 = 28.20;
    pub const WATERPLANE_AREA: f64 = 26.75;
    pub const PRISMATIC: f64 = 0.56;
    pub const MIDSHIP: f64 = 0.715;
    pub const LCB_PERCENT: f64 = -4.2;
    pub const LCF_PERCENT: f64 = -7.3;
}

fn spec() -> BoatSpec {
    BoatSpec::parse_ron(SPEC).expect("shipped boat file must be valid")
}

fn mesh(points_per_station: usize) -> TriMesh {
    let spec = spec();
    let hull = spec.hull.as_ref().expect("this fixture has geometry");
    loft_hull(hull, &LoftOptions { points_per_station })
}

fn body(spec: &BoatSpec) -> RigidBody {
    RigidBody::new(spec.mass_properties().expect("valid mass")).expect("valid rigid body")
}

#[test]
fn the_shipped_boat_parses_and_floats_level() {
    let spec = spec();
    let body = body(&spec);
    let water = Water::default();

    let flotation = solve_flotation(&mesh(16), &body, &water, &FlotationOptions::default())
        .expect("must float");

    assert_relative_eq!(
        flotation.hydrostatics.displacement(&water),
        body.mass_properties().mass(),
        max_relative = 1e-8
    );
    // The file's CoG sits at the measured LCB, so the boat is designed level.
    assert!(
        flotation.trim.abs() < 2e-3,
        "trim {} rad is not level",
        flotation.trim
    );
    assert!(flotation.heel.abs() < 1e-3);
    // Design canoe body draft is 0.40 m; the deck edge must stay clear.
    assert!(
        (0.38..0.43).contains(&flotation.draft),
        "draft {} m is off the design waterline",
        flotation.draft
    );
}

/// Watertightness on a hull with genuinely varying sections. For any closed
/// surface the area-weighted normals cancel exactly; a hole, a missing cap or
/// one flipped triangle leaves a residual, and every surface integral
/// downstream would be quietly wrong.
#[test]
fn the_lofted_hull_is_watertight() {
    let mesh = mesh(16);
    let residual: vela_core::geometry::Point = mesh.triangles().map(|t| t.area_normal()).sum();
    assert!(
        residual.norm() / mesh.surface_area() < 1e-12,
        "closure residual {} m^2 over {} m^2",
        residual.norm(),
        mesh.surface_area()
    );
}

/// The fitted geometry against the real YD-41's published coefficients.
///
/// Integrals and ratios of integrals — volume, LCB, LCF, waterline length —
/// come out essentially exact, because they average over the whole hull.
/// `Cm` and `Cp` are looser, and the cause is discretization rather than a bad
/// fit: with 17 stations the beam and section-area maxima fall *between*
/// stations, so the measured `A_m` reads slightly low and `Cp = V/(A_m L_wl)`
/// rises to compensate. Wetted surface is the interesting one — it was never a
/// fit target, and it lands within a percent anyway.
#[test]
fn the_fitted_hull_reproduces_the_published_form_coefficients() {
    let spec = spec();
    let mesh = mesh(24);
    let flotation = solve_flotation(
        &mesh,
        &body(&spec),
        &Water::default(),
        &FlotationOptions::default(),
    )
    .expect("must float");
    let options =
        FormOptions::at_draft(&mesh, flotation.draft).expect("hull is immersed at equilibrium");
    let form = hull_form(&mesh, &options).expect("form is measurable");

    // Measured values are in the comment beside each tolerance.
    assert_relative_eq!(
        form.waterline_length,
        published::WATERLINE_LENGTH,
        max_relative = 1e-6 // exact by construction
    );
    assert_relative_eq!(
        form.canoe_volume,
        published::CANOE_VOLUME,
        max_relative = 0.01
    ); // +0.2 %
    assert_relative_eq!(
        form.canoe_draft,
        published::CANOE_DRAFT,
        max_relative = 0.02
    ); // +0.8 %
    assert_relative_eq!(
        form.waterplane_area,
        published::WATERPLANE_AREA,
        max_relative = 0.02 // -0.7 %
    );
    assert_relative_eq!(
        form.wetted_surface,
        published::WETTED_SURFACE,
        max_relative = 0.02 // +0.7 %, and never a fit target
    );
    assert_relative_eq!(
        form.waterline_beam,
        published::WATERLINE_BEAM,
        max_relative = 0.03 // -1.6 %, peak falls between stations
    );
    assert_relative_eq!(form.prismatic, published::PRISMATIC, max_relative = 0.03); // +1.9 %
    assert_relative_eq!(form.midship, published::MIDSHIP, max_relative = 0.03); // -1.0 %

    // LCB and LCF are already fractions of Lwl and pass near zero, so compare
    // them by absolute difference in percentage points, not relatively.
    assert!(
        (form.lcb * 100.0 - published::LCB_PERCENT).abs() < 0.3,
        "LCB {} % against published {} %",
        form.lcb * 100.0,
        published::LCB_PERCENT
    );
    assert!(
        (form.lcf * 100.0 - published::LCF_PERCENT).abs() < 0.4,
        "LCF {} % against published {} %",
        form.lcf * 100.0,
        published::LCF_PERCENT
    );
}

/// The hull must sit inside the envelope of the models the DSYHS regressions
/// were fitted to — that is half the point of fitting it to a real design's
/// coefficients, since its predecessor at `Cp = 0.63` did not and could carry
/// no resistance calculation at all.
#[test]
fn the_fitted_hull_is_inside_the_dsyhs_envelope() {
    let hull = spec()
        .hull_parameters()
        .expect("the file declares its measured parameters");
    hull.check_envelope()
        .expect("a hull fitted to the YD-41's coefficients must be in range");
}

/// The declared parameters must still describe the offsets. This is what the
/// two-way hull description buys: an edit to the geometry that is not reflected
/// in the declared block, or vice versa, shows up here instead of silently
/// feeding wrong numbers to the resistance model.
#[test]
fn declared_parameters_still_match_the_geometry() {
    let spec = spec();
    let mesh = mesh(16);
    let flotation = solve_flotation(
        &mesh,
        &body(&spec),
        &Water::default(),
        &FlotationOptions::default(),
    )
    .expect("must float");
    let options = FormOptions::at_draft(&mesh, flotation.draft).expect("immersed");
    let derived = hull_form(&mesh, &options).expect("measurable");
    let declared = spec.hull_parameters().expect("declared");

    assert_relative_eq!(
        derived.canoe_volume,
        declared.canoe_volume,
        max_relative = 5e-3
    );
    assert_relative_eq!(derived.prismatic, declared.prismatic, max_relative = 5e-3);
    assert_relative_eq!(derived.midship, declared.midship, max_relative = 5e-3);
    assert!((derived.lcb * 100.0 - declared.lcb * 100.0).abs() < 0.05);
    assert!((derived.lcf * 100.0 - declared.lcf * 100.0).abs() < 0.05);
}

/// Draft must not drift materially as the mesh is refined. The bug this guards
/// against is real and was caught once already: arc-length resampling used to
/// chord across the corners of a station contour, which cost a rectangular
/// barge 4 % of its displacement at 16 points per station.
///
/// The residual drift is *not* that. This hull's stations already carry 13
/// offsets, so asking for 13 subdivides nothing and the contours are
/// geometrically identical at both settings. What changes is the surface
/// *between* stations: lofting two rings of N points and two rings of 4N points
/// produces different triangulations of the same ruled surface, whose enclosed
/// volume converges as `O(1/N²)`. Measured 0.4032 against 0.4037 m, 1.3e-3
/// relative — the right order for that, and three orders off the corner-cutting
/// failure this test exists to catch.
#[test]
fn draft_is_stable_across_resolutions() {
    let spec = spec();
    let body = body(&spec);
    let water = Water::default();
    let draft_at = |points: usize| {
        solve_flotation(&mesh(points), &body, &water, &FlotationOptions::default())
            .expect("must float")
            .draft
    };
    assert_relative_eq!(draft_at(13), draft_at(64), max_relative = 3e-3);
}
