//! Integration tests for the derived hull form parameters.
//!
//! Ground truth is a closed form throughout, never a recording of what this
//! code happened to print. Three shapes carry it: a box, whose every parameter
//! is a product of three numbers; a half-cylinder, whose immersed section is a
//! circular segment; and a wedge, whose waterplane is a trapezoid with a
//! textbook centroid. Where the shape-independent identities of a prism apply
//! they are asserted at machine precision, because they hold for the
//! discretized hull and not merely in the limit.

use approx::{assert_abs_diff_eq, assert_relative_eq};
use std::f64::consts::PI;
use vela_core::boat::{HullSpec, Offset, Station};
use vela_core::sections::{
    compare, hull_form, section_area, sectional_area_curve, FormError, FormOptions, HullForm,
};
use vela_core::{loft_hull, LoftOptions, TriMesh};

/// Lofts a hull from its two end stations.
fn loft(sections: [(f64, Vec<Offset>); 2], points_per_station: usize) -> TriMesh {
    let hull = HullSpec {
        stations: sections
            .into_iter()
            .map(|(x, points)| Station { x, points })
            .collect(),
    };
    loft_hull(&hull, &LoftOptions { points_per_station })
}

/// A rectangular section: flat bottom out to `half_beam`, vertical side up to
/// `depth`. Heights are measured up from the baseline, as the file frame does.
fn box_section(half_beam: f64, depth: f64) -> Vec<Offset> {
    vec![
        Offset { y: 0.0, z: 0.0 },
        Offset {
            y: half_beam,
            z: 0.0,
        },
        Offset {
            y: half_beam,
            z: depth,
        },
    ]
}

/// A rectangular barge, `length` by `2 * half_beam` in plan, `depth` deep.
fn barge(length: f64, half_beam: f64, depth: f64, points_per_station: usize) -> TriMesh {
    let section = box_section(half_beam, depth);
    loft(
        [(0.0, section.clone()), (length, section)],
        points_per_station,
    )
}

/// A box tapered in beam only: flat bottom, vertical sides at `± b(x)` with
/// `b` linear in `x`, `depth` deep.
///
/// Lofted at exactly three points per station so that `loft::refine` does
/// nothing at all. It distributes its extra points in proportion to segment
/// length, and with different beams at the two ends the two contours would come
/// out parameterized differently — the ruled surface between them would then
/// warp off the exact wedge, and the closed forms below would stop being closed
/// forms for the shape actually being measured.
fn wedge(length: f64, half_beam_aft: f64, half_beam_forward: f64, depth: f64) -> TriMesh {
    loft(
        [
            (0.0, box_section(half_beam_aft, depth)),
            (length, box_section(half_beam_forward, depth)),
        ],
        3,
    )
}

/// A half-cylinder of radius `radius`, flat side up, `length` long.
///
/// The section is inscribed in the circle with `steps` segments per quarter, so
/// it is slightly smaller than the circle and converges on it as `steps` grows.
fn half_cylinder(radius: f64, length: f64, steps: usize) -> TriMesh {
    let section: Vec<Offset> = (0..=steps)
        .map(|i| {
            let theta = 0.5 * PI * i as f64 / steps as f64;
            Offset {
                y: radius * theta.sin(),
                z: radius * (1.0 - theta.cos()),
            }
        })
        .collect();
    loft([(0.0, section.clone()), (length, section)], steps + 1)
}

/// Asserts that two parameter sets agree, using the comparison helper as the
/// assertion so that the helper is exercised by every drift test.
fn assert_forms_agree(reference: &HullForm, other: &HullForm, tolerance: f64) {
    let differences = compare(reference, other, tolerance);
    assert!(
        differences.is_empty(),
        "parameters drifted: {differences:?}"
    );
}

#[test]
fn a_barge_has_every_parameter_in_closed_form() {
    let (length, half_beam, depth, draft) = (10.0, 1.5, 2.0, 0.6);
    let beam = 2.0 * half_beam;
    let mesh = barge(length, half_beam, depth, 16);
    let form = hull_form(&mesh, &FormOptions::new(-draft)).expect("a barge floats");

    assert_relative_eq!(form.waterline_length, length, epsilon = 1e-12);
    assert_relative_eq!(form.waterline_beam, beam, epsilon = 1e-12);
    assert_relative_eq!(form.canoe_draft, draft, epsilon = 1e-12);
    assert_relative_eq!(form.canoe_volume, length * beam * draft, epsilon = 1e-12);
    assert_relative_eq!(form.waterplane_area, length * beam, epsilon = 1e-12);
    assert_relative_eq!(form.midship_area, beam * draft, epsilon = 1e-12);
    // Bottom, two sides, two ends. The deck is dry and the waterplane is not
    // hull, so neither counts.
    let wetted = length * beam + 2.0 * length * draft + 2.0 * beam * draft;
    assert_relative_eq!(form.wetted_surface, wetted, epsilon = 1e-12);
    // A box is its own prism and its own midship section.
    assert_relative_eq!(form.prismatic, 1.0, epsilon = 1e-12);
    assert_relative_eq!(form.midship, 1.0, epsilon = 1e-12);
    // Symmetric fore and aft, so both centroids are exactly amidships. A
    // relative test is meaningless against zero; an absolute one is the whole
    // point, since a sign error here would land near ±0.5.
    assert_abs_diff_eq!(form.lcb, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(form.lcf, 0.0, epsilon = 1e-12);
}

#[test]
fn a_barge_has_the_same_section_area_at_every_station() {
    let (length, half_beam, draft) = (10.0, 1.5, 0.6);
    let expected = 2.0 * half_beam * draft;
    let mesh = barge(length, half_beam, 2.0, 16);
    let options = FormOptions {
        waterline_z: -draft,
        station_count: 17,
    };

    let curve = sectional_area_curve(&mesh, &options);
    assert_eq!(curve.areas().len(), 17);
    for (station, area) in curve.stations().iter().zip(curve.areas()) {
        assert_relative_eq!(*area, expected, epsilon = 1e-12);
        // Stations are interval midpoints, so none of them lands on the
        // transom, where the area curve is genuinely discontinuous.
        assert!(
            *station > 0.0 && *station < length,
            "station {station} on end"
        );
    }

    // The whole curve rather than just its maximum: integrating it must return
    // the volume the divergence theorem gets by an unrelated route.
    assert_relative_eq!(
        curve.integrated_volume(),
        length * expected,
        epsilon = 1e-12
    );
    // And the one-off entry point must agree with the curve it is a slice of.
    assert_relative_eq!(section_area(&mesh, -draft, 5.0), expected, epsilon = 1e-12);
}

/// A half-cylinder floating at half its radius. The immersed section is a
/// circular segment, and for a chord a distance `a` below the centre the
/// segment area is `R² acos(a/R) − a √(R² − a²)`. Half immersed, `a = R/2`,
/// which gives `R² (π/3 − √3/4)`; the chord itself is `2 √(R² − a²) = R √3`.
#[test]
fn a_half_cylinder_matches_the_circular_segment() {
    let (radius, length, steps) = (1.0, 8.0, 128);
    let draft = 0.5 * radius;
    let beam = radius * 3.0f64.sqrt();
    let segment = radius * radius * (PI / 3.0 - 3.0f64.sqrt() / 4.0);

    let mesh = half_cylinder(radius, length, steps);
    let form = hull_form(&mesh, &FormOptions::new(-draft)).expect("a half-cylinder floats");

    // The section is a polygon inscribed in the circle, so every measured value
    // sits just inside the analytic one. Measured relative differences at 128
    // steps per quarter circle: area and volume -4.28e-5, beam -2.23e-5,
    // midship coefficient -2.05e-5. Halving the step count quadruples all of
    // them, which is the second-order convergence an inscribed polygon owes.
    assert_relative_eq!(form.midship_area, segment, max_relative = 1e-4);
    assert_relative_eq!(form.canoe_volume, segment * length, max_relative = 1e-4);
    assert_relative_eq!(form.waterline_beam, beam, max_relative = 1e-4);
    assert_relative_eq!(form.canoe_draft, draft, epsilon = 1e-12);
    assert_relative_eq!(form.waterline_length, length, epsilon = 1e-12);
    // C_m of a half-immersed circle: 0.7092, independent of the radius.
    assert_relative_eq!(form.midship, segment / (beam * draft), max_relative = 1e-4);
    assert_abs_diff_eq!(form.lcf, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(form.lcb, 0.0, epsilon = 1e-12);
}

/// `C_p = 1` for any prismatic hull, whatever the section shape. This is the
/// sharpest check available here precisely because it is shape-independent: it
/// holds for the *discretized* section too, since the volume of a prism is its
/// section area times its length whether that section is a circle or the
/// polygon inscribed in it. Nothing cancels unless the section area and the
/// volume integral agree exactly, so this catches a section area that is
/// systematically wrong by any factor at all — which a comparison against a
/// smooth analytic shape, with its convergence tolerance, would let through.
#[test]
fn the_prismatic_coefficient_is_one_for_every_prism() {
    for steps in [4, 8, 32, 64] {
        let mesh = half_cylinder(1.0, 8.0, steps);
        for draft in [0.25, 0.5, 0.9] {
            let form = hull_form(&mesh, &FormOptions::new(-draft))
                .unwrap_or_else(|error| panic!("{steps} steps at draft {draft}: {error}"));
            assert_relative_eq!(form.prismatic, 1.0, epsilon = 1e-12);
        }
    }

    let form = hull_form(&barge(10.0, 1.5, 2.0, 16), &FormOptions::new(-0.6)).expect("floats");
    assert_relative_eq!(form.prismatic, 1.0, epsilon = 1e-12);
}

/// A hull tapered at one end. With vertical sides the waterplane is a trapezoid
/// of half-beams `b_a` aft and `b_f` forward, whose centroid puts
/// `LCF = (b_f − b_a) / (6 (b_a + b_f))` of the length from midship. A fine bow
/// therefore lands aft of midship, which is *negative* in a frame where `x` is
/// forward. That is the sign the Delft series expects of a real yacht — its own
/// envelope for LCF runs from −9.5 % to −1.8 % — and getting it backwards would
/// stay invisible until it moved a resistance curve.
#[test]
fn a_fine_bow_puts_the_centre_of_flotation_aft() {
    let (length, depth, draft) = (10.0, 2.0, 0.6);
    let (aft_beam, forward_beam) = (1.5, 0.5);
    let mesh = wedge(length, aft_beam, forward_beam, depth);
    let options = FormOptions::new(-draft);
    let form = hull_form(&mesh, &options).expect("a wedge floats");

    let expected_lcf = (forward_beam - aft_beam) / (6.0 * (aft_beam + forward_beam));
    assert!(expected_lcf < 0.0, "the closed form must itself be aft");
    assert!(
        form.lcf < 0.0,
        "a fine bow must put the centre of flotation aft, got {}",
        form.lcf
    );
    assert_relative_eq!(form.lcf, expected_lcf, epsilon = 1e-12);
    // Sections are rectangles of width 2b(x), so volume is distributed along
    // the hull exactly as waterplane area is and the two centroids coincide.
    // An independent route to the same number: the volume integral rather than
    // Green's theorem on the waterplane.
    assert_relative_eq!(form.lcb, form.lcf, epsilon = 1e-12);

    assert_relative_eq!(form.waterline_length, length, epsilon = 1e-12);
    assert_relative_eq!(form.waterline_beam, 2.0 * aft_beam, epsilon = 1e-12);
    let waterplane = length * (aft_beam + forward_beam);
    assert_relative_eq!(form.waterplane_area, waterplane, epsilon = 1e-12);
    assert_relative_eq!(form.canoe_volume, draft * waterplane, epsilon = 1e-12);

    // The area curve of this hull is linear in `x`, which the midpoint rule
    // integrates exactly — so the curve is verified at every station, not just
    // where it peaks.
    let curve = sectional_area_curve(&mesh, &options);
    assert_relative_eq!(
        curve.integrated_volume(),
        form.canoe_volume,
        epsilon = 1e-12
    );
}

/// The mirror image, which must come out with the sign reversed. Asserting one
/// hull's sign shows only that the formula agrees with itself; asserting both
/// shows that it tracks the geometry.
#[test]
fn mirroring_the_taper_mirrors_the_longitudinal_centroids() {
    let options = FormOptions::new(-0.6);
    let fine_bow = hull_form(&wedge(10.0, 1.5, 0.5, 2.0), &options).expect("floats");
    let fine_stern = hull_form(&wedge(10.0, 0.5, 1.5, 2.0), &options).expect("floats");

    assert!(fine_bow.lcf < 0.0 && fine_stern.lcf > 0.0);
    assert_relative_eq!(fine_stern.lcf, -fine_bow.lcf, epsilon = 1e-12);
    assert_relative_eq!(fine_stern.lcb, -fine_bow.lcb, epsilon = 1e-12);
    // Reflection leaves everything that is not longitudinal alone.
    assert_relative_eq!(
        fine_stern.canoe_volume,
        fine_bow.canoe_volume,
        epsilon = 1e-12
    );
    assert_relative_eq!(
        fine_stern.waterplane_area,
        fine_bow.waterplane_area,
        epsilon = 1e-12
    );
    assert_relative_eq!(
        fine_stern.wetted_surface,
        fine_bow.wetted_surface,
        epsilon = 1e-12
    );
}

/// Refining the mesh must not move a parameter. `loft::refine` subdivides
/// rather than resamples, so a prismatic hull is represented exactly at every
/// resolution — which makes the parameters not merely convergent but identical,
/// and 1e-12 a real assertion rather than a concession to discretization.
#[test]
fn a_prisms_parameters_do_not_drift_with_mesh_resolution() {
    let options = FormOptions::new(-0.6);
    let reference = hull_form(&barge(10.0, 1.5, 2.0, 3), &options).expect("floats");
    for points_per_station in [4, 16, 64, 257] {
        let form = hull_form(&barge(10.0, 1.5, 2.0, points_per_station), &options).expect("floats");
        assert_forms_agree(&reference, &form, 1e-12);
    }
}

/// Nor with the longitudinal sampling, since every station of a prism sees the
/// same section — including the single-station degenerate case, which a naive
/// endpoint-based curve could not even represent.
#[test]
fn a_prisms_parameters_do_not_drift_with_station_count() {
    let mesh = barge(10.0, 1.5, 2.0, 16);
    let reference = hull_form(&mesh, &FormOptions::new(-0.6)).expect("floats");
    for station_count in [1, 2, 7, 50, 501] {
        let form = hull_form(
            &mesh,
            &FormOptions {
                waterline_z: -0.6,
                station_count,
            },
        )
        .expect("floats");
        assert_forms_agree(&reference, &form, 1e-12);
    }
}

#[test]
fn comparison_names_only_the_parameters_that_disagree() {
    let derived = hull_form(&barge(10.0, 1.5, 2.0, 16), &FormOptions::new(-0.6)).expect("floats");
    assert!(compare(&derived, &derived, 0.0).is_empty());

    let mut declared = derived;
    declared.prismatic *= 1.1;
    declared.waterplane_area *= 1.02;

    // Scored against the larger magnitude: 0.1/1.1 = 9.1 % for the prismatic
    // coefficient, 0.02/1.02 = 2.0 % for the waterplane area.
    let loose: Vec<&str> = compare(&derived, &declared, 0.05)
        .iter()
        .map(|difference| difference.parameter)
        .collect();
    assert_eq!(loose, ["Cp"]);

    let tight = compare(&derived, &declared, 0.01);
    let named: Vec<&str> = tight
        .iter()
        .map(|difference| difference.parameter)
        .collect();
    assert_eq!(named, ["Aw", "Cp"]);
    assert_relative_eq!(tight[1].relative, 0.1 / 1.1, epsilon = 1e-12);
    assert_relative_eq!(tight[1].derived, derived.prismatic, epsilon = 1e-12);
    assert_relative_eq!(tight[1].declared, declared.prismatic, epsilon = 1e-12);
}

/// A declared centre of flotation of zero against a measured one that is not.
/// The disagreement has to come out finite and equal to the difference in
/// fractions of length: a relative-only measure would report an infinite error
/// for the one parameter that legitimately passes through zero, which is
/// exactly the case a boat file gets wrong by leaving the field at its default.
#[test]
fn a_zero_declared_centroid_reports_a_finite_disagreement() {
    let derived = hull_form(&wedge(10.0, 1.5, 0.5, 2.0), &FormOptions::new(-0.6)).expect("floats");
    let mut declared = derived;
    declared.lcf = 0.0;

    let differences = compare(&derived, &declared, 0.01);
    assert_eq!(differences.len(), 1);
    assert_eq!(differences[0].parameter, "LCF");
    assert_relative_eq!(differences[0].relative, 1.0 / 12.0, epsilon = 1e-12);
    // 8.3 % of the waterline length is a real disagreement but a finite one, so
    // a tolerance above it must accept the hull rather than diverge.
    assert!(compare(&derived, &declared, 0.1).is_empty());
}

#[test]
fn a_dry_hull_has_no_form_parameters() {
    let mesh = barge(10.0, 1.5, 2.0, 16);
    // Waterplane on the keel: the bottom is wetted, nothing is immersed.
    assert_eq!(
        hull_form(&mesh, &FormOptions::new(0.0)),
        Err(FormError::Dry)
    );
    assert_eq!(
        hull_form(&TriMesh::default(), &FormOptions::new(-1.0)),
        Err(FormError::Dry)
    );
}

/// A wholly submerged hull has volume but no waterline, and the parameters
/// measured *on* the waterline do not exist for it. Returning the bounding-box
/// length instead — the tempting fudge — would feed a plausible lie into the
/// resistance model.
#[test]
fn a_submerged_hull_has_no_waterline() {
    let mesh = barge(10.0, 1.5, 2.0, 16);
    assert_eq!(
        hull_form(&mesh, &FormOptions::new(-5.0)),
        Err(FormError::NoWaterplane)
    );
}

#[test]
fn draft_is_measured_from_the_deepest_point_of_the_hull() {
    let mesh = barge(10.0, 1.5, 2.0, 16);
    let options = FormOptions::at_draft(&mesh, 0.6).expect("a non-empty mesh has a deepest point");
    assert_relative_eq!(options.waterline_z, -0.6, epsilon = 1e-12);

    let form = hull_form(&mesh, &options).expect("floats");
    assert_relative_eq!(form.canoe_draft, 0.6, epsilon = 1e-12);
    assert_eq!(FormOptions::at_draft(&TriMesh::default(), 0.6), None);
}
