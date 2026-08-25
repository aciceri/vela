//! Integration tests for buoyancy and flotation.
//!
//! Ground truth is analytic wherever a closed form exists — a rectangular barge
//! has one for every quantity here — and otherwise a conservation law or an
//! equilibrium condition that must hold for any hull.

use approx::assert_relative_eq;
use nalgebra::{UnitQuaternion, Vector3};
use std::f64::consts::PI;
use vela_core::boat::{HullSpec, Offset, Station};
use vela_core::hydrostatics::{hydrostatics, solve_flotation, FlotationError, FlotationOptions};
use vela_core::{loft_hull, BodyState, LoftOptions, MassProperties, RigidBody, TriMesh, Water};

const GRAVITY: f64 = 9.806_65;
const DENSITY: f64 = 1025.0;

fn water() -> Water {
    Water { density: DENSITY }
}

/// A rectangular barge: `length` by `2 * half_beam` in plan, `depth` deep.
fn barge(length: f64, half_beam: f64, depth: f64) -> TriMesh {
    let section = vec![
        Offset { y: 0.0, z: 0.0 },
        Offset {
            y: half_beam,
            z: 0.0,
        },
        Offset {
            y: half_beam,
            z: depth,
        },
    ];
    let hull = HullSpec {
        stations: vec![
            Station {
                x: 0.0,
                points: section.clone(),
            },
            Station {
                x: length,
                points: section,
            },
        ],
    };
    loft_hull(&hull, &LoftOptions::default())
}

/// A half-cylinder of radius `radius`, flat side up, `length` long.
///
/// Its section is discretized into `steps` segments per quarter circle, so
/// hydrostatic quantities approach the analytic circle values as `steps` grows.
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
    let hull = HullSpec {
        stations: vec![
            Station {
                x: 0.0,
                points: section.clone(),
            },
            Station {
                x: length,
                points: section,
            },
        ],
    };
    loft_hull(&hull, &LoftOptions::default())
}

fn body(mass: f64, cog: Vector3<f64>) -> RigidBody {
    let properties = MassProperties::from_gyradii(mass, cog, Vector3::new(1.0, 2.0, 2.0))
        .expect("valid mass properties");
    RigidBody::with_gravity(properties, GRAVITY).expect("valid rigid body")
}

/// Places the hull at a given sinkage, upright.
fn upright(sinkage: f64) -> BodyState {
    BodyState {
        position: Vector3::new(0.0, 0.0, sinkage),
        ..BodyState::default()
    }
}

#[test]
fn submerged_volume_of_a_barge_matches_draft_times_waterplane() {
    let mesh = barge(10.0, 1.5, 2.0);
    let draft = 0.6;
    let hydro = hydrostatics(&mesh, &upright(draft), &water(), GRAVITY);

    assert_relative_eq!(hydro.volume, 10.0 * 3.0 * draft, epsilon = 1e-9);
    assert_relative_eq!(hydro.waterplane_area, 10.0 * 3.0, epsilon = 1e-9);
    // Wetted area: bottom plus two sides plus two ends.
    let expected_wetted = 10.0 * 3.0 + 2.0 * 10.0 * draft + 2.0 * 3.0 * draft;
    assert_relative_eq!(hydro.wetted_area, expected_wetted, epsilon = 1e-9);
}

#[test]
fn centre_of_buoyancy_of_a_barge_is_at_half_the_draft() {
    let mesh = barge(10.0, 1.5, 2.0);
    let draft = 0.6;
    let hydro = hydrostatics(&mesh, &upright(draft), &water(), GRAVITY);

    assert_relative_eq!(hydro.centre_of_buoyancy.x, 5.0, epsilon = 1e-9);
    assert_relative_eq!(hydro.centre_of_buoyancy.y, 0.0, epsilon = 1e-9);
    // Body z is down and the keel is at z = 0, so half the draft up from the
    // bottom is a *negative* body z.
    assert_relative_eq!(hydro.centre_of_buoyancy.z, -draft / 2.0, epsilon = 1e-9);
}

/// The sharpest test in this file. Pressure integration and the divergence
/// theorem are two independent computations over the same clipped panels; both
/// are exact, so they must agree to machine precision. Any error in the
/// clipper, the winding, or the pressure integral breaks this.
#[test]
fn pressure_integral_equals_archimedes_and_acts_through_the_centre_of_buoyancy() {
    let mesh = barge(10.0, 1.5, 2.0);
    let state = upright(0.6);
    let hydro = hydrostatics(&mesh, &state, &water(), GRAVITY);

    // Magnitude: rho g V, directed upward, which is negative z in NED.
    let expected = DENSITY * GRAVITY * hydro.volume;
    assert_relative_eq!(hydro.buoyancy.force.x, 0.0, epsilon = 1e-6);
    assert_relative_eq!(hydro.buoyancy.force.y, 0.0, epsilon = 1e-6);
    assert_relative_eq!(hydro.buoyancy.force.z, -expected, epsilon = 1e-6);

    // Line of action: the moment about the body origin must equal the moment of
    // the resultant placed at the centre of buoyancy.
    let expected_moment = hydro.centre_of_buoyancy.cross(&hydro.buoyancy.force);
    assert_relative_eq!(hydro.buoyancy.moment, expected_moment, epsilon = 1e-6);
}

#[test]
fn the_cross_check_survives_heel_and_trim() {
    let mesh = barge(10.0, 1.5, 2.0);
    let state = BodyState {
        position: Vector3::new(0.0, 0.0, 0.8),
        attitude: UnitQuaternion::from_euler_angles(0.25, -0.06, 0.0),
        ..BodyState::default()
    };
    let hydro = hydrostatics(&mesh, &state, &water(), GRAVITY);

    assert!(hydro.volume > 0.0);
    assert_relative_eq!(
        hydro.buoyancy.force.norm(),
        DENSITY * GRAVITY * hydro.volume,
        epsilon = 1e-6
    );
    assert_relative_eq!(
        hydro.buoyancy.moment,
        hydro.centre_of_buoyancy.cross(&hydro.buoyancy.force),
        epsilon = 1e-6
    );
    // Buoyancy is vertical in the world frame whatever the attitude.
    let world = state.to_world(hydro.buoyancy.force);
    assert_relative_eq!(world.x, 0.0, epsilon = 1e-6);
    assert_relative_eq!(world.y, 0.0, epsilon = 1e-6);
}

#[test]
fn a_fully_submerged_hull_displaces_its_whole_volume() {
    let mesh = barge(10.0, 1.5, 2.0);
    let full = mesh.signed_volume();
    // Sink it well below the surface.
    let hydro = hydrostatics(&mesh, &upright(50.0), &water(), GRAVITY);
    assert_relative_eq!(hydro.volume, full, epsilon = 1e-9);
    assert_relative_eq!(hydro.waterplane_area, 0.0, epsilon = 1e-9);
}

#[test]
fn a_dry_hull_displaces_nothing() {
    let mesh = barge(10.0, 1.5, 2.0);
    let hydro = hydrostatics(&mesh, &upright(-5.0), &water(), GRAVITY);
    assert_relative_eq!(hydro.volume, 0.0, epsilon = 1e-12);
    assert_relative_eq!(hydro.buoyancy.force, Vector3::zeros(), epsilon = 1e-12);
}

/// A curved hull against the closed form for a half disc, at a fixed pose.
///
/// Deliberately not routed through the flotation solver: floating a
/// half-cylinder exactly at its diameter puts the waterline on the deck edge,
/// a degenerate configuration where the hull is on the verge of full
/// submersion. Worse, its discretized section is a polygon *inscribed* in the
/// circle, so it displaces slightly less than the circle and genuinely cannot
/// carry the analytic mass. Fixing the pose tests the geometry and the
/// integration, which is what this is about, and leaves the solver to the
/// tests below.
#[test]
fn half_cylinder_hydrostatics_match_the_analytic_half_disc() {
    let (radius, length, steps) = (1.0, 6.0, 96);
    let mesh = half_cylinder(radius, length, steps);
    let hydro = hydrostatics(&mesh, &upright(radius), &water(), GRAVITY);

    // An inscribed polygon is fine by O(step²): about 4e-5 relative here.
    let analytic_volume = 0.5 * PI * radius * radius * length;
    assert_relative_eq!(hydro.volume, analytic_volume, max_relative = 1e-4);

    // Centroid of a half disc lies 4R/(3π) from the flat face; body z is down
    // and the flat face is at body z = -radius.
    let expected = -(radius - 4.0 * radius / (3.0 * PI));
    assert_relative_eq!(hydro.centre_of_buoyancy.z, expected, max_relative = 1e-3);
    assert_relative_eq!(hydro.centre_of_buoyancy.x, length / 2.0, epsilon = 1e-9);
    // No waterplane assertion here: at this draft the flat face lies exactly on
    // the surface, so nothing straddles it and the cut set is legitimately
    // empty. The waterplane gets its own test at a draft where it exists.
}

/// Waterplane area of a circular section against its analytic chord. Half
/// immersed, the chord is `2√(R² − (R−d)²)`, which at `d = R/2` is `R√3`.
#[test]
fn half_cylinder_waterplane_matches_the_analytic_chord() {
    let (radius, length) = (1.0, 6.0);
    let mesh = half_cylinder(radius, length, 96);
    let draft = 0.5 * radius;
    let hydro = hydrostatics(&mesh, &upright(draft), &water(), GRAVITY);

    let chord = 2.0 * (radius * radius - (radius - draft).powi(2)).sqrt();
    assert_relative_eq!(hydro.waterplane_area, chord * length, max_relative = 1e-3);
}

/// The flotation solver on a hull that is *neutrally stable in heel*: a
/// circular section with its CoG on the centerline makes that Jacobian column
/// vanish, so every heel angle is an equilibrium. This is the case that plain
/// Newton cannot solve and Levenberg-Marquardt can.
#[test]
fn a_neutrally_stable_hull_still_reaches_equilibrium() {
    let (radius, length) = (1.0, 6.0);
    let mesh = half_cylinder(radius, length, 96);
    // Half of what it could carry: floats comfortably, well clear of the deck.
    let mass = 0.5 * DENSITY * 0.5 * PI * radius * radius * length;
    let rigid = body(mass, Vector3::new(length / 2.0, 0.0, -0.5 * radius));

    let flotation =
        solve_flotation(&mesh, &rigid, &water(), &FlotationOptions::default()).expect("must float");

    assert_relative_eq!(
        flotation.hydrostatics.displacement(&water()),
        mass,
        max_relative = 1e-6
    );
    assert!(flotation.draft > 0.0 && flotation.draft < radius);
    // Longitudinal balance is determined even though heel is not.
    let cob = flotation
        .state
        .to_world(flotation.hydrostatics.centre_of_buoyancy);
    let cog = flotation.state.to_world(rigid.mass_properties().cog());
    assert_relative_eq!(cob.x, cog.x, epsilon = 1e-6);
}

#[test]
fn a_barge_floats_at_the_analytic_draft() {
    let (length, half_beam, depth) = (10.0, 1.5, 2.0);
    let mesh = barge(length, half_beam, depth);
    let draft = 0.45;
    let mass = DENSITY * length * 2.0 * half_beam * draft;
    let rigid = body(mass, Vector3::new(length / 2.0, 0.0, -1.0));

    let flotation =
        solve_flotation(&mesh, &rigid, &water(), &FlotationOptions::default()).expect("must float");

    assert_relative_eq!(flotation.draft, draft, epsilon = 1e-6);
    assert_relative_eq!(flotation.heel, 0.0, epsilon = 1e-9);
    assert_relative_eq!(flotation.trim, 0.0, epsilon = 1e-9);
    assert_relative_eq!(
        flotation.hydrostatics.displacement(&water()),
        mass,
        max_relative = 1e-9
    );
}

/// The defining property of equilibrium, independent of hull shape: the centre
/// of buoyancy sits directly below the centre of gravity. Holds for any hull,
/// any mass distribution, and is not a restatement of the solver's residual.
#[test]
fn at_equilibrium_the_centre_of_buoyancy_lies_under_the_centre_of_gravity() {
    let (length, half_beam, depth) = (10.0, 1.5, 2.0);
    let mesh = barge(length, half_beam, depth);
    let mass = DENSITY * length * 2.0 * half_beam * 0.5;
    // Off-centre in both directions: the boat must heel and trim to suit.
    let cog = Vector3::new(0.6 * length, 0.25, -0.8);
    let rigid = body(mass, cog);

    let flotation =
        solve_flotation(&mesh, &rigid, &water(), &FlotationOptions::default()).expect("must float");

    let to_world = |p: Vector3<f64>| flotation.state.to_world(p);
    let buoyancy = to_world(flotation.hydrostatics.centre_of_buoyancy);
    let gravity = to_world(cog);

    assert_relative_eq!(buoyancy.x, gravity.x, epsilon = 1e-6);
    assert_relative_eq!(buoyancy.y, gravity.y, epsilon = 1e-6);
}

#[test]
fn weight_forward_trims_the_bow_down() {
    let (length, half_beam, depth) = (10.0, 1.5, 2.0);
    let mesh = barge(length, half_beam, depth);
    let mass = DENSITY * length * 2.0 * half_beam * 0.4;

    let forward = solve_flotation(
        &mesh,
        &body(mass, Vector3::new(0.65 * length, 0.0, -0.8)),
        &water(),
        &FlotationOptions::default(),
    )
    .expect("must float");
    let aft = solve_flotation(
        &mesh,
        &body(mass, Vector3::new(0.35 * length, 0.0, -0.8)),
        &water(),
        &FlotationOptions::default(),
    )
    .expect("must float");

    // Positive trim is bow up, so a forward CoG must give a negative trim, and
    // the two cases must be mirror images.
    assert!(forward.trim < 0.0, "forward weight must sink the bow");
    assert!(aft.trim > 0.0, "aft weight must lift the bow");
    assert_relative_eq!(forward.trim, -aft.trim, max_relative = 1e-6);
}

#[test]
fn ballast_to_starboard_heels_to_starboard() {
    let mesh = barge(10.0, 1.5, 2.0);
    let mass = DENSITY * 10.0 * 3.0 * 0.4;
    let flotation = solve_flotation(
        &mesh,
        &body(mass, Vector3::new(5.0, 0.4, -0.5)),
        &water(),
        &FlotationOptions::default(),
    )
    .expect("must float");

    assert!(
        flotation.heel > 0.0,
        "weight to starboard must heel to starboard, got {}",
        flotation.heel
    );
}

#[test]
fn an_overloaded_hull_is_rejected_rather_than_sunk_quietly() {
    let mesh = barge(10.0, 1.5, 2.0);
    let full_displacement = DENSITY * mesh.signed_volume();
    let rigid = body(full_displacement * 1.5, Vector3::new(5.0, 0.0, -1.0));

    let error = solve_flotation(&mesh, &rigid, &water(), &FlotationOptions::default())
        .expect_err("must not float");
    assert!(matches!(error, FlotationError::InsufficientBuoyancy { .. }));
}

#[test]
fn heave_stiffness_follows_the_waterplane() {
    let mesh = barge(12.0, 2.0, 2.0);
    let hydro = hydrostatics(&mesh, &upright(0.5), &water(), GRAVITY);
    let expected = DENSITY * GRAVITY * 12.0 * 4.0;
    assert_relative_eq!(
        hydro.heave_stiffness(&water(), GRAVITY),
        expected,
        epsilon = 1e-6
    );
}
