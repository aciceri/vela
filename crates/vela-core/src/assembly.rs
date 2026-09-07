//! Building a running simulation from a boat file.
//!
//! This is the one place that knows how the data format maps onto the force
//! modules, which keeps that mapping out of every frontend. A CLI, a test and a
//! renderer all ask for the same simulation and get the same one.
//!
//! # Why the result is captive in trim and yaw
//!
//! The assembled simulation is restrained in pitch and yaw — see
//! [`crate::sim::Captive`] for the mechanism. The reason is a gap in the data,
//! not a shortcut in the physics: the boat format carries no longitudinal
//! position for the keel, the rudder or the mast, because published yacht
//! particulars do not include them. Every force in the lateral plane and every
//! aerodynamic force therefore acts at `x = 0` and produces no yaw moment.
//!
//! Integrating those two modes would mean integrating an identically-zero
//! moment where the real one is not zero: the boat would settle to a heading
//! and a trim that look like results and are artefacts. Restraining them says
//! so in the mechanism. What remains free — surge, sway, heave, roll — is
//! exactly the balance a classical velocity prediction program solves, and it
//! needs no longitudinal information at all.
//!
//! Lifting this restriction is a schema change plus sourced positions, not a
//! change to any force model.

use crate::aero::{RigDimensions, Sail, SailSet};
use crate::appendages::{FoilPlanform, HullScalars, Keel};
use crate::balance::SailPlan as BalanceSailPlan;
use crate::boat::{
    Aerodynamics, AppendagesSpec, BoatSpec, FoilSpec, RigSpec, SailShapeSpec, SpecError,
};
use crate::controls::Controls;
use crate::cummins::{MemoryError, MemoryOptions, TransformOptions};
use crate::env::Environment;
use crate::flying::{Planform, Response, Shape};
use crate::geometry::Point;
use crate::lewis::{station_geometry, LewisForm};
use crate::loft::{loft_hull, LoftOptions};
use crate::mass::MassError;
use crate::modules::radiation::{Provenance, Radiation, RadiationSpectra};
use crate::modules::{Buoyancy, CanoeBody, LateralSystem, Sails};
use crate::rigid_body::RigidBody;
use crate::sail::{Member, Options as SailOptions};
use crate::sim::{Captive, ForceModule, Sim};
use crate::state::BodyState;
use crate::strip::{lateral_spectra, vertical_spectra, Strip};
use crate::tasai::{SectionSolver, TasaiOptions};
use crate::{SEA_WATER_DENSITY, STANDARD_GRAVITY};
use nalgebra::{Matrix6, Vector3};
use std::fmt;

/// Why a boat file could not be turned into a simulation.
///
/// Each variant names the block that is missing and what it would have been
/// used for, because "cannot simulate this boat" on its own sends a reader
/// hunting through a data file.
#[derive(Debug, Clone, PartialEq)]
pub enum AssemblyError {
    /// No station offsets, so there is no surface to integrate pressure over
    /// and therefore no buoyancy and no righting moment.
    MissingHullOffsets,
    /// No scalar form parameters, which the resistance regressions consume
    /// directly and cannot derive from geometry.
    MissingParameters,
    /// No keel and rudder, so nothing balances the side force of the sails.
    MissingAppendages,
    /// No rig, so there is nothing to drive the boat.
    MissingRig,
    /// The mass properties are not physical.
    Mass(MassError),
    /// The file itself is invalid.
    Spec(SpecError),
    /// The hull admits no stable fluid-memory model.
    Radiation(MemoryError),
    /// The lofted hull produced no triangles.
    DegenerateHull,
    /// The hull form is outside the range the DSYHS regressions were fitted
    /// to, so sailing it would run the resistance polynomials extrapolated —
    /// which they do silently and catastrophically. The file itself is fine
    /// and loads; it is *sailing* it on this resistance model that is refused.
    OutsideDsyhsEnvelope(crate::dsyhs::EnvelopeError),
}

impl fmt::Display for AssemblyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingHullOffsets => write!(
                f,
                "this boat has no hull offsets, so buoyancy and stability cannot be \
                 computed and it cannot be sailed; resistance-only commands still work"
            ),
            Self::MissingParameters => write!(
                f,
                "this boat declares no hull form parameters, which the resistance \
                 regressions need"
            ),
            Self::MissingAppendages => write!(
                f,
                "this boat has no keel or rudder, so nothing balances the sails"
            ),
            Self::MissingRig => write!(f, "this boat has no rig, so nothing drives it"),
            Self::Mass(error) => write!(f, "mass properties: {error}"),
            Self::Spec(error) => write!(f, "boat file: {error}"),
            Self::Radiation(error) => write!(
                f,
                "this hull admits no stable fluid-memory model ({error:?}); the sections \
                 may be outside what strip theory can represent"
            ),
            Self::DegenerateHull => {
                write!(f, "the hull offsets lofted to an empty mesh")
            }
            Self::OutsideDsyhsEnvelope(error) => write!(
                f,
                "{error}; the resistance regressions would be extrapolated, which they \
                 do silently and wrongly, so this hull cannot be sailed on them"
            ),
        }
    }
}

impl std::error::Error for AssemblyError {}

impl From<SpecError> for AssemblyError {
    fn from(error: SpecError) -> Self {
        Self::Spec(error)
    }
}

impl From<MassError> for AssemblyError {
    fn from(error: MassError) -> Self {
        Self::Mass(error)
    }
}

/// Assembles a boat, its environment and its four force modules into a
/// simulation restrained in trim and yaw.
///
/// The initial state is upright at the origin and at rest; a caller that wants
/// the boat already moving sets the state afterwards, which also re-anchors the
/// restraint.
///
/// # Errors
///
/// [`AssemblyError`] naming the block of the boat file that is missing.
pub fn velocity_prediction_sim(
    spec: &BoatSpec,
    env: Box<dyn Environment>,
    controls: Controls,
    loft: &LoftOptions,
) -> Result<Sim, AssemblyError> {
    let hull = spec
        .hull
        .as_ref()
        .ok_or(AssemblyError::MissingHullOffsets)?;
    let mesh = loft_hull(hull, loft);
    if mesh.triangle_count() == 0 {
        return Err(AssemblyError::DegenerateHull);
    }

    let body = RigidBody::new(spec.mass_properties()?)?;
    let modules = sailing_modules(spec, mesh)?;

    // Pitch is restrained even when the layout frees yaw. No resistance
    // component has a published line of action, so the steady trim moment of a
    // boat under way is missing rather than small, and a calm-water device has
    // nothing to gain from integrating a mode whose forcing it has not got.
    // [`sailing_sim`] frees it, for a reason that only exists in a seaway.
    let captive = if spec.layout.is_some() {
        Captive {
            pitch: true,
            ..Captive::free()
        }
    } else {
        Captive::velocity_prediction()
    };
    Ok(Sim::new(body, BodyState::default(), env, modules, controls).with_captive(captive))
}

/// The four modules a boat sails on: flotation, hull, foils, rig.
///
/// Shared so that the calm-water device and the seaway one cannot drift apart in
/// what a boat is made of. The water's memory is deliberately **not** here — it
/// is precisely what [`sailing_sim`] adds to [`velocity_prediction_sim`], and a
/// helper that hid the difference would defeat the point of having both.
///
/// Takes the mesh rather than lofting one, because the seaway assembly has
/// already cut sections from it and a second loft could disagree with the first
/// about the hull the sections came from.
///
/// # Errors
///
/// [`AssemblyError`] naming the block of the boat file that is missing, or
/// [`AssemblyError::OutsideDsyhsEnvelope`] for a hull the regressions cannot
/// be trusted on.
fn sailing_modules(
    spec: &BoatSpec,
    mesh: crate::geometry::TriMesh,
) -> Result<Vec<Box<dyn ForceModule>>, AssemblyError> {
    let parameters = spec
        .hull_parameters()
        .ok_or(AssemblyError::MissingParameters)?;
    // The gate the design promised and the hull module says belongs here:
    // a form outside the series is refused before a single polynomial runs
    // on it. `vela-cli hull` reports the same check as a diagnostic without
    // refusing, which is the right split — a file is allowed to describe a
    // hull the regressions cannot sail.
    parameters
        .check_envelope()
        .map_err(AssemblyError::OutsideDsyhsEnvelope)?;
    let appendages = spec.appendages.ok_or(AssemblyError::MissingAppendages)?;
    let rig = spec.rig.ok_or(AssemblyError::MissingRig)?;

    let keel = keel_of(&appendages);
    let rudder = planform_of(&appendages.rudder);
    let scalars = HullScalars {
        waterline_length: parameters.waterline_length,
        waterline_beam: parameters.waterline_beam,
        canoe_draft: parameters.canoe_draft,
        total_draft: parameters.canoe_draft + appendages.keel.span,
        canoe_volume: parameters.canoe_volume,
    };

    // The layout is what makes yaw solvable. Without it every lateral force acts
    // on the centreline at `x = 0`, so no combination of them yaws the boat, and
    // restraining yaw is the honest thing to do rather than integrating a moment
    // that is identically zero. With it, the sails and the two foils get their
    // arms and the rudder becomes a control that means something.
    let layout = spec.layout;
    let keel_at = layout.map_or(0.0, |it| it.keel_at);
    let rudder_at = layout.map_or(0.0, |it| it.rudder_at);
    let sails_at = layout.map_or(0.0, |it| {
        let plan = sail_plan_of(&rig);
        it.mast_at + plan.centre_of_effort_from_mast().0
    });

    let modules: Vec<Box<dyn ForceModule>> = vec![
        Box::new(Buoyancy::new(mesh)),
        Box::new(CanoeBody::new(parameters)),
        Box::new(LateralSystem::new(
            scalars,
            keel,
            lateral_centre(keel_at, parameters.canoe_draft, &keel.planform),
            rudder,
            lateral_centre(rudder_at, parameters.canoe_draft, &rudder),
        )),
        Box::new(Sails::new(rig_dimensions(&rig)).at(sails_at).with_geometry(
            match spec.aerodynamics {
                // The default, and the oracle: a boat that did not ask for
                // the geometric model gets none of it, not even a partial one.
                Aerodynamics::Tabular => Vec::new(),
                Aerodynamics::Geometric => {
                    sail_members(&rig, spec.sail_shapes.as_deref().unwrap_or_default())
                }
            },
        )),
    ];
    Ok(modules)
}

/// The sail plan geometry of a rig, for [`crate::balance`].
fn sail_plan_of(rig: &RigSpec) -> BalanceSailPlan {
    BalanceSailPlan {
        foretriangle_height: rig.foretriangle_height,
        foretriangle_base: rig.foretriangle_base,
        main_hoist: rig.main_hoist,
        main_foot: rig.main_foot,
        boom_above_sheer: rig.boom_above_sheer,
    }
}

/// Each sail's flying shape at a control setting, with the tack it hangs from.
///
/// # Why this is public
///
/// A sail is not flat. It takes a camber and a twist from the sheet, the traveller,
/// the outhaul, the cunningham and the leech, and [`crate::flying::Shape`] is the
/// engine's statement of what shape a given setting produces. This exposes that
/// statement so a renderer can *draw* it, rather than deriving a second version
/// of the same geometry and drifting from this one — the same reason
/// [`crate::seaway::Seaway::waves`] is public.
///
/// It is worth being precise about what a caller is getting, because the boat this
/// engine ships with sails on the **tabular** aerodynamic model, and that model
/// consumes only area, aspect ratio and angle: it never asks for a shape. That
/// does not make the shape fictional. The flying shape is what the sail does; the
/// tabular model is a lossy consumer of it, chosen because its coefficients are
/// measured. Drawing the shape and integrating the table is drawing the truth and
/// approximating the forces, which is the honest way round — the dishonest one
/// would be to draw a flat triangle because the approximation cannot see the
/// camber.
///
/// Returns the sails the geometric model would carry, in file order: the main and
/// the jib, and only those the boat file describes well enough to build. Empty for
/// a boat with no rig or no flying-shape block, which is a boat whose sails cannot
/// be drawn any better than as triangles.
///
/// Tacks are in the **file** frame — `x` forward, `z` up from the sheer — which is
/// the frame the rig's own dimensions are quoted in, not the body frame.
#[must_use]
pub fn sail_shapes(spec: &BoatSpec, controls: Controls) -> Vec<(Sail, Point, Shape)> {
    let Some(rig) = spec.rig else {
        return Vec::new();
    };
    sail_members(&rig, spec.sail_shapes.as_deref().unwrap_or_default())
        .into_iter()
        .map(|(sail, member)| {
            let shape = member.response.shape(member.planform, controls.shape);
            (sail, member.tack, shape)
        })
        .collect()
}

/// Turns a boat file's flying-shape block into members the geometric model can
/// carry, keyed by which sail each describes.
///
/// The luff and foot are **derived from the rig**, never declared: `P` and `E` for
/// the mainsail, and for a jib the forestay length `√(I² + J²)` with `LPG` as the
/// foot. That is the point of splitting the data this way — a boat file can declare
/// the shape a measurement rule cannot imply, and cannot declare a sail whose size
/// disagrees with the rig the tabular model is sailing.
///
/// Everything else is the file's. A sail whose declared travel runs the wrong way —
/// an outhaul that adds camber, leech tension that adds twist — is dropped here
/// rather than sailed, because [`crate::flying::Response`] refuses it and this
/// function has no business inventing a replacement.
///
/// Sails other than the main and the jib are skipped: a spinnaker's luff and foot
/// follow from no IOR letter, and downwind is where the literature has coefficients
/// rather than theory.
fn sail_members(rig: &RigSpec, shapes: &[SailShapeSpec]) -> Vec<(Sail, Member)> {
    let mut members = Vec::new();
    for shape in shapes {
        let (luff, foot) = match shape.sail {
            Sail::Main => (rig.main_hoist, rig.main_foot),
            Sail::Jib => (
                rig.foretriangle_height.hypot(rig.foretriangle_base),
                rig.jib_perpendicular,
            ),
            _ => continue,
        };
        let Some(planform) = Planform::new(luff, foot, shape.head_chord, shape.roach) else {
            continue;
        };
        let Some(response) = Response::new(
            (shape.angle.0.to_radians(), shape.angle.1.to_radians()),
            (shape.twist.0.to_radians(), shape.twist.1.to_radians()),
            shape.camber,
            shape.head_camber,
            shape.draft,
        ) else {
            continue;
        };
        members.push((
            shape.sail,
            Member {
                planform,
                response,
                // File frame: `x` forward, `z` up from the sheer. The tack's height
                // is the boom for a main and the deck for a jib, and both are
                // measured from the same datum the windage model uses.
                tack: Point::new(shape.tack_at, 0.0, shape.tack_above_water),
                options: SailOptions {
                    stall: shape.stall.to_radians(),
                    ..SailOptions::default()
                },
            },
        ));
    }
    members
}

/// The body-frame point a foil's side force is applied at.
///
/// Depth is the canoe body draft plus the planform area centroid below the
/// root, which is where a linear lift distribution over a trapezoid resolves
/// to. That is a *lift* centroid and is the right quantity here — unlike
/// `Z_CBk`, which is a displaced-volume centroid and belongs to the keel's
/// residuary resistance; the appendage model documents the distinction and why
/// the two differ sharply for a bulb keel.
///
/// `quarter_chord_at_waterline` is the boat file's number, taken where the
/// extended foil meets the waterline. The force does not act there: it acts at
/// the lift centroid, which is deeper, and a swept foil's quarter chord moves
/// **aft** with depth. Carrying the sweep down to the acting depth is not a
/// refinement — for this keel it is 0.14 m, which is more than half the whole
/// lead the rig is placed by.
fn lateral_centre(
    quarter_chord_at_waterline: f64,
    canoe_draft: f64,
    foil: &FoilPlanform,
) -> Vector3<f64> {
    let depth = canoe_draft + foil.planform_centroid_below_root();
    Vector3::new(
        quarter_chord_at_waterline - depth * foil.sweep.tan(),
        0.0,
        depth,
    )
}

fn planform_of(foil: &FoilSpec) -> FoilPlanform {
    FoilPlanform {
        root_chord: foil.root_chord,
        tip_chord: foil.tip_chord,
        span: foil.span,
        sweep: foil.sweep_deg.to_radians(),
    }
}

fn keel_of(appendages: &AppendagesSpec) -> Keel {
    Keel {
        planform: planform_of(&appendages.keel),
        volume: appendages.keel_volume,
        centre_of_buoyancy_below_hull_bottom: appendages.keel_cb_height,
    }
}

/// Maps the file's rig block onto the aerodynamic model's own notation.
///
/// A transcription rather than a conversion: both sides are in the published
/// IOR letters, and the only reason the two structs exist is that the force
/// model must not depend on the file format. The mizzen is `None` because the
/// format describes no mizzen — a ketch needs a schema addition, not a default.
fn rig_dimensions(rig: &RigSpec) -> RigDimensions {
    RigDimensions {
        main_hoist: rig.main_hoist,
        main_foot: rig.main_foot,
        foretriangle_height: rig.foretriangle_height,
        foretriangle_base: rig.foretriangle_base,
        jib_perpendicular: rig.jib_perpendicular,
        spinnaker_leech: rig.spinnaker_leech,
        boom_above_sheer: rig.boom_above_sheer,
        max_beam: rig.max_beam,
        average_freeboard: rig.average_freeboard,
        mast_height_above_sheer: rig.mast_above_sheer,
        mast_diameter: rig.mast_diameter,
        mizzen: None,
    }
}

/// Everything the radiation path needs out of a boat file, computed once.
///
/// Extracted because the frequency sweep is the load-time cost and two
/// assemblies want it: the vertical rig below and the seakeeping one after it.
/// Sharing the struct is also what keeps the two from disagreeing about which
/// waterline the sections were cut at, which is the one number the lateral
/// coefficients are sensitive to and the vertical ones are not.
struct Seakeeping {
    mesh: crate::geometry::TriMesh,
    strips: Vec<Strip>,
    solver: SectionSolver,
    grid: Vec<f64>,
    /// Height of the design waterline above the baseline, m.
    ///
    /// The body origin sits at the baseline and the sectional coefficients are
    /// computed about the waterline, so this is the lever arm between them —
    /// the source's `OG`. See [`crate::strip::lateral_coefficients`].
    waterline_height: f64,
}

fn seakeeping(
    spec: &BoatSpec,
    loft: &LoftOptions,
    options: RadiationOptions,
) -> Result<Seakeeping, AssemblyError> {
    let hull = spec
        .hull
        .as_ref()
        .ok_or(AssemblyError::MissingHullOffsets)?;
    let parameters = spec
        .hull_parameters()
        .ok_or(AssemblyError::MissingParameters)?;

    let mesh = loft_hull(hull, loft);
    if mesh.triangle_count() == 0 {
        return Err(AssemblyError::DegenerateHull);
    }

    // Sections at the design waterline. Dry stations are kept: they carry the
    // length over which the coefficients taper to nothing.
    let strips: Vec<Strip> = hull
        .stations
        .iter()
        .map(|station| Strip {
            x: station.x,
            form: station_geometry(station, parameters.canoe_draft)
                .map(|section| LewisForm::fit(&section)),
        })
        .collect();

    Ok(Seakeeping {
        mesh,
        strips,
        solver: SectionSolver::new(options.tasai),
        grid: frequency_grid(
            options.lowest_frequency,
            options.top_frequency,
            options.samples,
        ),
        waterline_height: parameters.canoe_draft,
    })
}

/// How many stations the Lewis fit had to clamp: the number `lewis` says
/// nobody should have to rediscover, counted here so the radiation modules
/// can publish it.
fn clamped_stations(strips: &[Strip]) -> usize {
    strips
        .iter()
        .filter(|strip| strip.form.is_some_and(|form| form.clamped))
        .count()
}

/// Fits the vertical pair, and returns it with the mass it adds.
fn vertical_radiation(
    setup: &Seakeeping,
    options: RadiationOptions,
) -> Result<(Radiation, Matrix6<f64>), AssemblyError> {
    let ((heave, coupling, pitch), sweep) = vertical_spectra(
        &setup.strips,
        &setup.grid,
        options.density,
        STANDARD_GRAVITY,
        &setup.solver,
    )
    .ok_or(AssemblyError::Radiation(MemoryError::Singular))?;
    // The fit reads each spectrum up to `fit_ceiling` times its own damping
    // peak; the band the provenance is judged on is the widest of the
    // diagonal ones, which is the one the fit actually consumed.
    let ceiling = options.memory.fit_ceiling * heave.peak_frequency().max(pitch.peak_frequency());
    let spectra =
        RadiationSpectra::vertical(heave, coupling, pitch).map_err(AssemblyError::Radiation)?;
    let infinite = spectra.infinite(options.transform);
    let fitted = Radiation::fit(&spectra, &infinite, options.memory)
        .map_err(AssemblyError::Radiation)?
        .with_provenance(Provenance::within(
            clamped_stations(&setup.strips),
            &sweep,
            ceiling,
        ));
    Ok((fitted, infinite.matrix()))
}

/// Fits the lateral triple, and returns it with the mass it adds.
fn lateral_radiation(
    setup: &Seakeeping,
    options: RadiationOptions,
) -> Result<(Radiation, Matrix6<f64>), AssemblyError> {
    let (spectra, sweep) = lateral_spectra(
        &setup.strips,
        setup.waterline_height,
        &setup.grid,
        options.density,
        STANDARD_GRAVITY,
        &setup.solver,
    )
    .ok_or(AssemblyError::Radiation(MemoryError::Singular))?;
    // Sway, roll and yaw are entries 0, 3 and 5 of the triangle — see
    // `lateral_spectra` for the order.
    let ceiling = options.memory.fit_ceiling
        * [0, 3, 5]
            .iter()
            .map(|&index| spectra[index].peak_frequency())
            .fold(0.0, f64::max);
    let spectra = RadiationSpectra::lateral(spectra).map_err(AssemblyError::Radiation)?;
    let infinite = spectra.infinite(options.transform);
    let fitted = Radiation::fit(&spectra, &infinite, options.memory)
        .map_err(AssemblyError::Radiation)?
        .with_provenance(Provenance::within(
            clamped_stations(&setup.strips),
            &sweep,
            ceiling,
        ));
    Ok((fitted, infinite.matrix()))
}

/// A simulation of the vertical modes alone, with the water's memory in it.
///
/// Heave and pitch free, everything else restrained, no wind and no sails: the
/// rig for watching a hull settle after it has been disturbed. Narrow on purpose,
/// and it stays narrow now that the lateral modes exist — restraining four
/// degrees of freedom is what makes a heave decrement mean only what it says.
/// For the boat that moves in all five, see [`seakeeping_sim`].
///
/// The frequency sweep happens here, which is the load-time cost: about two
/// hundred milliseconds for a hull of seventeen stations over a grid that
/// reaches far enough out for the transform. Everything after that is a
/// five-state matrix-vector product per mode per step.
///
/// `A_∞` goes into the mass matrix and the memory becomes a force module, which
/// is Cummins' split. See [`crate::modules::radiation`].
///
/// # Errors
///
/// [`AssemblyError`] naming the block of the boat file that is missing, or
/// [`AssemblyError::Radiation`] if the hull admits no stable memory model.
pub fn vertical_motion_sim(
    spec: &BoatSpec,
    env: Box<dyn Environment>,
    loft: &LoftOptions,
    options: RadiationOptions,
) -> Result<Sim, AssemblyError> {
    let setup = seakeeping(spec, loft, options)?;
    let (radiation, added) = vertical_radiation(&setup, options)?;

    let mut body = RigidBody::new(spec.mass_properties()?)?;
    body.add_added_mass(added)?;

    let modules: Vec<Box<dyn ForceModule>> =
        vec![Box::new(Buoyancy::new(setup.mesh)), Box::new(radiation)];

    Ok(Sim::new(
        body,
        BodyState::default(),
        env,
        modules,
        Controls::close_hauled(SailSet::upwind()),
    )
    .with_captive(Captive {
        surge: true,
        sway: true,
        heave: false,
        roll: true,
        pitch: false,
        yaw: true,
    }))
}

/// A simulation of every mode the water's memory is known for: five of six.
///
/// Sway, heave, roll, pitch and yaw free, with two radiation sets mounted — the
/// vertical pair and the lateral triple. Surge stays restrained, and that is the
/// honest boundary of this model rather than an omission: the source computes
/// two-dimensional surge coefficients by defining an equivalent longitudinal
/// section that sways, an empirical device this engine does not implement. A hull
/// free to surge would be carrying a zero where a force belongs.
///
/// Two [`Radiation`] modules rather than one six-mode module, because the sets do
/// not couple: for a hull symmetric about its centreline the vertical and lateral
/// problems are separate, which is the same symmetry strip theory already assumes
/// when it solves one half-section. Keeping them apart means each fits and reports
/// its own memory, and a bad fit says which half it came from.
///
/// No wind and no sails, like the vertical rig: this is an instrument for watching
/// a hull move in water, not a boat that goes anywhere.
///
/// # Errors
///
/// As [`vertical_motion_sim`], and additionally
/// [`AssemblyError::Radiation`] if the lateral coefficients admit no stable
/// memory model where the vertical ones did.
pub fn seakeeping_sim(
    spec: &BoatSpec,
    env: Box<dyn Environment>,
    loft: &LoftOptions,
    options: RadiationOptions,
) -> Result<Sim, AssemblyError> {
    let setup = seakeeping(spec, loft, options)?;
    let (vertical, vertical_mass) = vertical_radiation(&setup, options)?;
    let (lateral, lateral_mass) = lateral_radiation(&setup, options)?;

    let mut body = RigidBody::new(spec.mass_properties()?)?;
    // Disjoint blocks, so one call with the sum is the same as two calls, and
    // the sum is what a single Cholesky has to stay positive definite through.
    body.add_added_mass(vertical_mass + lateral_mass)?;

    let modules: Vec<Box<dyn ForceModule>> = vec![
        Box::new(Buoyancy::new(setup.mesh)),
        Box::new(vertical),
        Box::new(lateral),
    ];

    Ok(Sim::new(
        body,
        BodyState::default(),
        env,
        modules,
        Controls::close_hauled(SailSet::upwind()),
    )
    .with_captive(Captive {
        surge: true,
        sway: false,
        heave: false,
        roll: false,
        pitch: false,
        yaw: false,
    }))
}

/// A boat that sails, in water that remembers.
///
/// [`velocity_prediction_sim`] with the water's memory mounted and the vertical
/// modes released: the four sailing modules, both radiation sets, and a hull free
/// to heave, roll and pitch under a passing wave. This is the only assembly in
/// which the two halves of this engine meet — the rig drives the boat and the sea
/// moves it, through the same pressure integral.
///
/// # What the waves do, and what they do not
///
/// The **encounter frequency is not modelled — it happens.** [`Buoyancy`]
/// integrates pressure over whichever triangles are under the instantaneous
/// surface at the boat's own world position, so a boat driving into a head sea
/// meets crests faster than one running with them, and nothing in the code knows
/// the boat's speed relative to the waves. There is no Doppler term to get wrong
/// because there is no Doppler term.
///
/// What is *not* modelled is the forward-speed dependence of the radiation
/// coefficients themselves. Strip theory here solves stationary sections, so the
/// added mass and damping are the zero-speed ones and the classical
/// speed-correction terms are absent. The damping is therefore understated at
/// speed — the direction that makes motions larger rather than smaller, which is
/// at least the direction that shows.
///
/// # Which modes are free, and why they differ from both neighbours
///
/// **Surge is free here and restrained in [`seakeeping_sim`]**, because
/// [`CanoeBody`] supplies a surge force where the seakeeping rig has none:
/// resistance *is* the surge damping, and a far better model of it than a
/// radiation coefficient would be. Surge added mass is still absent, so the boat
/// accelerates fore-and-aft a few per cent too readily in a transient, and
/// steady sailing is untouched.
///
/// **Pitch is free here and restrained in [`velocity_prediction_sim`]**, which is
/// the more interesting reversal. The reason pitch is held in the calm-water
/// device — no resistance component has a published line of action, so the steady
/// trim moment is missing — has not gone away. But in a seaway the excitation and
/// the restoring in pitch are the dominant terms, both come out of the pressure
/// integral, and holding the mode would delete a boat's principal motion to avoid
/// a smaller error in its mean attitude. So the mean running trim of this
/// assembly is wrong by the missing moment, and the motion about it is not.
///
/// # The cross-check this unlocks
///
/// [`crate::equilibrium`] exists because a boat with no hydrodynamic damping in
/// heave or roll is an undamped oscillator that never settles, so steady sailing
/// had to be *solved* rather than reached. With radiation mounted it can be
/// reached: run this assembly in still water and it converges — and it must
/// converge to what the Newton solver says. Two independent routes to one answer,
/// which is the cross-check [`crate::sim`] promises in its own header.
///
/// # Errors
///
/// As [`velocity_prediction_sim`] and [`seakeeping_sim`] together: a missing
/// block of the boat file, or [`AssemblyError::Radiation`] if the hull admits no
/// stable memory model.
pub fn sailing_sim(
    spec: &BoatSpec,
    env: Box<dyn Environment>,
    controls: Controls,
    loft: &LoftOptions,
    options: RadiationOptions,
) -> Result<Sim, AssemblyError> {
    let setup = seakeeping(spec, loft, options)?;
    let (vertical, vertical_mass) = vertical_radiation(&setup, options)?;
    let (lateral, lateral_mass) = lateral_radiation(&setup, options)?;

    let mut body = RigidBody::new(spec.mass_properties()?)?;
    // Disjoint blocks, so one call with the sum is the same as two calls, and the
    // sum is what a single Cholesky has to stay positive definite through.
    body.add_added_mass(vertical_mass + lateral_mass)?;

    let mut modules = sailing_modules(spec, setup.mesh)?;
    modules.push(Box::new(vertical));
    modules.push(Box::new(lateral));

    // Every mode the forces can carry: pitch for the reason above, surge on the
    // hull's resistance rather than on a radiation coefficient it has not got.
    let captive = if spec.layout.is_some() {
        Captive::free()
    } else {
        Captive {
            yaw: true,
            ..Captive::free()
        }
    };
    Ok(Sim::new(body, BodyState::default(), env, modules, controls).with_captive(captive))
}

/// The frequencies the radiation problem is solved at: a uniform grid with a
/// geometric tail hung below it.
///
/// The uniform part is `top * i / samples` for `i` in `1..=samples`, which is
/// what this engine has always used, and it is uniform for a reason. The
/// retardation function comes out of a cosine transform,
/// `K(t) = (2/π)∫B(ω)cos(ωt)dω`, and the accuracy of that integral is set at
/// the *high* end of the grid, where `cos(ωt)` oscillates fastest across one
/// interval. A log-spaced grid — the obvious reflex when a low-frequency reach
/// is wanted — is coarsest exactly there, so it buys the long waves by
/// spending the transform, which is the wrong trade.
///
/// The trouble with the uniform grid alone is its first step,
/// `Δ = top / samples`: 0.25 rad/s at the defaults, a 25 s period. Below that
/// the memory model had nothing to fit and extrapolated, and a 25 s swell is
/// not an exotic sea. So below `Δ` the grid falls off geometrically at a ratio
/// of 1.5 until it passes `lowest`. That is cheap: from a 0.25 rad/s step down
/// to 0.02 rad/s takes six extra samples — a seventh would already be under
/// the floor — for a decade and a half of extra reach, against the ~110 extra
/// uniform samples the same reach would cost at `Δ = 0.02`.
///
/// If `lowest` is at or above `Δ` the tail is empty and the grid is exactly the
/// old uniform one, so the previous behaviour stays reachable. The result is
/// strictly increasing and strictly positive either way, which is what
/// [`crate::cummins::Spectrum::new`] insists on; its quadrature is a trapezoid
/// over the actual spacing, so a non-uniform grid needs nothing from it.
fn frequency_grid(lowest: f64, top: f64, samples: usize) -> Vec<f64> {
    /// How fast the tail falls away below the first uniform step.
    const TAIL_RATIO: f64 = 1.5;

    if samples == 0 {
        return Vec::new();
    }
    let step = top / samples as f64;

    // Built downwards from just under `step` — never at it, or the grid would
    // repeat a value — then reversed, because the grid has to increase.
    let mut tail = Vec::new();
    let mut frequency = step / TAIL_RATIO;
    while frequency > lowest {
        tail.push(frequency);
        frequency /= TAIL_RATIO;
    }

    let mut grid = Vec::with_capacity(tail.len() + samples);
    grid.extend(tail.into_iter().rev());
    grid.extend((1..=samples).map(|i| top * i as f64 / samples as f64));
    grid
}

/// How the radiation pipeline is run at assembly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadiationOptions {
    /// Water density, kg/m³.
    pub density: f64,
    /// Highest frequency sampled, rad/s.
    ///
    /// Has to be past where the damping has died, and for a yacht that is
    /// further than ship experience suggests: a narrow section reaches a given
    /// reduced frequency only at a high `ω`.
    pub top_frequency: f64,
    /// Lowest frequency sampled, rad/s.
    ///
    /// A 314 s period at the default 0.02, which is far below anything a yacht
    /// meets, and that is the point: the memory model is then fitted rather
    /// than extrapolated across the whole useful range. The uniform grid on its
    /// own starts at `top_frequency / samples`, or 30/120 = 0.25 rad/s at the
    /// defaults — a 25 s period, with the added mass and damping of every
    /// longer wave extrapolated. See [`frequency_grid`] for what reaching
    /// further down costs.
    pub lowest_frequency: f64,
    /// Number of frequencies sampled.
    pub samples: usize,
    /// How the sectional radiation problem is solved.
    pub tasai: TasaiOptions,
    /// How the time integral of Ogilvie's relation is taken.
    pub transform: TransformOptions,
    /// How the memory is fitted.
    pub memory: MemoryOptions,
}

impl Default for RadiationOptions {
    fn default() -> Self {
        Self {
            density: SEA_WATER_DENSITY,
            top_frequency: 30.0,
            lowest_frequency: 0.02,
            samples: 120,
            tasai: TasaiOptions::default(),
            transform: TransformOptions::default(),
            memory: MemoryOptions {
                order: 5,
                ..MemoryOptions::default()
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A grid that a spectrum would reject is worse than a coarse one, so this
    /// is checked over a spread of settings rather than at the defaults alone —
    /// including a `lowest` above the uniform step, where the tail vanishes.
    #[test]
    fn every_grid_is_positive_and_strictly_increasing() {
        for (lowest, top, samples) in [
            (0.02, 30.0, 120),
            (0.02, 30.0, 60),
            (0.5, 30.0, 60),
            (1.0, 4.0, 8),
            (0.001, 12.0, 200),
            (0.02, 30.0, 1),
        ] {
            let grid = frequency_grid(lowest, top, samples);
            assert!(
                grid.len() >= samples,
                "the uniform samples must all be there: {} of {samples}",
                grid.len()
            );
            for window in grid.windows(2) {
                assert!(
                    window[0] > 0.0 && window[1] > window[0],
                    "grid for ({lowest}, {top}, {samples}) is not increasing at {window:?}"
                );
            }
        }
    }

    /// The tail is only worth having if it arrives: one more step of the ratio
    /// would take the grid under the floor, so the first frequency sits within
    /// a factor of 1.5 of it.
    #[test]
    fn the_tail_reaches_the_lowest_frequency_asked_for() {
        for (lowest, top, samples) in [(0.02, 30.0, 120), (0.02, 30.0, 60), (0.005, 20.0, 40)] {
            let first = frequency_grid(lowest, top, samples)[0];
            assert!(
                first <= lowest * 1.5,
                "({lowest}, {top}, {samples}) reached only {first}"
            );
        }
    }

    /// The old behaviour has to stay reachable, and a `lowest` at or above the
    /// uniform step is how it is reached.
    #[test]
    fn a_floor_above_the_uniform_step_leaves_the_uniform_grid_alone() {
        let samples = 60;
        let top = 30.0;
        let uniform: Vec<f64> = (1..=samples)
            .map(|i| top * i as f64 / samples as f64)
            .collect();
        for lowest in [top / samples as f64, 1.0, 40.0] {
            assert_eq!(frequency_grid(lowest, top, samples), uniform);
        }
    }

    /// The tail is an addition, not a redistribution: the cosine transform
    /// still gets every uniform frequency it used to, `top` included.
    #[test]
    fn the_uniform_samples_survive_the_tail() {
        let (top, samples) = (30.0, 120);
        let grid = frequency_grid(0.02, top, samples);
        for i in [1, 2, 7, 60, 119, samples] {
            let expected = top * i as f64 / samples as f64;
            assert!(
                grid.contains(&expected),
                "the uniform sample at i={i} ({expected} rad/s) is missing"
            );
        }
    }

    /// Geometric spacing is what makes the low-frequency reach affordable; if
    /// the tail ever grew to the size of the uniform part, the sweep cost —
    /// seventeen stations by every frequency — would have doubled for it.
    #[test]
    fn the_tail_costs_only_a_handful_of_samples() {
        let options = RadiationOptions::default();
        let grid = frequency_grid(
            options.lowest_frequency,
            options.top_frequency,
            options.samples,
        );
        let extra = grid.len() - options.samples;
        assert!(extra < 15, "the tail added {extra} samples");
    }
}
