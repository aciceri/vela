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

use crate::aero::RigDimensions;
use crate::aero::SailSet;
use crate::appendages::{FoilPlanform, HullScalars, Keel};
use crate::balance::SailPlan as BalanceSailPlan;
use crate::boat::{AppendagesSpec, BoatSpec, FoilSpec, RigSpec, SpecError};
use crate::controls::Controls;
use crate::cummins::{MemoryError, MemoryOptions, TransformOptions};
use crate::env::Environment;
use crate::env::{StillWater, UniformWind};
use crate::lewis::{station_geometry, LewisForm};
use crate::loft::{loft_hull, LoftOptions};
use crate::mass::MassError;
use crate::modules::radiation::{Radiation, VerticalInfinite, VerticalSpectra};
use crate::modules::{Buoyancy, CanoeBody, LateralSystem, Sails};
use crate::rigid_body::RigidBody;
use crate::sim::{Captive, ForceModule, Sim};
use crate::state::BodyState;
use crate::strip::{vertical_spectra, Strip};
use crate::tasai::{SectionSolver, TasaiOptions};
use crate::{SEA_WATER_DENSITY, STANDARD_GRAVITY};
use nalgebra::Vector3;
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
    let parameters = spec
        .hull_parameters()
        .ok_or(AssemblyError::MissingParameters)?;
    let appendages = spec.appendages.ok_or(AssemblyError::MissingAppendages)?;
    let rig = spec.rig.ok_or(AssemblyError::MissingRig)?;
    let hull = spec
        .hull
        .as_ref()
        .ok_or(AssemblyError::MissingHullOffsets)?;

    let mesh = loft_hull(hull, loft);
    if mesh.triangle_count() == 0 {
        return Err(AssemblyError::DegenerateHull);
    }

    let body = RigidBody::new(spec.mass_properties()?)?;

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
        Box::new(Sails::new(rig_dimensions(&rig)).at(sails_at)),
    ];

    let captive = if layout.is_some() {
        Captive {
            pitch: true,
            ..Captive::free()
        }
    } else {
        Captive::velocity_prediction()
    };
    Ok(Sim::new(body, BodyState::default(), env, modules, controls).with_captive(captive))
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

/// A simulation of the vertical modes alone, with the water's memory in it.
///
/// Heave and pitch free, everything else restrained, no wind and no sails: the
/// rig for watching a hull settle after it has been disturbed. It is a
/// deliberately narrow assembly rather than a general one, because the vertical
/// modes are the only ones whose radiation this engine can compute — the sway
/// and roll sections are not written — and a simulation that pretended otherwise
/// would be quietly missing terms in four degrees of freedom.
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
    loft: &LoftOptions,
    options: RadiationOptions,
) -> Result<Sim, AssemblyError> {
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

    let solver = SectionSolver::new(options.tasai);
    let grid: Vec<f64> = (1..=options.samples)
        .map(|i| options.top_frequency * i as f64 / options.samples as f64)
        .collect();
    let (heave, coupling, pitch) =
        vertical_spectra(&strips, &grid, options.density, STANDARD_GRAVITY, &solver)
            .ok_or(AssemblyError::Radiation(MemoryError::Singular))?;
    let spectra = VerticalSpectra {
        heave,
        coupling,
        pitch,
    };
    let infinite = VerticalInfinite {
        heave: spectra.heave.infinite_added_mass(options.transform),
        coupling: spectra.coupling.infinite_added_mass(options.transform),
        pitch: spectra.pitch.infinite_added_mass(options.transform),
    };
    let radiation =
        Radiation::fit(&spectra, &infinite, options.memory).map_err(AssemblyError::Radiation)?;

    let mut body = RigidBody::new(spec.mass_properties()?)?;
    body.add_added_mass(Radiation::added_mass_matrix(&infinite))?;

    let modules: Vec<Box<dyn ForceModule>> =
        vec![Box::new(Buoyancy::new(mesh)), Box::new(radiation)];
    let env: Box<dyn Environment> = Box::new(StillWater::new(UniformWind::uniform(0.0, 0.0)));

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
