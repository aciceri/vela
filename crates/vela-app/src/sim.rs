//! The engine, as something a frame loop can drive.
//!
//! # The boundary this keeps
//!
//! `vela_core` has no clock, no frame rate and no filesystem: it advances
//! exactly when `Sim::step(dt)` is called, with a `dt` the caller chooses. That
//! is what makes it embeddable and deterministic, and it is the frontend's job
//! to hold up the other end — the engine must never learn what a frame is.
//!
//! So the accumulator lives here. Bevy's `FixedUpdate` already does this job,
//! and doing it twice would be worse than doing it once: the schedule is
//! configured with the engine's own step and the system inside it steps the
//! engine exactly once per tick. Render time that does not divide evenly stays in
//! Bevy's overstep, which is where interpolation would read it from if this ever
//! grows one.
//!
//! # Why the step is 1/200 and not 1/60
//!
//! The stiff mode in this problem is heave, whose natural period on the
//! reference boat is around a second, and roll's is two. Sixty hertz would be
//! ample for those. What sets the step is the appendage lift: it is linear in
//! angle of attack with no limiting term, so a keel three metres down on a
//! rolling boat can produce a very large force very quickly, and a step short
//! enough to resolve that is a step short enough to keep the explicit part of the
//! integrator well behaved. Two hundred hertz is what the seaway runs in
//! `vela-cli` use and what the cross-check test settles at, so it is also the
//! step the engine's behaviour has actually been measured at.

use bevy::prelude::*;
use vela_core::aero::SailSet;
use vela_core::assembly::{sailing_sim, velocity_prediction_sim, RadiationOptions};
use vela_core::boat::RigSpec;
use vela_core::equilibrium::{self, EquilibriumOptions};
use vela_core::seaway::SeaState;
use vela_core::{BoatSpec, Controls, LoftOptions, Seaway2D, Sim, StillWater, TriMesh, UniformWind};

/// The physics step, seconds. See the module documentation.
pub const STEP: f64 = 1.0 / 200.0;

/// The boat file this frontend sails.
///
/// Compiled in rather than loaded, because the engine takes boats from `&str`
/// and the browser has no filesystem to read one from. A file picker is a
/// feature; a hard dependency on one would have been an architecture.
const SPEC: &str = include_str!("../../../boats/yd41-form-study.ron");

/// True wind the boat is released into: m/s, and degrees off the bow.
///
/// The working breeze the test suite sails and the published polar is checked
/// in. It was six metres a second at forty-five degrees, and in that the keel
/// is past its stall — lift coefficient over two against a foil's 1.2, the
/// gap §5.4 of the functional analysis names — so the balanced helm sat at
/// its limit and the boat luffed along at under a knot, which a viewer read
/// as a boat that did not move against the sea. A knot less of wind and the
/// boat sails: helm at eighteen degrees, five knots, and the crests come past.
const WIND_SPEED: f64 = 5.0;
const WIND_ANGLE: f64 = 40.0;

/// The sea the boat is released into.
///
/// Half a metre at six seconds: a slight sea for a forty-footer. It was a metre
/// — the condition every seakeeping number in `docs/functional-analysis.md` is
/// measured at — and in a metre of sea this boat does not stay sailing. §5.4's
/// missing foil stall is why: the appendage lift is linear in angle of attack,
/// the angle at the keel includes the roll rate times its arm, and as the waves
/// slow the boat that ratio grows without bound — a keel lift coefficient of
/// 3.4 was read off the HUD twenty seconds after release, where a real foil
/// stalls near 1.2, with the boat down to a knot and the induced drag it was
/// charged pinning it there. The functional analysis says it in as many words:
/// seaway *motions* are usable and the seaway *speed loss* is not. Half a metre
/// keeps the roll rate where the linear model holds and the boat at the five
/// knots it was released at, which is the picture the frontend exists to show.
///
/// It was also once a metre and a half, to make the water look like it had
/// weather in it, and `examples/motion_scale` is why it came down from there:
/// seventeen degrees of trim and nearly three metres of sinkage against a hull
/// 1.84 m deep, twice a real forty-footer's response — the amplitude error of
/// no diffraction and no viscous damping, which §5.4 also names. Choosing the
/// sea to flatter the model would be the wrong lever; choosing it to stay
/// inside what the model can do is the right one, and the number goes back up
/// when the stall model lands.
///
/// Until then, `VELA_WAVE_HEIGHT` in the environment overrides it, for looking
/// at the *water* in weather: at two metres the sea reads as a sea, and the
/// boat, for the reasons above, does not sail in it.
const WAVE_HEIGHT: f64 = 0.5;
const WAVE_PERIOD: f64 = 6.0;

/// The significant wave height the boat is released into: the constant, or
/// the override; see [`WAVE_HEIGHT`].
fn wave_height() -> f64 {
    std::env::var("VELA_WAVE_HEIGHT")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|height: &f64| height.is_finite() && *height >= 0.0)
        .unwrap_or(WAVE_HEIGHT)
}

/// The simulation, and the mesh it was built from.
///
/// The mesh is kept because the renderer needs the same hull the physics is
/// integrating pressure over — not a visual stand-in that could disagree with
/// it. `vela_core`'s boat format deliberately carries no visual mesh for exactly
/// this reason: there is one hull, and it is lofted from the offsets.
#[derive(Resource)]
pub struct Engine {
    pub sim: Sim,
    pub hull: TriMesh,
    /// The rig, for drawing spars and sails at their published dimensions.
    ///
    /// Held rather than re-parsed so that the frontend draws the rig the force
    /// model is using, and so that nothing here has to invent a mast height.
    pub rig: RigSpec,
    /// The parsed boat file.
    ///
    /// Kept so the frontend can ask the engine what shape the sails have taken —
    /// `vela_core::assembly::sail_shapes` needs the file's flying-shape block, and
    /// re-parsing it per frame to draw a sail would be absurd. Nothing here reads
    /// it for anything the engine could answer instead.
    pub spec: BoatSpec,
    /// Where the mast stands along the hull, m forward of the aft perpendicular.
    ///
    /// `None` for a boat whose file declares no layout — which is also the boat
    /// whose yaw the engine restrains, so the two absences are the same absence.
    pub mast_at: Option<f64>,
    /// The sea's realisation, for the renderer to resynthesise.
    ///
    /// `None` in a calm, which is a state worth being able to represent rather
    /// than faking with a zero-height sea: a flat surface and a sea of no waves
    /// are the same picture and not the same object.
    pub sea: Option<vela_core::seaway::Seaway>,
    /// Rudder angle that balances the boat at the condition it was released in,
    /// radians.
    ///
    /// The helm a hand would hold, not amidships. A sailing boat is not balanced
    /// with the rudder centred — the sails' side force acts forward of the
    /// lateral plane's centre and the hull carries a permanent yaw moment against
    /// it, so a straight course needs a standing angle of weather helm. The
    /// equilibrium solve computes it, and this is where the answer is kept: it
    /// is the trim the helmsman's corrections sit on. See `crate::helm`.
    pub balanced_helm: f64,
    /// The heading the helmsman is holding, radians, `None` while a hand is on
    /// the wheel. Set to the heading the keys are released on — and on the
    /// first frame, to the heading the boat was released on.
    pub course: Option<f64>,
    /// The helmsman's learned offset from the balanced helm, radians: the
    /// slow integral of the heading error, which absorbs the difference
    /// between the balance the solve found and the one the free boat has.
    /// Reset whenever a hand takes the wheel.
    pub helm_bias: f64,
}

impl Engine {
    /// Length of the hull's stations, m: the bow is this far forward of the
    /// body origin along the hull's axis.
    #[must_use]
    pub fn hull_length(&self) -> f64 {
        self.spec
            .hull
            .as_ref()
            .map_or(0.0, vela_core::boat::HullSpec::length)
    }

    /// Greatest half-breadth of the hull, m.
    #[must_use]
    pub fn hull_half_beam(&self) -> f64 {
        self.spec
            .hull
            .as_ref()
            .map_or(0.0, vela_core::boat::HullSpec::max_half_breadth)
    }

    /// Builds the boat, solves its steady sailing condition, and releases it
    /// there.
    ///
    /// Released at the solved condition rather than from rest for the same reason
    /// `vela-cli seaway` does it: the first seconds of a run started anywhere
    /// else are a boat falling into the water, which is neither interesting to
    /// watch nor anything to do with the sea.
    ///
    /// # Errors
    ///
    /// A string naming what the boat file lacks, or what the solver could not
    /// find. Returned rather than panicked because a frontend that fails to
    /// start should say why on the screen.
    pub fn launch() -> Result<Self, String> {
        let spec = BoatSpec::parse_ron(SPEC).map_err(|error| format!("boat file: {error}"))?;
        let parameters = spec
            .hull_parameters()
            .ok_or("the boat file has no parameters block")?;
        let hull_spec = spec.hull.as_ref().ok_or("the boat file has no hull")?;
        let rig = spec
            .rig
            .ok_or("the boat file has no rig, so it has no sails")?;
        let loft = LoftOptions::default();
        let hull = vela_core::loft_hull(hull_spec, &loft);

        let wind = UniformWind::uniform(WIND_SPEED, WIND_ANGLE.to_radians());
        let base = Controls::close_hauled(SailSet::upwind());
        let start = EquilibriumOptions {
            initial_sinkage: parameters.canoe_draft,
            ..EquilibriumOptions::default()
        };

        let mut vpp = velocity_prediction_sim(&spec, Box::new(StillWater::new(wind)), base, &loft)
            .map_err(|error| format!("assembly: {error}"))?;
        let helm = equilibrium::solve_with_helm(&mut vpp, parameters.waterline_length, &start)
            .map_err(|error| format!("the boat will not sail: {error}"))?;
        let solved = helm.equilibrium;
        info!(
            "released at TWS {WIND_SPEED} m/s, TWA {WIND_ANGLE} deg: speed {:.2} m/s, heel {:.1} deg, helm {:.1} deg",
            solved.speed,
            solved.heel.to_degrees(),
            helm.rudder_angle.to_degrees()
        );

        let sea = Seaway2D::new(
            wind,
            SeaState {
                significant_height: wave_height(),
                peak_period: WAVE_PERIOD,
                // Travelling towards the boat: waves come from where the wind does.
                heading: WIND_ANGLE.to_radians() + std::f64::consts::PI,
                // What a sea looks like: crests pinched, troughs flat. Costs
                // half a step again over the linear sea (`frame_cost`), and the
                // physics feels the asymmetry too, which is right.
                choppiness: 0.8,
                ..SeaState::default()
            },
        );
        let realisation = sea.sea().clone();

        let mut sim = sailing_sim(
            &spec,
            Box::new(sea),
            base.with_rudder(helm.rudder_angle),
            &loft,
            RadiationOptions::default(),
        )
        .map_err(|error| format!("assembly with radiation: {error}"))?;
        sim.set_state(solved.state);

        Ok(Self {
            sim,
            hull,
            rig,
            mast_at: spec.layout.map(|layout| layout.mast_at),
            spec,
            sea: Some(realisation),
            balanced_helm: helm.rudder_angle,
            course: None,
            helm_bias: 0.0,
        })
    }
}

/// Advances the engine by one fixed step.
///
/// One step per tick, unconditionally. Bevy's `FixedUpdate` runs this as many
/// times as the accumulated render time allows, which is the whole of the
/// frame-rate independence: a slow frame produces several physics steps rather
/// than one long one, and the engine never sees a variable `dt`.
pub fn step(mut engine: ResMut<Engine>) {
    engine.sim.step(STEP);
}

/// Registers the engine and its step.
pub struct EnginePlugin;

impl Plugin for EnginePlugin {
    fn build(&self, app: &mut App) {
        match Engine::launch() {
            Ok(engine) => {
                app.insert_resource(engine)
                    .insert_resource(Time::<Fixed>::from_seconds(STEP))
                    .add_systems(FixedUpdate, step);
            }
            // A frontend that cannot build its boat has nothing to draw. Saying
            // so and stopping beats opening a window onto an empty sea.
            Err(message) => panic!("vela could not start: {message}"),
        }
    }
}
