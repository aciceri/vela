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

/// True wind the boat is released into, m/s and radians off the bow.
const WIND_SPEED: f64 = 6.0;
const WIND_ANGLE: f64 = 45.0;

/// The sea the boat is released into.
///
/// A metre at six seconds: a moderate sea for a forty-footer, and the condition
/// every seakeeping number quoted in `docs/functional-analysis.md` was measured
/// at.
///
/// It was briefly raised to a metre and a half, to make the water look like it
/// had weather in it. `examples/motion_scale` is why it came back down: at that
/// height the boat swings seventeen degrees of trim and nearly three metres of
/// sinkage against a hull only 1.84 m deep, which is roughly twice a real
/// forty-footer's response. The natural periods are right — heave 2.2 s, pitch
/// 2.1 s, both inside the published band — so this is not a scale error but an
/// amplitude one, and §5.4 already names its two causes: no diffraction, so the
/// Froude-Krylov excitation is over-predicted at wavelengths near the hull's own,
/// and no viscous damping, so nothing limits the response where the spectrum
/// overlaps the pitch resonance.
///
/// Choosing the sea to flatter the model would have been the wrong lever, and
/// choosing it to stay inside what the model is measured at is the right one.
const WAVE_HEIGHT: f64 = 1.0;
const WAVE_PERIOD: f64 = 6.0;

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
    /// equilibrium solve computes it, and this is where the answer is kept so the
    /// frontend can return the helm *there* when nobody is steering.
    ///
    /// Centring on zero instead, which is what this file did first, releases a
    /// perfectly trimmed boat with its rudder in the wrong place: it luffs up
    /// within seconds and stops. That looked like a physics problem and was a
    /// frontend one.
    pub balanced_helm: f64,
}

impl Engine {
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

        let sea = Seaway2D::new(
            wind,
            SeaState {
                significant_height: WAVE_HEIGHT,
                peak_period: WAVE_PERIOD,
                // Travelling towards the boat: waves come from where the wind does.
                heading: WIND_ANGLE.to_radians() + std::f64::consts::PI,
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
