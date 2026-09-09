//! The force-module contract and the time-domain loop.
//!
//! # What this is
//!
//! A velocity prediction program solves a force balance for the one speed and
//! attitude that make it vanish. This does not: it integrates the equations of
//! motion, so a tack, a gust or a broach is a trajectory rather than a
//! forbidden state. Steady sailing is then whatever the trajectory settles to —
//! which is also how the classical answer is recovered, and comparing the two
//! is a free cross-check on both.
//!
//! # Why the kinematics live here
//!
//! Every force model needs the same handful of derived quantities: heel, speed
//! through the water, leeway, apparent wind. Each of them carries a sign
//! convention, and each of those conventions is a way to produce forces that
//! look entirely plausible and are wrong — the surrounding modules say so
//! repeatedly, in those words.
//!
//! So they are derived **once**, here, on [`StepCtx`], and no force module is
//! permitted its own definition. A module that recomputed leeway from the body
//! velocity would be free to pick the other sign, and nothing would catch it:
//! the boat would still sail, just with the keel lifting the wrong way. This is
//! the single most important structural decision in this file.
//!
//! # Contract
//!
//! - Modules receive an immutable [`StepCtx`] and return a [`Wrench`] in the
//!   **body frame, about the body origin** — never about the centre of
//!   gravity, which moves.
//! - Modules never see each other. Coupling either lives inside one module —
//!   which is why keel and rudder are one module, not two — or passes through
//!   the state.
//! - Modules are stateful by design (`&mut self`): a cached factorization or an
//!   internal ODE state is expected. Determinism comes from a fixed step and a
//!   fixed call order, not from purity.
//! - Gravity is not a module. [`RigidBody`] owns it, because it is exact.

use crate::aero::ApparentWind;
use crate::controls::Controls;
use crate::env::Environment;
use crate::rigid_body::RigidBody;
use crate::state::BodyState;
use crate::telemetry::Telemetry;
use crate::wrench::Wrench;
use nalgebra::{UnitQuaternion, Vector3};

/// Which way is leeward, given an apparent wind angle.
///
/// Returns `-1` when the wind is on the starboard bow (positive angle), because
/// leeward is then to port, and `+1` on the other tack. Multiply an
/// athwartships force that the aerodynamic model reports as "positive to
/// leeward" by this to land it on the body `y` axis, which points to starboard.
///
/// A free function rather than a method so that the one line of tack logic in
/// this engine has exactly one home.
#[must_use]
pub fn leeward_sign(apparent_wind_angle: f64) -> f64 {
    if apparent_wind_angle >= 0.0 {
        -1.0
    } else {
        1.0
    }
}

/// The water flow past a body-fixed point, as a foil there would feel it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalFlow {
    /// Speed of the water past the point, m/s.
    pub speed: f64,
    /// Angle of the flow to the centreline, radians, **positive when it
    /// produces side force towards `+y`** — the same sense as
    /// [`StepCtx::leeway`], of which this is the generalisation to a point off
    /// the origin.
    pub angle: f64,
}
/// Everything a force module may read, for one step.
pub struct StepCtx<'a> {
    /// Pose and velocities.
    pub state: &'a BodyState,
    /// Crew inputs.
    pub controls: &'a Controls,
    /// Wind, water and free surface.
    pub env: &'a dyn Environment,
    /// Simulation time, s.
    pub time: f64,
    /// The step about to be taken, s.
    pub dt: f64,
}

impl StepCtx<'_> {
    /// Heel angle, radians, positive starboard-down.
    #[must_use]
    pub fn heel(&self) -> f64 {
        self.state.euler_angles().0
    }

    /// Trim angle, radians, positive bow-up.
    ///
    /// The pitch Euler angle unchanged. The sign is genuinely not obvious in a
    /// z-down frame and an earlier version of this function had it backwards,
    /// so it is pinned by geometry rather than by argument: a positive rotation
    /// about the starboard-pointing body `y` axis carries the downward axis
    /// towards the bow, which lifts the bow. The test
    /// `positive_trim_lifts_the_bow` asserts that as a fact about the rotated
    /// forward vector, so it cannot drift back.
    #[must_use]
    pub fn trim_angle(&self) -> f64 {
        self.state.euler_angles().1
    }

    /// Heading, radians, measured as the compass bearing of the centreline.
    #[must_use]
    pub fn heading(&self) -> f64 {
        self.state.euler_angles().2
    }

    /// Speed through the water, m/s: the magnitude of the horizontal velocity.
    ///
    /// Horizontal rather than the full body-frame speed, because that is what
    /// every resistance regression means by `V`, and because taking the body
    /// speed would fold heave — a wave response, not progress — into the
    /// resistance.
    #[must_use]
    pub fn speed_through_water(&self) -> f64 {
        let world = self.state.world_velocity();
        world.xy().norm()
    }

    /// Leeway angle, radians.
    ///
    /// **Positive leeway produces side force towards `+y`, to starboard.** That
    /// is the sign the appendage model is written in, and it is the reason for
    /// the negation below: a boat slipping to port has a negative sway velocity
    /// and needs its keel to lift to starboard, back to windward.
    ///
    /// Derived from body-frame surge and sway, so it is the angle between the
    /// centreline and the track — which is what leeway means — and not the
    /// angle to the apparent wind.
    #[must_use]
    pub fn leeway(&self) -> f64 {
        let surge = self.state.velocity.x;
        let sway = self.state.velocity.y;
        if surge == 0.0 && sway == 0.0 {
            return 0.0;
        }
        (-sway).atan2(surge)
    }

    /// The water flow past a body-fixed point.
    ///
    /// [`StepCtx::leeway`] is this at the body origin. A foil is not at the
    /// origin, and the difference is not a refinement: a keel a metre and a
    /// third below the origin, on a boat rolling at half a radian per second,
    /// sees two thirds of a metre per second of athwartships flow it would not
    /// otherwise see. Against four metres per second of boat speed that is nine
    /// degrees of angle of attack — and the lift it produces is a moment
    /// opposing the roll.
    ///
    /// That moment is **the dominant damping of a keelboat's roll**, and it
    /// comes out of the already-transcribed lift model for free, as a
    /// consequence of asking the foil what flow it is actually in. Taking it
    /// from the origin instead leaves roll undamped, which is not a small error
    /// in a simulation meant to be sailed.
    #[must_use]
    pub fn local_flow_at(&self, body_point: Vector3<f64>) -> LocalFlow {
        let velocity = self.state.point_velocity(body_point);
        let surge = velocity.x;
        let sway = velocity.y;
        LocalFlow {
            speed: (surge * surge + sway * sway).sqrt(),
            angle: if surge == 0.0 && sway == 0.0 {
                0.0
            } else {
                (-sway).atan2(surge)
            },
        }
    }

    /// The apparent wind at a body-fixed point, in the plane of the heeled sail
    /// plan.
    ///
    /// Takes a point rather than a height so that the rotational part of the
    /// point's velocity is included: at the centre of effort of a rig, roll
    /// rate contributes metres per second of athwartships motion, and dropping
    /// it removes the aerodynamic damping of roll entirely.
    ///
    /// The returned angle is measured from the bow, positive when the wind is
    /// on the starboard side.
    ///
    /// # Heel
    ///
    /// Heel enters here rather than in the sail coefficients, which is where
    /// the source puts it. Larsson, Eliasson & Orych, *Principles of Yacht
    /// Design*, 5th ed., Fig 8.22, describing Hazen's model: *"Rather than
    /// modifying all coefficients, the apparent wind speed and direction are
    /// computed in a plane that heels with the yacht. The component of the
    /// apparent velocity along the hull is unchanged by heel, while the
    /// component at right angles thereto is proportional to the cosine of the
    /// heel angle."*
    ///
    /// That is the whole of it: the along-hull component survives, the
    /// athwartships one is scaled by `cos φ`, and the speed and angle are
    /// rebuilt from the pair. The text also notes that leeway is neglected in
    /// this transformation, which is why the resolution is onto the hull axis
    /// and not onto the track.
    ///
    /// This matters more than a refinement usually does. Without it a yacht in
    /// a working breeze solves to nearly 40 degrees of heel, because nothing
    /// ever spills the wind out of the rig as it lies over.
    #[must_use]
    pub fn apparent_wind_at(&self, body_point: Vector3<f64>) -> ApparentWind {
        let world_point = self.state.position + self.state.to_world(body_point);
        let true_wind = self.env.wind(world_point, self.time);
        let point_velocity = self.state.to_world(self.state.point_velocity(body_point));

        // Velocity of the air relative to the point, horizontal components only.
        let relative = (true_wind - point_velocity).xy();
        // The wind is named by where it comes *from*.
        let from = -relative;

        let heading = self.horizontal_heading();
        // Starboard in the horizontal plane, for a z-down world: z x heading.
        let starboard = Vector3::new(-heading.y, heading.x, 0.0);

        let along = from.dot(&heading.xy());
        let across = from.dot(&starboard.xy()) * self.heel().cos();

        ApparentWind {
            speed: (along * along + across * across).sqrt(),
            angle: across.atan2(along),
        }
    }

    /// The centreline projected onto the horizontal plane, unit length.
    ///
    /// Falls back to due north when the boat is pitched so far that the
    /// centreline is vertical and a heading is undefined. That is a pitchpole,
    /// not a sailing condition, and returning a NaN direction from it would
    /// poison every force downstream instead of merely being wrong about a boat
    /// that is already lost.
    fn horizontal_heading(&self) -> Vector3<f64> {
        let forward = self.state.to_world(Vector3::x());
        let horizontal = Vector3::new(forward.x, forward.y, 0.0);
        let norm = horizontal.norm();
        if norm < 1e-9 {
            Vector3::x()
        } else {
            horizontal / norm
        }
    }
}

/// One superposed contribution to the total force on the boat.
///
/// # Why `Send + Sync`
///
/// A [`Sim`] holds its modules and its environment as trait objects, so those
/// bounds are what make the simulation itself movable between threads — and a
/// simulation that is not is a simulation a frontend cannot own. Bevy requires
/// it of a resource, a host running several boats at once requires it, and a
/// worker thread requires it.
///
/// It costs nothing real: a force module is a transcription of coefficients plus
/// some cached state, and everything in this crate satisfies the bound without
/// trying. It is stated rather than left to inference so that a module which
/// reached for a `Rc` or a raw pointer fails at its definition instead of at
/// some distant call site.
pub trait ForceModule: Send + Sync {
    /// Stable identifier, used in telemetry keys and error messages.
    fn name(&self) -> &'static str;

    /// The wrench this module contributes, body frame, about the body origin.
    fn step(&mut self, ctx: &StepCtx<'_>) -> Wrench;

    /// Publishes the module's named components.
    ///
    /// Called after [`ForceModule::step`], so a module may report what it just
    /// computed. Defaulted to nothing so that a module without a meaningful
    /// breakdown is not forced to invent one.
    fn telemetry(&self, out: &mut Telemetry) {
        let _ = out;
    }
}

/// Degrees of freedom held fixed while the rest are integrated.
///
/// This is not a numerical convenience: it is how hydrodynamic data has always
/// been taken. A towing-tank model on a dynamometer is restrained in the modes
/// nobody is measuring and free in the ones they are, and a *captive* test is
/// the standard name for it. The same device answers a question this engine
/// genuinely faces.
///
/// The boat data format carries no longitudinal position for the keel, the
/// rudder or the mast — the published particulars of a design simply do not
/// include them, and inventing one would silently fabricate a yaw balance. So
/// the yaw and trim moments of this engine are not modelled, and a run that
/// integrated them would be integrating zero where the physics is not zero.
/// Holding those two modes states that limitation in the mechanism instead of
/// hiding it in a doc comment, and what remains free — surge, sway, heave,
/// roll — is exactly the force balance a classical velocity prediction solves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Captive {
    pub surge: bool,
    pub sway: bool,
    pub heave: bool,
    pub roll: bool,
    pub pitch: bool,
    pub yaw: bool,
}

impl Captive {
    /// Everything free: the full six degrees of freedom.
    #[must_use]
    pub const fn free() -> Self {
        Self {
            surge: false,
            sway: false,
            heave: false,
            roll: false,
            pitch: false,
            yaw: false,
        }
    }

    /// Trim and yaw held, the rest free.
    ///
    /// The condition a velocity prediction program is posed in, and the only
    /// one the present force models have the data to support. See the type
    /// documentation.
    #[must_use]
    pub const fn velocity_prediction() -> Self {
        Self {
            pitch: true,
            yaw: true,
            ..Self::free()
        }
    }

    #[must_use]
    const fn any_rotation(self) -> bool {
        self.roll || self.pitch || self.yaw
    }

    /// The restraint as the generalized mask [`crate::rigid_body::RigidBody`]
    /// wants, in the order surge, sway, heave, roll, pitch, yaw.
    #[must_use]
    pub const fn held(self) -> [bool; 6] {
        [
            self.surge, self.sway, self.heave, self.roll, self.pitch, self.yaw,
        ]
    }
}

/// A boat, its environment, and its force modules, advanced in fixed steps.
pub struct Sim {
    body: RigidBody,
    state: BodyState,
    env: Box<dyn Environment>,
    modules: Vec<Box<dyn ForceModule>>,
    controls: Controls,
    time: f64,
    telemetry: Telemetry,
    last_wrench: Wrench,
    captive: Captive,
    restrained_pose: BodyState,
}

impl Sim {
    /// Assembles a simulation, fully free in all six degrees of freedom.
    ///
    /// Module order is fixed by the caller and never changes afterwards. Since
    /// modules only read shared state, the order cannot change the physics —
    /// pinning it pins determinism, nothing more.
    #[must_use]
    pub fn new(
        body: RigidBody,
        state: BodyState,
        env: Box<dyn Environment>,
        modules: Vec<Box<dyn ForceModule>>,
        controls: Controls,
    ) -> Self {
        Self {
            body,
            state: state.clone(),
            env,
            modules,
            controls,
            time: 0.0,
            telemetry: Telemetry::new(),
            last_wrench: Wrench::zero(),
            captive: Captive::free(),
            restrained_pose: state,
        }
    }

    /// Restrains the given degrees of freedom at the current pose.
    ///
    /// The pose is captured here rather than at construction so that a caller
    /// can settle a boat, then restrain it about the attitude it settled at.
    ///
    /// The restraint goes into the mass matrix — see
    /// [`RigidBody::restrain`] for why it has to, and what goes wrong when a
    /// held mode's force is allowed to reach the free ones through the added
    /// mass before being cancelled.
    #[must_use]
    pub fn with_captive(mut self, captive: Captive) -> Self {
        self.captive = captive;
        self.body.restrain(captive.held());
        self.restrained_pose = self.state.clone();
        self
    }

    #[must_use]
    pub fn captive(&self) -> Captive {
        self.captive
    }

    pub fn set_controls(&mut self, controls: Controls) {
        self.controls = controls;
    }

    #[must_use]
    pub fn controls(&self) -> &Controls {
        &self.controls
    }

    #[must_use]
    pub fn state(&self) -> &BodyState {
        &self.state
    }

    /// Replaces the state, and with it the pose any captive degrees of freedom
    /// are held at: setting the state is how a caller says where the boat is,
    /// and a restraint that went on pointing at the old pose would silently
    /// drag it back there.
    pub fn set_state(&mut self, state: BodyState) {
        self.state = state.clone();
        self.restrained_pose = state;
    }

    /// Replaces the environment the boat is in - the wind and the water -
    /// from the next step on.
    ///
    /// The modules are not told. None of them holds a copy of the sea: each
    /// reads it through the context at every step, so a new sea is simply the
    /// one the next step evaluates. What a swap does not do is fade: a boat on
    /// a calm handed a two-metre sea meets its first crest in one step, with
    /// the jolt that implies, and the radiation memory carries on from the
    /// motion it had. That is the honest behaviour of a change of weather
    /// nobody asked the physics to smooth; a caller wanting a gentle change
    /// hands over a sea that starts gentle.
    pub fn set_environment(&mut self, env: Box<dyn Environment>) {
        self.env = env;
    }

    #[must_use]
    pub fn body(&self) -> &RigidBody {
        &self.body
    }

    #[must_use]
    pub fn time(&self) -> f64 {
        self.time
    }

    /// The summed external wrench of the most recent step, excluding gravity.
    #[must_use]
    pub fn last_wrench(&self) -> Wrench {
        self.last_wrench
    }

    #[must_use]
    pub fn telemetry(&self) -> &Telemetry {
        &self.telemetry
    }

    /// Sums the module wrenches at the current state without integrating.
    ///
    /// This is what an equilibrium solver drives to zero, and what a test uses
    /// to inspect a force balance at a pose it constructed by hand.
    pub fn evaluate(&mut self, dt: f64) -> Wrench {
        // Disjoint field borrows: the context reads `state`, `controls` and
        // `env` while the modules are borrowed mutably.
        let ctx = StepCtx {
            state: &self.state,
            controls: &self.controls,
            env: self.env.as_ref(),
            time: self.time,
            dt,
        };

        let mut total = Wrench::zero();
        for module in &mut self.modules {
            total += module.step(&ctx);
        }

        self.telemetry.clear();
        for module in &self.modules {
            module.telemetry(&mut self.telemetry);
        }

        self.last_wrench = total;
        total
    }

    /// The module sum plus weight: the whole force the boat actually feels.
    ///
    /// This is the quantity an equilibrium solver drives to zero. Kept separate
    /// from [`Sim::evaluate`] because the two answer different questions and
    /// confusing them is a whole-displacement error: the module sum excludes
    /// gravity, since [`RigidBody`] owns it, so at equilibrium `evaluate`
    /// returns the weight rather than nothing.
    pub fn applied_wrench(&mut self, dt: f64) -> Wrench {
        let external = self.evaluate(dt);
        external + self.body.gravity_wrench(&self.state)
    }

    /// Advances the simulation by exactly one fixed step.
    pub fn step(&mut self, dt: f64) {
        let total = self.evaluate(dt);
        self.body.step(&mut self.state, total, dt);
        self.enforce_captive();
        self.time += dt;
    }

    /// Anchors the pose of every restrained degree of freedom.
    ///
    /// The *dynamics* of the restraint live in the mass matrix
    /// ([`RigidBody::restrain`]), which is what makes a held mode's
    /// acceleration exactly zero and stops its unbalanced force leaking into the
    /// free modes. Two jobs are left over for this, and both are about the pose
    /// rather than the forces:
    ///
    /// - A caller may hand in a state that is already moving in a held mode
    ///   through [`Sim::set_state`]. Zero acceleration would preserve that
    ///   velocity forever; the restraint means it should not have one.
    /// - Zero body-frame pitch rate is **not** a frozen Euler pitch angle. With
    ///   `φ` of heel and a yaw rate `r`, `θ̇ = q cos φ - r sin φ`, so a boat held
    ///   at `q = 0` still changes trim as it turns. Rebuilding the attitude from
    ///   the angles is what actually holds the trim.
    ///
    /// The applied wrench on a held mode is never deleted, before or after: a
    /// dynamometer absorbs a force rather than removing it, and that residual
    /// reading stays in [`Sim::last_wrench`] and in the telemetry.
    fn enforce_captive(&mut self) {
        let captive = self.captive;

        if captive.surge {
            self.state.velocity.x = 0.0;
        }
        if captive.sway {
            self.state.velocity.y = 0.0;
        }
        if captive.heave {
            self.state.velocity.z = 0.0;
            self.state.position.z = self.restrained_pose.position.z;
        }
        if captive.roll {
            self.state.angular_velocity.x = 0.0;
        }
        if captive.pitch {
            self.state.angular_velocity.y = 0.0;
        }
        if captive.yaw {
            self.state.angular_velocity.z = 0.0;
        }

        if !captive.any_rotation() {
            return;
        }

        // Rebuild the attitude, taking each restrained angle from the pose the
        // restraint was set at and each free angle from the integration.
        let (roll, pitch, yaw) = self.state.euler_angles();
        let (held_roll, held_pitch, held_yaw) = self.restrained_pose.euler_angles();
        self.state.attitude = UnitQuaternion::from_euler_angles(
            if captive.roll { held_roll } else { roll },
            if captive.pitch { held_pitch } else { pitch },
            if captive.yaw { held_yaw } else { yaw },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aero::SailSet;
    use crate::env::{StillWater, UniformWind};
    use crate::mass::MassProperties;
    use approx::assert_relative_eq;
    use nalgebra::UnitQuaternion;
    use std::f64::consts::FRAC_PI_2;

    fn controls() -> Controls {
        Controls::close_hauled(SailSet::upwind())
    }

    fn body() -> RigidBody {
        let mass = MassProperties::from_gyradii(
            3800.0,
            Vector3::new(0.0, 0.0, 0.5),
            Vector3::new(1.1, 2.6, 2.7),
        )
        .expect("valid mass properties");
        RigidBody::new(mass).expect("valid rigid body")
    }

    #[test]
    fn a_boat_slipping_to_port_has_positive_leeway() {
        // Sailing forward and sliding to port: sway is negative in a
        // starboard-positive frame. The keel must then lift to starboard, which
        // is positive leeway by this engine's convention.
        let state = BodyState {
            velocity: Vector3::new(3.0, -0.2, 0.0),
            ..BodyState::default()
        };
        let env = StillWater::new(UniformWind::uniform(6.0, 0.0));
        let controls = controls();
        let ctx = StepCtx {
            state: &state,
            controls: &controls,
            env: &env,
            time: 0.0,
            dt: 0.01,
        };
        assert!(ctx.leeway() > 0.0);
        assert_relative_eq!(ctx.leeway(), (0.2f64 / 3.0).atan(), epsilon = 1e-12);
    }

    #[test]
    fn a_stationary_boat_has_no_leeway_rather_than_a_nan() {
        let state = BodyState::default();
        let env = StillWater::new(UniformWind::uniform(6.0, 0.0));
        let controls = controls();
        let ctx = StepCtx {
            state: &state,
            controls: &controls,
            env: &env,
            time: 0.0,
            dt: 0.01,
        };
        assert_relative_eq!(ctx.leeway(), 0.0);
    }

    #[test]
    fn a_head_wind_on_a_stationary_boat_reads_as_dead_ahead() {
        // Boat heading north, wind from the north.
        let state = BodyState::default();
        let env = StillWater::new(UniformWind::uniform(8.0, 0.0));
        let controls = controls();
        let ctx = StepCtx {
            state: &state,
            controls: &controls,
            env: &env,
            time: 0.0,
            dt: 0.01,
        };
        let wind = ctx.apparent_wind_at(Vector3::new(0.0, 0.0, -8.0));
        assert_relative_eq!(wind.speed, 8.0, epsilon = 1e-9);
        assert_relative_eq!(wind.angle, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn a_beam_wind_from_starboard_reads_as_a_positive_angle() {
        // Boat heading north, wind from the east: that is the starboard beam.
        let state = BodyState::default();
        let env = StillWater::new(UniformWind::uniform(8.0, FRAC_PI_2));
        let controls = controls();
        let ctx = StepCtx {
            state: &state,
            controls: &controls,
            env: &env,
            time: 0.0,
            dt: 0.01,
        };
        let wind = ctx.apparent_wind_at(Vector3::new(0.0, 0.0, -8.0));
        assert_relative_eq!(wind.angle, FRAC_PI_2, epsilon = 1e-9);
        assert_relative_eq!(leeward_sign(wind.angle), -1.0);
    }

    #[test]
    fn boat_speed_shifts_the_apparent_wind_forward() {
        // Beam true wind plus boat speed must come from forward of the beam.
        let state = BodyState {
            velocity: Vector3::new(5.0, 0.0, 0.0),
            ..BodyState::default()
        };
        let env = StillWater::new(UniformWind::uniform(5.0, FRAC_PI_2));
        let controls = controls();
        let ctx = StepCtx {
            state: &state,
            controls: &controls,
            env: &env,
            time: 0.0,
            dt: 0.01,
        };
        let wind = ctx.apparent_wind_at(Vector3::new(0.0, 0.0, -8.0));
        assert!(wind.angle < FRAC_PI_2, "apparent wind must move forward");
        assert_relative_eq!(wind.angle, FRAC_PI_2 / 2.0, epsilon = 1e-9);
        assert_relative_eq!(wind.speed, (50.0f64).sqrt(), epsilon = 1e-9);
    }

    #[test]
    fn roll_rate_moves_the_apparent_wind_at_the_masthead() {
        // The masthead of a rolling boat sweeps sideways, and the rig feels it.
        // Without this term aerodynamic roll damping does not exist.
        let still = BodyState::default();
        let rolling = BodyState {
            angular_velocity: Vector3::new(0.5, 0.0, 0.0),
            ..BodyState::default()
        };
        let env = StillWater::new(UniformWind::uniform(8.0, 0.0));
        let controls = controls();
        let masthead = Vector3::new(0.0, 0.0, -15.0);

        let quiet = StepCtx {
            state: &still,
            controls: &controls,
            env: &env,
            time: 0.0,
            dt: 0.01,
        }
        .apparent_wind_at(masthead);
        let moving = StepCtx {
            state: &rolling,
            controls: &controls,
            env: &env,
            time: 0.0,
            dt: 0.01,
        }
        .apparent_wind_at(masthead);

        assert!(
            (moving.angle - quiet.angle).abs() > 1e-3,
            "roll rate must change the apparent wind aloft"
        );
    }

    #[test]
    fn heel_and_heading_carry_the_signs_a_naval_architect_expects() {
        let state = BodyState {
            attitude: UnitQuaternion::from_euler_angles(0.3, 0.05, 1.0),
            ..BodyState::default()
        };
        let env = StillWater::new(UniformWind::uniform(6.0, 0.0));
        let controls = controls();
        let ctx = StepCtx {
            state: &state,
            controls: &controls,
            env: &env,
            time: 0.0,
            dt: 0.01,
        };
        assert_relative_eq!(ctx.heel(), 0.3, epsilon = 1e-9);
        assert_relative_eq!(ctx.trim_angle(), 0.05, epsilon = 1e-9);
        assert_relative_eq!(ctx.heading(), 1.0, epsilon = 1e-9);
    }

    /// Pins the trim sign to geometry rather than to a convention name.
    ///
    /// A previous version of `trim_angle` returned the negated pitch on the
    /// strength of a plausible-sounding argument about the right-hand rule in a
    /// z-down frame, and it was backwards. The only defence against that is an
    /// assertion about where the bow actually points: in NED, upward is
    /// negative `z`, so a bow-up attitude must give the rotated forward vector
    /// a negative `z` component.
    #[test]
    fn positive_trim_lifts_the_bow() {
        let state = BodyState {
            attitude: UnitQuaternion::from_euler_angles(0.0, 0.15, 0.0),
            ..BodyState::default()
        };
        let forward_in_world = state.to_world(Vector3::x());
        assert!(
            forward_in_world.z < 0.0,
            "positive pitch must point the bow above the horizon, got z = {}",
            forward_in_world.z
        );

        let env = StillWater::new(UniformWind::uniform(6.0, 0.0));
        let controls = controls();
        let ctx = StepCtx {
            state: &state,
            controls: &controls,
            env: &env,
            time: 0.0,
            dt: 0.01,
        };
        assert!(
            ctx.trim_angle() > 0.0,
            "an attitude with the bow up must report positive trim"
        );
    }

    #[test]
    fn speed_through_water_ignores_heave() {
        let state = BodyState {
            velocity: Vector3::new(4.0, 0.0, 2.0),
            ..BodyState::default()
        };
        let env = StillWater::new(UniformWind::uniform(6.0, 0.0));
        let controls = controls();
        let ctx = StepCtx {
            state: &state,
            controls: &controls,
            env: &env,
            time: 0.0,
            dt: 0.01,
        };
        assert_relative_eq!(ctx.speed_through_water(), 4.0, epsilon = 1e-12);
    }

    #[test]
    fn a_simulation_with_no_modules_falls_under_gravity_alone() {
        let env = StillWater::new(UniformWind::uniform(0.0, 0.0));
        let mut sim = Sim::new(
            body(),
            BodyState::default(),
            Box::new(env),
            Vec::new(),
            controls(),
        );
        sim.step(1.0 / 120.0);
        assert!(
            sim.state().velocity.z > 0.0,
            "with no buoyancy the boat must sink"
        );
        assert_eq!(sim.last_wrench(), Wrench::zero());
    }

    /// A module that applies one constant wrench, for exercising the loop.
    struct ConstantForce(Wrench);

    impl ForceModule for ConstantForce {
        fn name(&self) -> &'static str {
            "constant"
        }

        fn step(&mut self, _ctx: &StepCtx<'_>) -> Wrench {
            self.0
        }

        fn telemetry(&self, out: &mut Telemetry) {
            out.set("constant.force.x", self.0.force.x);
        }
    }

    fn sim_with(wrench: Wrench, captive: Captive) -> Sim {
        let env = StillWater::new(UniformWind::uniform(0.0, 0.0));
        Sim::new(
            RigidBody::with_gravity(body().mass_properties().clone(), 0.0)
                .expect("valid rigid body"),
            BodyState::default(),
            Box::new(env),
            vec![Box::new(ConstantForce(wrench))],
            controls(),
        )
        .with_captive(captive)
    }

    #[test]
    fn a_restrained_yaw_mode_does_not_turn_the_boat() {
        let yawing = Wrench::from_moment(Vector3::new(0.0, 0.0, 5_000.0));
        let mut sim = sim_with(yawing, Captive::velocity_prediction());
        for _ in 0..240 {
            sim.step(1.0 / 240.0);
        }
        assert_relative_eq!(sim.state().angular_velocity.z, 0.0, epsilon = 1e-12);
        assert_relative_eq!(sim.state().euler_angles().2, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn a_restraint_absorbs_the_force_without_hiding_it() {
        // The whole point of a captive measurement: the restrained mode's
        // force is the reading, so it must remain visible.
        let yawing = Wrench::from_moment(Vector3::new(0.0, 0.0, 5_000.0));
        let mut sim = sim_with(yawing, Captive::velocity_prediction());
        sim.step(1.0 / 240.0);
        assert_relative_eq!(sim.last_wrench().moment.z, 5_000.0);
        assert_eq!(sim.telemetry().get("constant.force.x"), Some(0.0));
    }

    #[test]
    fn a_free_mode_still_moves_while_its_neighbours_are_held() {
        // Roll is free in the velocity-prediction condition and must respond.
        let heeling = Wrench::from_moment(Vector3::new(4_000.0, 0.0, 0.0));
        let mut sim = sim_with(heeling, Captive::velocity_prediction());
        for _ in 0..240 {
            sim.step(1.0 / 240.0);
        }
        assert!(
            sim.state().euler_angles().0 > 0.01,
            "a free roll mode must heel under a heeling moment"
        );
    }

    #[test]
    fn restraints_are_taken_at_the_pose_they_were_set_at() {
        let env = StillWater::new(UniformWind::uniform(0.0, 0.0));
        let heeled = BodyState {
            attitude: UnitQuaternion::from_euler_angles(0.0, 0.2, 0.0),
            ..BodyState::default()
        };
        let mut sim = Sim::new(
            RigidBody::with_gravity(body().mass_properties().clone(), 0.0)
                .expect("valid rigid body"),
            heeled,
            Box::new(env),
            vec![Box::new(ConstantForce(Wrench::from_moment(Vector3::new(
                0.0, 3_000.0, 0.0,
            ))))],
            controls(),
        )
        .with_captive(Captive::velocity_prediction());

        for _ in 0..120 {
            sim.step(1.0 / 240.0);
        }
        // Trim was held at the 0.2 rad it was restrained at, not driven to zero.
        assert_relative_eq!(sim.state().euler_angles().1, 0.2, epsilon = 1e-9);
    }
}
