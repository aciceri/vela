//! The lateral plane: keel and rudder, their side force, their induced
//! resistance, and the keel's own residuary resistance.
//!
//! # Provenance
//!
//! Every number here comes from [`crate::appendages`], which is transcribed
//! from Larsson, Eliasson & Orych, *Principles of Yacht Design*, 5th ed. —
//! Fig 6.13 (side force and downwash), Fig 6.14 (induced resistance), Fig 5.19
//! (keel residuary resistance) and Fig 5.23 (its change with heel). This file
//! adds no coefficient, no correction and no constant of its own: it converts
//! flow state into that model's [`FlowState`] and its
//! [`AppendageForces`](crate::appendages::AppendageForces) into a body-frame
//! wrench, and nothing else.
//!
//! # Why keel and rudder are one module
//!
//! The keel's circulation sets the downwash that changes the rudder's angle of
//! attack, so the two cannot be superposed independently — solving them apart
//! gives a yaw balance that looks reasonable and is wrong. That coupling is
//! resolved inside [`appendage_forces`], in the order the figures require, and
//! this module calls it exactly once per step for exactly that reason.
//!
//! # Accuracy, in the source's own words
//!
//! The book states that Fig 6.13's accuracy is **too low for optimizing
//! appendages**: it estimates the total side force and no more. The
//! keel-versus-rudder split this module publishes is therefore a plausible
//! distribution, not a measurement. It is good enough to give a boat a helm
//! that responds in the right direction and to show how balance shifts with
//! heel; it is not good enough to choose between two candidate rudders, and the
//! telemetry breakdown must not be read as though it were.
//!
//! # Conventions
//!
//! * The returned wrench is in the **body frame** (`x` forward, `y` starboard,
//!   `z` down) with the moment about the **body origin**.
//! * `ctx.leeway()` is positive when the resulting side force points to `+y`.
//!   That is the convention of both [`StepCtx`] and the appendage model, so the
//!   side forces go onto the body `y` axis with their own sign and *no* tack
//!   correction. [`crate::sim::leeward_sign`] is deliberately **not** used
//!   here: it exists for models that report a force "positive to leeward",
//!   which the appendage model does not — its sign is carried by the leeway
//!   angle, which is a hydrodynamic quantity and knows nothing about where the
//!   wind is. Multiplying by it would make the keel lift to leeward on one
//!   tack.
//!
//! # Known gaps
//!
//! * **Centres of effort are inputs.** Figs 6.13 and 6.14 give force
//!   magnitudes and no line of action, so `keel_at` and `rudder_at` are
//!   constructor arguments. Deriving them from the planform would mean
//!   inventing a chordwise and spanwise centre of pressure that no figure here
//!   supplies, and every yaw and heeling moment this module produces would then
//!   rest on that invention.
//! * **The side force is placed horizontally, as the figure states it.**
//!   Fig 6.13's `F_h` is the horizontal component of the blade's lift; the lift
//!   itself is perpendicular to the span, larger by `1/cos φ`, and therefore
//!   has a vertical component `F_h tan φ`. This module applies `F_h` along body
//!   `y` and no vertical force, because that is the quantity the figure
//!   publishes. Where the distinction changes a published force it is not lost:
//!   [`crate::appendages::induced_resistance`] builds its lift coefficient from
//!   `F_h / cos φ` internally.
//! * **No appendage friction.** It is a wetted-surface calculation on the real
//!   blades, not on the extended planforms this model uses, and neither the
//!   wetted surfaces nor a reference length are inputs to this module. It is
//!   absent rather than estimated.
//! * **No stall.** Fig 6.13's lift is linear in angle of attack with no
//!   limiting term, so a rudder held hard over keeps gaining side force. In a
//!   time-domain simulation, which can reach angles a velocity prediction
//!   program never visits, this overstates what a stalled rudder can do.
//! * **Resistance passes through the body origin.** The figures give
//!   magnitudes only, and the constructor's application points are documented
//!   as centres of effort for the *side* force; the keel's residuary resistance
//!   is wave making by its volume and has no published line of action at all.
//!   So appendage drag contributes no trim moment here. Attributing one would
//!   mean choosing a lever the source does not give.
//! * **Froude envelope.** Below `Fn = 0.20` the keel residuary resistance
//!   tapers to zero and above `Fn = 0.60` the last tabulated row is held; both
//!   are [`crate::appendages`]'s documented edge treatments, inherited here.

use crate::appendages::{
    appendage_forces, AppendageForces, FlowState, FoilPlanform, HullScalars, Keel,
};
use crate::sim::{ForceModule, StepCtx};
use crate::telemetry::Telemetry;
use crate::wrench::Wrench;
use nalgebra::Vector3;

/// Keel and rudder as one force module.
///
/// Holds the last computed [`AppendageForces`] so that
/// [`ForceModule::telemetry`] reports what the step actually produced rather
/// than recomputing it — a second evaluation could disagree with the first if
/// anything about the flow state were read twice.
pub struct LateralSystem {
    hull: HullScalars,
    keel: Keel,
    keel_at: Vector3<f64>,
    rudder: FoilPlanform,
    rudder_at: Vector3<f64>,
    /// `None` until the first step: before then there is nothing to report,
    /// and publishing zeros would be indistinguishable from a boat that really
    /// is generating no side force.
    last: Option<AppendageForces>,
}

impl LateralSystem {
    /// Builds the module from the two foils and the body-frame points their
    /// side forces act at.
    ///
    /// `keel_at` and `rudder_at` are the centres of effort, in body-frame
    /// metres, with `z` positive downwards — a keel below the origin has a
    /// positive `z`. They are **arguments, not derived**: the figures this
    /// module wraps tabulate force magnitudes and say nothing about where those
    /// forces act, so the only honest place for them is the caller, who knows
    /// the boat's geometry.
    ///
    /// `rudder` is the rudder planform **extended to the bottom of the canoe
    /// body**, which is what Figs 6.13 and 6.14 mean by the blade; see
    /// [`FoilPlanform`]. A rudder that stops short of the hull must be extended
    /// by the caller.
    #[must_use]
    pub fn new(
        hull: HullScalars,
        keel: Keel,
        keel_at: Vector3<f64>,
        rudder: FoilPlanform,
        rudder_at: Vector3<f64>,
    ) -> Self {
        Self {
            hull,
            keel,
            keel_at,
            rudder,
            rudder_at,
            last: None,
        }
    }

    /// The unit vector of the boat's track, in the body frame.
    ///
    /// Built from the **horizontal** world velocity, because that is the motion
    /// every resistance regression is fitted against, and then rotated into the
    /// body frame so the drag can be returned there. Under heel this direction
    /// acquires a body `z` component, which is correct: the force is horizontal
    /// in the world and the boat is not.
    ///
    /// Returns `None` when the boat is not moving horizontally, so that a
    /// division by zero speed never reaches a force. The magnitude comes from
    /// [`StepCtx::speed_through_water`] rather than from a second norm, so the
    /// direction and the speed the model was given cannot drift apart.
    fn track_direction(ctx: &StepCtx<'_>, speed: f64) -> Option<Vector3<f64>> {
        if speed <= 0.0 {
            return None;
        }
        let world = ctx.state.world_velocity();
        let horizontal = Vector3::new(world.x, world.y, 0.0);
        Some(ctx.state.to_body(horizontal) / speed)
    }
}

impl ForceModule for LateralSystem {
    fn name(&self) -> &'static str {
        "lateral"
    }

    fn step(&mut self, ctx: &StepCtx<'_>) -> Wrench {
        let speed = ctx.speed_through_water();
        let flow = FlowState {
            speed,
            leeway: ctx.leeway(),
            heel: ctx.heel(),
            rudder_angle: ctx.controls.rudder_angle,
        };

        // The wake factor on the rudder's inflow, the downwash on its angle of
        // attack, and the ordering of the two are all inside this call. There
        // is deliberately no second entry point.
        let forces = appendage_forces(
            &self.hull,
            &self.keel,
            &self.rudder,
            &flow,
            ctx.env.water().density,
            ctx.env.gravity(),
        );
        self.last = Some(forces);

        // Positive leeway lifts to starboard, which is `+y`: the appendage
        // model and `StepCtx` share that convention, so the sign of the force
        // is the sign the model returned and nothing is applied to it. The
        // rudder obeys the same rule at its own angle of attack, which already
        // includes the keel's downwash and the helm angle.
        //
        // Applying each force at its centre of effort makes the heeling moment
        // and the yaw moment fall out of the geometry: a starboard side force
        // below the origin gives a negative roll moment, heeling the boat to
        // port, and the same force abaft the origin gives a negative yaw
        // moment. Neither is assembled by hand, so neither can be given the
        // wrong sign independently of the other.
        let keel =
            Wrench::from_force_at(Vector3::new(0.0, forces.keel_side_force, 0.0), self.keel_at);
        let rudder = Wrench::from_force_at(
            Vector3::new(0.0, forces.rudder_side_force, 0.0),
            self.rudder_at,
        );

        // Induced resistance, keel residuary and its heel correction all oppose
        // motion along the track. The sum is signed by the regressions — Fig
        // 5.23's heel term can come out negative for some hulls — so it is
        // passed through as it stands rather than clamped.
        let drag = match Self::track_direction(ctx, speed) {
            Some(track) => Wrench::new(-forces.resistance() * track, Vector3::zeros()),
            None => Wrench::zero(),
        };

        keel + rudder + drag
    }

    fn telemetry(&self, out: &mut Telemetry) {
        let Some(forces) = self.last else {
            return;
        };

        out.set("lateral.keel.side_force", forces.keel_side_force);
        out.set("lateral.rudder.side_force", forces.rudder_side_force);
        out.set("lateral.side_force_total", forces.side_force());

        out.set(
            "lateral.keel.induced_resistance",
            forces.keel_induced_resistance,
        );
        out.set(
            "lateral.rudder.induced_resistance",
            forces.rudder_induced_resistance,
        );
        out.set("lateral.keel.residuary", forces.keel_residuary_resistance);
        out.set(
            "lateral.keel.residuary_heel_delta",
            forces.keel_heel_residuary_delta,
        );
        out.set("lateral.resistance_total", forces.resistance());

        // The two numbers a helm-balance investigation actually needs: the
        // rudder is working at the angle it is given, not at the leeway, and
        // the difference is the downwash. Published as angles rather than left
        // to be inferred from the force ratio, which also depends on planform.
        out.set("lateral.downwash_angle", forces.downwash_angle);
        out.set(
            "lateral.rudder.angle_of_attack",
            forces.rudder_angle_of_attack,
        );

        // Lift coefficients say how hard each foil is working, which is the
        // only warning available that the linear model has been pushed past
        // where a real blade would have stalled.
        out.set(
            "lateral.keel.lift_coefficient",
            forces.keel_lift_coefficient,
        );
        out.set(
            "lateral.rudder.lift_coefficient",
            forces.rudder_lift_coefficient,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aero::SailSet;
    use crate::controls::Controls;
    use crate::env::{StillWater, UniformWind};
    use crate::state::BodyState;
    use approx::assert_relative_eq;
    use nalgebra::UnitQuaternion;

    /// YD-41 particulars, Appendix 1 of the source, half-loaded displacement —
    /// the same fixture the appendage model's own validation tests use, so the
    /// operating point below is one whose forces are already pinned elsewhere.
    fn yd41_hull() -> HullScalars {
        HullScalars {
            waterline_length: 11.90,
            waterline_beam: 3.18,
            canoe_draft: 0.40,
            total_draft: 2.30,
            canoe_volume: 6.05,
        }
    }

    fn yd41_keel() -> Keel {
        let planform = FoilPlanform {
            root_chord: 1.00,
            tip_chord: 0.78,
            span: 1.90,
            sweep: 5.5_f64.to_radians(),
        };
        Keel {
            volume: 0.275,
            centre_of_buoyancy_below_hull_bottom: planform.planform_centroid_below_root(),
            planform,
        }
    }

    fn yd41_rudder() -> FoilPlanform {
        FoilPlanform {
            root_chord: 0.48,
            tip_chord: 0.22,
            span: 1.15,
            sweep: 10.0_f64.to_radians(),
        }
    }

    /// Application points chosen for these tests only, not measured from the
    /// boat: the keel a little abaft the origin and well below it, the rudder
    /// far aft and shallower. That is all the geometry the moment signs need —
    /// keel below the origin, rudder abaft the keel — and it is exactly the
    /// information the module refuses to guess for itself.
    const KEEL_AT: Vector3<f64> = Vector3::new(-0.2, 0.0, 1.3);
    const RUDDER_AT: Vector3<f64> = Vector3::new(-4.5, 0.0, 0.9);

    /// The speed of the source's worked examples, m/s: `Fn = 0.35` on an
    /// 11.90 m waterline, a tabulated row of Fig 5.19 rather than a point
    /// between two.
    const SPEED: f64 = 3.7816;

    fn system() -> LateralSystem {
        LateralSystem::new(yd41_hull(), yd41_keel(), KEEL_AT, yd41_rudder(), RUDDER_AT)
    }

    fn environment() -> StillWater {
        // Wind is irrelevant to every assertion here — the appendages see water
        // — but an `Environment` has to supply one.
        StillWater::new(UniformWind::uniform(8.0, 0.0))
    }

    /// Sailing at `speed` with the given leeway and heel, heading north.
    ///
    /// The body sway is negative for positive leeway, which is what
    /// [`StepCtx::leeway`] turns back into a positive angle: a boat slipping to
    /// port needs its keel to lift to starboard.
    fn state(speed: f64, leeway: f64, heel: f64) -> BodyState {
        BodyState {
            attitude: UnitQuaternion::from_euler_angles(heel, 0.0, 0.0),
            velocity: Vector3::new(speed * leeway.cos(), -speed * leeway.sin(), 0.0),
            ..BodyState::default()
        }
    }

    /// Steps the module once and returns the wrench together with the telemetry
    /// it published for that step.
    fn run(state: &BodyState, rudder_angle: f64) -> (Wrench, Telemetry) {
        let mut module = system();
        let env = environment();
        let controls = Controls::close_hauled(SailSet::upwind()).with_rudder(rudder_angle);
        let ctx = StepCtx {
            state,
            controls: &controls,
            env: &env,
            time: 0.0,
            dt: 0.01,
        };
        let wrench = module.step(&ctx);
        let mut telemetry = Telemetry::new();
        module.telemetry(&mut telemetry);
        (wrench, telemetry)
    }

    fn published(telemetry: &Telemetry, key: &str) -> f64 {
        telemetry
            .get(key)
            .unwrap_or_else(|| panic!("{key} must be published"))
    }

    #[test]
    fn positive_leeway_lifts_the_lateral_plane_to_starboard() {
        let (wrench, telemetry) = run(&state(SPEED, 4.0_f64.to_radians(), 0.0), 0.0);

        assert!(
            wrench.force.y > 0.0,
            "positive leeway must give a positive y force, got {}",
            wrench.force.y
        );
        // Both foils pull the same way: the rudder shares the leeway.
        assert!(published(&telemetry, "lateral.keel.side_force") > 0.0);
        assert!(published(&telemetry, "lateral.rudder.side_force") > 0.0);
        // The rudder is the minor partner. A missing wake factor or a lost
        // downwash would show up as the rudder carrying far more than this.
        let total = published(&telemetry, "lateral.side_force_total");
        let rudder_share = published(&telemetry, "lateral.rudder.side_force") / total;
        assert!(
            (0.0..0.25).contains(&rudder_share),
            "rudder share {rudder_share} is not a rudder's share"
        );
    }

    #[test]
    fn the_two_tacks_are_mirror_images() {
        // Same sailing condition, other tack: leeway and heel both reversed.
        let leeway = 4.0_f64.to_radians();
        let heel = 20.0_f64.to_radians();
        let (starboard, _) = run(&state(SPEED, leeway, heel), 0.0);
        let (port, _) = run(&state(SPEED, -leeway, -heel), 0.0);

        // Odd in the side force and in the moments it makes.
        assert_relative_eq!(port.force.y, -starboard.force.y, max_relative = 1e-12);
        assert_relative_eq!(port.moment.x, -starboard.moment.x, max_relative = 1e-12);
        assert_relative_eq!(port.moment.z, -starboard.moment.z, max_relative = 1e-12);
        // Even in the drag: heeling to port must cost what heeling to starboard
        // costs, and a boat does not go faster on one tack than the other.
        assert_relative_eq!(port.force.x, starboard.force.x, max_relative = 1e-12);
        assert_relative_eq!(port.force.z, starboard.force.z, max_relative = 1e-12);
    }

    #[test]
    fn downwash_puts_the_rudder_below_the_leeway_angle() {
        let leeway = 4.0_f64.to_radians();
        let heel = 20.0_f64.to_radians();
        let (_, telemetry) = run(&state(SPEED, leeway, heel), 0.0);

        let downwash = published(&telemetry, "lateral.downwash_angle");
        let alpha = published(&telemetry, "lateral.rudder.angle_of_attack");

        // The value the appendage model's own documentation cites for this
        // yacht at 4 degrees of leeway: 1.94 degrees, roughly half the leeway.
        //
        // Compared to the precision it was quoted at, not to machine
        // precision. A number printed as "1.94" asserts only that it lies in
        // [1.935, 1.945]; the transcription gives 1.9379 here, which agrees.
        // A tighter tolerance would be testing the rounding of the citation
        // rather than the correctness of the model.
        assert!(
            (downwash.to_degrees() - 1.94).abs() < 0.005,
            "downwash {} deg disagrees with the cited 1.94 deg",
            downwash.to_degrees()
        );
        assert!(
            alpha < leeway && alpha > 0.0,
            "rudder angle of attack {alpha} must sit between zero and the leeway {leeway}"
        );
        assert_relative_eq!(alpha, leeway - downwash, max_relative = 1e-12);
    }

    #[test]
    fn downwash_still_unloads_the_rudder_on_the_other_tack() {
        // A downwash whose sign was not restored would *add* to the angle of
        // attack here, and the model would be silently one-tack-only.
        let leeway = -4.0_f64.to_radians();
        let (_, telemetry) = run(&state(SPEED, leeway, -20.0_f64.to_radians()), 0.0);

        let downwash = published(&telemetry, "lateral.downwash_angle");
        let alpha = published(&telemetry, "lateral.rudder.angle_of_attack");
        assert!(downwash < 0.0, "downwash must follow the leeway's sign");
        assert!(
            alpha < 0.0 && alpha.abs() < leeway.abs(),
            "rudder must be unloaded, not loaded further: {alpha}"
        );
    }

    #[test]
    fn rudder_deflection_swings_the_yaw_moment_one_way() {
        let upright = state(SPEED, 0.0, 0.0);
        let (amidships, _) = run(&upright, 0.0);
        let (helm, telemetry) = run(&upright, 5.0_f64.to_radians());

        // Positive helm adds to the rudder's angle of attack, so its side force
        // goes to starboard; the rudder is abaft the origin, so `x F_y` is a
        // negative yaw moment. Sign confusion between the two would show up as
        // a boat that turns the wrong way.
        assert!(published(&telemetry, "lateral.rudder.side_force") > 0.0);
        assert!(
            helm.moment.z < amidships.moment.z,
            "positive helm must swing the yaw moment negative: {} vs {}",
            helm.moment.z,
            amidships.moment.z
        );
        // Amidships and upright with no leeway there is no side force at all,
        // so the only yaw moment is the one the helm just made.
        assert_relative_eq!(amidships.moment.z, 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            helm.moment.z,
            RUDDER_AT.x * published(&telemetry, "lateral.rudder.side_force"),
            max_relative = 1e-12
        );
        // More helm, more moment, monotonically.
        let (harder, _) = run(&upright, 10.0_f64.to_radians());
        assert!(harder.moment.z < helm.moment.z);
    }

    #[test]
    fn a_side_force_below_the_origin_heels_the_boat_away_from_it() {
        let (wrench, telemetry) = run(&state(SPEED, 4.0_f64.to_radians(), 0.0), 0.0);

        // Both foils lift to starboard from below, so the boat rolls to port:
        // a negative roll moment in a z-down frame.
        assert!(
            wrench.moment.x < 0.0,
            "a starboard force below the origin must heel to port, got {}",
            wrench.moment.x
        );
        // And the moment is the geometry, not a separate calculation: the drag
        // passes through the origin and contributes nothing to it.
        let expected = -(KEEL_AT.z * published(&telemetry, "lateral.keel.side_force")
            + RUDDER_AT.z * published(&telemetry, "lateral.rudder.side_force"));
        assert_relative_eq!(wrench.moment.x, expected, max_relative = 1e-12);
    }

    #[test]
    fn resistance_opposes_the_track() {
        let heel = 20.0_f64.to_radians();
        let sailing = state(SPEED, 4.0_f64.to_radians(), heel);
        let (wrench, telemetry) = run(&sailing, 0.0);

        // Induced resistance exists at this operating point, so this is not a
        // test of the residuary term alone.
        assert!(published(&telemetry, "lateral.keel.induced_resistance") > 0.0);
        assert!(published(&telemetry, "lateral.rudder.induced_resistance") > 0.0);

        // Strip the side forces off and what remains must be the total
        // resistance pointing exactly back along the track.
        let side = Vector3::new(0.0, published(&telemetry, "lateral.side_force_total"), 0.0);
        let drag = wrench.force - side;
        let world = sailing.world_velocity();
        let track = sailing
            .to_body(Vector3::new(world.x, world.y, 0.0))
            .normalize();
        let expected = -published(&telemetry, "lateral.resistance_total") * track;

        assert_relative_eq!(drag.x, expected.x, max_relative = 1e-9);
        assert_relative_eq!(drag.y, expected.y, max_relative = 1e-9);
        assert_relative_eq!(drag.z, expected.z, max_relative = 1e-9);
        assert!(drag.dot(&track) < 0.0, "drag must oppose the track");
    }

    #[test]
    fn upright_with_no_leeway_the_only_force_is_drag_straight_aft() {
        let (wrench, telemetry) = run(&state(SPEED, 0.0, 0.0), 0.0);

        assert_relative_eq!(wrench.force.y, 0.0, epsilon = 1e-12);
        assert_relative_eq!(wrench.force.z, 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            wrench.force.x,
            -published(&telemetry, "lateral.resistance_total"),
            max_relative = 1e-12
        );
        assert!(wrench.force.x < 0.0);
        // No lift, so no induced resistance: what is left is the keel's own
        // wave making, which the boat pays for at this Froude number.
        assert_relative_eq!(
            published(&telemetry, "lateral.keel.induced_resistance"),
            0.0,
            epsilon = 1e-12
        );
        assert!(published(&telemetry, "lateral.keel.residuary") > 0.0);
        assert_relative_eq!(wrench.moment.x, 0.0, epsilon = 1e-12);
        assert_relative_eq!(wrench.moment.z, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn a_boat_at_rest_produces_nothing_rather_than_a_nan() {
        // The track direction is undefined at rest; dividing by the speed
        // anyway would poison every force downstream with a NaN.
        let (wrench, telemetry) = run(&BodyState::default(), 5.0_f64.to_radians());

        assert_eq!(wrench, Wrench::zero());
        for (key, value) in telemetry.iter() {
            assert!(value.is_finite(), "{key} is not finite: {value}");
        }
    }

    #[test]
    fn nothing_is_published_before_the_first_step() {
        // A module that reported zeros here would be claiming a measurement it
        // has not made.
        let module = system();
        let mut telemetry = Telemetry::new();
        module.telemetry(&mut telemetry);
        assert!(telemetry.is_empty());
    }
}
