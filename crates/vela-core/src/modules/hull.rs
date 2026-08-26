//! Canoe body resistance as a force module.
//!
//! # Provenance
//!
//! This file contains no physics. Every number it produces comes out of
//! [`crate::dsyhs`], which transcribes the Delft Systematic Yacht Hull Series
//! from *Principles of Yacht Design*, 5th ed. — including its Froude envelope
//! behaviour, its low-speed taper and its `0.7 × Lwl` friction length. Those
//! decisions are documented and argued there and are deliberately not
//! re-litigated here: this module's whole job is to turn one scalar resistance
//! into a body-frame wrench with the right direction.
//!
//! # Scope
//!
//! The canoe body only: friction, viscous pressure, upright residuary, and the
//! change of residuary with heel. Appendage resistance — induced drag, the
//! keel's own residuary and its change with heel — belongs to
//! [`crate::modules::lateral`], and adding any of it here would double count a
//! component that is invisible in a total.
//!
//! # Direction
//!
//! Resistance opposes the motion of the hull **through the water**, which is
//! along the track and not along the centreline. At leeway those differ by a
//! few degrees, so the force acquires a small athwartships component; dropping
//! it would silently credit the boat with side force it has not got.
//!
//! The track direction is built from [`StepCtx::leeway`], never from
//! `state.velocity`. Leeway carries a sign convention that this engine defines
//! in exactly one place, and a module with its own version of it is free to
//! pick the other sign while still looking entirely plausible.
//!
//! # Known gaps
//!
//! - **No application point.** The DSYHS regressions report a force and nothing
//!   about where it acts, so there is no sourced centre of effort to apply it
//!   at. The force is therefore applied at the body origin and produces **no
//!   moment at all**: the resistance-induced bow-down trim moment of a real
//!   hull is missing, and so is the yaw moment that the athwartships component
//!   would exert about a point off the origin. Inventing a plausible-looking
//!   point — a fraction of the waterline length, say — would put an unsourced
//!   term under every force this module produces, so the gap is recorded
//!   instead.
//! - **Steady water only.** The series is a calm-water regression evaluated
//!   quasi-statically each step. There is no added resistance in waves and no
//!   memory of the hull's own wake.
//! - **The envelope is not checked here.** A hull outside the series' parameter
//!   range makes these polynomials diverge; see
//!   [`HullParameters::check_envelope`], which the boat-loading path is the
//!   right place to call.

use crate::dsyhs::{hull_resistance, HullParameters, HullResistance as ResistanceComponents};
use crate::sim::{ForceModule, StepCtx};
use crate::telemetry::Telemetry;
use crate::wrench::Wrench;
use nalgebra::Vector3;

/// Canoe body resistance, evaluated from the DSYHS regressions each step.
pub struct CanoeBody {
    /// The hull as the regressions see it: nine form parameters, fixed for the
    /// run.
    parameters: HullParameters,
    /// Breakdown from the last [`ForceModule::step`].
    ///
    /// Cached rather than recomputed because [`ForceModule::telemetry`] takes
    /// `&self` and must report what was actually applied — recomputing there
    /// would be free to report a different number from the one integrated.
    components: ResistanceComponents,
    /// Froude number of the last step, published as telemetry because it is
    /// what tells a reader which row of the coefficient tables the numbers
    /// above came from.
    froude: f64,
}

impl CanoeBody {
    /// Wraps a set of hull form parameters as a force module.
    ///
    /// Takes no application point, no correction factor and no reference area:
    /// everything else this module needs is in `parameters` or in the
    /// environment, which is the reason it can be constructed from one
    /// argument.
    #[must_use]
    pub fn new(parameters: HullParameters) -> Self {
        Self {
            parameters,
            components: ResistanceComponents {
                friction: 0.0,
                viscous_pressure: 0.0,
                residuary: 0.0,
                heel_residuary: 0.0,
            },
            froude: 0.0,
        }
    }
}

impl ForceModule for CanoeBody {
    fn name(&self) -> &'static str {
        "hull"
    }

    fn step(&mut self, ctx: &StepCtx<'_>) -> Wrench {
        let water = ctx.env.water();
        let gravity = ctx.env.gravity();
        let speed = ctx.speed_through_water();

        self.froude = self.parameters.froude(speed, gravity);
        self.components =
            hull_resistance(&self.parameters, speed, ctx.heel(), water.density, gravity);

        // A hull at rest has no resistance *and* no direction of motion. Both
        // facts matter: the components above are already zero, but the track
        // direction would be a normalized zero vector, so it is never formed.
        if speed <= 0.0 {
            return Wrench::zero();
        }

        // The track in the body frame. Leeway is the angle from the centreline
        // to the track, positive when the boat needs side force to starboard —
        // which is the boat slipping to port, hence the negative `y`. This is
        // the one place the athwartships share of the resistance is decided.
        let leeway = ctx.leeway();
        let track = Vector3::new(leeway.cos(), -leeway.sin(), 0.0);

        // Applied at the body origin: see the module note on the missing
        // centre of effort. `from_force_at` states that choice rather than
        // leaving a bare zero moment for a reader to interpret.
        Wrench::from_force_at(-self.components.total() * track, Vector3::zeros())
    }

    fn telemetry(&self, out: &mut Telemetry) {
        // One key per component the model actually returns, so the four sum to
        // the total and a reader can see which one is hurting. Keys are public
        // API: extend, never rename.
        out.set("hull.friction", self.components.friction);
        out.set("hull.viscous_pressure", self.components.viscous_pressure);
        out.set("hull.residuary", self.components.residuary);
        out.set("hull.heel_residuary", self.components.heel_residuary);
        out.set("hull.total", self.components.total());
        out.set("hull.froude", self.froude);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aero::SailSet;
    use crate::controls::Controls;
    use crate::dsyhs::heel_residuary_delta;
    use crate::env::{Environment, StillWater, UniformWind};
    use crate::state::BodyState;
    use approx::assert_relative_eq;
    use nalgebra::UnitQuaternion;

    /// YD-41 at half-loaded displacement — the design yacht of the series' own
    /// textbook, the same particulars the `dsyhs` validation tests use
    /// (*Principles of Yacht Design*, 5th ed., Appendix 1). A sourced hull
    /// rather than invented numbers, and one known to sit inside the envelope.
    fn yd41() -> HullParameters {
        HullParameters {
            waterline_length: 11.90,
            waterline_beam: 3.18,
            canoe_draft: 0.40,
            canoe_volume: 6.05,
            wetted_surface: 28.20,
            waterplane_area: 26.75,
            prismatic: 0.56,
            midship: 0.715,
            lcb: -0.042,
            lcf: -0.073,
        }
    }

    fn environment() -> StillWater {
        StillWater::new(UniformWind::uniform(8.0, 0.0))
    }

    /// The module ignores the controls entirely; something valid is still
    /// needed to build a context.
    fn controls() -> Controls {
        Controls::close_hauled(SailSet::upwind())
    }

    /// Runs `use_ctx` against a context built from a body-frame velocity and a
    /// heel angle.
    ///
    /// Handing the context to the caller rather than only the wrench lets a
    /// test ask [`StepCtx`] for the leeway it is about to check the force
    /// against, instead of restating the sign convention a second time.
    fn with_ctx<R>(
        velocity: Vector3<f64>,
        heel: f64,
        use_ctx: impl FnOnce(&StepCtx<'_>) -> R,
    ) -> R {
        let state = BodyState {
            attitude: UnitQuaternion::from_euler_angles(heel, 0.0, 0.0),
            velocity,
            ..BodyState::default()
        };
        let env = environment();
        let controls = controls();
        let ctx = StepCtx {
            state: &state,
            controls: &controls,
            env: &env,
            time: 0.0,
            dt: 0.01,
        };
        use_ctx(&ctx)
    }

    /// Steps the module once at a body-frame velocity and heel angle.
    fn step(module: &mut CanoeBody, velocity: Vector3<f64>, heel: f64) -> Wrench {
        with_ctx(velocity, heel, |ctx| module.step(ctx))
    }

    #[test]
    fn resistance_opposes_a_boat_moving_ahead() {
        let mut module = CanoeBody::new(yd41());
        let wrench = step(&mut module, Vector3::new(5.0, 0.0, 0.0), 0.0);

        assert!(
            wrench.force.x < 0.0,
            "resistance must retard the boat, got {}",
            wrench.force.x
        );
        // Dead upright, no leeway: nothing athwartships and nothing vertical.
        assert_relative_eq!(wrench.force.y, 0.0, epsilon = 1e-12);
        assert_relative_eq!(wrench.force.z, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn the_force_acts_at_the_body_origin_and_makes_no_moment() {
        // The stated gap, asserted so that a later change which quietly invents
        // an application point fails here rather than in a trim reading.
        let mut module = CanoeBody::new(yd41());
        let wrench = step(&mut module, Vector3::new(5.0, -0.3, 0.0), 0.2);

        assert_relative_eq!(wrench.moment.norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn a_boat_at_rest_has_exactly_no_resistance_and_no_nan() {
        let mut module = CanoeBody::new(yd41());
        let wrench = step(&mut module, Vector3::zeros(), 0.0);

        assert!(
            wrench
                .force
                .iter()
                .chain(wrench.moment.iter())
                .all(|c| c.is_finite()),
            "a boat at rest must not produce a NaN, got {wrench:?}"
        );
        assert_relative_eq!(wrench.force.norm(), 0.0);
        assert_relative_eq!(wrench.moment.norm(), 0.0);
    }

    #[test]
    fn heave_alone_is_not_progress_and_makes_no_resistance() {
        // Speed through the water is horizontal by definition, so a hull rising
        // on a wave has no resistance to show for it.
        let mut module = CanoeBody::new(yd41());
        let wrench = step(&mut module, Vector3::new(0.0, 0.0, -1.0), 0.0);

        assert_relative_eq!(wrench.force.norm(), 0.0);
    }

    #[test]
    fn at_leeway_the_force_is_antiparallel_to_the_track() {
        // Sailing forward while sliding to port: positive leeway by this
        // engine's convention, so the resistance leans to starboard, opposing
        // the slip.
        let mut module = CanoeBody::new(yd41());
        let velocity = Vector3::new(5.0, -0.3, 0.0);
        let (wrench, leeway) = with_ctx(velocity, 0.0, |ctx| (module.step(ctx), ctx.leeway()));

        assert!(leeway > 0.0, "the fixture must actually be at leeway");
        assert!(wrench.force.x < 0.0, "still mostly retarding");
        assert!(
            wrench.force.y > 0.0,
            "resistance must oppose a slip to port, got {}",
            wrench.force.y
        );

        // Anti-parallel to the track: the cross product with the track
        // direction vanishes, and the athwartships share is exactly the tangent
        // of the leeway angle the context reports.
        let track = Vector3::new(leeway.cos(), -leeway.sin(), 0.0);
        assert_relative_eq!(wrench.force.cross(&track).norm(), 0.0, epsilon = 1e-9);
        assert_relative_eq!(
            wrench.force.y / -wrench.force.x,
            leeway.tan(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn heel_moves_the_total_the_way_the_model_says() {
        let hull = yd41();
        let env = environment();
        let heel = 20.0_f64.to_radians(); // The angle the Fig 5.22 table is quoted at.
        let speed = 5.0;

        let expected = heel_residuary_delta(&hull, speed, heel, env.water().density, env.gravity());
        assert!(
            expected != 0.0,
            "the test is only meaningful if heel changes the model's answer"
        );

        let mut module = CanoeBody::new(hull);
        let upright = step(&mut module, Vector3::new(speed, 0.0, 0.0), 0.0);
        let heeled = step(&mut module, Vector3::new(speed, 0.0, 0.0), heel);

        // Both forces point dead aft, so the change in total resistance is the
        // change in the surge component, with the sign the wrapper gives it.
        assert_relative_eq!(
            upright.force.x - heeled.force.x,
            expected,
            max_relative = 1e-9
        );
        if expected > 0.0 {
            assert!(heeled.force.norm() > upright.force.norm());
        } else {
            assert!(heeled.force.norm() < upright.force.norm());
        }
    }

    #[test]
    fn heel_is_read_from_the_context_not_from_the_velocity() {
        // A heeled boat with the same speed through the water must differ from
        // an upright one; if `heel()` were dropped, these would be identical.
        let mut module = CanoeBody::new(yd41());
        let upright = step(&mut module, Vector3::new(5.0, 0.0, 0.0), 0.0);
        let heeled = step(&mut module, Vector3::new(5.0, 0.0, 0.0), 0.4);

        assert!((heeled.force.x - upright.force.x).abs() > 0.0);
    }

    #[test]
    fn telemetry_components_sum_to_the_reported_total() {
        let hull = yd41();
        let env = environment();
        let speed = 5.0;

        let mut module = CanoeBody::new(hull);
        let wrench = step(&mut module, Vector3::new(speed, 0.0, 0.0), 0.3);

        let mut out = Telemetry::new();
        module.telemetry(&mut out);

        let value = |key: &str| out.get(key).unwrap_or_else(|| panic!("missing {key}"));
        let parts = value("hull.friction")
            + value("hull.viscous_pressure")
            + value("hull.residuary")
            + value("hull.heel_residuary");
        let total = value("hull.total");

        assert_relative_eq!(parts, total, max_relative = 1e-12);
        // And the total is what was actually applied, not a second opinion.
        assert_relative_eq!(wrench.force.norm(), total, max_relative = 1e-12);
        assert_relative_eq!(
            value("hull.froude"),
            yd41().froude(speed, env.gravity()),
            max_relative = 1e-12
        );
    }

    #[test]
    fn telemetry_before_the_first_step_reports_zero_rather_than_stale_numbers() {
        let module = CanoeBody::new(yd41());
        let mut out = Telemetry::new();
        module.telemetry(&mut out);

        assert_relative_eq!(out.get("hull.total").expect("published"), 0.0);
        assert_relative_eq!(out.get("hull.froude").expect("published"), 0.0);
    }

    #[test]
    fn the_module_name_is_the_telemetry_prefix() {
        // The keys above are a public contract and are prefixed by this name;
        // they must not drift apart.
        assert_eq!(CanoeBody::new(yd41()).name(), "hull");
    }
}
