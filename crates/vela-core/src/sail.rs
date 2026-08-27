//! A sail: geometry in, wrench out.
//!
//! # What this module is
//!
//! The assembly of phase 2. [`crate::flying`] turns controls into a shape,
//! [`crate::vlm`] turns a shape into panel forces, [`crate::stall`] corrects them
//! where potential flow has run out of validity. This module is the one that owns
//! all three and answers the question a boat actually asks: given this trim and
//! this wind, what force and what moment.
//!
//! Named [`Sail`] rather than something more elaborate, and distinct from
//! [`crate::aero::Sail`], which is an enum naming *which* sail. This is a model of
//! one.
//!
//! # Two costs, deliberately visible
//!
//! Building a `Sail` factorises an influence matrix — fifteen milliseconds at three
//! hundred panels, measured. Asking one for a wrench solves a right-hand side —
//! 0.15 ms, a hundred times less. Both numbers are in [`crate::vlm`]'s own
//! documentation and they are the reason this type exists as a *thing you keep*
//! rather than a function you call.
//!
//! So the expensive operation is a separate method with a name that says so
//! ([`Sail::retrim`]), and the type will tell a caller when it is needed
//! ([`Sail::wake_drift`]) rather than deciding for it. The alternative — noticing
//! internally that the wind angle moved and quietly refactorising — would put a
//! fifteen-millisecond stall inside a sixty-hertz loop at a moment nobody chose.
//!
//! # The stall anchor rides on the working factorisation
//!
//! [`crate::stall::Separated`] has to be anchored on the attached model's own lift
//! and drag *at the stall angle*, which is a different attitude from the one being
//! sailed. Done properly that is a second factorisation, doubling the cost of every
//! trim change.
//!
//! It is done improperly on purpose: the anchor is evaluated by changing the onset
//! direction while keeping the wake the current factorisation was built for.
//! [`crate::vlm`]'s own measurement of that approximation is under one per cent
//! below six degrees and a few per cent by twelve; at a stall angle near twenty it
//! is a handful. The quantity it perturbs is the anchor of an empirical branch whose
//! cross-tunnel uncertainty is ten to fifteen per cent, so the error is an order
//! below the noise it enters. Paying a second `O(N³)` for it would be buying
//! precision in the one place it cannot be spent.
//!
//! # How the blend becomes a wrench
//!
//! [`crate::stall`] corrects *coefficients*; a boat needs a force and a moment. The
//! conversion could be done on the totals, but then the moment needs a point to act
//! at, and the centre of effort past stall is not something this engine knows.
//!
//! Instead the correction is applied to **every panel**, as the same pair of ratios:
//! each panel's force is split into its along-flow part, scaled by the drag ratio,
//! and everything else, scaled by the lift ratio. Summing gives exactly the blended
//! lift and drag, and the moment comes out of the panels that produced it — so the
//! centre of effort is the lattice's, by construction, with nothing assumed about
//! where it goes.
//!
//! That the *vertical* force scales with lift rather than with drag is a choice and
//! is the right one: it is lift, resolved onto a different axis, and it comes from
//! the same circulation.
//!
//! # What it does not do
//!
//! One sail. A rig's sails interact — a headsail's downwash is what makes a
//! mainsail's leading edge work — and that coupling belongs to whatever assembles
//! several of these, in one lattice, because the interaction is between
//! circulations and not between forces.
//!
//! Uniform onset flow. The lattice takes an arbitrary field and would carry wind
//! shear for free, but the shear profile belongs to a wind model that is not built,
//! and half of one here would be a number nobody chose.

use nalgebra::Vector3;

use crate::flying::{Controls, Planform, Response, Shape};
use crate::geometry::Point;
use crate::stall::{Blend, Separated};
use crate::vlm::Solver;

/// Below this wind speed a sail makes no force worth computing.
const STILL_AIR: f64 = 1e-6;

/// How finely to resolve a sail, and where it stalls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Options {
    /// Panels along the chord and up the sail.
    ///
    /// Ten by twenty-four is the working default: 240 panels, a 0.15 ms solve, and
    /// a lift slope within a per cent of what forty-eight spanwise panels give.
    pub panels: (usize, usize),
    /// Mean incidence at which separation begins, radians.
    ///
    /// A property of the sail — camber, Reynolds number, leading-edge geometry,
    /// cloth — with no closed form. The literature's anchors are about 17° for a
    /// twist-free rigid model and 20° for a twisting full-scale sail.
    pub stall: f64,
    /// Width of the handover onto the separated branch, radians.
    ///
    /// A smoothing scale and not a measurement; see [`crate::stall`].
    pub blend_width: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            panels: (10, 24),
            stall: 18.0_f64.to_radians(),
            blend_width: 8.0_f64.to_radians(),
        }
    }
}

/// What a sail produced at one wind.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wrench {
    /// Force, N, in the file frame.
    pub force: Vector3<f64>,
    /// Moment, N·m, about the point it was asked for.
    pub moment: Vector3<f64>,
    /// Lift coefficient on planform area, after blending.
    pub lift_coefficient: f64,
    /// Drag coefficient on planform area, after blending. Induced plus whatever
    /// the separated branch adds; there is no viscous term here.
    pub drag_coefficient: f64,
    /// Lift coefficient the lattice alone gave, before blending.
    ///
    /// Carried because the difference between this and the blended one is the whole
    /// empirical content of the answer, and a caller debugging a polar needs to see
    /// which of the two moved.
    pub attached_lift_coefficient: f64,
    /// Weight the separated branch was given, 0 to 1.
    pub separated_fraction: f64,
    /// Area-weighted mean incidence, radians — what the blend was read against.
    pub mean_incidence: f64,
    /// Height of the centre of effort above the sail's tack, m.
    ///
    /// Rolling moment over horizontal force, which is the convention the
    /// wind-tunnel literature reports and therefore the one that can be compared
    /// against it.
    pub centre_of_effort: f64,
}

/// A sail: a shape, a factorised lattice, and a handover to separated flow.
#[derive(Debug, Clone)]
pub struct Sail {
    planform: Planform,
    response: Response,
    options: Options,
    shape: Shape,
    solver: Solver,
    blend: Blend,
    wake: Vector3<f64>,
}

impl Sail {
    /// Builds a sail at a trim, factorised for a wake direction.
    ///
    /// `wake` is the direction the trailing vorticity leaves along, which is the
    /// mean flow's — so in practice the apparent wind the sail is expected to work
    /// in. It is baked into the factorisation; see the module documentation.
    ///
    /// Returns `None` for a degenerate wake direction, a panel count with a zero in
    /// it, a stall angle outside the first quadrant, or a handover that would run
    /// past ninety degrees.
    #[must_use]
    pub fn new(
        planform: Planform,
        response: Response,
        controls: Controls,
        wake: Vector3<f64>,
        options: Options,
    ) -> Option<Self> {
        let shape = response.shape(planform, controls);
        let solver = Solver::new(shape.lattice(options.panels.0, options.panels.1)?, wake)?;
        let blend = Self::fit_blend(&planform, &shape, &solver, wake, &options)?;
        Some(Self {
            planform,
            response,
            options,
            shape,
            solver,
            blend,
            wake,
        })
    }

    /// Anchors the separated branch on the attached model at the stall attitude.
    ///
    /// Evaluated on the caller's factorisation rather than a fresh one — the trade
    /// the module documentation states and prices.
    fn fit_blend(
        planform: &Planform,
        shape: &Shape,
        solver: &Solver,
        wake: Vector3<f64>,
        options: &Options,
    ) -> Option<Blend> {
        let speed = wake.norm();
        if speed < STILL_AIR {
            return None;
        }
        // The onset that puts the sail's *mean* incidence at the stall angle. The
        // mean is linear in the wind angle, so one evaluation locates it — and the
        // direction of the turn is the trap: rotating the flow by `+θ` about `z`
        // *reduces* the incidence by `θ`, because the chord and the flow are
        // compared the other way round. So the turn is `mean - stall`, not
        // `stall - mean`, and the sign is asserted rather than reasoned about in
        // `the_anchor_sits_at_the_stall_angle`.
        let reference = wake / speed;
        let turn = shape.mean_incidence(reference) - options.stall;
        let (sine, cosine) = turn.sin_cos();
        let stalled = Vector3::new(
            reference.x * cosine - reference.y * sine,
            reference.x * sine + reference.y * cosine,
            reference.z,
        );

        let onset = stalled * speed;
        let solution = solver.solve(|_| onset, 1.0)?;
        let (lift, drag) = Self::split(solution.force(), onset, planform, 1.0);
        Separated::fit(planform.aspect_ratio(), options.stall, lift, drag)
            .and_then(|separated| Blend::new(separated, options.blend_width))
    }

    /// Lift and drag coefficients of a force, on planform area.
    ///
    /// Lift across the flow and horizontal, toward leeward; drag along it. The
    /// vertical component belongs to neither and is not returned — it is carried
    /// through the per-panel scaling instead, which is where it can be kept
    /// consistent with the lift it came from.
    fn split(
        force: Vector3<f64>,
        onset: Vector3<f64>,
        planform: &Planform,
        density: f64,
    ) -> (f64, f64) {
        let speed = onset.norm();
        let along = onset / speed;
        let across = Self::leeward(along);
        let dynamic = 0.5 * density * speed * speed * planform.area();
        (force.dot(&across) / dynamic, force.dot(&along) / dynamic)
    }

    /// The horizontal unit vector across a flow, pointing to leeward.
    ///
    /// The flow turned a quarter turn the way a starboard-tack sail's camber
    /// bulges, so that a positive lift coefficient is a sail pulling to leeward and
    /// not a sign convention waiting to be discovered.
    fn leeward(along: Vector3<f64>) -> Vector3<f64> {
        let horizontal = Vector3::new(along.x, along.y, 0.0);
        let norm = horizontal.norm();
        if norm < STILL_AIR {
            return Vector3::y();
        }
        Vector3::new(horizontal.y, -horizontal.x, 0.0) / norm
    }

    /// The shape the controls left.
    #[must_use]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// The wake direction baked into the factorisation.
    #[must_use]
    pub fn wake(&self) -> Vector3<f64> {
        self.wake
    }

    /// How far a wind has drifted from the baked wake, radians.
    ///
    /// The number a caller watches to decide when to pay for [`Sail::retrim`]. The
    /// cost of *not* paying is measured in [`crate::vlm`]: under a per cent of lift
    /// below six degrees of drift, a few per cent by twelve, six by twenty-five.
    #[must_use]
    pub fn wake_drift(&self, wind: Vector3<f64>) -> f64 {
        let (baked, current) = (self.wake.norm(), wind.norm());
        if baked < STILL_AIR || current < STILL_AIR {
            return 0.0;
        }
        (self.wake.dot(&wind) / (baked * current))
            .clamp(-1.0, 1.0)
            .acos()
    }

    /// Rebuilds the shape and the factorisation. **Expensive** — `O(N³)` plus the
    /// `N²` ring evaluations that dominate it.
    ///
    /// Both arguments, because the two reasons to rebuild are a trim change and a
    /// wake that has drifted, and a caller doing one will usually want the other.
    /// Returns `None` and leaves the sail untouched if the new geometry is
    /// degenerate, so a failed retrim cannot leave a sail unusable.
    pub fn retrim(&mut self, controls: Controls, wake: Vector3<f64>) -> Option<()> {
        let rebuilt = Self::new(self.planform, self.response, controls, wake, self.options)?;
        *self = rebuilt;
        Some(())
    }

    /// The wrench at a wind, about a point. **Cheap** — one right-hand side.
    ///
    /// `wind` is the air's velocity in the file frame, so a wind blowing aft is
    /// negative in `x`. Returns a zero wrench in still air rather than a division
    /// by a vanishing speed.
    #[must_use]
    pub fn wrench(&self, wind: Vector3<f64>, density: f64, about: Point) -> Wrench {
        let speed = wind.norm();
        let mean_incidence = self.shape.mean_incidence(wind);
        if speed < STILL_AIR || !density.is_finite() || density <= 0.0 {
            return Wrench {
                force: Vector3::zeros(),
                moment: Vector3::zeros(),
                lift_coefficient: 0.0,
                drag_coefficient: 0.0,
                attached_lift_coefficient: 0.0,
                separated_fraction: self.blend.weight(mean_incidence),
                mean_incidence,
                centre_of_effort: 0.0,
            };
        }

        let Some(solution) = self.solver.solve(|_| wind, density) else {
            return Wrench {
                force: Vector3::zeros(),
                moment: Vector3::zeros(),
                lift_coefficient: 0.0,
                drag_coefficient: 0.0,
                attached_lift_coefficient: 0.0,
                separated_fraction: self.blend.weight(mean_incidence),
                mean_incidence,
                centre_of_effort: 0.0,
            };
        };

        let attached = Self::split(solution.force(), wind, &self.planform, density);
        let blended = self.blend.coefficients(attached, mean_incidence);

        // The same two ratios on every panel: along-flow scaled by drag, everything
        // else by lift. Summing reproduces the blended coefficients exactly and
        // leaves the centre of effort where the lattice put it.
        let ratio = |blended: f64, attached: f64| {
            if attached.abs() < 1e-12 {
                1.0
            } else {
                blended / attached
            }
        };
        let (lift_ratio, drag_ratio) = (ratio(blended.0, attached.0), ratio(blended.1, attached.1));

        let along = wind / speed;
        let mut force = Vector3::zeros();
        let mut moment = Vector3::zeros();
        for (panel, &at) in solution.forces().iter().zip(solution.positions()) {
            let streamwise = along * panel.dot(&along);
            let corrected = streamwise * drag_ratio + (panel - streamwise) * lift_ratio;
            force += corrected;
            moment += (at - about).cross(&corrected);
        }

        let horizontal = force.x.hypot(force.y);
        let centre_of_effort = if horizontal < STILL_AIR {
            0.0
        } else {
            moment.x.hypot(moment.y) / horizontal
        };

        Wrench {
            force,
            moment,
            lift_coefficient: blended.0,
            drag_coefficient: blended.1,
            attached_lift_coefficient: attached.0,
            separated_fraction: self.blend.weight(mean_incidence),
            mean_incidence,
            centre_of_effort,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    use crate::flying::Distribution;

    const AIR: f64 = 1.225;
    const SPEED: f64 = 8.0;

    fn planform() -> Planform {
        Planform::new(5.185, 2.0, 0.10, 0.387).expect("a real sail's dimensions")
    }

    fn response() -> Response {
        Response::new(
            (25.0_f64.to_radians(), 4.0_f64.to_radians()),
            (22.0_f64.to_radians(), 3.0_f64.to_radians()),
            (0.16, 0.08),
            0.55,
            (0.48, 0.34),
        )
        .expect("a real sail's travel")
    }

    /// A wind of `SPEED` blowing from `degrees` off the bow, on starboard tack.
    fn wind(degrees: f64) -> Vector3<f64> {
        let beta = degrees.to_radians();
        Vector3::new(-SPEED * beta.cos(), SPEED * beta.sin(), 0.0)
    }

    fn sail(controls: Controls, at: f64) -> Sail {
        Sail::new(
            planform(),
            response(),
            controls,
            wind(at),
            Options::default(),
        )
        .expect("a real sail")
    }

    /// The wrench's coefficients are the ones the panels sum to.
    ///
    /// The load-bearing property of the per-panel correction: the blend is defined
    /// on totals and applied to parts, and the parts have to add back up. If they
    /// did not, the reported coefficients and the force actually fed to the boat
    /// would be different numbers, and every polar would be a lie the telemetry
    /// could not catch.
    #[test]
    fn the_panels_sum_to_the_reported_coefficients() {
        for at in [8.0_f64, 16.0, 24.0, 40.0] {
            let sail = sail(Controls::HARD, at);
            let wrench = sail.wrench(wind(at), AIR, Point::zeros());
            let (lift, drag) = Sail::split(wrench.force, wind(at), &planform(), AIR);
            assert_relative_eq!(lift, wrench.lift_coefficient, max_relative = 1e-10);
            assert_relative_eq!(drag, wrench.drag_coefficient, max_relative = 1e-10);
        }
    }

    /// The correction leaves the centre of effort where the lattice put it.
    ///
    /// A uniform pair of ratios cannot move it far, and this pins how far: the
    /// blended and unblended centres agree to well under a per cent even where the
    /// separated branch has taken over most of the force. That is the reason for
    /// correcting panels rather than totals — a total needs a point to act at, and
    /// the centre of effort past stall is not something this engine knows.
    #[test]
    fn the_correction_does_not_move_the_centre_of_effort() {
        let at = 30.0;
        let sail = sail(Controls::HARD, at);
        let blended = sail.wrench(wind(at), AIR, Point::zeros());
        assert!(
            blended.separated_fraction > 0.9,
            "this test is not exercising the branch: {}",
            blended.separated_fraction
        );

        // The lattice's own centre of effort, from an unblended solve.
        let solution = sail.solver.solve(|_| wind(at), AIR).expect("a sail solves");
        let force = solution.force();
        let moment = solution.moment_about(Point::zeros());
        let raw = moment.x.hypot(moment.y) / force.x.hypot(force.y);

        assert_relative_eq!(blended.centre_of_effort, raw, max_relative = 0.01);
        // And it is a sensible height on this sail.
        assert!((0.3..0.6).contains(&(blended.centre_of_effort / 5.185)));
    }

    /// Forces scale with dynamic pressure, and coefficients do not.
    ///
    /// The definition of a coefficient, and the check that the reference area and
    /// the density are being applied once each.
    #[test]
    fn forces_scale_with_dynamic_pressure_and_coefficients_do_not() {
        let sail = sail(Controls::HARD, 20.0);
        let base = sail.wrench(wind(20.0), AIR, Point::zeros());
        let faster = sail.wrench(wind(20.0) * 2.0, AIR, Point::zeros());
        let denser = sail.wrench(wind(20.0), AIR * 3.0, Point::zeros());

        for k in 0..3 {
            assert_relative_eq!(faster.force[k], 4.0 * base.force[k], max_relative = 1e-9);
            assert_relative_eq!(denser.force[k], 3.0 * base.force[k], max_relative = 1e-9);
        }
        assert_relative_eq!(
            faster.lift_coefficient,
            base.lift_coefficient,
            max_relative = 1e-9
        );
        assert_relative_eq!(
            denser.drag_coefficient,
            base.drag_coefficient,
            max_relative = 1e-9
        );
    }

    /// Still air makes no force, and does not divide by zero doing it.
    #[test]
    fn still_air_makes_no_force() {
        let sail = sail(Controls::HARD, 20.0);
        for wind in [Vector3::zeros(), Vector3::new(1e-15, 0.0, 0.0)] {
            let wrench = sail.wrench(wind, AIR, Point::zeros());
            assert_eq!(wrench.force, Vector3::zeros());
            assert_eq!(wrench.moment, Vector3::zeros());
            assert!(wrench.lift_coefficient.is_finite());
            assert!(wrench.centre_of_effort.is_finite());
        }
        // A nonsense density is refused the same way rather than propagated.
        let wrench = sail.wrench(wind(20.0), 0.0, Point::zeros());
        assert_eq!(wrench.force, Vector3::zeros());
    }

    /// The moment transforms with its reference point.
    ///
    /// `M(b) = M(a) + (a - b) × F`, through the per-panel correction. Pinned because
    /// the heeling moment is the sail's most consequential output and it is taken
    /// about whatever point the boat hands over.
    #[test]
    fn the_moment_transforms_with_its_reference_point() {
        let sail = sail(Controls::HARD, 22.0);
        let a = Point::new(0.3, -0.2, 1.1);
        let b = Point::new(-1.7, 2.4, -0.6);
        let at_a = sail.wrench(wind(22.0), AIR, a);
        let at_b = sail.wrench(wind(22.0), AIR, b);
        let shifted = at_a.moment + (a - b).cross(&at_a.force);
        for k in 0..3 {
            assert_relative_eq!(at_b.moment[k], shifted[k], max_relative = 1e-9);
        }
    }

    /// The drift from the baked wake is measured, and refactorising removes it.
    ///
    /// The cost this type refuses to hide. A caller that sails away from the angle
    /// its sail was factorised at gets a number saying how far, and a method that
    /// costs `O(N³)` to fix it.
    #[test]
    fn the_wake_drift_is_reported_and_retrimming_clears_it() {
        let mut sail = sail(Controls::HARD, 20.0);
        assert_relative_eq!(sail.wake_drift(wind(20.0)), 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            sail.wake_drift(wind(32.0)),
            12.0_f64.to_radians(),
            max_relative = 1e-9
        );
        // Symmetric, and blind to speed.
        assert_relative_eq!(
            sail.wake_drift(wind(8.0)),
            12.0_f64.to_radians(),
            max_relative = 1e-9
        );
        assert_relative_eq!(
            sail.wake_drift(wind(32.0) * 5.0),
            sail.wake_drift(wind(32.0)),
            max_relative = 1e-12
        );

        sail.retrim(Controls::HARD, wind(32.0))
            .expect("a real retrim");
        assert_relative_eq!(sail.wake_drift(wind(32.0)), 0.0, epsilon = 1e-12);

        // And the drift costs what the lattice says it costs. Measured on the
        // *attached* coefficient, which is the one the wake approximation acts on:
        // 1.2 % at three degrees, 2.5 at six, 5.5 at twelve — the same shape
        // `crate::vlm` reports for an ellipse, arriving on a real sail.
        let mut stale = sail.clone();
        stale
            .retrim(Controls::HARD, wind(20.0))
            .expect("a real retrim");
        let mut worst: f64 = 0.0;
        for at in [23.0_f64, 26.0, 32.0] {
            let fresh = super::tests::sail(Controls::HARD, at);
            let aligned = fresh
                .wrench(wind(at), AIR, Point::zeros())
                .attached_lift_coefficient;
            let drifted = stale
                .wrench(wind(at), AIR, Point::zeros())
                .attached_lift_coefficient;
            let error = (drifted / aligned - 1.0).abs();
            assert!(
                error > worst,
                "the cost of drift is not growing: {error:.4}"
            );
            worst = error;
        }
        assert!(
            (0.03..0.09).contains(&worst),
            "twelve degrees of wake drift cost {:.1} % of the attached lift",
            100.0 * worst
        );
    }

    /// A failed retrim leaves the sail usable.
    #[test]
    fn a_failed_retrim_changes_nothing() {
        let mut sail = sail(Controls::HARD, 20.0);
        let before = sail.wrench(wind(20.0), AIR, Point::zeros());
        assert!(sail.retrim(Controls::HARD, Vector3::zeros()).is_none());
        let after = sail.wrench(wind(20.0), AIR, Point::zeros());
        assert_eq!(before.force, after.force);
        assert_relative_eq!(sail.wake_drift(wind(20.0)), 0.0, epsilon = 1e-12);
    }

    /// The anchor of the separated branch sits at the stall angle.
    ///
    /// The invariant that catches a sign error nothing else would. Anchoring the
    /// branch means evaluating the attached model at the attitude whose *mean*
    /// incidence is the stall angle, and reaching that attitude means turning the
    /// onset flow — in the direction that raises the incidence, which is the
    /// opposite of the one a first reading of the rotation suggests.
    ///
    /// Getting it backwards evaluates the anchor at `2·mean - stall`, which is
    /// *close enough to be invisible* when the sail happens to be trimmed near its
    /// stall angle and wildly wrong away from it. It cost a forty-three per cent
    /// error in the blended lift at forty degrees before this test existed, and
    /// every coefficient check in the module passed throughout.
    ///
    /// Checked against the attitude located independently — by inverting
    /// `mean_incidence` rather than by repeating the rotation — so the test cannot
    /// agree with the code by sharing its mistake.
    #[test]
    fn the_anchor_sits_at_the_stall_angle() {
        for at in [12.0_f64, 20.0, 30.0, 40.0] {
            let sail = sail(Controls::HARD, at);
            let stall = sail.options.stall;

            // Mean incidence is linear in the wind angle with unit slope, so the
            // attitude that stalls the sail is one subtraction away.
            let offset = at.to_radians() - sail.shape().mean_incidence(wind(at));
            let stalling = wind((stall + offset).to_degrees());
            assert_relative_eq!(
                sail.shape().mean_incidence(stalling),
                stall,
                max_relative = 1e-9
            );

            let attached = Sail::split(
                sail.solver
                    .solve(|_| stalling, AIR)
                    .expect("a sail solves")
                    .force(),
                stalling,
                &planform(),
                AIR,
            );
            assert_relative_eq!(
                sail.blend.separated().lift(stall),
                attached.0,
                max_relative = 1e-9
            );
            assert_relative_eq!(
                sail.blend.separated().drag(stall),
                attached.1,
                max_relative = 1e-9
            );
        }
    }

    /// The driving force has a maximum inside the sheet's travel.
    ///
    /// The end-to-end statement that all three modules are wired with consistent
    /// signs, and a better one than "sheeting in goes faster" — because sheeting in
    /// *stops* going faster. Over-trimming stalls the sail and loses drive, which is
    /// the single most familiar fact about sail trim and the reason a trimmer has a
    /// job. A model without an interior optimum would say the fastest trim is always
    /// hard on.
    ///
    /// The mechanism is visible in the numbers: drive climbs from 0.2 to 0.8 of the
    /// sheet's travel and falls at 1.0, where the mean incidence has passed the
    /// stall onset and the blend has taken 65 % of the force.
    #[test]
    fn the_driving_force_has_a_maximum_inside_the_sheets_travel() {
        let at = 28.0;
        let drive_at = |sheet: f64| {
            // Sheet and vang together, which is how a sheet is actually eased.
            let controls = Controls {
                sheet,
                traveller: 1.0,
                vang: sheet,
                ..Controls::EASED
            };
            let sail = sail(controls, at);
            let wrench = sail.wrench(wind(at), AIR, Point::zeros());
            // `x` is forward in the file frame, so the drive is `+force.x`.
            (wrench.force.x, wrench.force.y, wrench.separated_fraction)
        };

        let travel: Vec<_> = [0.2_f64, 0.4, 0.6, 0.8, 1.0]
            .iter()
            .map(|&s| drive_at(s))
            .collect();
        let drives: Vec<f64> = travel.iter().map(|&(drive, _, _)| drive).collect();

        // Rising over most of the travel...
        for pair in drives[..4].windows(2) {
            assert!(
                pair[1] > pair[0],
                "drive did not build with sheet: {drives:?}"
            );
        }
        // ...and falling at the end, because the sail is over-trimmed.
        assert!(
            drives[4] < drives[3],
            "over-sheeting did not cost drive: {drives:?}"
        );
        assert!(
            travel[4].2 > 0.5 && travel[3].2 == 0.0,
            "the loss is not the stall taking over: {:?}",
            travel.iter().map(|&(_, _, s)| s).collect::<Vec<_>>()
        );
        // And the sail pulls to leeward throughout, which is the tack convention.
        for &(_, side, _) in &travel {
            assert!(side > 0.0, "the sail pulled to windward");
        }
    }

    /// Dropping the traveller depowers more than it slows.
    ///
    /// The trim a sailor reaches for in a gust, end to end: the heeling moment falls
    /// faster than the driving force, because that is what depowering *means*. It
    /// works here only because the angle and the leech are computed by different
    /// rules in [`crate::flying::Controls`] — a single sheet scalar could not
    /// express an open angle with a closed leech.
    ///
    /// The mechanism is that induced drag falls as the square of the lift while the
    /// drive is a difference between lift and drag terms, so the drive gives up less
    /// than the side force does. Measured across the car's whole travel, heel falls
    /// 67 % where drive falls 43 %.
    #[test]
    fn dropping_the_traveller_depowers_more_than_it_slows() {
        let at = 26.0;
        let powered = Controls {
            sheet: 1.0,
            traveller: 1.0,
            vang: 1.0,
            outhaul: 0.0,
            cunningham: 0.0,
        };

        let measure = |traveller: f64| {
            let sail = sail(
                Controls {
                    traveller,
                    ..powered
                },
                at,
            );
            let wrench = sail.wrench(wind(at), AIR, Point::zeros());
            // Drive forward along the centreline, heeling moment about the tack.
            (wrench.force.x, wrench.moment.x.abs())
        };
        let (drive_on, heel_on) = measure(1.0);

        // Heel sheds faster than drive at every setting of the car, not only at the
        // ends — so this is the shape of the trade rather than one lucky pair.
        for traveller in [0.8_f64, 0.6, 0.4, 0.2, 0.0] {
            let (drive, heel) = measure(traveller);
            assert!(heel < heel_on, "the car did not shed heel at {traveller}");
            assert!(
                (1.0 - heel / heel_on) > (1.0 - drive / drive_on),
                "at {traveller} heel fell {:.1} % and drive {:.1} %, which is not \
                 depowering",
                100.0 * (1.0 - heel / heel_on),
                100.0 * (1.0 - drive / drive_on)
            );
        }
        // And dropping it all the way does cost real drive - it is a trade, not a
        // free lunch.
        let (drive_off, _) = measure(0.0);
        assert!(drive_off < 0.7 * drive_on);
    }

    /// Past the stall the blend takes over and the lattice's runaway is gone.
    ///
    /// End to end, through the assembly this module exists to make: the attached
    /// coefficient the lattice reports keeps climbing with wind angle, and the one
    /// the sail returns does not.
    #[test]
    fn a_sail_stalls_where_the_lattice_would_not() {
        let mut attached = Vec::new();
        let mut blended = Vec::new();
        for at in [10.0_f64, 18.0, 26.0, 34.0, 45.0] {
            // Refactorised at each angle, so the wake is never the excuse.
            let sail = sail(Controls::HARD, at);
            let wrench = sail.wrench(wind(at), AIR, Point::zeros());
            attached.push(wrench.attached_lift_coefficient);
            blended.push(wrench.lift_coefficient);
        }

        for pair in attached.windows(2) {
            assert!(
                pair[1] > pair[0],
                "the lattice stopped climbing: {attached:?}"
            );
        }
        let peak = blended.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            *blended.last().expect("a sample") < peak,
            "the sail never stalled: {blended:?}"
        );
        // Below the onset the two are the same number, bit for bit.
        assert_eq!(blended[0].to_bits(), attached[0].to_bits());
    }

    /// Twist delays the stall, through the whole chain.
    ///
    /// The published finding, arriving at the level a boat sees it: a sail with the
    /// leech eased carries attached flow to a wider wind angle, because the blend
    /// reads the area-weighted mean incidence and twist lowers it.
    #[test]
    fn an_eased_leech_carries_attached_flow_further() {
        let onset_of = |controls: Controls| {
            let mut previous = 0.0;
            for step in 0..=120 {
                let at = 5.0 + step as f64 * 0.5;
                let sail = sail(controls, at);
                let fraction = sail
                    .wrench(wind(at), AIR, Point::zeros())
                    .separated_fraction;
                if fraction > 0.0 {
                    return previous;
                }
                previous = at;
            }
            f64::INFINITY
        };

        let tight = onset_of(Controls::HARD);
        let eased = onset_of(Controls {
            sheet: 0.0,
            vang: 0.0,
            ..Controls::HARD
        });
        assert!(
            eased > tight + 1.0,
            "easing the leech did not delay the onset: {eased} against {tight}"
        );
    }

    /// A degenerate sail is refused rather than built.
    #[test]
    fn a_degenerate_sail_is_refused() {
        let build = |wake: Vector3<f64>, options: Options| {
            Sail::new(planform(), response(), Controls::HARD, wake, options)
        };
        assert!(build(Vector3::zeros(), Options::default()).is_none());
        assert!(build(Vector3::new(f64::NAN, 0.0, 0.0), Options::default()).is_none());
        assert!(build(
            wind(20.0),
            Options {
                panels: (0, 10),
                ..Options::default()
            }
        )
        .is_none());
        assert!(build(
            wind(20.0),
            Options {
                stall: 0.0,
                ..Options::default()
            }
        )
        .is_none());
        assert!(build(
            wind(20.0),
            Options {
                stall: PI,
                ..Options::default()
            }
        )
        .is_none());
        assert!(build(
            wind(20.0),
            Options {
                blend_width: 0.0,
                ..Options::default()
            }
        )
        .is_none());
        // A shape that solves is still required, not merely a shape that parses.
        let flat = Planform::new(5.0, 2.0, 0.0, 0.0).expect("a triangle");
        let uniform = Response::new((0.4, 0.1), (0.4, 0.05), (0.12, 0.06), 0.5, (0.45, 0.35))
            .expect("a real travel");
        assert!(Sail::new(
            flat,
            uniform,
            Controls::HARD,
            wind(20.0),
            Options::default()
        )
        .is_some());
        let _ = Distribution::uniform(0.0);
    }
}
