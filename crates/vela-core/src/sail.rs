//! A sail plan: geometry in, wrench out.
//!
//! # What this module is
//!
//! The assembly of phase 2. [`crate::flying`] turns controls into a shape,
//! [`crate::vlm`] turns shapes into panel forces, [`crate::stall`] corrects them
//! where potential flow has run out of validity. This module owns all three and
//! answers the question a boat asks: given this trim and this wind, what force and
//! what moment.
//!
//! [`Plan`] rather than `Sail`, because the unit is the plan and not the sail. A
//! sloop's main and headsail are one lifting system — they share a linear system in
//! [`crate::vlm::Solver::coupled`] — and the distinction from
//! [`crate::aero::SailPlan`] is that this one computes forces from geometry where
//! that one reads them from a table.
//!
//! # Two costs, deliberately visible
//!
//! Building a `Plan` factorises an influence matrix — fifteen milliseconds at three
//! hundred panels, measured, and cubic in the total, so two sails of 240 cost about
//! eight times one of them rather than twice. Asking one for a wrench solves a
//! right-hand side, a hundred times less. Both numbers are in [`crate::vlm`]'s own
//! documentation and they are why this is a *thing you keep* rather than a function
//! you call.
//!
//! So the expensive operation has a name that says so ([`Plan::retrim`]), and the
//! type reports when it is needed ([`Plan::wake_drift`]) rather than deciding.
//! Noticing internally that the wind angle moved and quietly refactorising would
//! put a fifty-millisecond stall inside a sixty-hertz loop at a moment nobody chose.
//!
//! # Every sail blends on its own incidence
//!
//! Not the plan's. A headsail and a mainsail do not stall together — the main sits
//! in the headsail's wake and is back-winded first — and with the panels already
//! attributed per surface by the coupled solve, correcting each sail against its own
//! mean incidence costs nothing extra and is simply more nearly true.
//!
//! Each sail's stall angle is its own too, declared per [`Member`], because a flat
//! genoa and a full main are different cloth at different Reynolds numbers.
//!
//! # The stall anchor rides on the working factorisation
//!
//! [`crate::stall::Separated`] has to be anchored on the attached model's own lift
//! and drag *at the stall angle*, which is a different attitude from the one being
//! sailed. Done properly that is a second factorisation per sail.
//!
//! It is done improperly on purpose: the anchor is evaluated by changing the onset
//! direction while keeping the wake the current factorisation was built for.
//! [`crate::vlm`]'s measurement of that approximation is under one per cent below
//! six degrees of drift and a few per cent by twelve. The quantity it perturbs is
//! the anchor of an empirical branch whose cross-tunnel uncertainty is ten to
//! fifteen per cent, so the error is an order below the noise it enters.
//!
//! One thing the shortcut gets *right* rather than merely cheaply: because the solve
//! is coupled, a sail's anchor is measured with its neighbours present, so the
//! interaction is inside the empirical branch as well as inside the attached one.
//!
//! # How the blend becomes a wrench
//!
//! [`crate::stall`] corrects *coefficients*; a boat needs a force and a moment. The
//! conversion could be done on the totals, but then the moment needs a point to act
//! at, and the centre of effort past stall is not something this engine knows.
//!
//! Instead the correction is applied to **every panel of that sail**, as the same
//! pair of ratios: each panel's force is split into its along-flow part, scaled by
//! the drag ratio, and everything else, scaled by the lift ratio. Summing gives
//! exactly the blended lift and drag, and the moment comes out of the panels that
//! produced it — so the centre of effort is the lattice's, by construction, with
//! nothing assumed about where it goes.
//!
//! That the *vertical* force scales with lift rather than with drag is a choice and
//! is the right one: it is lift, resolved onto a different axis, and it comes from
//! the same circulation.
//!
//! # What it does not do
//!
//! Uniform onset flow. The lattice takes an arbitrary field and would carry wind
//! shear for free, but the shear profile belongs to a wind model that is not built,
//! and half of one here would be a number nobody chose.
//!
//! No viscous drag. Everything here is induced or separated; the friction of cloth
//! and the parasitic drag of mast and topsides belong to whoever assembles a boat,
//! and [`crate::aero`] already has them.

use nalgebra::Vector3;

use crate::flying::{Controls, Planform, Response, Shape};
use crate::geometry::Point;
use crate::stall::{Blend, Separated};
use crate::vlm::{Lattice, Solver};

/// Below this wind speed a sail makes no force worth computing.
const STILL_AIR: f64 = 1e-6;

/// Most sails one plan can carry.
///
/// Five, which is what [`crate::aero::Sail`] enumerates — main, headsail,
/// spinnaker, mizzen, mizzen staysail. A fixed bound rather than a growable one so
/// that a per-frame wrench carries its breakdown on the stack instead of
/// allocating.
pub const MAX_SAILS: usize = 5;

/// How finely to resolve a sail, and where it stalls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Options {
    /// Panels along the chord and up the sail.
    ///
    /// Ten by twenty-four is the working default: 240 panels, and a lift slope
    /// within a per cent of what forty-eight spanwise panels give.
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

/// One sail: what it is, where its tack sits, and how it is resolved.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Member {
    /// The outline.
    pub planform: Planform,
    /// What its controls reach.
    pub response: Response,
    /// Where its tack sits in the file frame.
    ///
    /// The placement [`crate::flying`] deliberately refuses to hold: a shape puts
    /// its luff on the `z` axis at the origin, and a rig is what knows where on the
    /// boat that is. Two sails at the same tack would be one sail solved twice.
    pub tack: Point,
    /// Resolution and stall.
    pub options: Options,
}

/// One sail's share of a plan's wrench.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Contribution {
    /// Force, N, in the file frame.
    pub force: Vector3<f64>,
    /// Moment, N·m, about the point the wrench was asked for.
    pub moment: Vector3<f64>,
    /// Lift coefficient on this sail's planform area, after blending.
    pub lift_coefficient: f64,
    /// Drag coefficient on this sail's planform area, after blending. Induced plus
    /// whatever the separated branch adds; there is no viscous term.
    pub drag_coefficient: f64,
    /// Lift coefficient the lattice alone gave, before blending.
    ///
    /// The difference between this and the blended one is the whole empirical
    /// content of the answer, and a caller debugging a polar needs to see which of
    /// the two moved.
    pub attached_lift_coefficient: f64,
    /// Weight this sail's separated branch was given, 0 to 1.
    pub separated_fraction: f64,
    /// This sail's area-weighted mean incidence, radians.
    pub mean_incidence: f64,
    /// Height of this sail's centre of effort above its own tack, m.
    ///
    /// Rolling moment over horizontal force, which is the convention the
    /// wind-tunnel literature reports and therefore the one that can be compared
    /// against it.
    pub centre_of_effort: f64,
}

impl Contribution {
    const ZERO: Self = Self {
        force: Vector3::new(0.0, 0.0, 0.0),
        moment: Vector3::new(0.0, 0.0, 0.0),
        lift_coefficient: 0.0,
        drag_coefficient: 0.0,
        attached_lift_coefficient: 0.0,
        separated_fraction: 0.0,
        mean_incidence: 0.0,
        centre_of_effort: 0.0,
    };
}

/// What a plan produced at one wind.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wrench {
    /// Total force, N, in the file frame.
    pub force: Vector3<f64>,
    /// Total moment, N·m, about the point it was asked for.
    pub moment: Vector3<f64>,
    breakdown: [Contribution; MAX_SAILS],
    carried: usize,
}

impl Wrench {
    /// Each sail's share, in the order the members were declared.
    #[must_use]
    pub fn sails(&self) -> &[Contribution] {
        &self.breakdown[..self.carried]
    }
}

/// A sail plan: several surfaces in one linear system.
#[derive(Debug, Clone)]
pub struct Plan {
    members: Vec<Member>,
    shapes: Vec<Shape>,
    blends: Vec<Blend>,
    solver: Solver,
    wake: Vector3<f64>,
}

impl Plan {
    /// Builds a plan at a trim, factorised for a wake direction.
    ///
    /// `wake` is the direction the trailing vorticity leaves along, which is the
    /// mean flow's — in practice the apparent wind the plan is expected to work in.
    /// It is baked into the factorisation; see the module documentation.
    ///
    /// One `Controls` per member, in the same order. Returns `None` for an empty
    /// plan, more than [`MAX_SAILS`], a control list of the wrong length, a
    /// degenerate wake, or any member whose geometry or stall model will not stand
    /// up.
    #[must_use]
    pub fn new(members: Vec<Member>, controls: &[Controls], wake: Vector3<f64>) -> Option<Self> {
        if members.is_empty() || members.len() > MAX_SAILS || controls.len() != members.len() {
            return None;
        }
        let shapes: Vec<Shape> = members
            .iter()
            .zip(controls)
            .map(|(member, &controls)| member.response.shape(member.planform, controls))
            .collect();

        let mut lattices = Vec::with_capacity(members.len());
        for (member, shape) in members.iter().zip(&shapes) {
            let (chordwise, spanwise) = member.options.panels;
            lattices.push(Lattice::from_shape(chordwise, spanwise, |along, up| {
                shape.point(along, up) + member.tack
            })?);
        }
        let solver = Solver::coupled(lattices, wake)?;

        let mut blends = Vec::with_capacity(members.len());
        for (index, member) in members.iter().enumerate() {
            blends.push(Self::fit_blend(
                index,
                member,
                &shapes[index],
                &solver,
                wake,
            )?);
        }

        let wake = solver_wake(&solver, wake);
        Some(Self {
            members,
            shapes,
            blends,
            solver,
            wake,
        })
    }

    /// Anchors one sail's separated branch on the attached model at *its* stall
    /// attitude.
    ///
    /// Evaluated on the caller's factorisation rather than a fresh one — the trade
    /// the module documentation states and prices.
    fn fit_blend(
        index: usize,
        member: &Member,
        shape: &Shape,
        solver: &Solver,
        wake: Vector3<f64>,
    ) -> Option<Blend> {
        let speed = wake.norm();
        if speed < STILL_AIR {
            return None;
        }
        // The onset that puts this sail's *mean* incidence at its stall angle. The
        // mean is linear in the wind angle, so one evaluation locates it — and the
        // direction of the turn is the trap: rotating the flow by `+θ` about `z`
        // *reduces* the incidence by `θ`, because the chord and the flow are
        // compared the other way round. So the turn is `mean - stall`, not
        // `stall - mean`, and the sign is asserted rather than reasoned about in
        // `the_anchor_sits_at_the_stall_angle`.
        let reference = wake / speed;
        let turn = shape.mean_incidence(reference) - member.options.stall;
        let (sine, cosine) = turn.sin_cos();
        let stalled = Vector3::new(
            reference.x * cosine - reference.y * sine,
            reference.x * sine + reference.y * cosine,
            reference.z,
        );

        let onset = stalled * speed;
        let solution = solver.solve(|_| onset, 1.0)?;
        let (lift, drag) = split(solution.force_on(index)?, onset, &member.planform, 1.0);
        Separated::fit(
            member.planform.aspect_ratio(),
            member.options.stall,
            lift,
            drag,
        )
        .and_then(|separated| Blend::new(separated, member.options.blend_width))
    }

    /// The sails this plan carries.
    #[must_use]
    pub fn members(&self) -> &[Member] {
        &self.members
    }

    /// One sail's shape, as the controls left it.
    #[must_use]
    pub fn shape(&self, sail: usize) -> Option<&Shape> {
        self.shapes.get(sail)
    }

    /// The wake direction baked into the factorisation.
    #[must_use]
    pub fn wake(&self) -> Vector3<f64> {
        self.wake
    }

    /// How far a wind has drifted from the baked wake, radians.
    ///
    /// The number a caller watches to decide when to pay for [`Plan::retrim`]. The
    /// cost of *not* paying is measured: 1.2 % of attached lift at three degrees of
    /// drift, 2.5 at six, 5.5 at twelve.
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

    /// Rebuilds the shapes and the factorisation. **Expensive** — cubic in the
    /// total panel count, plus the `N²` ring evaluations that dominate it.
    ///
    /// Both arguments, because the two reasons to rebuild are a trim change and a
    /// wake that has drifted, and a caller doing one will usually want the other.
    /// Returns `None` and leaves the plan untouched if the new geometry is
    /// degenerate, so a failed retrim cannot leave a plan unusable.
    pub fn retrim(&mut self, controls: &[Controls], wake: Vector3<f64>) -> Option<()> {
        let rebuilt = Self::new(self.members.clone(), controls, wake)?;
        *self = rebuilt;
        Some(())
    }

    /// The wrench at a wind, about a point. **Cheap** — one right-hand side.
    ///
    /// `wind` is the air's velocity in the file frame, so a wind blowing aft is
    /// negative in `x`. Returns a zero wrench in still air rather than a division by
    /// a vanishing speed.
    #[must_use]
    pub fn wrench(&self, wind: Vector3<f64>, density: f64, about: Point) -> Wrench {
        let mut breakdown = [Contribution::ZERO; MAX_SAILS];
        let carried = self.members.len();
        for (index, shape) in self.shapes.iter().enumerate() {
            breakdown[index].mean_incidence = shape.mean_incidence(wind);
            breakdown[index].separated_fraction =
                self.blends[index].weight(breakdown[index].mean_incidence);
        }

        let speed = wind.norm();
        let usable = speed >= STILL_AIR && density.is_finite() && density > 0.0;
        let solution = if usable {
            self.solver.solve(|_| wind, density)
        } else {
            None
        };
        let Some(solution) = solution else {
            return Wrench {
                force: Vector3::zeros(),
                moment: Vector3::zeros(),
                breakdown,
                carried,
            };
        };

        let along = wind / speed;
        let mut total_force = Vector3::zeros();
        let mut total_moment = Vector3::zeros();

        for (index, member) in self.members.iter().enumerate() {
            let range = solution
                .range(index)
                .expect("a member's panels are in range by construction");
            let attached = split(
                solution
                    .force_on(index)
                    .expect("a member's panels are in range by construction"),
                wind,
                &member.planform,
                density,
            );
            let blended =
                self.blends[index].coefficients(attached, breakdown[index].mean_incidence);

            // The same two ratios on every panel of this sail: along-flow scaled by
            // drag, everything else by lift. Summing reproduces the blended
            // coefficients exactly and leaves the centre of effort where the lattice
            // put it.
            let ratio = |blended: f64, attached: f64| {
                if attached.abs() < 1e-12 {
                    1.0
                } else {
                    blended / attached
                }
            };
            let (lift_ratio, drag_ratio) =
                (ratio(blended.0, attached.0), ratio(blended.1, attached.1));

            let mut force = Vector3::zeros();
            let mut moment = Vector3::zeros();
            let mut about_tack = Vector3::zeros();
            for panel in range {
                let raw = solution.forces()[panel];
                let at = solution.positions()[panel];
                let streamwise = along * raw.dot(&along);
                let corrected = streamwise * drag_ratio + (raw - streamwise) * lift_ratio;
                force += corrected;
                moment += (at - about).cross(&corrected);
                about_tack += (at - member.tack).cross(&corrected);
            }

            let horizontal = force.x.hypot(force.y);
            breakdown[index] = Contribution {
                force,
                moment,
                lift_coefficient: blended.0,
                drag_coefficient: blended.1,
                attached_lift_coefficient: attached.0,
                centre_of_effort: if horizontal < STILL_AIR {
                    0.0
                } else {
                    about_tack.x.hypot(about_tack.y) / horizontal
                },
                ..breakdown[index]
            };
            total_force += force;
            total_moment += moment;
        }

        Wrench {
            force: total_force,
            moment: total_moment,
            breakdown,
            carried,
        }
    }
}

/// The normalised wake a solver actually kept, so the plan and it cannot disagree.
fn solver_wake(solver: &Solver, fallback: Vector3<f64>) -> Vector3<f64> {
    let kept = solver.wake();
    if kept.norm() < STILL_AIR {
        fallback
    } else {
        kept
    }
}

/// Lift and drag coefficients of a force, on a planform's area.
///
/// Lift across the flow and horizontal, toward leeward; drag along it. The vertical
/// component belongs to neither and is not returned — it is carried through the
/// per-panel scaling instead, which is where it can be kept consistent with the
/// lift it came from.
fn split(
    force: Vector3<f64>,
    onset: Vector3<f64>,
    planform: &Planform,
    density: f64,
) -> (f64, f64) {
    let speed = onset.norm();
    let along = onset / speed;
    let across = leeward(along);
    let dynamic = 0.5 * density * speed * speed * planform.area();
    (force.dot(&across) / dynamic, force.dot(&along) / dynamic)
}

/// The horizontal unit vector across a flow, pointing to leeward.
///
/// The flow turned a quarter turn the way a starboard-tack sail's camber bulges, so
/// that a positive lift coefficient is a sail pulling to leeward and not a sign
/// convention waiting to be discovered.
fn leeward(along: Vector3<f64>) -> Vector3<f64> {
    let horizontal = Vector3::new(along.x, along.y, 0.0);
    let norm = horizontal.norm();
    if norm < STILL_AIR {
        return Vector3::y();
    }
    Vector3::new(horizontal.y, -horizontal.x, 0.0) / norm
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

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

    fn member(tack: Point) -> Member {
        Member {
            planform: planform(),
            response: response(),
            tack,
            options: Options::default(),
        }
    }

    /// A wind of `SPEED` blowing from `degrees` off the bow, on starboard tack.
    fn wind(degrees: f64) -> Vector3<f64> {
        let beta = degrees.to_radians();
        Vector3::new(-SPEED * beta.cos(), SPEED * beta.sin(), 0.0)
    }

    fn one(controls: Controls, at: f64) -> Plan {
        Plan::new(vec![member(Point::zeros())], &[controls], wind(at)).expect("a real plan")
    }

    /// A sloop: headsail forward, mainsail aft, both on the centreline.
    fn sloop(controls: Controls, at: f64) -> Plan {
        Plan::new(
            vec![member(Point::new(4.0, 0.0, 0.0)), member(Point::zeros())],
            &[controls, controls],
            wind(at),
        )
        .expect("a real plan")
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
            let plan = one(Controls::HARD, at);
            let wrench = plan.wrench(wind(at), AIR, Point::zeros());
            let (lift, drag) = split(wrench.force, wind(at), &planform(), AIR);
            let sail = wrench.sails()[0];
            assert_relative_eq!(lift, sail.lift_coefficient, max_relative = 1e-10);
            assert_relative_eq!(drag, sail.drag_coefficient, max_relative = 1e-10);
        }
    }

    /// The breakdown adds up to the total, force and moment.
    #[test]
    fn the_breakdown_adds_up_to_the_total() {
        let at = 26.0;
        let plan = sloop(Controls::HARD, at);
        let about = Point::new(1.1, -0.4, 0.9);
        let wrench = plan.wrench(wind(at), AIR, about);
        assert_eq!(wrench.sails().len(), 2);

        let force: Vector3<f64> = wrench.sails().iter().map(|s| s.force).sum();
        let moment: Vector3<f64> = wrench.sails().iter().map(|s| s.moment).sum();
        for k in 0..3 {
            assert_relative_eq!(force[k], wrench.force[k], max_relative = 1e-12);
            assert_relative_eq!(moment[k], wrench.moment[k], max_relative = 1e-12);
        }
    }

    /// The anchor of each sail's separated branch sits at that sail's stall angle.
    ///
    /// The invariant that catches a sign error nothing else would. Anchoring means
    /// evaluating the attached model at the attitude whose *mean* incidence is the
    /// stall angle, and reaching that attitude means turning the onset flow — in the
    /// direction that raises the incidence, which is the opposite of the one a first
    /// reading of the rotation suggests.
    ///
    /// Getting it backwards evaluates the anchor at `2·mean - stall`, which is
    /// *close enough to be invisible* when the sail happens to be trimmed near its
    /// stall angle and wildly wrong away from it. It cost a forty-three per cent
    /// error in blended lift at forty degrees before this test existed, and every
    /// coefficient check in the module passed throughout.
    ///
    /// Checked against the attitude located independently — by inverting
    /// `mean_incidence` rather than by repeating the rotation — so the test cannot
    /// agree with the code by sharing its mistake.
    #[test]
    fn the_anchor_sits_at_the_stall_angle() {
        for at in [12.0_f64, 20.0, 30.0, 40.0] {
            let plan = sloop(Controls::HARD, at);
            for index in 0..2 {
                let stall = plan.members[index].options.stall;
                let shape = plan.shape(index).expect("a member");

                // Mean incidence is linear in the wind angle with unit slope, so the
                // attitude that stalls this sail is one subtraction away.
                let offset = at.to_radians() - shape.mean_incidence(wind(at));
                let stalling = wind((stall + offset).to_degrees());
                assert_relative_eq!(shape.mean_incidence(stalling), stall, max_relative = 1e-9);

                let attached = split(
                    plan.solver
                        .solve(|_| stalling, AIR)
                        .expect("a plan solves")
                        .force_on(index)
                        .expect("in range"),
                    stalling,
                    &planform(),
                    AIR,
                );
                assert_relative_eq!(
                    plan.blends[index].separated().lift(stall),
                    attached.0,
                    max_relative = 1e-9
                );
                assert_relative_eq!(
                    plan.blends[index].separated().drag(stall),
                    attached.1,
                    max_relative = 1e-9
                );
            }
        }
    }

    /// The correction leaves the centre of effort where the lattice put it.
    #[test]
    fn the_correction_does_not_move_the_centre_of_effort() {
        let at = 30.0;
        let plan = one(Controls::HARD, at);
        let blended = plan.wrench(wind(at), AIR, Point::zeros());
        assert!(
            blended.sails()[0].separated_fraction > 0.9,
            "this test is not exercising the branch: {}",
            blended.sails()[0].separated_fraction
        );

        let solution = plan.solver.solve(|_| wind(at), AIR).expect("a plan solves");
        let force = solution.force();
        let moment = solution.moment_about(Point::zeros());
        let raw = moment.x.hypot(moment.y) / force.x.hypot(force.y);

        assert_relative_eq!(
            blended.sails()[0].centre_of_effort,
            raw,
            max_relative = 0.01
        );
        assert!((0.3..0.6).contains(&(blended.sails()[0].centre_of_effort / 5.185)));
    }

    /// Forces scale with dynamic pressure, and coefficients do not.
    #[test]
    fn forces_scale_with_dynamic_pressure_and_coefficients_do_not() {
        let plan = one(Controls::HARD, 20.0);
        let base = plan.wrench(wind(20.0), AIR, Point::zeros());
        let faster = plan.wrench(wind(20.0) * 2.0, AIR, Point::zeros());
        let denser = plan.wrench(wind(20.0), AIR * 3.0, Point::zeros());

        for k in 0..3 {
            assert_relative_eq!(faster.force[k], 4.0 * base.force[k], max_relative = 1e-9);
            assert_relative_eq!(denser.force[k], 3.0 * base.force[k], max_relative = 1e-9);
        }
        assert_relative_eq!(
            faster.sails()[0].lift_coefficient,
            base.sails()[0].lift_coefficient,
            max_relative = 1e-9
        );
        assert_relative_eq!(
            denser.sails()[0].drag_coefficient,
            base.sails()[0].drag_coefficient,
            max_relative = 1e-9
        );
    }

    /// Still air makes no force, and does not divide by zero doing it.
    #[test]
    fn still_air_makes_no_force() {
        let plan = one(Controls::HARD, 20.0);
        for calm in [Vector3::zeros(), Vector3::new(1e-15, 0.0, 0.0)] {
            let wrench = plan.wrench(calm, AIR, Point::zeros());
            assert_eq!(wrench.force, Vector3::zeros());
            assert_eq!(wrench.moment, Vector3::zeros());
            assert!(wrench.sails()[0].lift_coefficient.is_finite());
            assert!(wrench.sails()[0].centre_of_effort.is_finite());
        }
        let wrench = plan.wrench(wind(20.0), 0.0, Point::zeros());
        assert_eq!(wrench.force, Vector3::zeros());
        // The breakdown is still the right length, so a caller's telemetry does not
        // change shape when the wind drops.
        assert_eq!(wrench.sails().len(), 1);
    }

    /// The moment transforms with its reference point.
    #[test]
    fn the_moment_transforms_with_its_reference_point() {
        let plan = sloop(Controls::HARD, 22.0);
        let a = Point::new(0.3, -0.2, 1.1);
        let b = Point::new(-1.7, 2.4, -0.6);
        let at_a = plan.wrench(wind(22.0), AIR, a);
        let at_b = plan.wrench(wind(22.0), AIR, b);
        let shifted = at_a.moment + (a - b).cross(&at_a.force);
        for k in 0..3 {
            assert_relative_eq!(at_b.moment[k], shifted[k], max_relative = 1e-9);
        }
    }

    /// The drift from the baked wake is measured, and refactorising removes it.
    #[test]
    fn the_wake_drift_is_reported_and_retrimming_clears_it() {
        let mut plan = one(Controls::HARD, 20.0);
        assert_relative_eq!(plan.wake_drift(wind(20.0)), 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            plan.wake_drift(wind(32.0)),
            12.0_f64.to_radians(),
            max_relative = 1e-9
        );
        // Symmetric, and blind to speed.
        assert_relative_eq!(
            plan.wake_drift(wind(8.0)),
            12.0_f64.to_radians(),
            max_relative = 1e-9
        );
        assert_relative_eq!(
            plan.wake_drift(wind(32.0) * 5.0),
            plan.wake_drift(wind(32.0)),
            max_relative = 1e-12
        );

        plan.retrim(&[Controls::HARD], wind(32.0))
            .expect("a real retrim");
        assert_relative_eq!(plan.wake_drift(wind(32.0)), 0.0, epsilon = 1e-12);

        // And the drift costs what the lattice says it costs, on the *attached*
        // coefficient the approximation acts on: 1.2 % at three degrees, 2.5 at six,
        // 5.5 at twelve.
        let stale = one(Controls::HARD, 20.0);
        let mut worst: f64 = 0.0;
        for at in [23.0_f64, 26.0, 32.0] {
            let fresh = one(Controls::HARD, at);
            let aligned =
                fresh.wrench(wind(at), AIR, Point::zeros()).sails()[0].attached_lift_coefficient;
            let drifted =
                stale.wrench(wind(at), AIR, Point::zeros()).sails()[0].attached_lift_coefficient;
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

    /// A failed retrim leaves the plan usable.
    #[test]
    fn a_failed_retrim_changes_nothing() {
        let mut plan = one(Controls::HARD, 20.0);
        let before = plan.wrench(wind(20.0), AIR, Point::zeros());
        assert!(plan.retrim(&[Controls::HARD], Vector3::zeros()).is_none());
        // And a control list of the wrong length is refused too.
        assert!(plan
            .retrim(&[Controls::HARD, Controls::HARD], wind(20.0))
            .is_none());
        let after = plan.wrench(wind(20.0), AIR, Point::zeros());
        assert_eq!(before.force, after.force);
        assert_relative_eq!(plan.wake_drift(wind(20.0)), 0.0, epsilon = 1e-12);
    }

    /// The driving force has a maximum inside the sheet's travel.
    ///
    /// The end-to-end statement that all three modules are wired with consistent
    /// signs, and a better one than "sheeting in goes faster" — because sheeting in
    /// *stops* going faster. Over-trimming stalls the sail and loses drive, which is
    /// the single most familiar fact about sail trim and the reason a trimmer has a
    /// job. A model without an interior optimum would say the fastest trim is always
    /// hard on.
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
            let plan = one(controls, at);
            let wrench = plan.wrench(wind(at), AIR, Point::zeros());
            // `x` is forward in the file frame, so the drive is `+force.x`.
            (
                wrench.force.x,
                wrench.force.y,
                wrench.sails()[0].separated_fraction,
            )
        };

        let travel: Vec<_> = [0.2_f64, 0.4, 0.6, 0.8, 1.0]
            .iter()
            .map(|&s| drive_at(s))
            .collect();
        let drives: Vec<f64> = travel.iter().map(|&(drive, _, _)| drive).collect();

        for pair in drives[..4].windows(2) {
            assert!(
                pair[1] > pair[0],
                "drive did not build with sheet: {drives:?}"
            );
        }
        assert!(
            drives[4] < drives[3],
            "over-sheeting did not cost drive: {drives:?}"
        );
        assert!(
            travel[4].2 > 0.5 && travel[3].2 == 0.0,
            "the loss is not the stall taking over"
        );
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
            let plan = one(
                Controls {
                    traveller,
                    ..powered
                },
                at,
            );
            let wrench = plan.wrench(wind(at), AIR, Point::zeros());
            (wrench.force.x, wrench.moment.x.abs())
        };
        let (drive_on, heel_on) = measure(1.0);

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
        let (drive_off, _) = measure(0.0);
        assert!(drive_off < 0.7 * drive_on);
    }

    /// Past the stall the blend takes over and the lattice's runaway is gone.
    #[test]
    fn a_sail_stalls_where_the_lattice_would_not() {
        let mut attached = Vec::new();
        let mut blended = Vec::new();
        for at in [10.0_f64, 18.0, 26.0, 34.0, 45.0] {
            let plan = one(Controls::HARD, at);
            let sail = plan.wrench(wind(at), AIR, Point::zeros()).sails()[0];
            attached.push(sail.attached_lift_coefficient);
            blended.push(sail.lift_coefficient);
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
        assert_eq!(blended[0].to_bits(), attached[0].to_bits());
    }

    /// The mainsail is back-winded by the headsail ahead of it, and the headsail
    /// gains.
    ///
    /// The reason a plan is one linear system. The main sits in the headsail's wake
    /// and loses lift; the headsail sees the main's bound vorticity as upwash and
    /// gains. Both directions are what a sailor and a wind tunnel report, and
    /// neither can appear at all if the sails are solved separately and added.
    ///
    /// Solved apart, the pair over-predicts its own total — which is the error a
    /// coefficient model papers over with a rig-dependent fudge factor.
    #[test]
    fn the_mainsail_is_back_winded_by_the_headsail() {
        let at = 26.0;
        let coupled = sloop(Controls::HARD, at);
        let together = coupled.wrench(wind(at), AIR, Point::zeros());

        // The same two sails, each in clean air.
        let headsail_alone = Plan::new(
            vec![member(Point::new(4.0, 0.0, 0.0))],
            &[Controls::HARD],
            wind(at),
        )
        .expect("a real plan")
        .wrench(wind(at), AIR, Point::zeros());
        let main_alone = one(Controls::HARD, at);
        let main_alone = main_alone.wrench(wind(at), AIR, Point::zeros());

        let head = together.sails()[0].attached_lift_coefficient
            / headsail_alone.sails()[0].attached_lift_coefficient;
        let main = together.sails()[1].attached_lift_coefficient
            / main_alone.sails()[0].attached_lift_coefficient;

        assert!(
            main < 0.95,
            "the mainsail did not lose to the headsail's wake: {main:.4}"
        );
        assert!(
            head > 1.01,
            "the headsail did not gain from the mainsail's upwash: {head:.4}"
        );
        // And the pair is worth less than the sum of its parts.
        let sum = headsail_alone.force + main_alone.force;
        assert!(together.force.norm() < sum.norm());
    }

    /// Each sail blends on its own incidence, so one can be stalled while the other
    /// is not.
    ///
    /// The reason the handover is per sail rather than per plan. Given different
    /// cloth — declared as different stall angles — the plan must be able to report
    /// one sail separated and the other attached at the same instant, because that
    /// is what happens on a boat when the main goes soft before the jib.
    #[test]
    fn one_sail_can_stall_while_the_other_does_not() {
        let at = 24.0;
        let plan = Plan::new(
            vec![
                Member {
                    options: Options {
                        stall: 30.0_f64.to_radians(),
                        ..Options::default()
                    },
                    ..member(Point::new(4.0, 0.0, 0.0))
                },
                Member {
                    options: Options {
                        stall: 14.0_f64.to_radians(),
                        ..Options::default()
                    },
                    ..member(Point::zeros())
                },
            ],
            &[Controls::HARD, Controls::HARD],
            wind(at),
        )
        .expect("a real plan");

        let wrench = plan.wrench(wind(at), AIR, Point::zeros());
        assert_eq!(wrench.sails()[0].separated_fraction, 0.0);
        assert!(wrench.sails()[1].separated_fraction > 0.5);
        // The attached sail's numbers are untouched, bit for bit.
        assert_eq!(
            wrench.sails()[0].lift_coefficient.to_bits(),
            wrench.sails()[0].attached_lift_coefficient.to_bits()
        );
        assert!(wrench.sails()[1].lift_coefficient < wrench.sails()[1].attached_lift_coefficient);
    }

    /// A degenerate plan is refused rather than built.
    #[test]
    fn a_degenerate_plan_is_refused() {
        let ok = member(Point::zeros());
        assert!(Plan::new(vec![], &[], wind(20.0)).is_none());
        assert!(Plan::new(vec![ok], &[], wind(20.0)).is_none());
        assert!(Plan::new(vec![ok], &[Controls::HARD], Vector3::zeros()).is_none());
        assert!(Plan::new(
            vec![ok],
            &[Controls::HARD],
            Vector3::new(f64::NAN, 0.0, 0.0)
        )
        .is_none());
        assert!(Plan::new(
            vec![ok; MAX_SAILS + 1],
            &[Controls::HARD; MAX_SAILS + 1],
            wind(20.0)
        )
        .is_none());

        let bad_options = |options: Options| {
            Plan::new(
                vec![Member { options, ..ok }],
                &[Controls::HARD],
                wind(20.0),
            )
        };
        assert!(bad_options(Options {
            panels: (0, 10),
            ..Options::default()
        })
        .is_none());
        assert!(bad_options(Options {
            stall: 0.0,
            ..Options::default()
        })
        .is_none());
        assert!(bad_options(Options {
            stall: PI,
            ..Options::default()
        })
        .is_none());
        assert!(bad_options(Options {
            blend_width: 0.0,
            ..Options::default()
        })
        .is_none());

        // Two sails at the same tack are one sail solved twice, and the influence
        // matrix says so by being singular.
        assert!(Plan::new(vec![ok, ok], &[Controls::HARD; 2], wind(20.0)).is_none());
    }
}
