//! The aerodynamic force module: sails plus rig windage.
//!
//! # Provenance
//!
//! No physics of its own, and no numbers of its own. Every force here comes out
//! of [`crate::aero`] — the model of Hazen (1980) as transcribed from Larsson,
//! Eliasson & Orych, *Principles of Yacht Design*, 5th ed., Fig 8.19 and
//! Table 8.1. There is not one coefficient, area, height or drag constant in
//! this file, and there must never be one: a number invented in a wrapper is
//! indistinguishable from a number the source published. The unit tests do
//! declare a rig — lengths, the same YD-41 dimensions the [`crate::aero`]
//! validation suite sails, because a test has to sail something — and assert
//! signs, mirrors and relations on it, never a value.
//!
//! What this module contributes is a **change of description**. The
//! coefficient model answers in the wind's terms — lift and drag resolved into
//! a driving force along the centreline and a heeling force *to leeward*, with
//! no notion of which tack the yacht is on — while the simulation wants a
//! [`Wrench`] in the body frame, about the body origin. Turning the one into
//! the other is three decisions (which axis leeward is, where the force is
//! applied, what not to add on top), and each of them is a way to produce
//! forces that look entirely plausible and are wrong. They are made once, here.
//!
//! # Scope
//!
//! Everything the air acts on: the set sails, and the mast-and-topsides
//! windage that is all that remains under bare poles. Nothing below the
//! waterline — the canoe body is [`super::hull`]'s, the appendages are
//! [`super::lateral`]'s, and hydrostatic pressure is [`super::buoyancy`]'s.
//!
//! # Conventions
//!
//! Body frame: `x` forward, `y` to **starboard**, `z` **down**. The returned
//! moment is about the **body origin**, never about the centre of gravity.
//!
//! The sail force has no body-frame `z` component, because the coefficient
//! model resolves lift and drag in the horizontal plane and has no vertical
//! term. That is not a gap in the wrench: a heeled yacht's sail force *does*
//! have a world-frame vertical component, and it appears by itself when the
//! integrator rotates a body-frame force through the attitude. Putting one in
//! the body frame as well would count heel twice.
//!
//! # Known gaps, all inherited and none patched over
//!
//! - **No longitudinal centre of effort, hence no yaw moment.** Fig 8.19 gives
//!   the centres of effort as *heights* only; the boat data carries no
//!   longitudinal position for the mast, exactly as [`crate::sim::Captive`]
//!   says of the keel and the rudder. So the centre of effort is placed on the
//!   body `z` axis, in the athwartships plane through the origin, and this
//!   module's yaw moment is identically zero. Inventing a lead here would
//!   fabricate a yaw balance out of nothing, and a fabricated yaw balance is
//!   invisible in a trajectory that looks like sailing.
//! - **No heel correction on the apparent wind angle**, as
//!   [`crate::sim::StepCtx::apparent_wind_at`] records: sail forces at large
//!   heel are overestimated.
//! - **Under bare poles the centre of effort collapses to the freeboard**,
//!   because [`crate::aero`] does not locate the centre of pressure of mast and
//!   topsides. The windage force is right; its arm understates the mast.

use crate::aero::{
    ApparentWind, RigDimensions, Sail, SailForces, SailPlan, SailPlanError, SailSet,
};
use crate::flying::{Controls as SailTrim, Planform};
use crate::frames::file_to_body;
use crate::geometry::Point;
use crate::sail::{Member, Plan};
use crate::sim::{leeward_sign, ForceModule, StepCtx};
use crate::telemetry::Telemetry;
use crate::wrench::Wrench;
use nalgebra::Vector3;

/// Wake drift past which the geometric plan is refactorised, radians.
///
/// Four degrees, where [`crate::vlm`] measures the cost of a stale wake at under
/// two per cent of attached lift. Rebuilding is cubic in the panel count, so the
/// threshold is the whole trade: smaller and the loop stalls on refactorisations,
/// larger and the forces drift. It is a constant rather than a parameter because
/// nothing about a particular boat changes it — it is a property of the method.
const WAKE_TOLERANCE: f64 = 4.0 * std::f64::consts::PI / 180.0;

/// The aerodynamic module for one rig.
///
/// The rig is fixed for the life of the module — it is the yacht's spars, not a
/// control — while the sail set and the trim arrive from [`crate::Controls`] on
/// every step, because they are what the crew changes while sailing.
#[derive(Debug)]
pub struct Sails {
    rig: RigDimensions,
    /// Longitudinal position of the centre of effort, m, body frame.
    ///
    /// Zero unless the boat file declares a layout. A rig with no arm makes no
    /// yawing moment, which is a visible absence rather than a silent one: the
    /// telemetry publishes the moment and it is exactly zero.
    centre_of_effort_at: f64,
    /// The sail set [`Sails::plan`] was built for: the cache key.
    set: SailSet,
    /// The plan for `set`, or the reason that set cannot be sailed on this rig.
    plan: Result<SailPlan, SailPlanError>,
    /// Flying-shape data per sail, from the boat file. Empty means this rig has
    /// none and every sail set runs on the tabular model.
    geometry: Vec<(Sail, Member)>,
    /// The geometric plan in force, and the set it was built for.
    ///
    /// Built lazily inside [`ForceModule::step`], because it needs a wake direction
    /// and the wake direction is the apparent wind — which the constructor cannot
    /// know. So the first step of a run, and any step after the wind has swung more
    /// than [`WAKE_TOLERANCE`], pays a factorisation. That cost is real and is the
    /// reason it is bounded by a threshold rather than paid every step.
    geometric: Option<((SailSet, SailTrim, u64), Plan)>,
    /// What the last step computed, kept for [`ForceModule::telemetry`].
    /// `None` after a step whose plan could not be built.
    last: Option<Reported>,
}

/// The last step's results, in the form telemetry publishes them.
///
/// Stored rather than recomputed in `telemetry`, so that what is published is
/// what was actually applied — in particular `side_force`, which carries the
/// tack. A recomputation would be free to pick the other sign.
#[derive(Debug)]
struct Reported {
    wind: ApparentWind,
    forces: SailForces,
    /// The athwartships force as it went onto the body `y` axis, N.
    side_force: f64,
    /// What the geometric model contributed, if it was the model that ran.
    geometric: Option<Geometric>,
}

/// Everything one step of the geometric path needs, in one argument.
///
/// A struct rather than seven parameters. The five travel together, come from the
/// same place and are read once each - and a call site with seven positional
/// arguments of which three are `f64` is a swap waiting to compile.
#[derive(Debug, Clone, Copy)]
struct Condition {
    wind: ApparentWind,
    trim: crate::aero::Trim,
    shape: SailTrim,
    density: f64,
    /// Body frame, and where the empirical drag is applied.
    centre_of_effort: Vector3<f64>,
}

/// The geometric model's own numbers, for telemetry.
///
/// Published under `aero.geometric.` alongside the shared keys, because they are
/// the quantities that say *why* the answer differs from the tabular one — and a
/// caller comparing the two models needs to see the mean incidence and the
/// separated fraction, which the tabular model has no notion of.
#[derive(Debug, Clone, Copy)]
struct Geometric {
    mean_incidence: f64,
    separated_fraction: f64,
    attached_lift_coefficient: f64,
    wake_drift: f64,
    sails: usize,
}

impl Sails {
    /// Smallest reef factor the geometry is built at.
    ///
    /// A reef factor of zero is a sail of no size, and a lattice over it has no
    /// normals. Clamped rather than refused, because a solver walking a power path
    /// will ask for it and the honest answer is the smallest real sail.
    const MINIMUM_REEF: f64 = 0.05;

    /// Wraps a rig.
    ///
    /// Infallible, although [`SailPlan::new`] is not: the rig alone cannot say
    /// whether it will be asked for a sail plan it can build, since the sail
    /// set only arrives with the controls. The plan is therefore built here for
    /// bare poles — the one set every rig can carry — and rebuilt when the crew
    /// sets something. A rig whose dimensions are degenerate fails even that,
    /// and the failure is kept in [`Sails::plan_error`] rather than turned into
    /// a constructor that every caller has to unwrap for a condition that
    /// belongs to the boat file.
    #[must_use]
    pub fn new(rig: RigDimensions) -> Self {
        let set = SailSet::none();
        let plan = SailPlan::new(&rig, set);
        Self {
            rig,
            centre_of_effort_at: 0.0,
            set,
            plan,
            geometry: Vec::new(),
            geometric: None,
            last: None,
        }
    }

    /// Hands over the boat file's flying-shape data.
    ///
    /// With it, a sail set every entry of which has shape data runs on the
    /// geometric model of [`crate::sail`]; a set that is not fully covered — bare
    /// poles, or anything carrying a spinnaker — runs on the tabular one. That
    /// fallback is per *set* rather than per boat, so a yacht can compute its
    /// upwind forces from geometry and its downwind forces from a table, which is
    /// where each of the two is actually the better model.
    #[must_use]
    pub fn with_geometry(mut self, geometry: Vec<(Sail, Member)>) -> Self {
        self.geometry = geometry;
        self.geometric = None;
        self
    }

    /// Places the centre of effort along the hull.
    ///
    /// Consumed by the assembly from the boat file's layout block. Without it
    /// the rig drives the boat but cannot turn it.
    #[must_use]
    pub fn at(mut self, centre_of_effort_at: f64) -> Self {
        self.centre_of_effort_at = centre_of_effort_at;
        self
    }

    /// The sail plan currently in force, or `None` if the current set cannot be
    /// built on this rig.
    #[must_use]
    pub fn plan(&self) -> Option<&SailPlan> {
        self.plan.as_ref().ok()
    }

    /// Why the current sail set produces no force, if it produces none.
    ///
    /// [`ForceModule::step`] cannot return an error and must not panic — it
    /// runs inside a fixed-step loop, and a boat file that asks for a sail the
    /// rig has no dimensions for is a data problem, not a reason to abort a
    /// simulation. So the error is kept here, where a user interface, a test or
    /// a batch run can find out *why* the sails are doing nothing instead of
    /// inferring it from a zero.
    #[must_use]
    pub fn plan_error(&self) -> Option<SailPlanError> {
        self.plan.as_ref().err().copied()
    }

    /// Rebuilds the cached plan when the crew has changed sails.
    ///
    /// [`SailPlan`] says that changing sails means building a new plan and that
    /// this costs a handful of multiplications, so rebuilding on every step
    /// would be affordable. It is still not done, for a reason that is about
    /// clarity as much as cost: a plan depends on the rig and the set and on
    /// *nothing else* — not on the trim, which enters [`SailPlan::forces`], and
    /// not on the state. Keying the cache on the set alone therefore states
    /// that dependency in the code, and the check is one integer comparison
    /// against a rebuild at 120-plus steps a second.
    ///
    /// A set that failed to build is not retried while it is still the current
    /// set: the outcome is a deterministic function of the rig and the set, and
    /// the rig cannot change.
    fn refresh_plan(&mut self, set: SailSet) {
        if set == self.set {
            return;
        }
        self.set = set;
        self.plan = SailPlan::new(&self.rig, set);
    }

    /// The members for a sail set, or `None` if any sail in it has no shape data.
    ///
    /// All or nothing on purpose. Half a plan from geometry and half from a table
    /// would be two models sharing a reference area and double-counting the
    /// interaction between the sails they each think they own.
    fn members_for(&self, set: SailSet) -> Option<Vec<Member>> {
        if self.geometry.is_empty() || set.is_empty() {
            return None;
        }
        set.iter()
            .map(|sail| {
                self.geometry
                    .iter()
                    .find(|(which, _)| *which == sail)
                    .map(|(_, member)| *member)
            })
            .collect()
    }

    /// The apparent wind as a file-frame velocity, folded onto starboard tack.
    ///
    /// [`crate::sail`] builds its shapes cambered to port, which is a sail on
    /// starboard tack, so a port-tack boat is the same problem mirrored in `y`. The
    /// fold happens here and the un-fold happens to the force, which keeps the
    /// mirror in one place instead of putting a `±1` through the geometry.
    fn folded_flow(wind: ApparentWind) -> Vector3<f64> {
        let angle = wind.angle.abs();
        // A wind *from* `angle` off the bow blows aft and to leeward.
        Vector3::new(-angle.cos(), angle.sin(), 0.0) * wind.speed.max(0.0)
    }

    /// Rebuilds the geometric plan if the trim, the set or the wake has moved.
    ///
    /// # Reefing is geometry, so it is applied to the geometry
    ///
    /// The tabular model's reef factor scales an *area* and hands the same
    /// coefficients back. Here it scales the sail's **outline** — luff, foot and
    /// head chord together — because that is what shortening sail does, and a
    /// smaller similar sail has its own lift slope, its own aspect ratio and its own
    /// centre of effort rather than a fraction of the full sail's.
    ///
    /// The flattening factor is deliberately *not* read. Flattening is what an
    /// outhaul and a traveller do, and those are in the line positions already; a
    /// second scalar doing the same job would double-count whatever a caller set.
    ///
    /// Returns whether a plan is available for this set.
    fn refresh_geometry(
        &mut self,
        set: SailSet,
        wind: ApparentWind,
        trim: SailTrim,
        reef: f64,
    ) -> bool {
        let Some(members) = self.members_for(set) else {
            self.geometric = None;
            return false;
        };
        // A wake needs a direction and not a speed, so a becalmed boat can still
        // hold a factorisation: the angle survives the speed going to zero.
        let angle = wind.angle.abs();
        let wake = Vector3::new(-angle.cos(), angle.sin(), 0.0);
        let reef = reef.clamp(Self::MINIMUM_REEF, 1.0);

        let stale = match &self.geometric {
            Some((built, plan)) => {
                *built != (set, trim, reef.to_bits()) || plan.wake_drift(wake) > WAKE_TOLERANCE
            }
            None => true,
        };
        if !stale {
            return true;
        }

        let reefed: Option<Vec<Member>> = members
            .iter()
            .map(|member| {
                let full = member.planform;
                Planform::new(
                    full.luff() * reef,
                    full.foot() * reef,
                    full.head() * reef,
                    full.roach(),
                )
                .map(|planform| Member {
                    planform,
                    ..*member
                })
            })
            .collect();
        let Some(reefed) = reefed else {
            self.geometric = None;
            return false;
        };

        // One trim for every sail. A real crew trims the main and the headsail
        // separately, and `Plan` takes a control per sail so that it can; what is
        // missing is a place in `Controls` to say so, and inventing a split here
        // would be inventing a trimmer's choices.
        let controls = vec![trim; reefed.len()];
        self.geometric =
            Plan::new(reefed, &controls, wake).map(|plan| ((set, trim, reef.to_bits()), plan));
        self.geometric.is_some()
    }

    /// The geometric wrench, or `None` if this set has no geometry.
    ///
    /// # What comes from where
    ///
    /// Lift and induced drag come from [`crate::sail`], out of the geometry. The
    /// **viscous and parasitic drag do not** — a lattice has no viscosity, and the
    /// friction of cloth and the windage of mast and topsides are real forces that
    /// the tabular model already carries from Hazen's own coefficients. So they are
    /// added on top, at the tabular model's own centre of effort, which is exactly
    /// the composition §4.1 of the design describes.
    ///
    /// Leaving them out would have been the quiet mistake: a boat with the same
    /// lift and less drag sails faster, and the polar would improve for a reason
    /// that is not physics.
    fn geometric_wrench(
        &mut self,
        tabular: &SailPlan,
        set: SailSet,
        at: Condition,
    ) -> Option<(Wrench, SailForces, f64, Geometric)> {
        let Condition {
            wind,
            trim,
            shape,
            density,
            centre_of_effort,
        } = at;
        if !self.refresh_geometry(set, wind, shape, trim.reef) {
            return None;
        }
        let (_, plan) = self.geometric.as_ref()?;

        let flow = Self::folded_flow(wind);
        let file = plan.wrench(flow, density, Point::zeros());

        // Un-fold: on port tack the whole problem was mirrored in `y`, so the force
        // mirrors back and the moment — a pseudovector — mirrors the other way.
        let port = wind.angle < 0.0;
        let mirror_force = |v: Vector3<f64>| {
            if port {
                Vector3::new(v.x, -v.y, v.z)
            } else {
                v
            }
        };
        let mirror_moment = |v: Vector3<f64>| {
            if port {
                Vector3::new(-v.x, v.y, -v.z)
            } else {
                v
            }
        };

        let force_body = file_to_body(mirror_force(file.force));
        let moment_body = file_to_body(mirror_moment(file.moment));

        // The empirical drag the lattice cannot see, along the wind, at the tabular
        // model's own arm.
        let area = tabular.nominal_area();
        let dynamic = 0.5 * density * wind.speed * wind.speed * area;
        let empirical = tabular.viscous_drag_coefficient(wind.angle) * trim.reef
            + tabular.parasitic_drag_coefficient();
        let along = if wind.speed > 0.0 {
            flow / wind.speed
        } else {
            Vector3::zeros()
        };
        let extra = file_to_body(mirror_force(along * (empirical * dynamic)));
        let windage = Wrench::from_force_at(extra, centre_of_effort);

        // Lift and drag of the geometric part, in the wind's own axes, before the
        // mirror — which is where "to leeward" is unambiguous.
        let leeward = Vector3::new(along.y, -along.x, 0.0);
        let lift = file.force.dot(&leeward);
        let drag = file.force.dot(&along) + empirical * dynamic;
        let scale = if dynamic > 0.0 { 1.0 / dynamic } else { 0.0 };

        let heeling_force = lift * along.x.abs() - file.force.dot(&along) * along.y.abs();
        let sails = file.sails();
        let weight = |pick: fn(&crate::sail::Contribution) -> f64| {
            let total: f64 = sails.iter().map(|s| s.force.norm()).sum();
            if total <= 0.0 {
                return sails.first().map_or(0.0, pick);
            }
            sails.iter().map(|s| pick(s) * s.force.norm()).sum::<f64>() / total
        };

        let forces = SailForces {
            lift,
            drag,
            driving_force: force_body.x + extra.x,
            heeling_force: heeling_force.abs(),
            heeling_moment_about_waterline: heeling_force.abs() * (-centre_of_effort.z),
            centre_of_effort_height: -centre_of_effort.z,
            lift_coefficient: lift * scale,
            viscous_drag_coefficient: tabular.viscous_drag_coefficient(wind.angle) * trim.reef,
            induced_drag_coefficient: file.force.dot(&along) * scale,
            parasitic_drag_coefficient: tabular.parasitic_drag_coefficient(),
            aspect_ratio: tabular.aspect_ratio(trim.effective_span),
            nominal_area: area,
        };
        let reported = Geometric {
            mean_incidence: weight(|s| s.mean_incidence),
            separated_fraction: weight(|s| s.separated_fraction),
            // On the *nominal* area, like `aero.lift_coefficient`, and not on each
            // sail's own. The only reason to publish this number is to be compared
            // against the blended one, and two coefficients on two reference areas
            // are not comparable however alike their names look.
            attached_lift_coefficient: sails
                .iter()
                .zip(plan.members())
                .map(|(sail, member)| sail.attached_lift_coefficient * member.planform.area())
                .sum::<f64>()
                / area,
            wake_drift: plan.wake_drift(Vector3::new(
                -wind.angle.abs().cos(),
                wind.angle.abs().sin(),
                0.0,
            )),
            sails: sails.len(),
        };

        let side_force = force_body.y + extra.y;

        Some((
            Wrench::new(force_body + extra, moment_body + windage.moment),
            forces,
            side_force,
            reported,
        ))
    }
}

impl ForceModule for Sails {
    /// `aero`, not `sails`: this is the name the telemetry keys are namespaced
    /// under, and the namespace covers rig windage as well as cloth.
    fn name(&self) -> &'static str {
        "aero"
    }

    /// The aerodynamic wrench, body frame, about the body origin.
    ///
    /// # Why the model's heeling moment is not added
    ///
    /// [`crate::aero::SailForces::heeling_moment_about_waterline`] is the
    /// heeling force times the height of the centre of effort above the water —
    /// the very product [`Wrench::from_force_at`] forms below, from the same two
    /// numbers. Adding it to the wrench as well would double the roll moment of
    /// every sail force, and a doubled roll moment does not look like an error:
    /// the boat simply sails at the wrong heel, with a plausible-looking
    /// stability curve.
    ///
    /// It is also a moment about the **waterline**, not about the body origin,
    /// and it deliberately excludes the arm of the side force on keel and
    /// rudder, which acts *below* the waterline and belongs to
    /// [`super::lateral`]. So it is a reporting quantity — published in
    /// telemetry because it is what a trim display and the classical literature
    /// call *the* heeling moment — and never a contribution.
    fn step(&mut self, ctx: &StepCtx<'_>) -> Wrench {
        self.refresh_plan(ctx.controls.sails);

        let Ok(plan) = self.plan else {
            // No plan, no force, no panic. See `Sails::plan_error`.
            self.last = None;
            return Wrench::zero();
        };

        let trim = ctx.controls.trim;
        let air_density = ctx.env.air_density();

        // Where the centre of effort is depends on the sail set and on the reef
        // factor, and not at all on the wind — so one evaluation at zero wind
        // speed asks the model that purely geometric question and answers it
        // exactly. The alternative, re-deriving `reef · CoE + freeboard` here,
        // would copy a rule whose subtlety (the freeboard is added *after* the
        // reef factor, and the reef factor is clamped) lives in
        // `SailPlan::forces` and is free to change there. A second coefficient
        // lookup is a cheap price for that rule having exactly one home.
        let probe = ApparentWind {
            speed: 0.0,
            angle: 0.0,
        };
        let height = plan
            .forces(probe, trim, air_density)
            .centre_of_effort_height;

        // Above the water is *up*, and body `z` is down. The longitudinal
        // position comes from the boat file's layout block by way of the
        // assembly; a boat that declares none gets zero, which makes no yawing
        // moment and is exactly why such a boat has yaw restrained.
        let centre_of_effort = Vector3::new(self.centre_of_effort_at, 0.0, -height);

        // At the centre of effort, not at the origin: the rotational part of
        // that point's velocity is what makes the apparent wind fall as the rig
        // falls away to leeward, and dropping it removes the aerodynamic
        // damping of roll altogether.
        let wind = ctx.apparent_wind_at(centre_of_effort);

        // The geometric model first, when this set has shape data for every sail.
        // It replaces the lift and the induced drag with numbers computed from the
        // cloth's actual shape; it does not replace the empirical viscous and
        // parasitic terms, which it adds on top. See `Sails::geometric_wrench`.
        let condition = Condition {
            wind,
            trim,
            shape: ctx.controls.shape,
            density: air_density,
            centre_of_effort,
        };
        if let Some((wrench, forces, side_force, geometric)) =
            self.geometric_wrench(&plan, self.set, condition)
        {
            self.last = Some(Reported {
                wind,
                forces,
                side_force,
                geometric: Some(geometric),
            });
            return wrench;
        }
        let forces = plan.forces(wind, trim, air_density);

        // The model reports the heeling force positive **to leeward** and knows
        // nothing of tacks; the tack is in the sign of the apparent wind angle,
        // and `leeward_sign` is the single place in this engine that reads it.
        // Wind on the starboard bow (positive angle) puts leeward to port, so
        // the force lands on negative body `y` — the yacht heels away from the
        // wind, which is the whole of the sign's justification.
        let side_force = leeward_sign(wind.angle) * forces.heeling_force;

        // `driving_force` is positive forward and is not clamped: pinched up,
        // the drag term beats the lift term and the sails genuinely push the
        // yacht backwards.
        let force = Vector3::new(forces.driving_force, side_force, 0.0);

        self.last = Some(Reported {
            wind,
            forces,
            side_force,
            geometric: None,
        });

        // Both components at the centre of effort, so the heeling moment — and
        // any yaw moment the geometry ever gains — follows from the arm rather
        // than from a second hand-computed product.
        Wrench::from_force_at(force, centre_of_effort)
    }

    /// Publishes the breakdown under `aero.`.
    ///
    /// The whole breakdown, not the totals: a component-wise regression test
    /// compares the three drag coefficients separately, and a trim display that
    /// can only show a total cannot tell a crew *why* the boat is slow. The
    /// three drags sum to `aero.drag_coefficient` by
    /// [`crate::aero::SailForces::drag_coefficient`], so publishing all four is
    /// redundant on purpose — the sum is what a consumer plots, the parts are
    /// what a consumer diagnoses.
    ///
    /// Two keys need reading carefully, because they are not the same number:
    /// `aero.heeling_force` is the model's own value, positive **to leeward**
    /// and identical on both tacks, while `aero.side_force` is what went onto
    /// the body `y` axis, so its sign *is* the tack.
    ///
    /// After a step whose sail plan could not be built, only the forces that
    /// were applied — all zero — and `aero.sail_plan.valid` are published. The
    /// coefficients are omitted rather than reported as zeros, because the model
    /// did not produce a zero coefficient; it produced nothing at all.
    fn telemetry(&self, out: &mut Telemetry) {
        let Some(reported) = self.last.as_ref() else {
            out.set("aero.sail_plan.valid", 0.0);
            out.set("aero.driving_force", 0.0);
            out.set("aero.heeling_force", 0.0);
            out.set("aero.side_force", 0.0);
            return;
        };

        let forces = &reported.forces;
        out.set("aero.sail_plan.valid", 1.0);
        out.set("aero.apparent_wind.speed", reported.wind.speed);
        out.set("aero.apparent_wind.angle", reported.wind.angle);
        out.set("aero.driving_force", forces.driving_force);
        out.set("aero.heeling_force", forces.heeling_force);
        out.set("aero.side_force", reported.side_force);
        out.set("aero.lift", forces.lift);
        out.set("aero.drag", forces.drag);
        out.set("aero.lift_coefficient", forces.lift_coefficient);
        out.set("aero.drag_coefficient", forces.drag_coefficient());
        out.set(
            "aero.viscous_drag_coefficient",
            forces.viscous_drag_coefficient,
        );
        out.set(
            "aero.induced_drag_coefficient",
            forces.induced_drag_coefficient,
        );
        out.set(
            "aero.parasitic_drag_coefficient",
            forces.parasitic_drag_coefficient,
        );
        out.set("aero.aspect_ratio", forces.aspect_ratio);
        out.set("aero.nominal_area", forces.nominal_area);
        out.set(
            "aero.centre_of_effort.height",
            forces.centre_of_effort_height,
        );
        // Reporting only, and about the waterline. Not part of the wrench —
        // see `ForceModule::step`.
        out.set(
            "aero.heeling_moment_about_waterline",
            forces.heeling_moment_about_waterline,
        );

        // Which model ran, and the quantities only the geometric one has. A caller
        // comparing the two needs to see the mean incidence and the separated
        // fraction, because those are what say *why* the answers differ — and
        // `attached_lift_coefficient` against `lift_coefficient` is the whole
        // empirical content of the geometric answer, in two numbers.
        let Some(geometric) = reported.geometric else {
            out.set("aero.geometric.active", 0.0);
            return;
        };
        out.set("aero.geometric.active", 1.0);
        out.set("aero.geometric.sails", geometric.sails as f64);
        out.set("aero.geometric.mean_incidence", geometric.mean_incidence);
        out.set(
            "aero.geometric.separated_fraction",
            geometric.separated_fraction,
        );
        out.set(
            "aero.geometric.attached_lift_coefficient",
            geometric.attached_lift_coefficient,
        );
        out.set("aero.geometric.wake_drift", geometric.wake_drift);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aero::Sail;
    use crate::controls::Controls;
    use crate::env::{Environment, StillWater, UniformWind};
    use crate::state::BodyState;
    use approx::assert_relative_eq;

    /// Apparent wind speed and boat speed for the sign tests. Arbitrary: every
    /// assertion below is a sign, a mirror or an inequality, so the dynamic
    /// pressure cancels out of all of them.
    const WIND_SPEED: f64 = 8.0;
    const BOAT_SPEED: f64 = 3.0;

    /// True wind bearing, degrees, for a boat heading north: the wind is then
    /// on the starboard bow and the yacht is on starboard tack. Negated for the
    /// mirror case.
    const FROM_STARBOARD: f64 = 30.0;

    /// The YD-41 rig, the same dimensions the [`crate::aero`] validation suite
    /// uses. Only signs and relations are asserted on it here, so the four
    /// dimensions that are assumed rather than published do no harm.
    fn rig() -> RigDimensions {
        RigDimensions {
            main_hoist: 16.7,
            main_foot: 5.6,
            foretriangle_height: 16.2,
            foretriangle_base: 5.1,
            jib_perpendicular: 16.2 * 5.1 / 16.2_f64.hypot(5.1),
            spinnaker_leech: 18.0,
            boom_above_sheer: 1.0,
            max_beam: 4.20,
            average_freeboard: 1.2,
            mast_height_above_sheer: 17.7,
            mast_diameter: 0.15,
            mizzen: None,
        }
    }

    /// Upright, heading north, making way, rolling at `roll_rate` (rad/s,
    /// positive starboard-down).
    fn state(roll_rate: f64) -> BodyState {
        let mut state = BodyState::at_rest(Vector3::zeros());
        state.velocity = Vector3::new(BOAT_SPEED, 0.0, 0.0);
        state.angular_velocity = Vector3::new(roll_rate, 0.0, 0.0);
        state
    }

    fn wind_from(degrees: f64) -> StillWater {
        StillWater::new(UniformWind::uniform(WIND_SPEED, degrees.to_radians()))
    }

    fn step_once(
        sails: &mut Sails,
        controls: &Controls,
        state: &BodyState,
        env: &dyn Environment,
    ) -> (Wrench, Telemetry) {
        let ctx = StepCtx {
            state,
            controls,
            env,
            time: 0.0,
            dt: 1.0 / 120.0,
        };
        let wrench = sails.step(&ctx);
        let mut telemetry = Telemetry::new();
        sails.telemetry(&mut telemetry);
        (wrench, telemetry)
    }

    /// One step of a fresh module: the common case of every test here.
    fn sail(wind_bearing: f64, set: SailSet, roll_rate: f64) -> (Wrench, Telemetry) {
        let mut sails = Sails::new(rig());
        let controls = Controls::close_hauled(set);
        let env = wind_from(wind_bearing);
        step_once(&mut sails, &controls, &state(roll_rate), &env)
    }

    fn value(telemetry: &Telemetry, key: &str) -> f64 {
        telemetry
            .get(key)
            .unwrap_or_else(|| panic!("{key} should be published"))
    }

    #[test]
    fn close_hauled_on_starboard_tack_drives_forward_and_heels_to_port() {
        let (wrench, telemetry) = sail(FROM_STARBOARD, SailSet::upwind(), 0.0);

        // The premise: the wind really is on the starboard bow.
        assert!(value(&telemetry, "aero.apparent_wind.angle") > 0.0);

        assert!(wrench.force.x > 0.0, "close-hauled sails drive forward");
        assert!(wrench.force.y < 0.0, "leeward is to port on starboard tack");

        // The model's heeling force is tack-free and positive to leeward; the
        // body-frame force is the same magnitude, signed by the tack.
        let heeling = value(&telemetry, "aero.heeling_force");
        assert!(heeling > 0.0);
        assert_relative_eq!(wrench.force.y, -heeling);
        // Telemetry reports what was applied, not a recomputation.
        assert_relative_eq!(value(&telemetry, "aero.side_force"), wrench.force.y);
        assert_relative_eq!(value(&telemetry, "aero.driving_force"), wrench.force.x);
    }

    #[test]
    fn the_centre_of_effort_being_above_the_water_sets_the_moments() {
        let (wrench, telemetry) = sail(FROM_STARBOARD, SailSet::upwind(), 0.0);
        let height = value(&telemetry, "aero.centre_of_effort.height");
        assert!(height > 0.0, "the rig is above the water");

        // Heeling to port is a negative roll moment: body x is forward and
        // positive roll is starboard-down.
        assert!(wrench.moment.x < 0.0);
        // And it is exactly the arm times the force, which is the reason the
        // model's own `heeling_moment_about_waterline` must not be added too.
        assert_relative_eq!(wrench.moment.x, height * wrench.force.y);
        assert_relative_eq!(
            wrench.moment.x.abs(),
            value(&telemetry, "aero.heeling_moment_about_waterline")
        );

        // Drive applied above the waterline pitches the bow down: a positive
        // moment about the starboard-pointing y axis lifts the bow.
        assert_relative_eq!(wrench.moment.y, -height * wrench.force.x);
        assert!(wrench.moment.y < 0.0);

        // No yaw moment: the model gives the centre of effort no longitudinal
        // position, and this module refuses to invent one.
        assert_relative_eq!(wrench.moment.z, 0.0);
    }

    #[test]
    fn the_two_tacks_are_mirror_images() {
        let (starboard, _) = sail(FROM_STARBOARD, SailSet::upwind(), 0.0);
        let (port, telemetry) = sail(-FROM_STARBOARD, SailSet::upwind(), 0.0);

        assert!(value(&telemetry, "aero.apparent_wind.angle") < 0.0);
        assert!(port.force.y > 0.0, "leeward is to starboard on port tack");

        assert_relative_eq!(port.force.x, starboard.force.x, epsilon = 1e-12);
        assert_relative_eq!(port.force.y, -starboard.force.y, epsilon = 1e-12);
        assert_relative_eq!(port.moment.x, -starboard.moment.x, epsilon = 1e-12);
        assert_relative_eq!(port.moment.y, starboard.moment.y, epsilon = 1e-12);
    }

    #[test]
    fn bare_poles_still_makes_windage_drag_and_no_lift() {
        let (wrench, telemetry) = sail(FROM_STARBOARD, SailSet::none(), 0.0);

        assert_relative_eq!(value(&telemetry, "aero.lift"), 0.0);
        assert_relative_eq!(value(&telemetry, "aero.lift_coefficient"), 0.0);
        assert_relative_eq!(value(&telemetry, "aero.viscous_drag_coefficient"), 0.0);
        assert_relative_eq!(value(&telemetry, "aero.induced_drag_coefficient"), 0.0);

        // Mast and topsides are still standing there.
        assert!(value(&telemetry, "aero.parasitic_drag_coefficient") > 0.0);
        assert!(value(&telemetry, "aero.drag") > 0.0);
        assert!(wrench.force.x < 0.0, "windage forward of the beam retards");
        assert!(wrench.force.y < 0.0, "and still pushes to leeward");

        // The arm is the freeboard alone: with no sail area there is no
        // centroid to take, and the model does not locate the windage centre.
        assert_relative_eq!(
            value(&telemetry, "aero.centre_of_effort.height"),
            rig().average_freeboard
        );
    }

    #[test]
    fn a_sail_plan_that_cannot_be_built_makes_no_force_instead_of_panicking() {
        // A sloop cannot set a mizzen staysail, and the model has no area for
        // one on any rig.
        let mut sails = Sails::new(rig());
        let controls = Controls::close_hauled(SailSet::upwind().with(Sail::MizzenStaysail));
        let env = wind_from(FROM_STARBOARD);
        let (wrench, telemetry) = step_once(&mut sails, &controls, &state(0.0), &env);

        assert_eq!(wrench, Wrench::zero());
        assert_relative_eq!(value(&telemetry, "aero.sail_plan.valid"), 0.0);
        assert_relative_eq!(value(&telemetry, "aero.driving_force"), 0.0);
        assert!(
            telemetry.get("aero.lift_coefficient").is_none(),
            "a coefficient that was never computed is not published as zero"
        );
        assert_eq!(
            sails.plan_error(),
            Some(SailPlanError::AreaUnavailable(Sail::MizzenStaysail))
        );
        assert!(sails.plan().is_none());
    }

    #[test]
    fn a_degenerate_rig_fails_at_construction_without_panicking() {
        let mut sails = Sails::new(RigDimensions {
            main_hoist: 0.0,
            ..rig()
        });
        assert!(sails.plan_error().is_some());

        let controls = Controls::close_hauled(SailSet::upwind());
        let env = wind_from(FROM_STARBOARD);
        let (wrench, telemetry) = step_once(&mut sails, &controls, &state(0.0), &env);
        assert_eq!(wrench, Wrench::zero());
        assert_relative_eq!(value(&telemetry, "aero.sail_plan.valid"), 0.0);
    }

    #[test]
    fn changing_sails_rebuilds_the_plan() {
        let mut sails = Sails::new(rig());
        let env = wind_from(FROM_STARBOARD);

        let upwind = Controls::close_hauled(SailSet::upwind());
        let (_, telemetry) = step_once(&mut sails, &upwind, &state(0.0), &env);
        let with_jib = value(&telemetry, "aero.lift_coefficient");
        assert_eq!(sails.plan().map(SailPlan::set), Some(SailSet::upwind()));

        let downwind = Controls::close_hauled(SailSet::downwind());
        let (_, telemetry) = step_once(&mut sails, &downwind, &state(0.0), &env);
        let with_spinnaker = value(&telemetry, "aero.lift_coefficient");
        assert_eq!(sails.plan().map(SailPlan::set), Some(SailSet::downwind()));

        // Close-hauled, a spinnaker makes no lift where a jib makes its most,
        // so the cache cannot have been reused.
        assert!(with_spinnaker < with_jib);
    }

    #[test]
    fn rolling_to_leeward_reduces_the_apparent_wind_at_the_centre_of_effort() {
        // The reason the apparent wind is taken at the centre of effort rather
        // than at the origin: on starboard tack leeward is to port, so a
        // negative roll rate carries the rig away from the wind and must take
        // the force with it. Without that, roll has no aerodynamic damping at
        // all and a gust sets the boat rolling for ever.
        let (steady, steady_telemetry) = sail(FROM_STARBOARD, SailSet::upwind(), 0.0);
        let (falling, falling_telemetry) = sail(FROM_STARBOARD, SailSet::upwind(), -0.3);

        assert!(
            value(&falling_telemetry, "aero.apparent_wind.speed")
                < value(&steady_telemetry, "aero.apparent_wind.speed")
        );
        assert!(falling.force.y.abs() < steady.force.y.abs());

        // And rolling to windward increases it, which is the same mechanism
        // with the other sign — not a one-sided fudge.
        let (climbing, climbing_telemetry) = sail(FROM_STARBOARD, SailSet::upwind(), 0.3);
        assert!(
            value(&climbing_telemetry, "aero.apparent_wind.speed")
                > value(&steady_telemetry, "aero.apparent_wind.speed")
        );
        assert!(climbing.force.y.abs() > steady.force.y.abs());
    }
}
