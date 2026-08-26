//! Hydrostatic pressure over the wetted hull: the vertical force and the
//! restoring moments.
//!
//! # Provenance
//!
//! Every number here comes from [`crate::hydrostatics::hydrostatics`], which
//! returns the exact body-frame wrench at any pose in closed form. This module
//! adds no coefficient, no correction and no constant of its own — density and
//! gravity are read from the [`crate::env::Environment`], the geometry is the
//! hull mesh it was constructed with, and nothing else enters. Its value is in
//! the boundary it documents and the breakdown it publishes, not in arithmetic.
//!
//! # Why the module exists
//!
//! Without it, heel and sinkage would have to be *prescribed*: handed to the
//! force models as inputs, the way a velocity prediction program is given an
//! attitude to evaluate. With it they are genuine degrees of freedom, because
//! the righting moment is not a stiffness coefficient but a consequence of
//! which triangles happen to be under water at this instant. That is the whole
//! point of integrating pressure over a clipped mesh rather than tabulating a
//! GZ curve — a knockdown, the loss of stability past a submerged deck edge,
//! and the recovery from either come out of the geometry instead of having to
//! be modelled one at a time.
//!
//! # The free surface this module sits on
//!
//! There are two descriptions of the water surface in this engine, and this
//! module is exactly where they meet:
//!
//! - [`crate::hydrostatics`] integrates against the **flat world plane
//!   `z = 0`**, and depends on that flatness twice: the pressure it integrates
//!   is `ρ g z` with no unsteady term, and the waterplane lid can be left
//!   ungenerated only because it is coplanar with the world origin and lies
//!   where `p = 0`.
//! - [`crate::env::Environment::depth`] is the **general signed-depth
//!   surface** — positive below, negative above, zero on it — which a wave
//!   field will supply.
//!
//! Today the two agree exactly. The water is still, so `depth` returns the
//! world `z` of the point and the flat plane *is* the free surface; calling
//! `hydrostatics` here is not an approximation. But that agreement is a
//! property of still water, not a property of this module, and it is worth
//! being explicit about which of the two is being relied on.
//!
//! The generalisation path runs through [`crate::clip`], whose signed-depth
//! closure is already the right shape: clipping against `|p| env.depth(p, t)`
//! instead of `|p| p.z` gives the wetted geometry under a wave, and the
//! pressure integral then needs the incident-wave pressure in place of `ρ g z`.
//! Both changes live *inside* `hydrostatics`. This module's interface does not
//! move — `step` would call the same function with the same arguments.
//!
//! # What is therefore missing today
//!
//! **The Froude-Krylov force of an incident wave.** A passing wave changes both
//! the wetted geometry and the pressure field within it, and neither effect is
//! present here. In flat water that absence costs nothing; in a seaway it is
//! the dominant wave-exciting force, so any seakeeping result obtained from
//! this module as it stands is not a seakeeping result. It is deliberately not
//! implemented and deliberately not approximated: a plausible stand-in would be
//! indistinguishable from the real term inside a total, which is the one kind
//! of error this engine has no way to catch.
//!
//! Diffraction and radiation are absent for the same reason and belong
//! elsewhere in any case — radiation is its own module with its own memory
//! states and its own added-mass contribution, not a correction to
//! hydrostatics.
//!
//! Also absent, by ownership rather than by omission:
//!
//! - **Resistance of any kind.** Hydrostatic pressure on a rigid hull under a
//!   flat surface is normal to the surface and integrates to a purely vertical
//!   resultant, so there is no drag to be had here even in principle. The
//!   canoe body's resistance belongs to [`crate::modules::hull`] and the
//!   appendages' to [`crate::modules::lateral`].
//! - **Weight.** [`crate::rigid_body::RigidBody`] owns gravity because it is
//!   exact. The wrench returned here is buoyancy alone and therefore does
//!   **not** vanish at equilibrium; it cancels the weight wrench, which this
//!   module never sees.

use crate::geometry::TriMesh;
use crate::hydrostatics::{hydrostatics, Hydrostatics, Water};
use crate::sim::{ForceModule, StepCtx};
use crate::telemetry::Telemetry;
use crate::wrench::Wrench;

/// Buoyancy of a hull mesh, re-integrated from the geometry every step.
pub struct Buoyancy {
    /// The hull, in the **body frame**.
    ///
    /// Closed and outward-oriented by the [`TriMesh`] contract. The winding is
    /// load-bearing: `hydrostatics` reads it as the direction pressure pushes
    /// against, so an inward-wound mesh would produce a hull that sinks with
    /// perfect internal consistency. Orientation is the builder's job
    /// ([`crate::loft`] does it) and is not re-checked on every step, because a
    /// per-step check would cost a full surface integral to catch a
    /// construction-time bug.
    mesh: TriMesh,

    /// What the last [`ForceModule::step`] found, retained only so that
    /// [`ForceModule::telemetry`] can report it.
    ///
    /// `None` before the first step, and telemetry then publishes nothing at
    /// all rather than zeros. A zero displacement is a boat out of the water —
    /// a physical claim this module has no business making before it has looked
    /// at a pose.
    last: Option<Snapshot>,
}

/// One step's hydrostatics together with the fluid it was computed in.
///
/// The water and gravity are kept alongside because two of the published
/// quantities — displacement and heave stiffness — are properties of the hull
/// *and* the fluid, while [`ForceModule::telemetry`] receives no [`StepCtx`]
/// from which to recover them.
struct Snapshot {
    hydrostatics: Hydrostatics,
    water: Water,
    gravity: f64,
}

impl Buoyancy {
    /// Wraps a hull mesh given in the **body frame**.
    ///
    /// Takes no centre of buoyancy, no application point and no stiffness.
    /// Unlike the appendage and sail modules, this one has nothing it could be
    /// told: the centre of buoyancy is an *output* of the geometry at each
    /// pose, and accepting one as a parameter would be accepting a number the
    /// geometry is free to contradict.
    #[must_use]
    pub fn new(mesh: TriMesh) -> Self {
        Self { mesh, last: None }
    }

    /// The hydrostatics of the most recent step, or `None` before the first.
    ///
    /// Exposed as the struct rather than only as telemetry strings because the
    /// consumers that care most about these numbers — a trim display, a
    /// component-wise regression against an oracle — want them typed, and
    /// re-deriving them by looking up keys would reintroduce the stringly
    /// typed layer this avoids.
    #[must_use]
    pub fn last(&self) -> Option<&Hydrostatics> {
        self.last.as_ref().map(|snapshot| &snapshot.hydrostatics)
    }
}

impl ForceModule for Buoyancy {
    fn name(&self) -> &'static str {
        "buoyancy"
    }

    fn step(&mut self, ctx: &StepCtx<'_>) -> Wrench {
        // Sampled once: the environment is behind a trait object, and the same
        // fluid must underlie both the wrench and the derived quantities
        // telemetry publishes from it.
        let water = ctx.env.water();
        let gravity = ctx.env.gravity();

        // The full pose is passed straight through, and this module reads
        // neither `ctx.heel()` nor `ctx.trim_angle()`. That is not the forbidden
        // direction of the same mistake: a pressure integral needs the attitude
        // itself, and rebuilding a rotation out of Euler angles would create a
        // second description of the pose free to disagree with the first.
        let snapshot = Snapshot {
            hydrostatics: hydrostatics(&self.mesh, ctx.state, &water, gravity),
            water,
            gravity,
        };

        let buoyancy = snapshot.hydrostatics.buoyancy;
        self.last = Some(snapshot);
        buoyancy
    }

    /// Publishes the flotation state under `buoyancy.hull.*`.
    ///
    /// The full wrench is published, not just the vertical force, because at
    /// heel the buoyant force has body-frame `y` and `z` components that a
    /// single "vertical force" would hide, and because the yaw moment being
    /// identically zero in still water is an invariant worth being able to
    /// watch rather than assert once in a test.
    ///
    /// Forces and moments are named by their degree of freedom — surge, sway,
    /// heave, roll, pitch, yaw — which is what the equations of motion call
    /// them. The centre of buoyancy is published as body-frame coordinates
    /// (`cob_x` longitudinal, `cob_y` transverse, `cob_z` vertical, positive
    /// down) rather than as LCB/TCB/VCB, so that the frame is unambiguous in
    /// the key itself.
    fn telemetry(&self, out: &mut Telemetry) {
        let Some(Snapshot {
            hydrostatics,
            water,
            gravity,
        }) = self.last.as_ref()
        else {
            return;
        };

        out.set("buoyancy.hull.volume", hydrostatics.volume);
        out.set(
            "buoyancy.hull.displacement",
            hydrostatics.displacement(water),
        );
        out.set("buoyancy.hull.wetted_area", hydrostatics.wetted_area);
        out.set(
            "buoyancy.hull.waterplane_area",
            hydrostatics.waterplane_area,
        );
        out.set(
            "buoyancy.hull.heave_stiffness",
            hydrostatics.heave_stiffness(water, *gravity),
        );

        let centre = hydrostatics.centre_of_buoyancy;
        out.set("buoyancy.hull.cob_x", centre.x);
        out.set("buoyancy.hull.cob_y", centre.y);
        out.set("buoyancy.hull.cob_z", centre.z);

        let Wrench { force, moment } = hydrostatics.buoyancy;
        out.set("buoyancy.hull.surge_force", force.x);
        out.set("buoyancy.hull.sway_force", force.y);
        out.set("buoyancy.hull.heave_force", force.z);
        out.set("buoyancy.hull.roll_moment", moment.x);
        out.set("buoyancy.hull.pitch_moment", moment.y);
        out.set("buoyancy.hull.yaw_moment", moment.z);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aero::SailSet;
    use crate::controls::Controls;
    use crate::env::{Environment, StillWater, UniformWind};
    use crate::geometry::Point;
    use crate::state::BodyState;
    use approx::assert_relative_eq;
    use nalgebra::UnitQuaternion;

    /// An axis-aligned box, closed and outward oriented, spanning `lo..hi`.
    ///
    /// Written here because there is nothing public to reuse: [`crate::loft`]
    /// builds hulls from station offsets, and [`crate::geometry`] keeps a box
    /// only inside its own test module. A box is also the right shape for these
    /// tests — it is the only one whose displaced volume, centre of buoyancy
    /// and waterplane area can be written down by hand *at heel* as well as
    /// upright, which is what lets a sign be checked against a number rather
    /// than against another implementation.
    fn box_mesh(lo: Point, hi: Point) -> TriMesh {
        let vertices = vec![
            Point::new(lo.x, lo.y, lo.z),
            Point::new(hi.x, lo.y, lo.z),
            Point::new(hi.x, hi.y, lo.z),
            Point::new(lo.x, hi.y, lo.z),
            Point::new(lo.x, lo.y, hi.z),
            Point::new(hi.x, lo.y, hi.z),
            Point::new(hi.x, hi.y, hi.z),
            Point::new(lo.x, hi.y, hi.z),
        ];
        let indices = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        let mut mesh = TriMesh::new(vertices, indices);
        mesh.orient_outward();
        mesh
    }

    /// A barge 6 m long and 4 m in beam, 2 m deep, straddling `z = 0` so that
    /// the body origin sits on the upright waterplane and half of it — 24 m³ —
    /// is immersed.
    ///
    /// The section is symmetric under a half turn about the body `x` axis, so
    /// any plane through the origin cuts it into congruent halves: the
    /// displaced volume is exactly 24 m³ at *any* heel that keeps the cut
    /// inside the box. That makes a heel test a test of the moment alone,
    /// rather than of the moment tangled up with a change of displacement.
    fn barge() -> TriMesh {
        box_mesh(Point::new(-3.0, -2.0, -1.0), Point::new(3.0, 2.0, 1.0))
    }

    const HALF_IMMERSED_VOLUME: f64 = 24.0;

    fn environment() -> StillWater {
        // Nothing in this module reads the wind, but an `Environment` supplies
        // one, and a calm makes it obvious that no aerodynamic term leaks in.
        StillWater::new(UniformWind::uniform(0.0, 0.0))
    }

    /// The buoyant force magnitude the environment implies for a given volume.
    /// Derived from the environment, never written as a literal.
    fn archimedes(env: &StillWater, volume: f64) -> f64 {
        env.water().density * env.gravity() * volume
    }

    fn pose(sinkage: f64, heel: f64) -> BodyState {
        BodyState {
            position: Point::new(0.0, 0.0, sinkage),
            attitude: UnitQuaternion::from_euler_angles(heel, 0.0, 0.0),
            ..BodyState::default()
        }
    }

    fn step(module: &mut Buoyancy, state: &BodyState, env: &StillWater) -> Wrench {
        let controls = Controls::close_hauled(SailSet::upwind());
        let ctx = StepCtx {
            state,
            controls: &controls,
            env,
            time: 0.0,
            dt: 0.01,
        };
        module.step(&ctx)
    }

    #[test]
    fn the_module_is_named_for_its_telemetry_prefix() {
        assert_eq!(Buoyancy::new(barge()).name(), "buoyancy");
    }

    #[test]
    fn a_half_immersed_hull_is_pushed_up_with_the_weight_of_the_water_it_moves() {
        let env = environment();
        let mut module = Buoyancy::new(barge());
        let wrench = step(&mut module, &pose(0.0, 0.0), &env);

        let expected = archimedes(&env, HALF_IMMERSED_VOLUME);
        // Up is negative z: this is the single sign that, wrong, would make
        // every hull in the engine sink while looking entirely reasonable.
        assert!(wrench.force.z < 0.0, "buoyancy must act upward, towards -z");
        assert_relative_eq!(wrench.force.z, -expected, epsilon = 1e-6);

        // Upright and symmetric: no horizontal resultant, and in particular no
        // resistance, which this module does not own.
        assert_relative_eq!(wrench.force.x, 0.0, epsilon = 1e-6);
        assert_relative_eq!(wrench.force.y, 0.0, epsilon = 1e-6);
        assert_relative_eq!(wrench.moment.x, 0.0, epsilon = 1e-6);
        assert_relative_eq!(wrench.moment.y, 0.0, epsilon = 1e-6);
        assert_relative_eq!(wrench.moment.z, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn the_buoyant_force_is_read_from_the_environment_not_baked_in() {
        // Halving the density must halve the force exactly. Nothing in this
        // module may hold a density of its own for that to hold.
        let heavy = environment();
        let light = environment().with_water(Water {
            density: heavy.water().density / 2.0,
        });

        let mut module = Buoyancy::new(barge());
        let in_heavy = step(&mut module, &pose(0.0, 0.0), &heavy);
        let in_light = step(&mut module, &pose(0.0, 0.0), &light);

        assert_relative_eq!(in_light.force.z, in_heavy.force.z / 2.0, epsilon = 1e-9);
    }

    #[test]
    fn a_hull_lifted_clear_of_the_water_contributes_exactly_nothing() {
        let env = environment();
        let mut module = Buoyancy::new(barge());
        // Keel at world z = -1.5, deck at -3.5: entirely above the surface.
        let wrench = step(&mut module, &pose(-2.5, 0.0), &env);

        // Exact, not approximate: clipping yields no triangles at all, so the
        // sums are untouched. A near-zero here would mean a panel had been
        // wrongly retained.
        assert_eq!(wrench, Wrench::zero());
        assert_relative_eq!(module.last().expect("stepped").volume, 0.0);
    }

    #[test]
    fn heeling_a_symmetric_hull_produces_a_moment_opposing_the_heel() {
        let env = environment();
        let mut module = Buoyancy::new(barge());
        let heel = 10.0_f64.to_radians();

        let heeled = step(&mut module, &pose(0.0, heel), &env);
        let displaced = module.last().expect("stepped").volume;

        // The wedge that immerses to starboard is congruent to the one that
        // emerges to port, so displacement is untouched and the moment below is
        // a pure shift of the centre of buoyancy.
        assert_relative_eq!(displaced, HALF_IMMERSED_VOLUME, epsilon = 1e-9);

        // Positive heel is starboard down, so a restoring roll moment is
        // negative. This is the term that makes heel a degree of freedom.
        assert!(
            heeled.moment.x < 0.0,
            "roll moment must oppose positive heel, got {}",
            heeled.moment.x
        );

        // Buoyancy is world-vertical, so in the body frame of a boat heeled to
        // starboard it leans to port. Getting this backwards would add a
        // spurious side force in exactly the direction the sails push.
        assert!(heeled.force.y < 0.0, "buoyancy leans to port when heeled");
        assert_relative_eq!(
            heeled.force.norm(),
            archimedes(&env, HALF_IMMERSED_VOLUME),
            epsilon = 1e-6
        );

        // A hull symmetric about its centreline must right itself equally on
        // either tack.
        let to_port = step(&mut module, &pose(0.0, -heel), &env);
        assert!(to_port.moment.x > 0.0);
        assert_relative_eq!(to_port.moment.x, -heeled.moment.x, epsilon = 1e-6);
        assert_relative_eq!(to_port.force.y, -heeled.force.y, epsilon = 1e-6);
    }

    #[test]
    fn the_vertical_force_grows_monotonically_with_immersion() {
        let env = environment();
        let mut module = Buoyancy::new(barge());

        // From barely wetted to nearly submerged, in the same pose otherwise.
        let mut previous = 0.0;
        for tenth in 1..=19 {
            let sinkage = -1.0 + 0.1 * f64::from(tenth);
            let upward = -step(&mut module, &pose(sinkage, 0.0), &env).force.z;
            assert!(
                upward > previous,
                "immersing further must push harder: {upward} at sinkage {sinkage} \
                 did not exceed {previous}"
            );
            previous = upward;
        }

        // The end of the sweep is a hull immersed 1.9 m of its 2 m depth.
        assert_relative_eq!(previous, archimedes(&env, 6.0 * 4.0 * 1.9), epsilon = 1e-6);
    }

    #[test]
    fn the_moment_is_taken_about_the_body_origin_not_the_centre_of_buoyancy() {
        // A box whose centre of buoyancy sits 4 m forward and 2.5 m to
        // starboard of the body origin, still half immersed.
        let env = environment();
        let mut module = Buoyancy::new(box_mesh(
            Point::new(1.0, 0.5, -1.0),
            Point::new(7.0, 4.5, 1.0),
        ));
        let wrench = step(&mut module, &pose(0.0, 0.0), &env);
        let force = archimedes(&env, HALF_IMMERSED_VOLUME);

        let centre = module.last().expect("stepped").centre_of_buoyancy;
        assert_relative_eq!(centre, Point::new(4.0, 2.5, 0.5), epsilon = 1e-9);

        // r x F with r the centre of buoyancy and F pointing up (-z). Referring
        // the moment to the centre of buoyancy instead would give zero here,
        // and a boat whose trim never responded to where its volume is.
        assert_relative_eq!(wrench.moment.x, -2.5 * force, epsilon = 1e-6);
        assert_relative_eq!(wrench.moment.y, 4.0 * force, epsilon = 1e-6);
        assert_relative_eq!(wrench.moment.z, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn telemetry_is_silent_until_the_module_has_looked_at_a_pose() {
        let module = Buoyancy::new(barge());
        let mut out = Telemetry::new();
        module.telemetry(&mut out);
        assert!(out.is_empty(), "a zero displacement is not a safe default");
    }

    #[test]
    fn telemetry_reports_the_flotation_that_was_just_computed() {
        let env = environment();
        let mut module = Buoyancy::new(barge());
        let wrench = step(&mut module, &pose(0.0, 0.0), &env);

        let mut out = Telemetry::new();
        module.telemetry(&mut out);

        let get = |key: &str| out.get(key).unwrap_or_else(|| panic!("missing {key}"));

        assert_relative_eq!(
            get("buoyancy.hull.volume"),
            HALF_IMMERSED_VOLUME,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            get("buoyancy.hull.displacement"),
            env.water().density * HALF_IMMERSED_VOLUME,
            epsilon = 1e-6
        );
        // Bottom, two sides and two ends of the immersed half: 24 + 2*6 + 2*4.
        assert_relative_eq!(get("buoyancy.hull.wetted_area"), 44.0, epsilon = 1e-9);
        assert_relative_eq!(get("buoyancy.hull.waterplane_area"), 24.0, epsilon = 1e-9);
        assert_relative_eq!(
            get("buoyancy.hull.heave_stiffness"),
            archimedes(&env, 1.0) * 24.0,
            epsilon = 1e-6
        );

        assert_relative_eq!(get("buoyancy.hull.cob_x"), 0.0, epsilon = 1e-9);
        assert_relative_eq!(get("buoyancy.hull.cob_y"), 0.0, epsilon = 1e-9);
        assert_relative_eq!(get("buoyancy.hull.cob_z"), 0.5, epsilon = 1e-9);

        // The published wrench is the wrench that was returned, not a second
        // computation that could drift from it.
        assert_relative_eq!(get("buoyancy.hull.surge_force"), wrench.force.x);
        assert_relative_eq!(get("buoyancy.hull.sway_force"), wrench.force.y);
        assert_relative_eq!(get("buoyancy.hull.heave_force"), wrench.force.z);
        assert_relative_eq!(get("buoyancy.hull.roll_moment"), wrench.moment.x);
        assert_relative_eq!(get("buoyancy.hull.pitch_moment"), wrench.moment.y);
        assert_relative_eq!(get("buoyancy.hull.yaw_moment"), wrench.moment.z);
    }
}
