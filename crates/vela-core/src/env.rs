//! The environment a boat sails in: wind, water, and the free surface.
//!
//! # Why a trait
//!
//! Force modules never ask *what kind* of sea they are in. Still water today
//! and an FFT wave field later differ only in the implementation behind
//! [`Environment::depth`] and [`Environment::wind`], so nothing downstream has
//! to change when waves arrive. The signed-depth signature is the same
//! generalization point [`crate::clip`] already builds on.
//!
//! # Conventions
//!
//! Positions are in the **world frame** (NED: x north, y east, z down), so a
//! point below the still-water surface has positive `z`, and
//! [`Environment::depth`] returns a positive number for it.
//!
//! Wind is returned as a **velocity vector of the air**, in the world frame —
//! not as a speed and a bearing. That is deliberate: "the wind is 15 knots from
//! 220°" needs a convention to become a vector, and having every caller apply
//! that convention itself is how sign errors get in. The conversion happens
//! once, in [`UniformWind`].

use crate::hydrostatics::Water;
use crate::seaway::{SeaState, Seaway};
use nalgebra::Vector3;

/// Everything outside the boat.
///
/// `Send + Sync` for the same reason [`crate::sim::ForceModule`] is: a [`Sim`]
/// holds one as a trait object, and that bound is what lets a simulation be
/// owned by a frontend, a worker thread, or a host running several at once.
/// Every environment here is a handful of numbers and a wave realisation, so the
/// bound is free.
///
/// [`Sim`]: crate::sim::Sim
pub trait Environment: Send + Sync {
    /// True wind velocity of the air at a world-frame point, m/s.
    ///
    /// A vector, not a speed: see the module documentation. The vertical
    /// component is zero for every implementation here, but the signature
    /// allows one because a gust front does not have to be horizontal.
    fn wind(&self, position: Vector3<f64>, time: f64) -> Vector3<f64>;

    /// Signed depth of a world-frame point below the free surface, m —
    /// positive below, negative above, zero on it.
    fn depth(&self, position: Vector3<f64>, time: f64) -> f64;

    /// Pressure at a world-frame point divided by `ρ g`, m.
    ///
    /// Defaults to the depth, which is exactly right for still water — and is a
    /// separate question from it in a seaway, where a wave's dynamic pressure
    /// decays with depth. See [`crate::hydrostatics::FreeSurface`], whose two
    /// methods these mirror, for why collapsing the two over-drives a deep keel
    /// in short waves.
    fn pressure_head(&self, position: Vector3<f64>, time: f64) -> f64 {
        self.depth(position, time)
    }

    /// Both questions at one point, for callers that always ask both.
    ///
    /// Defaults to asking them separately, which is correct and is what a still
    /// water or a hand-written test environment wants. An implementation whose two
    /// answers share work — a seaway, where each is a sum over wave components —
    /// should override this: a hull clip asks the pair thousands of times a step
    /// and the sharing is worth a third of it.
    ///
    /// An override must agree with [`Environment::depth`] and
    /// [`Environment::pressure_head`] evaluated separately. Nothing enforces
    /// that, so it is the implementor's contract to keep.
    fn depth_and_pressure_head(&self, position: Vector3<f64>, time: f64) -> (f64, f64) {
        (
            self.depth(position, time),
            self.pressure_head(position, time),
        )
    }

    /// Water properties: density and kinematic viscosity.
    fn water(&self) -> Water;

    /// Air density, kg/m³.
    fn air_density(&self) -> f64;

    /// Gravitational acceleration, m/s².
    fn gravity(&self) -> f64;
}

/// A steady wind, optionally with a vertical gradient.
///
/// # Direction
///
/// `from_direction` is the compass bearing the wind blows **from**, in radians:
/// 0 is a northerly, π/2 an easterly. That is the meteorological and sailing
/// convention, and it is the opposite of the velocity direction — hence the
/// negation when the vector is built, done once here.
///
/// # Gradient
///
/// The profile is a power law, `V(h) = V_ref (h / h_ref)^α`. There is
/// deliberately **no default exponent**: a boundary-layer exponent is a
/// modelling choice with a real effect on heeling moment, and none of this
/// project's transcribed sources supplies one, so the caller states it.
/// [`UniformWind::uniform`] is the honest zero-knowledge choice and is what
/// reproduces a coefficient sail model, which has no gradient correction of its
/// own.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UniformWind {
    speed: f64,
    from_direction: f64,
    reference_height: f64,
    shear_exponent: f64,
}

impl UniformWind {
    /// A wind with no vertical gradient: the same vector at every height.
    #[must_use]
    pub fn uniform(speed: f64, from_direction: f64) -> Self {
        Self {
            speed,
            from_direction,
            reference_height: 10.0,
            shear_exponent: 0.0,
        }
    }

    /// A wind following a power-law profile.
    ///
    /// `speed` is the speed at `reference_height` above the surface. The
    /// exponent is the caller's choice; see the type documentation for why no
    /// default is offered.
    #[must_use]
    pub fn sheared(
        speed: f64,
        from_direction: f64,
        reference_height: f64,
        shear_exponent: f64,
    ) -> Self {
        Self {
            speed,
            from_direction,
            reference_height,
            shear_exponent,
        }
    }

    /// Wind speed at a height above the surface, m/s.
    ///
    /// At or below the surface the profile is held at its surface value rather
    /// than being allowed to go to zero or to a negative power: a sail is never
    /// there, and a submerged sample must not produce a NaN that propagates
    /// into a force.
    #[must_use]
    pub fn speed_at(&self, height_above_surface: f64) -> f64 {
        if self.shear_exponent == 0.0 || height_above_surface <= 0.0 {
            return self.speed;
        }
        self.speed * (height_above_surface / self.reference_height).powf(self.shear_exponent)
    }

    /// The wind velocity vector at a height, world frame.
    #[must_use]
    pub fn velocity_at(&self, height_above_surface: f64) -> Vector3<f64> {
        let speed = self.speed_at(height_above_surface);
        // Blowing *from* `from_direction`, hence the negation.
        Vector3::new(
            -speed * self.from_direction.cos(),
            -speed * self.from_direction.sin(),
            0.0,
        )
    }

    /// The reference speed the profile was built from, m/s.
    #[must_use]
    pub fn reference_speed(&self) -> f64 {
        self.speed
    }

    /// The bearing the wind blows from, radians.
    #[must_use]
    pub fn from_direction(&self) -> f64 {
        self.from_direction
    }
}

/// Still water with a steady wind: the flat-water sailing condition.
///
/// The free surface is the world plane `z = 0`, which is the same surface
/// [`crate::hydrostatics`] already integrates against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StillWater {
    wind: UniformWind,
    water: Water,
    air_density: f64,
    gravity: f64,
}

impl StillWater {
    #[must_use]
    pub fn new(wind: UniformWind) -> Self {
        Self {
            wind,
            water: Water::default(),
            air_density: crate::AIR_DENSITY,
            gravity: crate::STANDARD_GRAVITY,
        }
    }

    #[must_use]
    pub fn with_water(mut self, water: Water) -> Self {
        self.water = water;
        self
    }

    #[must_use]
    pub fn wind_profile(&self) -> UniformWind {
        self.wind
    }
}

impl Environment for StillWater {
    fn wind(&self, position: Vector3<f64>, _time: f64) -> Vector3<f64> {
        // World z is down, so height above the surface is -z.
        self.wind.velocity_at(-position.z)
    }

    fn depth(&self, position: Vector3<f64>, _time: f64) -> f64 {
        position.z
    }

    fn water(&self) -> Water {
        self.water
    }

    fn air_density(&self) -> f64 {
        self.air_density
    }

    fn gravity(&self) -> f64 {
        self.gravity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn a_northerly_blows_towards_the_south() {
        // Wind *from* north (bearing 0) moves air towards -x in NED.
        let wind = UniformWind::uniform(10.0, 0.0);
        assert_relative_eq!(
            wind.velocity_at(10.0),
            Vector3::new(-10.0, 0.0, 0.0),
            epsilon = 1e-12
        );
    }

    #[test]
    fn an_easterly_blows_towards_the_west() {
        let wind = UniformWind::uniform(10.0, FRAC_PI_2);
        assert_relative_eq!(
            wind.velocity_at(10.0),
            Vector3::new(0.0, -10.0, 0.0),
            epsilon = 1e-12
        );
    }

    #[test]
    fn a_uniform_wind_ignores_height() {
        let wind = UniformWind::uniform(8.0, 1.0);
        assert_relative_eq!(wind.speed_at(1.0), wind.speed_at(20.0));
    }

    #[test]
    fn the_power_law_matches_the_reference_speed_at_its_own_height() {
        let wind = UniformWind::sheared(8.0, 0.0, 10.0, 0.11);
        assert_relative_eq!(wind.speed_at(10.0), 8.0, epsilon = 1e-12);
        assert!(wind.speed_at(15.0) > 8.0, "wind must increase with height");
        assert!(wind.speed_at(5.0) < 8.0);
    }

    #[test]
    fn the_profile_is_held_at_and_below_the_surface() {
        // A negative height must not raise a negative base to a fractional
        // power and produce NaN.
        let wind = UniformWind::sheared(8.0, 0.0, 10.0, 0.11);
        assert!(wind.speed_at(-1.0).is_finite());
        assert_relative_eq!(wind.speed_at(0.0), 8.0);
    }

    #[test]
    fn depth_is_positive_below_the_surface() {
        let env = StillWater::new(UniformWind::uniform(5.0, 0.0));
        assert_relative_eq!(env.depth(Vector3::new(0.0, 0.0, 1.5), 0.0), 1.5);
        assert_relative_eq!(env.depth(Vector3::new(0.0, 0.0, -0.5), 0.0), -0.5);
    }

    #[test]
    fn wind_is_sampled_at_the_height_of_the_point() {
        let env = StillWater::new(UniformWind::sheared(10.0, 0.0, 10.0, 0.2));
        // z = -20 in NED is 20 m up.
        let high = env.wind(Vector3::new(0.0, 0.0, -20.0), 0.0).norm();
        let low = env.wind(Vector3::new(0.0, 0.0, -2.0), 0.0).norm();
        assert!(high > low, "sheared wind must be stronger aloft");
    }
}

/// A wind over an irregular sea.
///
/// The environment that makes the radiation work of [`crate::cummins`] mean
/// something: until there were waves, a boat settled after being pushed and was
/// never pushed.
///
/// Holds a [`Seaway`], which is a realisation and not a height field — the same
/// seed gives the same water to the physics here and to anything else that
/// evaluates the stated synthesis, a renderer included.
#[derive(Debug, Clone)]
pub struct Seaway2D {
    wind: UniformWind,
    sea: Seaway,
    water: Water,
    air_density: f64,
    gravity: f64,
}

impl Seaway2D {
    /// Wraps a wind and a sea state.
    ///
    /// Gravity is baked into the realisation through the dispersion relation, so
    /// it is taken here rather than asked for later.
    #[must_use]
    pub fn new(wind: UniformWind, state: SeaState) -> Self {
        Self {
            wind,
            sea: Seaway::new(state, crate::STANDARD_GRAVITY),
            water: Water::default(),
            air_density: crate::AIR_DENSITY,
            gravity: crate::STANDARD_GRAVITY,
        }
    }

    /// The realisation, for anything that has to draw or measure the same water.
    #[must_use]
    pub fn sea(&self) -> &Seaway {
        &self.sea
    }
}

impl Environment for Seaway2D {
    fn wind(&self, position: Vector3<f64>, _time: f64) -> Vector3<f64> {
        // Height above the *mean* surface, not above the local wave. A gust does
        // not follow the water, and the difference is a metre in a sea whose
        // gradient is stated over tens of metres.
        self.wind.velocity_at(-position.z)
    }

    fn depth(&self, position: Vector3<f64>, time: f64) -> f64 {
        self.sea.depth(position.x, position.y, position.z, time)
    }

    fn pressure_head(&self, position: Vector3<f64>, time: f64) -> f64 {
        self.sea
            .pressure_head(position.x, position.y, position.z, time)
    }

    fn depth_and_pressure_head(&self, position: Vector3<f64>, time: f64) -> (f64, f64) {
        self.sea
            .depth_and_pressure_head(position.x, position.y, position.z, time)
    }

    fn water(&self) -> Water {
        self.water
    }

    fn air_density(&self) -> f64 {
        self.air_density
    }

    fn gravity(&self) -> f64 {
        self.gravity
    }
}
