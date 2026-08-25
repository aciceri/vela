//! Force and moment pairs in the body frame.

use nalgebra::{Vector3, Vector6};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul, Neg};

/// A force and a moment, both in the **body frame**, with the moment taken
/// about the **body origin** — never about the centre of gravity.
///
/// This is the output type of every force module. Referencing moments to the
/// body origin rather than the CoG means a shifting CoG (crew, ballast) changes
/// nothing in any force module: the origin-to-CoG transfer is the integrator's
/// job alone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wrench {
    /// Force in the body frame, N.
    pub force: Vector3<f64>,
    /// Moment about the body origin in the body frame, N·m.
    pub moment: Vector3<f64>,
}

impl Wrench {
    /// A wrench with no force and no moment.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            force: Vector3::zeros(),
            moment: Vector3::zeros(),
        }
    }

    #[must_use]
    pub fn new(force: Vector3<f64>, moment: Vector3<f64>) -> Self {
        Self { force, moment }
    }

    /// A pure force applied at `point` (body frame), producing the moment it
    /// exerts about the body origin.
    ///
    /// This is how buoyancy, gravity and every panel-integrated pressure force
    /// enter: as a force with an application point, not a hand-computed moment.
    #[must_use]
    pub fn from_force_at(force: Vector3<f64>, point: Vector3<f64>) -> Self {
        Self {
            moment: point.cross(&force),
            force,
        }
    }

    /// A pure moment, no net force (e.g. a damping couple).
    #[must_use]
    pub fn from_moment(moment: Vector3<f64>) -> Self {
        Self {
            force: Vector3::zeros(),
            moment,
        }
    }

    /// Stacks into the generalized force vector `[force; moment]` used by the
    /// 6-DOF equations of motion.
    #[must_use]
    pub fn to_generalized(self) -> Vector6<f64> {
        Vector6::new(
            self.force.x,
            self.force.y,
            self.force.z,
            self.moment.x,
            self.moment.y,
            self.moment.z,
        )
    }
}

impl Default for Wrench {
    fn default() -> Self {
        Self::zero()
    }
}

impl Add for Wrench {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            force: self.force + rhs.force,
            moment: self.moment + rhs.moment,
        }
    }
}

impl AddAssign for Wrench {
    fn add_assign(&mut self, rhs: Self) {
        self.force += rhs.force;
        self.moment += rhs.moment;
    }
}

impl Neg for Wrench {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            force: -self.force,
            moment: -self.moment,
        }
    }
}

impl Mul<f64> for Wrench {
    type Output = Self;
    fn mul(self, scale: f64) -> Self {
        Self {
            force: self.force * scale,
            moment: self.moment * scale,
        }
    }
}

impl Sum for Wrench {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, w| acc + w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn force_at_a_point_produces_the_right_moment() {
        // A downward force one metre to starboard: r x F with r = (0,1,0) and
        // F = (0,0,10) gives (+10,0,0). Positive roll is starboard-down in this
        // convention, which is exactly what that force must produce.
        let w = Wrench::from_force_at(Vector3::new(0.0, 0.0, 10.0), Vector3::new(0.0, 1.0, 0.0));
        assert_relative_eq!(w.moment, Vector3::new(10.0, 0.0, 0.0));
    }

    #[test]
    fn force_through_the_origin_produces_no_moment() {
        let w = Wrench::from_force_at(Vector3::new(1.0, 2.0, 3.0), Vector3::zeros());
        assert_relative_eq!(w.moment, Vector3::zeros());
    }

    #[test]
    fn force_along_the_lever_arm_produces_no_moment() {
        let arm = Vector3::new(2.0, 0.0, 0.0);
        let w = Wrench::from_force_at(arm * 5.0, arm);
        assert_relative_eq!(w.moment, Vector3::zeros());
    }

    #[test]
    fn summing_wrenches_adds_components() {
        let a = Wrench::new(Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 2.0, 0.0));
        let b = Wrench::new(Vector3::new(0.0, 3.0, 0.0), Vector3::new(0.0, 0.0, 4.0));
        let total: Wrench = [a, b].into_iter().sum();
        assert_relative_eq!(total.force, Vector3::new(1.0, 3.0, 0.0));
        assert_relative_eq!(total.moment, Vector3::new(0.0, 2.0, 4.0));
    }
}
