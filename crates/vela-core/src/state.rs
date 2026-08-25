//! Rigid-body state: pose in the world frame, velocities in the body frame.

use nalgebra::{UnitQuaternion, Vector3};

/// The complete kinematic state of the boat.
///
/// Pose lives in the world frame (NED), velocities in the body frame — the
/// standard marine-craft split. Body-frame velocities are what force models
/// actually want (surge, leeway, heel rate), so keeping them native avoids a
/// rotation on every force evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct BodyState {
    /// Position of the **body origin** in the world frame (NED), m.
    pub position: Vector3<f64>,
    /// Attitude: rotates body-frame vectors into the world frame.
    pub attitude: UnitQuaternion<f64>,
    /// Velocity of the body origin, expressed in the **body frame**, m/s.
    pub velocity: Vector3<f64>,
    /// Angular velocity, body frame, rad/s.
    pub angular_velocity: Vector3<f64>,
}

impl BodyState {
    /// At rest at the given world position, upright and heading north.
    #[must_use]
    pub fn at_rest(position: Vector3<f64>) -> Self {
        Self {
            position,
            attitude: UnitQuaternion::identity(),
            velocity: Vector3::zeros(),
            angular_velocity: Vector3::zeros(),
        }
    }

    /// Velocity of the body origin expressed in the world frame.
    #[must_use]
    pub fn world_velocity(&self) -> Vector3<f64> {
        self.attitude * self.velocity
    }

    /// Rotates a world-frame vector into the body frame.
    #[must_use]
    pub fn to_body(&self, world: Vector3<f64>) -> Vector3<f64> {
        self.attitude.inverse_transform_vector(&world)
    }

    /// Rotates a body-frame vector into the world frame.
    #[must_use]
    pub fn to_world(&self, body: Vector3<f64>) -> Vector3<f64> {
        self.attitude * body
    }

    /// Velocity of a body-fixed point, in the body frame: `v_p = v_o + ω × r`.
    ///
    /// Needed wherever a force depends on local flow: appendage inflow, panel
    /// velocities, apparent wind at the masthead.
    #[must_use]
    pub fn point_velocity(&self, point: Vector3<f64>) -> Vector3<f64> {
        self.velocity + self.angular_velocity.cross(&point)
    }

    /// Roll, pitch and yaw (rad). Reporting only — the simulation integrates a
    /// quaternion and never Euler angles, which is what keeps it free of
    /// gimbal lock.
    #[must_use]
    pub fn euler_angles(&self) -> (f64, f64, f64) {
        self.attitude.euler_angles()
    }
}

impl Default for BodyState {
    fn default() -> Self {
        Self::at_rest(Vector3::zeros())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn body_and_world_conversions_round_trip() {
        let state = BodyState {
            attitude: UnitQuaternion::from_euler_angles(0.3, -0.2, 1.1),
            ..BodyState::default()
        };
        let v = Vector3::new(1.0, -2.0, 3.0);
        assert_relative_eq!(state.to_body(state.to_world(v)), v, epsilon = 1e-12);
    }

    #[test]
    fn yaw_of_ninety_degrees_maps_forward_to_east() {
        let state = BodyState {
            attitude: UnitQuaternion::from_euler_angles(0.0, 0.0, FRAC_PI_2),
            ..BodyState::default()
        };
        // Body x (forward) becomes world y (east) after a quarter turn of yaw.
        assert_relative_eq!(state.to_world(Vector3::x()), Vector3::y(), epsilon = 1e-12);
    }

    #[test]
    fn point_velocity_includes_the_rotational_term() {
        let state = BodyState {
            velocity: Vector3::new(5.0, 0.0, 0.0),
            angular_velocity: Vector3::new(0.0, 0.0, 2.0),
            ..BodyState::default()
        };
        // A point 3 m forward of the origin, yawing at 2 rad/s, gains 6 m/s of
        // starboard velocity.
        assert_relative_eq!(
            state.point_velocity(Vector3::new(3.0, 0.0, 0.0)),
            Vector3::new(5.0, 6.0, 0.0)
        );
    }
}
