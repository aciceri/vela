//! The one place two coordinate conventions meet.
//!
//! # Why this module exists at all
//!
//! `vela-core` works in the naval convention: **x forward (north), y to
//! starboard (east), z down**. Bevy works in the graphics convention: **y up**.
//! Neither is negotiable — the engine's is what every hydrodynamic source in
//! its bibliography is written in, and Bevy's is what its camera, its light and
//! its meshes assume.
//!
//! So there is a transform, and the only interesting decision about it is that
//! there is exactly **one**. A sign error here is the classic way to build a
//! simulator where the boat heels the correct amount in the wrong direction, and
//! it is invisible in a screenshot of a symmetric hull. Every conversion in this
//! frontend goes through [`to_render`] and [`rotation`]; nothing else is
//! permitted to write `-position.z`.
//!
//! # The mapping
//!
//! ```text
//! render.x =  engine.x   (north, the bow)
//! render.y = -engine.z   (up)
//! render.z =  engine.y   (east, starboard)
//! ```
//!
//! That is a proper rotation and not a reflection, which matters: a reflection
//! would turn every quaternion into its mirror image and every right-handed
//! moment into a left-handed one. The matrix is a −90° turn about `x`, its
//! determinant is `+1`, and `the_mapping_is_a_rotation` asserts exactly that
//! rather than trusting the arithmetic above.

use bevy::prelude::*;
use nalgebra::{UnitQuaternion, Vector3};

/// An engine-frame vector, in render coordinates.
#[must_use]
pub fn to_render(engine: Vector3<f64>) -> Vec3 {
    Vec3::new(engine.x as f32, -engine.z as f32, engine.y as f32)
}

/// An engine-frame attitude, as a render-frame rotation.
///
/// The conjugation `R q R⁻¹` rather than a direct component swap, because an
/// attitude is not a vector: it has to be re-expressed in the new basis on both
/// sides. Writing it as a component permutation is the mistake this function
/// exists to prevent, and it produces a boat that rolls when it should pitch.
#[must_use]
pub fn rotation(attitude: &UnitQuaternion<f64>) -> Quat {
    let axis = attitude.scaled_axis();
    let rotated = to_render(axis);
    let angle = rotated.length();
    if angle < 1e-12 {
        Quat::IDENTITY
    } else {
        Quat::from_axis_angle(rotated / angle, angle)
    }
}

/// A length, unchanged: both frames are metres.
///
/// Exists so that a call site reads as a deliberate conversion rather than as an
/// `as f32` someone will later wonder about.
#[must_use]
pub fn metres(engine: f64) -> f32 {
    engine as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::FRAC_PI_2;

    /// The basis vectors, one at a time, which is where a swap would show.
    #[test]
    fn the_axes_land_where_the_convention_says() {
        assert_eq!(to_render(Vector3::new(1.0, 0.0, 0.0)), Vec3::X);
        assert_eq!(to_render(Vector3::new(0.0, 1.0, 0.0)), Vec3::Z);
        // Down in the engine is *minus* up in the renderer. This is the sign the
        // whole module exists for.
        assert_eq!(to_render(Vector3::new(0.0, 0.0, 1.0)), -Vec3::Y);
    }

    /// A rotation, not a reflection: handedness has to survive.
    ///
    /// Checked by the cross product rather than by a determinant, because that is
    /// the property that actually matters downstream — if `x × y ≠ z` after the
    /// mapping, every moment in the simulation is drawn backwards.
    #[test]
    fn the_mapping_is_a_rotation() {
        let x = to_render(Vector3::new(1.0, 0.0, 0.0));
        let y = to_render(Vector3::new(0.0, 1.0, 0.0));
        let z = to_render(Vector3::new(0.0, 0.0, 1.0));
        assert_relative_eq!(x.cross(y).x, z.x, epsilon = 1e-6);
        assert_relative_eq!(x.cross(y).y, z.y, epsilon = 1e-6);
        assert_relative_eq!(x.cross(y).z, z.z, epsilon = 1e-6);
    }

    /// Heel is a roll about the engine's `x`, and must come out as a roll about
    /// the renderer's `x` too — the one axis the two frames share.
    ///
    /// A boat heeled to starboard must have its masthead lean towards `+z` in
    /// render coordinates, which is east. Written as a mast rather than as a
    /// quaternion comparison because that is the thing a viewer would notice.
    #[test]
    fn heeling_to_starboard_leans_the_mast_east() {
        let heel = UnitQuaternion::from_euler_angles(0.4, 0.0, 0.0);
        let mast_engine = Vector3::new(0.0, 0.0, -15.0);
        let expected = to_render(heel * mast_engine);

        let mast_render = rotation(&heel) * to_render(mast_engine);

        assert_relative_eq!(mast_render.x, expected.x, epsilon = 1e-5);
        assert_relative_eq!(mast_render.y, expected.y, epsilon = 1e-5);
        assert_relative_eq!(mast_render.z, expected.z, epsilon = 1e-5);
        assert!(
            mast_render.z > 1.0,
            "starboard heel must lean the mast east, got z = {}",
            mast_render.z
        );
    }

    /// The general case: rotating then converting is converting then rotating.
    ///
    /// This is the whole correctness condition of [`rotation`], and it fails for
    /// a component permutation of the quaternion — which is why the function does
    /// the conjugation instead.
    #[test]
    fn rotating_and_converting_commute() {
        let attitude = UnitQuaternion::from_euler_angles(0.3, -0.2, 1.1);
        for point in [
            Vector3::new(6.0, 0.0, 0.0),
            Vector3::new(0.0, 2.5, -1.0),
            Vector3::new(-3.0, 1.0, 4.0),
        ] {
            let converted_then_rotated = rotation(&attitude) * to_render(point);
            let rotated_then_converted = to_render(attitude * point);
            assert_relative_eq!(
                converted_then_rotated.x,
                rotated_then_converted.x,
                epsilon = 1e-5
            );
            assert_relative_eq!(
                converted_then_rotated.y,
                rotated_then_converted.y,
                epsilon = 1e-5
            );
            assert_relative_eq!(
                converted_then_rotated.z,
                rotated_then_converted.z,
                epsilon = 1e-5
            );
        }
    }

    /// A quarter turn is a quarter turn: no angle is being scaled or halved.
    #[test]
    fn the_angle_survives() {
        let quarter = UnitQuaternion::from_euler_angles(FRAC_PI_2, 0.0, 0.0);
        assert_relative_eq!(
            rotation(&quarter).to_axis_angle().1,
            FRAC_PI_2 as f32,
            epsilon = 1e-6
        );
    }
}
