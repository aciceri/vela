//! Coordinate frame conventions and the single conversion between them.
//!
//! # File frame (boat data files)
//!
//! Naval-architecture convention, chosen because that is how hull offsets,
//! sail plans and appendage positions are drawn and tabulated:
//!
//! - origin at the intersection of baseline, centerline and aft perpendicular,
//! - `x` forward, `y` to port, `z` up (right-handed).
//!
//! # Body frame (all dynamics)
//!
//! Marine-craft convention (Fossen 2011):
//!
//! - `x` forward, `y` to starboard, `z` down (right-handed),
//! - origin at a fixed body reference point, *not* the centre of gravity: the
//!   CoG moves (crew, ballast, tanks) and force modules must not care.
//!
//! # World frame
//!
//! NED: `x` north, `y` east, `z` down. Gravity is therefore `+z`, and positive
//! heave is downward — consistent with the body frame, which is the entire
//! point of picking NED over an up-positive world frame.
//!
//! # The conversion
//!
//! File → body is a rotation of π about the shared `x` axis: `(x, y, z)` maps to
//! `(x, -y, -z)`. Its determinant is `+1`, so it is a proper rotation: cross
//! products, moments and handedness survive it. It is its own inverse.

use nalgebra::{Matrix3, Vector3};

/// Converts a vector or point from the file frame to the body frame.
///
/// Self-inverse: applying it twice is the identity, so the same function serves
/// both directions. Use [`body_to_file`] when the direction matters to a reader.
#[inline]
#[must_use]
pub fn file_to_body(v: Vector3<f64>) -> Vector3<f64> {
    Vector3::new(v.x, -v.y, -v.z)
}

/// Converts a vector or point from the body frame to the file frame.
///
/// Identical to [`file_to_body`]; exists so call sites can state intent.
#[inline]
#[must_use]
pub fn body_to_file(v: Vector3<f64>) -> Vector3<f64> {
    file_to_body(v)
}

/// Converts an inertia tensor (or any rank-2 tensor) from file to body axes.
///
/// For the frame rotation `R = diag(1, -1, -1)` this is `R I Rᵀ`, which flips
/// the sign of the `xy` and `xz` products of inertia and leaves the rest alone.
#[must_use]
pub fn tensor_file_to_body(inertia: &Matrix3<f64>) -> Matrix3<f64> {
    let r = Matrix3::new(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0);
    r * inertia * r.transpose()
}

/// Skew-symmetric cross-product matrix: `skew(a) * b == a.cross(&b)`.
///
/// Written out rather than assembled elementwise because it appears in the
/// generalized mass matrix and the rigid-body equations, where a transcription
/// error is expensive to find.
#[must_use]
pub fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn file_to_body_is_an_involution() {
        let v = Vector3::new(1.5, -2.5, 3.5);
        assert_relative_eq!(body_to_file(file_to_body(v)), v);
    }

    #[test]
    fn file_to_body_preserves_handedness() {
        // A proper rotation commutes with the cross product; an improper one
        // (a reflection) would flip its sign. This is the guard against
        // "just negate y" style mistakes.
        let a = Vector3::new(1.0, 2.0, 3.0);
        let b = Vector3::new(-4.0, 0.5, 2.0);
        assert_relative_eq!(
            file_to_body(a).cross(&file_to_body(b)),
            file_to_body(a.cross(&b))
        );
    }

    #[test]
    fn port_and_up_map_to_starboard_and_down() {
        // Port in file frame becomes negative starboard in body frame.
        assert_relative_eq!(file_to_body(Vector3::y()), -Vector3::y());
        // Up in file frame becomes negative down in body frame.
        assert_relative_eq!(file_to_body(Vector3::z()), -Vector3::z());
        // Forward is shared.
        assert_relative_eq!(file_to_body(Vector3::x()), Vector3::x());
    }

    #[test]
    fn skew_matches_cross_product() {
        let a = Vector3::new(0.3, -1.2, 4.0);
        let b = Vector3::new(2.0, 0.7, -0.5);
        assert_relative_eq!(skew(&a) * b, a.cross(&b));
    }

    #[test]
    fn tensor_conversion_flips_the_expected_products() {
        let inertia = Matrix3::new(10.0, 1.0, 2.0, 1.0, 20.0, 3.0, 2.0, 3.0, 30.0);
        let converted = tensor_file_to_body(&inertia);
        // Diagonal is untouched.
        assert_relative_eq!(converted[(0, 0)], 10.0);
        assert_relative_eq!(converted[(1, 1)], 20.0);
        assert_relative_eq!(converted[(2, 2)], 30.0);
        // xy and xz flip, yz survives.
        assert_relative_eq!(converted[(0, 1)], -1.0);
        assert_relative_eq!(converted[(0, 2)], -2.0);
        assert_relative_eq!(converted[(1, 2)], 3.0);
    }
}
