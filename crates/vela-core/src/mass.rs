//! Mass properties and the generalized 6-DOF mass matrix.

use crate::frames::skew;
use nalgebra::{Matrix3, Matrix6, Vector3};
use std::fmt;

/// Rejected mass properties. These are data errors (a bad boat file) rather
/// than programming errors, so they are reported rather than asserted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MassError {
    /// Mass must be finite and strictly positive.
    NonPositiveMass(f64),
    /// The inertia tensor must be symmetric and positive definite. A tensor
    /// violating this describes no physical body — usually a typo in gyradii or
    /// a sign error in a product of inertia.
    InvalidInertia,
}

impl fmt::Display for MassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonPositiveMass(m) => write!(f, "mass must be positive, got {m}"),
            Self::InvalidInertia => {
                write!(f, "inertia tensor is not symmetric positive definite")
            }
        }
    }
}

impl std::error::Error for MassError {}

/// Mass, centre of gravity and rotational inertia of the boat, in the body
/// frame.
///
/// The CoG is stored as an offset from the body origin and is expected to move
/// during a simulation (crew weight, water ballast, canting keel). Everything
/// derived from it is recomputed rather than cached by force modules.
#[derive(Debug, Clone, PartialEq)]
pub struct MassProperties {
    mass: f64,
    cog: Vector3<f64>,
    inertia_cog: Matrix3<f64>,
}

impl MassProperties {
    /// # Errors
    ///
    /// Returns [`MassError`] if the mass is not positive or the inertia tensor
    /// is not symmetric positive definite.
    pub fn new(mass: f64, cog: Vector3<f64>, inertia_cog: Matrix3<f64>) -> Result<Self, MassError> {
        if !mass.is_finite() || mass <= 0.0 {
            return Err(MassError::NonPositiveMass(mass));
        }
        let asymmetry = (inertia_cog - inertia_cog.transpose()).norm();
        if asymmetry > 1e-9 * inertia_cog.norm() || inertia_cog.cholesky().is_none() {
            return Err(MassError::InvalidInertia);
        }
        Ok(Self {
            mass,
            cog,
            inertia_cog,
        })
    }

    /// Builds from radii of gyration about the body axes through the CoG, which
    /// is how yacht mass data is normally quoted.
    ///
    /// # Errors
    ///
    /// As [`MassProperties::new`]; a non-positive radius yields
    /// [`MassError::InvalidInertia`].
    pub fn from_gyradii(
        mass: f64,
        cog: Vector3<f64>,
        gyradii: Vector3<f64>,
    ) -> Result<Self, MassError> {
        if gyradii.iter().any(|r| !r.is_finite() || *r <= 0.0) {
            return Err(MassError::InvalidInertia);
        }
        let inertia = Matrix3::from_diagonal(&Vector3::new(
            mass * gyradii.x * gyradii.x,
            mass * gyradii.y * gyradii.y,
            mass * gyradii.z * gyradii.z,
        ));
        Self::new(mass, cog, inertia)
    }

    #[must_use]
    pub fn mass(&self) -> f64 {
        self.mass
    }

    /// Centre of gravity, body frame, relative to the body origin.
    #[must_use]
    pub fn cog(&self) -> Vector3<f64> {
        self.cog
    }

    /// Inertia tensor about the centre of gravity, body axes.
    #[must_use]
    pub fn inertia_cog(&self) -> Matrix3<f64> {
        self.inertia_cog
    }

    /// Inertia tensor about the **body origin**, by the parallel axis theorem.
    ///
    /// `-m S(r) S(r)` equals `m (|r|² I - r rᵀ)`, the usual translation term;
    /// the skew form is used because it is the same building block as the
    /// coupling blocks of [`MassProperties::generalized`].
    #[must_use]
    pub fn inertia_origin(&self) -> Matrix3<f64> {
        let s = skew(&self.cog);
        self.inertia_cog - self.mass * s * s
    }

    /// The 6×6 generalized rigid-body mass matrix about the body origin,
    /// ordered `[linear; angular]`:
    ///
    /// ```text
    /// M = | m I      -m S(r) |
    ///     | m S(r)    I_o    |
    /// ```
    ///
    /// The off-diagonal blocks are the translation-rotation coupling introduced
    /// by referencing motion to the origin instead of the CoG; they vanish only
    /// when the CoG sits exactly on the origin. `M` is symmetric because
    /// `S(r)ᵀ = -S(r)`, and positive definite because it is a kinetic-energy
    /// form — which is what makes a Cholesky factorization valid.
    #[must_use]
    pub fn generalized(&self) -> Matrix6<f64> {
        let s = skew(&self.cog);
        let mut m = Matrix6::zeros();
        m.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(Matrix3::identity() * self.mass));
        m.fixed_view_mut::<3, 3>(0, 3).copy_from(&(-self.mass * s));
        m.fixed_view_mut::<3, 3>(3, 0).copy_from(&(self.mass * s));
        m.fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&self.inertia_origin());
        m
    }

    /// Moves the centre of gravity, e.g. crew shifting to windward. The inertia
    /// about the CoG is unchanged by construction; only origin-referenced
    /// quantities move.
    pub fn set_cog(&mut self, cog: Vector3<f64>) {
        self.cog = cog;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn sample() -> MassProperties {
        MassProperties::from_gyradii(
            3800.0,
            Vector3::new(0.2, 0.0, 0.35),
            Vector3::new(1.1, 2.6, 2.7),
        )
        .expect("valid mass properties")
    }

    #[test]
    fn rejects_non_positive_mass() {
        let err = MassProperties::from_gyradii(0.0, Vector3::zeros(), Vector3::new(1.0, 1.0, 1.0));
        assert_eq!(err, Err(MassError::NonPositiveMass(0.0)));
    }

    #[test]
    fn rejects_indefinite_inertia() {
        let bogus = Matrix3::from_diagonal(&Vector3::new(1.0, -1.0, 1.0));
        assert_eq!(
            MassProperties::new(10.0, Vector3::zeros(), bogus),
            Err(MassError::InvalidInertia)
        );
    }

    #[test]
    fn rejects_asymmetric_inertia() {
        // Positive definite but not symmetric: still not a physical tensor.
        let bogus = Matrix3::new(10.0, 5.0, 0.0, -5.0, 10.0, 0.0, 0.0, 0.0, 10.0);
        assert_eq!(
            MassProperties::new(10.0, Vector3::zeros(), bogus),
            Err(MassError::InvalidInertia)
        );
    }

    #[test]
    fn parallel_axis_theorem_matches_the_closed_form() {
        let m = sample();
        let r = m.cog();
        let expected = m.inertia_cog()
            + m.mass() * (r.norm_squared() * Matrix3::identity() - r * r.transpose());
        assert_relative_eq!(m.inertia_origin(), expected, epsilon = 1e-9);
    }

    #[test]
    fn generalized_mass_matrix_is_symmetric_and_positive_definite() {
        let m = sample().generalized();
        assert_relative_eq!(m, m.transpose(), epsilon = 1e-9);
        assert!(
            m.cholesky().is_some(),
            "kinetic energy form must be positive definite"
        );
    }

    #[test]
    fn coupling_blocks_vanish_when_cog_is_at_the_origin() {
        let centred =
            MassProperties::from_gyradii(1000.0, Vector3::zeros(), Vector3::new(1.0, 2.0, 3.0))
                .expect("valid mass properties");
        let m = centred.generalized();
        assert_relative_eq!(m.fixed_view::<3, 3>(0, 3).into_owned(), Matrix3::zeros());
        assert_relative_eq!(m.fixed_view::<3, 3>(3, 0).into_owned(), Matrix3::zeros());
        assert_relative_eq!(centred.inertia_origin(), centred.inertia_cog());
    }
}
