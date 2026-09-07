//! Exact 6-DOF rigid-body dynamics with a fixed-step integrator.
//!
//! # The equations
//!
//! Newton-Euler in the body frame, referenced to the **body origin** rather
//! than the centre of gravity (so that a moving CoG stays invisible to force
//! modules). With `r` the CoG offset, `v` the origin velocity, `ω` the angular
//! velocity, `v_g = v + ω × r` the CoG velocity and `a = ω × v_g`:
//!
//! ```text
//! | m I     -m S(r) | | v̇ |   | F   - m a                    |
//! | m S(r)   I_o    | | ω̇ | = | M_o - ω × (I_g ω) - m (r × a) |
//! ```
//!
//! The left-hand matrix is [`MassProperties::generalized`]; it is constant for
//! a given CoG, symmetric and positive definite, so it is Cholesky-factorized
//! once at construction and only reused thereafter.
//!
//! # Why this shape
//!
//! Writing the dynamics as a 6×6 solve rather than two decoupled 3-vector
//! equations is deliberate: hydrodynamic **added mass** — of the same order as
//! the boat's own mass — enters exactly here, as a constant matrix summed into
//! the left-hand side before factorization. Treating it as a force instead
//! would require the acceleration being solved for, and is the classic source
//! of instability in marine simulators. [`RigidBody::add_added_mass`] is the
//! registration point; `modules::radiation` supplies the coefficients.
//!
//! # Integrator
//!
//! Fixed step, **one force evaluation per step** — non-negotiable, because a
//! force evaluation here means a vortex-lattice solve plus a mesh clip against
//! a wave surface. An RK4 step would cost four of them.
//!
//! Within that budget the scheme is a hybrid, because the two halves of the
//! problem have different failure modes:
//!
//! - **Applied forces**: frozen at the start of the step, semi-implicit Euler
//!   (velocities first, then pose from the updated velocities). First-order,
//!   and well-behaved with the stiff restoring forces — buoyancy, righting
//!   moment — that dominate this problem.
//! - **Inertial terms** `C(ν) ν`: iterated to the *midpoint* velocity, which
//!   is the implicit midpoint rule and therefore conserves quadratic
//!   invariants. This costs only back-substitutions against an
//!   already-factorized matrix, no force re-evaluation.
//!
//! The second point is load-bearing rather than decorative. Evaluating the
//! gyroscopic term `ω × (I ω)` explicitly makes a freely rotating asymmetric
//! body *gain* energy — measured at +1.6 % over 60 s at 120 Hz, an instability
//! wearing the costume of truncation error. Evaluating it at the end of the
//! step instead merely flips the sign of the same error. The midpoint
//! evaluation drops the drift by six orders of magnitude, to −2e-8 over the
//! same interval, and scales as `dt³`. See the conservation tests.

use crate::mass::{MassError, MassProperties};
use crate::state::BodyState;
use crate::wrench::Wrench;
use crate::STANDARD_GRAVITY;
use nalgebra::{Cholesky, Const, Matrix6, UnitQuaternion, Vector3, Vector6};

/// Fixed-point iterations used to reach the midpoint velocity at which the
/// inertial terms are evaluated in [`RigidBody::step`].
///
/// The iteration converges fast because `C(ν) ν` is only quadratic in the
/// velocity; two passes put the residual well below the truncation error at any
/// step size this simulation uses. Zero would make the treatment explicit —
/// which injects energy — so this is a correctness knob, not a tuning one.
const INERTIAL_ITERATIONS: usize = 2;

/// Body-frame accelerations produced by a force balance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Acceleration {
    /// Linear acceleration of the body origin, body frame, m/s².
    pub linear: Vector3<f64>,
    /// Angular acceleration, body frame, rad/s².
    pub angular: Vector3<f64>,
}

/// The boat as a rigid body: mass properties plus the factorized mass matrix.
///
/// Rebuild it (or call [`RigidBody::set_cog`]) when mass properties change;
/// that is the only time the factorization is redone.
#[derive(Debug, Clone)]
pub struct RigidBody {
    mass: MassProperties,
    /// The hydrodynamic added mass registered so far, kept apart from the
    /// body's own so that a moving centre of gravity can rebuild the sum
    /// rather than lose it. Zero until [`RigidBody::add_added_mass`].
    added: Matrix6<f64>,
    /// `mass.generalized() + added`.
    mass_matrix: Matrix6<f64>,
    /// Cholesky of the **restrained** mass matrix, which equals `mass_matrix`
    /// when nothing is held. See [`RigidBody::restrain`].
    factorization: Cholesky<f64, Const<6>>,
    /// Which generalized coordinates are held, in the order surge, sway, heave,
    /// roll, pitch, yaw.
    held: [bool; 6],
    gravity: f64,
}

impl RigidBody {
    /// # Errors
    ///
    /// Returns [`MassError::InvalidInertia`] if the generalized mass matrix is
    /// not positive definite, which no physical body produces.
    pub fn new(mass: MassProperties) -> Result<Self, MassError> {
        Self::with_gravity(mass, STANDARD_GRAVITY)
    }

    /// Same as [`RigidBody::new`] with an explicit gravitational acceleration.
    /// `gravity = 0.0` isolates the rotational dynamics, which is how the
    /// torque-free conservation tests are written.
    ///
    /// # Errors
    ///
    /// As [`RigidBody::new`].
    pub fn with_gravity(mass: MassProperties, gravity: f64) -> Result<Self, MassError> {
        let mass_matrix = mass.generalized();
        let factorization = mass_matrix.cholesky().ok_or(MassError::InvalidInertia)?;
        Ok(Self {
            mass,
            added: Matrix6::zeros(),
            mass_matrix,
            factorization,
            held: [false; 6],
            gravity,
        })
    }

    #[must_use]
    pub fn mass_properties(&self) -> &MassProperties {
        &self.mass
    }

    #[must_use]
    pub fn gravity(&self) -> f64 {
        self.gravity
    }

    /// Moves the centre of gravity and refactorizes the mass matrix.
    ///
    /// The added mass survives the move: it is a property of the hull in the
    /// water, not of where the crew is sitting, and it is registered about the
    /// body origin, which does not move. An earlier version rebuilt the body
    /// from its own mass alone and silently dropped it.
    ///
    /// # Errors
    ///
    /// As [`RigidBody::new`].
    pub fn set_cog(&mut self, cog: Vector3<f64>) -> Result<(), MassError> {
        let mut mass = self.mass.clone();
        mass.set_cog(cog);
        let combined = mass.generalized() + self.added;
        combined.cholesky().ok_or(MassError::InvalidInertia)?;
        self.mass = mass;
        self.mass_matrix = combined;
        self.restrain(self.held);
        Ok(())
    }

    /// Holds the given generalized coordinates at zero acceleration, in the
    /// **equations of motion** rather than after them.
    ///
    /// # Why this cannot be done afterwards
    ///
    /// Cancelling a restrained mode's velocity once the step has been taken
    /// looks equivalent and is not. The mass matrix couples modes — through
    /// `-m S(r)` whenever the centre of gravity is off the origin, and far more
    /// strongly through hydrodynamic added mass — so an unbalanced force on a
    /// held mode accelerates the *free* ones, and deleting the held mode's own
    /// motion afterwards leaves that behind.
    ///
    /// It is not a small effect. A yacht held in pitch carries an unbalanced
    /// trim moment, because nothing balanced it; through the heave-pitch added
    /// mass that moment drives a heave acceleration of order 0.4 m/s², and the
    /// boat rises until it has bought an equal and opposite buoyancy error. The
    /// result is a steady state carrying a fifth of the boat's weight in
    /// unbalanced vertical force, reported by the telemetry and denied by the
    /// motion — the exact shape of error this engine is built to make impossible.
    ///
    /// # What it does
    ///
    /// Replaces the held rows and columns of the mass matrix with the identity
    /// and zeroes the held rows of the right-hand side. The free block `M_ff` is
    /// then inverted alone, which is the reduced system of the constrained
    /// dynamics, and the held accelerations come out as exactly zero rather than
    /// as something small. The constraint force is not computed because nothing
    /// needs it: the applied wrench on the held modes stays visible in
    /// [`crate::sim::Sim::last_wrench`] and in the telemetry, which is what a
    /// captive measurement is *for*.
    ///
    /// # Panics
    ///
    /// Never. A principal submatrix of a positive definite matrix is positive
    /// definite and the identity is positive definite, so the restrained matrix
    /// factorizes whenever the free one does — which construction has already
    /// established. The `expect` states that theorem rather than returning an
    /// error no caller could act on.
    pub fn restrain(&mut self, held: [bool; 6]) {
        self.held = held;
        self.factorization = restrained(&self.mass_matrix, held)
            .cholesky()
            .expect("a restrained positive definite matrix is positive definite");
    }

    /// Which generalized coordinates are held.
    #[must_use]
    pub fn held(&self) -> [bool; 6] {
        self.held
    }

    /// Adds a hydrodynamic added-mass matrix and refactorizes.
    ///
    /// This is where added mass belongs, and the reason is a stability one rather
    /// than a matter of taste. Added mass multiplies acceleration, so treating it
    /// as a force means computing a force from the acceleration the same step is
    /// about to produce. Explicit schemes handle that by using the *previous*
    /// step's acceleration, which for a yacht in heave — where the added mass is
    /// several times the hull's own — is a feedback loop that diverges. In the
    /// mass matrix it is exact, unconditional, and paid once.
    ///
    /// `added` is the infinite-frequency added mass in the body frame about the
    /// body origin, in the same generalized ordering as
    /// [`RigidBody::mass_matrix`]. Only the entries a caller has actually
    /// computed need be non-zero: the matrix is added, not replaced.
    ///
    /// # Errors
    ///
    /// Returns [`MassError::InvalidInertia`] if the sum is not positive definite.
    /// That is a real failure and not a numerical nicety — an indefinite mass
    /// matrix accelerates a boat against the force applied to it — and it is the
    /// reason this returns a `Result` rather than asserting.
    pub fn add_added_mass(&mut self, added: Matrix6<f64>) -> Result<(), MassError> {
        let combined = self.mass_matrix + added;
        // Checked on the *unrestrained* sum: an indefinite added mass is a
        // physical error whether or not the modes it corrupts happen to be held,
        // and a restraint that hid it would let a bad fit through.
        combined.cholesky().ok_or(MassError::InvalidInertia)?;
        self.added += added;
        self.mass_matrix = combined;
        self.restrain(self.held);
        Ok(())
    }

    /// Weight as a body-frame wrench about the body origin.
    ///
    /// Gravity is uniform, so it exerts no moment about the CoG — but it does
    /// about the origin whenever the CoG is offset, and that moment is the
    /// righting moment of a ballasted boat. It is computed here rather than in
    /// a force module because it is exact and unconditional.
    #[must_use]
    pub fn gravity_wrench(&self, state: &BodyState) -> Wrench {
        let weight_world = Vector3::new(0.0, 0.0, self.mass.mass() * self.gravity);
        Wrench::from_force_at(state.to_body(weight_world), self.mass.cog())
    }

    /// Velocity-dependent inertial terms — the `C(ν) ν` of the equations of
    /// motion — as a generalized vector to be subtracted from the applied
    /// wrench.
    ///
    /// Built from the **whole** mass matrix, added mass included, in Fossen's
    /// parametrisation for a symmetric `M = [[M₁₁, M₁₂], [M₂₁, M₂₂]]`:
    ///
    /// ```text
    /// p = M₁₁ v + M₁₂ ω          (momentum)
    /// h = M₂₁ v + M₂₂ ω          (angular momentum about the origin)
    /// C(ν) ν = ( ω × p ,  v × p + ω × h )
    /// ```
    ///
    /// With `M` the rigid body's own this is exactly the classical `ω × m v_g`
    /// centripetal and `ω × I ω` gyroscopic terms — the two forms agree by the
    /// Jacobi identity, and the conservation tests are the check. With added
    /// mass in `M` it is what transports the fluid's momentum with the body: an
    /// earlier version built `C` from the rigid mass alone, and a body with a
    /// heavy heave added mass tumbling in free flight lost 146 % of its world
    /// momentum in ten seconds.
    ///
    /// # The one term deliberately left out
    ///
    /// `v × M₁₁ v` is the **Munk moment**: the moment a body feels translating
    /// steadily through a fluid whose added mass is anisotropic. It is zero for
    /// the rigid part (`M₁₁ = m I`) and it is *not* zero for the added part, and
    /// it is left out on purpose, for two reasons that are the same reason.
    /// First, the added mass here is strip theory's: the canoe body alone, at
    /// infinite frequency, with no surge entry at all — which is the right
    /// matrix for radiation and the wrong one for steady flow, where the
    /// free-surface condition is the other limit and the keel carries most of
    /// the sway added mass. Second, the hull and foil forces come from tank
    /// regressions fitted to heeled hulls at leeway, whose measured totals
    /// already contain whatever Munk moment the real flow produces; adding a
    /// potential-flow one on top would count it twice, with the wrong
    /// coefficients. Measured on the YD-41 at the solved upwind condition, the
    /// term would have been 2 kN·m in roll, 11 kN·m in pitch and 1 kN·m in
    /// yaw — three quarters of a degree of heel and 1.3 % of speed away from
    /// the velocity prediction the polar was validated against.
    ///
    /// So: everything that depends on a rate is here, and the steady
    /// translational moment is the empirical models'. The cost is that the
    /// matrix is skew only up to that term, so kinetic energy is conserved to
    /// `ω · (v × A₁₁ v)`, which vanishes at steady sailing and is small beside
    /// the radiation damping everywhere else.
    fn inertial_terms(&self, velocity: Vector3<f64>, omega: Vector3<f64>) -> Vector6<f64> {
        let m = &self.mass_matrix;
        let translational = m.fixed_view::<3, 3>(0, 0);
        let coupling = m.fixed_view::<3, 3>(0, 3);
        let coupling_transposed = m.fixed_view::<3, 3>(3, 0);
        let rotational = m.fixed_view::<3, 3>(3, 3);

        let momentum = translational * velocity + coupling * omega;
        let angular_momentum = coupling_transposed * velocity + rotational * omega;

        let linear = omega.cross(&momentum);
        // `v × (M₁₂ ω)` and not `v × p`: the steady `v × M₁₁ v` is the Munk
        // moment, left to the empirical models — see above.
        let angular = velocity.cross(&(coupling * omega)) + omega.cross(&angular_momentum);
        Vector6::new(
            linear.x, linear.y, linear.z, angular.x, angular.y, angular.z,
        )
    }

    /// Solves `M ν̇ = τ - C(ν) ν` for a given applied wrench and trial velocity.
    fn solve_balance(
        &self,
        applied: Wrench,
        velocity: Vector3<f64>,
        omega: Vector3<f64>,
    ) -> Acceleration {
        let mut rhs = applied.to_generalized() - self.inertial_terms(velocity, omega);
        for (row, held) in self.held.iter().enumerate() {
            if *held {
                rhs[row] = 0.0;
            }
        }
        let solution = self.factorization.solve(&rhs);
        Acceleration {
            linear: solution.fixed_rows::<3>(0).into_owned(),
            angular: solution.fixed_rows::<3>(3).into_owned(),
        }
    }

    /// Solves the force balance for body-frame accelerations at the current
    /// state.
    ///
    /// `external` is the sum of every force module's wrench, in the body frame
    /// about the body origin. Gravity is added here and must not be included by
    /// the caller.
    #[must_use]
    pub fn acceleration(&self, state: &BodyState, external: Wrench) -> Acceleration {
        self.solve_balance(
            external + self.gravity_wrench(state),
            state.velocity,
            state.angular_velocity,
        )
    }

    /// Advances the state by exactly one fixed step.
    ///
    /// Velocities update first, then the pose from the *updated* velocities
    /// (semi-implicit). The attitude increment uses the exponential map of the
    /// body-frame rotation vector, so a constant angular velocity integrates
    /// without small-angle error; renormalization bounds the drift of the
    /// unit-norm constraint under repeated products.
    ///
    /// The applied wrench is evaluated **once** per step; the inertial terms
    /// `C(ν) ν` are then iterated to the midpoint velocity, which is what keeps
    /// the scheme from injecting energy. See the module documentation for the
    /// measurements behind that choice.
    pub fn step(&self, state: &mut BodyState, external: Wrench, dt: f64) {
        let applied = external + self.gravity_wrench(state);
        let velocity = state.velocity;
        let omega = state.angular_velocity;

        let half = 0.5 * dt;
        let mut acceleration = self.solve_balance(applied, velocity, omega);
        for _ in 0..INERTIAL_ITERATIONS {
            acceleration = self.solve_balance(
                applied,
                velocity + acceleration.linear * half,
                omega + acceleration.angular * half,
            );
        }

        state.velocity = velocity + acceleration.linear * dt;
        state.angular_velocity = omega + acceleration.angular * dt;

        state.position += state.attitude * state.velocity * dt;
        let increment = UnitQuaternion::from_scaled_axis(state.angular_velocity * dt);
        state.attitude = UnitQuaternion::new_normalize((state.attitude * increment).into_inner());
    }

    /// Kinetic energy, `½ νᵀ M ν`, J.
    #[must_use]
    pub fn kinetic_energy(&self, state: &BodyState) -> f64 {
        let nu = Vector6::new(
            state.velocity.x,
            state.velocity.y,
            state.velocity.z,
            state.angular_velocity.x,
            state.angular_velocity.y,
            state.angular_velocity.z,
        );
        0.5 * (nu.transpose() * self.mass_matrix * nu)[0]
    }

    /// Angular momentum about the centre of gravity, expressed in the world
    /// frame, kg·m²/s.
    ///
    /// Conserved whenever the only external force is uniform gravity, which
    /// makes it the sharpest available check on the rotational integration.
    #[must_use]
    pub fn angular_momentum_world(&self, state: &BodyState) -> Vector3<f64> {
        state.to_world(self.mass.inertia_cog() * state.angular_velocity)
    }

    /// Linear momentum in the world frame, kg·m/s.
    #[must_use]
    pub fn linear_momentum_world(&self, state: &BodyState) -> Vector3<f64> {
        let cog_velocity = state.point_velocity(self.mass.cog());
        state.to_world(self.mass.mass() * cog_velocity)
    }

    /// The generalized mass matrix in use — the body's own plus every added
    /// mass registered — for inspection.
    #[must_use]
    pub fn mass_matrix(&self) -> &Matrix6<f64> {
        &self.mass_matrix
    }
}

/// The mass matrix a restrained body inverts: held rows and columns replaced by
/// the identity.
///
/// Zeroing the off-diagonal entries is the whole point — those are the terms
/// through which a held mode's unbalanced force reaches the free ones. What is
/// left on the free block is `M_ff` exactly, which is the reduced system of the
/// constrained dynamics, and the identity on the diagonal makes the held
/// accelerations come out as zero instead of as a division by nothing.
fn restrained(matrix: &Matrix6<f64>, held: [bool; 6]) -> Matrix6<f64> {
    let mut out = *matrix;
    for (index, is_held) in held.iter().enumerate() {
        if !is_held {
            continue;
        }
        for other in 0..6 {
            out[(index, other)] = 0.0;
            out[(other, index)] = 0.0;
        }
        out[(index, index)] = 1.0;
    }
    out
}
