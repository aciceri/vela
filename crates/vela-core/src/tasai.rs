//! Two-dimensional heave added mass and damping by the Ursell-Tasai method.
//!
//! # Provenance
//!
//! Transcribed from Journée, J.M.J., *Theoretical Manual of SEAWAY (Release
//! 4.19)*, Ship Hydromechanics Laboratory, Delft University of Technology,
//! Report 1216a, 2001, §4.1.1, which presents the method of [[Ursell, 1949]],
//! [[Tasai, 1959]] and [[Tasai, 1960]] in the form a working program uses.
//!
//! The manual was chosen over the textbook derivation deliberately. A textbook
//! gives the equations; a program manual gives the index ranges, where to
//! truncate the series, and which of two algebraically identical expressions to
//! evaluate — and those are the parts a transcription gets wrong.
//!
//! # What it computes
//!
//! For one section, replaced by its [`LewisForm`], and one frequency: the
//! two-dimensional hydrodynamic mass and damping in heave, per unit length of
//! hull. Integrating these along the hull is what gives a boat heave and pitch
//! added mass and radiation damping, which is the missing term that leaves the
//! vertical modes of this simulator undamped.
//!
//! # Method
//!
//! An oscillating cylinder radiates two wave systems: a standing set that dies
//! away with distance, and a progressive set that carries energy off. The
//! standing set is a multipole series about the origin whose strengths `P_2m`
//! and `Q_2m` are unknown; the progressive set is known in closed form. The
//! condition that the flow follow the section's surface turns the unknowns into
//! two linear systems which share their matrix and differ only in their right
//! hand side — so one factorisation answers both.
//!
//! The pressure that the resulting potential exerts on the surface, integrated
//! over it, is the force; split into the parts in phase with acceleration and
//! with velocity, it is the added mass and the damping.
//!
//! # Why the result can be trusted
//!
//! The source gives two independent routes to the damping: one through the
//! pressure integral, one through the energy the radiated waves carry away.
//! Equating them yields an identity that must hold at every frequency,
//!
//! ```text
//! M_0 A_0 - N_0 B_0 = π² / 2
//! ```
//!
//! and it involves every quantity in the method. A dropped factor, a sign, an
//! index off by one — any of them breaks it. It is reported as
//! [`HeaveCoefficients::energy_residual`] rather than merely asserted in a test,
//! because it doubles as the truncation diagnostic: the identity is exact in the
//! limit, so how far off it sits says whether `multipoles` was large enough for
//! this section at this frequency.

use crate::lewis::LewisForm;
use nalgebra::{Complex, DMatrix, DVector};
use std::f64::consts::PI;

/// Euler-Mascheroni constant, for the series expansion of `E_1`.
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// Where the series gives way to the continued fraction.
///
/// This is a bound on `|w| + Re w`, not on `|w|`, and the difference is the
/// whole point. The series' largest term is of order `e^|w|` while the answer
/// it is building is of order `e^(Re w)`, so the digits it loses to
/// cancellation go as `e^(|w| + Re w)` — which is `e^(|w|(1 + cos arg w))`, and
/// therefore *vanishes* as the argument approaches the negative real axis.
///
/// That matters because the arguments this module produces all have
/// non-positive real part, and the ones near the keel sit almost exactly on
/// that axis. A limit on `|w|` alone sends those to the continued fraction,
/// which is the one place it fails: it needs thousands of iterations at
/// `arg w = -174°` and does not converge at all at `-179.9°`. The criterion
/// below sends them to the series instead, where the cancellation it was
/// chosen to avoid does not happen. Twelve leaves eleven digits.
const SERIES_CANCELLATION_LIMIT: f64 = 12.0;

/// `e^w · E_1(w)`, the exponential integral scaled by its own exponential.
///
/// `E_1(w) = ∫_w^∞ e^-s / s ds`, on the principal branch with the cut along the
/// negative real axis. The scaled combination is computed rather than `E_1`
/// itself because it is what the caller needs and because it is the
/// well-conditioned one: `E_1` alone spans hundreds of orders of magnitude over
/// the arguments here, and multiplying the overflow back out afterwards throws
/// away the precision that computing it separately cost.
///
/// Neither evaluation is in the source. The source avoids this function by
/// expanding the integral it comes from as a power series after
/// [[Porter, 1960]], because it calls the integral's numerical convergence
/// *"very slowly"*. The integral has a closed form in terms of a standard
/// function, so that is used instead — see [`progressive_wave`] for the
/// derivation, and `the_wave_integral_matches_direct_quadrature` for the
/// evidence that the substitution is one.
fn scaled_exp1(w: Complex<f64>) -> Complex<f64> {
    if w.norm() + w.re < SERIES_CANCELLATION_LIMIT {
        scaled_exp1_series(w)
    } else {
        scaled_exp1_fraction(w)
    }
}

/// [`scaled_exp1`] by power series: exact near the branch cut, and only there.
fn scaled_exp1_series(w: Complex<f64>) -> Complex<f64> {
    // E_1(w) = -γ - ln w - Σ_{n≥1} (-1)^n w^n / (n · n!)
    let mut sum = Complex::new(0.0, 0.0);
    let mut power = Complex::new(1.0, 0.0);
    for n in 1..=400 {
        power *= w / n as f64;
        let term = power / n as f64;
        if n % 2 == 1 {
            sum -= term;
        } else {
            sum += term;
        }
        if term.norm() < 1e-18 * sum.norm() {
            break;
        }
    }
    w.exp() * (-Complex::new(EULER_GAMMA, 0.0) - w.ln() - sum)
}

/// [`scaled_exp1`] by continued fraction, in modified Lentz form.
///
/// `e^w E_1(w) = 1 / (w + 1 - 1²/(w + 3 - 2²/(w + 5 - ...)))`, which converges
/// in a dozen steps away from the negative real axis and not at all on it.
fn scaled_exp1_fraction(w: Complex<f64>) -> Complex<f64> {
    let mut b = w + 1.0;
    let mut c = Complex::new(1e300, 0.0);
    let mut d = b.inv();
    let mut h = d;
    for i in 1..=2000 {
        let a = Complex::new(-((i * i) as f64), 0.0);
        b += 2.0;
        d = (a * d + b).inv();
        c = b + a / c;
        let delta = c * d;
        h *= delta;
        if (delta.re - 1.0).abs() + delta.im.abs() < 1e-17 {
            break;
        }
    }
    h
}

/// The slowly-convergent integral in the progressive-wave functions.
///
/// The source writes the progressive wave potential and stream function with a
/// term
///
/// ```text
/// ∫_0^∞ (ν cos(k y) + k sin(k y)) / (k² + ν²) · e^(-k x) dk        (real part)
/// ∫_0^∞ (ν sin(k y) - k cos(k y)) / (k² + ν²) · e^(-k x) dk   (imaginary part)
/// ```
///
/// and notes that its numerical convergence is very slow. Both are parts of one
/// complex integral, because `(ν - i k)/(k² + ν²) = 1/(ν + i k)`:
///
/// ```text
/// J = ∫_0^∞ e^(-k (x - i y)) / (ν + i k) dk = -i e^w E_1(w),   w = -ν (y + i x)
/// ```
///
/// which is closed form and costs one [`scaled_exp1`]. Returns `J`, whose real
/// part is the first integral and imaginary part the second.
///
/// The argument has non-positive real part, so it sits on the side of the origin
/// where `E_1` keeps its branch cut, and reaches the cut itself when `x` is
/// exactly zero. Nothing here evaluates that point: the quadrature over the
/// section is open, so it excludes the keel, and the one point evaluated by hand
/// is the waterline, where `x` is the half beam. Points *near* the cut are
/// visited constantly, though, which is what [`SERIES_CANCELLATION_LIMIT`] is
/// about.
fn progressive_wave(nu: f64, x: f64, y: f64) -> Complex<f64> {
    let w = Complex::new(-nu * y, -nu * x);
    -Complex::<f64>::i() * scaled_exp1(w)
}

/// The progressive wave system evaluated at one point of the section surface.
///
/// Four numbers that all fall out of one [`progressive_wave`] call: the stream
/// function and the velocity potential, each in the phase with `cos(ωt)` and the
/// phase with `sin(ωt)`. Kept together because the exponential integral behind
/// them is the dominant cost of a solve, and computing them one at a time repeats
/// it.
#[derive(Debug, Clone, Copy)]
struct ProgressiveWaves {
    /// `ψ_B0c`.
    stream_cos: f64,
    /// `ψ_B0s`.
    stream_sin: f64,
    /// `φ_B0c`.
    potential_cos: f64,
    /// `φ_B0s`.
    potential_sin: f64,
}

/// Gauss-Legendre nodes and weights on `[a, b]`.
///
/// Computed rather than tabulated, by Newton's method on the Legendre
/// polynomial. The rule is open, which matters here: the integrands are
/// evaluated on `(0, π/2)` and the keel end of that interval is exactly where
/// the wave integral touches its branch cut.
fn gauss_legendre(n: usize, a: f64, b: f64) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);
    let half = 0.5 * (b - a);
    let mid = 0.5 * (b + a);

    for i in 0..n {
        // Chebyshev-like starting guess, then Newton on P_n.
        let mut t = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut derivative = 0.0;
        for _ in 0..100 {
            let (value, d) = legendre(n, t);
            derivative = d;
            let step = -value / d;
            t += step;
            if step.abs() < 1e-16 {
                break;
            }
        }
        nodes.push(mid + half * t);
        weights.push(2.0 * half / ((1.0 - t * t) * derivative * derivative));
    }
    (nodes, weights)
}

/// The Legendre polynomial of degree `n` and its derivative at `t`.
fn legendre(n: usize, t: f64) -> (f64, f64) {
    let mut previous = 1.0;
    let mut current = t;
    for k in 2..=n {
        let next = ((2 * k - 1) as f64 * t * current - (k - 1) as f64 * previous) / k as f64;
        previous = current;
        current = next;
    }
    let derivative = n as f64 * (t * current - previous) / (t * t - 1.0);
    (current, derivative)
}

/// How far the multipole series and the quadrature are taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TasaiOptions {
    /// Number of multipoles `M` in the standing-wave series.
    ///
    /// The source advises only `M ≥ N`, where `N = 2` is the number of Lewis
    /// mapping coefficients, which is far short of what the accuracy actually
    /// needs. This is the one knob that governs it: the quadrature below is
    /// converged well before its default, so every part of
    /// [`HeaveCoefficients::energy_residual`] that is not machine noise is this
    /// truncation.
    ///
    /// What it has to be depends on the *non-dimensional* frequency
    /// `ξ_b = ω² B / 2g`, not on `ω`. Twelve holds the identity to a few parts
    /// in a hundred thousand up to `ξ_b ≈ 3`, which covers a sailing yacht's
    /// sections over the whole range of encounter frequencies that matter —
    /// a 3.5 m section reaches `ξ_b = 3` only at 4 rad/s. Beyond that the series
    /// converges algebraically and slowly, because short waves concentrate the
    /// flow near the waterline and a series about the origin is a poor way to
    /// describe that. Raising this is cheap, since it is paid once per section
    /// per frequency at load rather than per frame.
    pub multipoles: usize,
    /// Number of Gauss-Legendre points across `(0, π/2)`.
    ///
    /// The integrands carry `sin(ν x)` and `cos(ν x)` over the section, so this
    /// has to resolve an oscillation that gets shorter with frequency. The
    /// default is past the point where the answer stops moving: at `ξ_b = 8`,
    /// quadrupling it changes the energy residual in no digit at all.
    pub quadrature: usize,
}

impl Default for TasaiOptions {
    fn default() -> Self {
        Self {
            multipoles: 12,
            quadrature: 96,
        }
    }
}

/// Two-dimensional heave coefficients for one section at one frequency.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeaveCoefficients {
    /// `M'_33`, hydrodynamic mass per unit length, kg/m.
    pub added_mass: f64,
    /// `N'_33`, hydrodynamic damping per unit length, kg/(m·s).
    pub damping: f64,
    /// Ratio of radiated wave amplitude to heave amplitude, dimensionless.
    ///
    /// `η_a / y_a`. Published because it is the physical content of the damping —
    /// a section damps heave exactly to the extent that it makes waves — and
    /// because it is the quantity Tasai's own figures plot.
    pub wave_amplitude_ratio: f64,
    /// Relative departure from the source's energy identity, dimensionless.
    ///
    /// `(M_0 A_0 - N_0 B_0) / (π²/2) - 1`. Zero in exact arithmetic at every
    /// frequency. What it measures in practice is truncation: it grows when
    /// [`TasaiOptions::multipoles`] is too small for the frequency, and it is
    /// the honest way to ask whether a computed coefficient can be trusted.
    pub energy_residual: f64,
}

/// Two-dimensional sway and roll coefficients for one section at one frequency.
///
/// The antisymmetric problem, §4.1.2 and §4.1.3 of the same source, solved in one
/// pass. Sway and roll share the standing-wave basis, the progressive-wave system
/// and therefore the Gram matrix; they differ only in the surface shape that
/// forces them. Solving them together costs two extra right-hand sides rather
/// than a second factorisation — and it is what makes the reciprocity check free,
/// which is the strongest oracle this module has.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LateralCoefficients {
    /// `M'_22`, sway added mass per unit length, kg/m.
    pub sway_added_mass: f64,
    /// `N'_22`, sway damping per unit length, kg/(m·s).
    pub sway_damping: f64,
    /// `M'_44`, roll added moment of inertia per unit length, kg·m.
    pub roll_added_inertia: f64,
    /// `N'_44`, roll damping per unit length, kg·m/s.
    pub roll_damping: f64,
    /// `M'_42`, the sway-into-roll added-mass coupling, kg.
    ///
    /// Taken from the sway solve, which is the better conditioned of the two
    /// routes to it: sway's forcing shape `g(θ)` is order one over the whole
    /// section, where roll's `μ(θ) - 1` vanishes at the waterline and so carries
    /// less of the projection.
    pub coupling_added_mass: f64,
    /// `N'_42`, the sway-into-roll damping coupling, kg/s.
    pub coupling_damping: f64,
    /// `η_a / x_a`, radiated wave amplitude per unit sway amplitude.
    pub sway_wave_amplitude_ratio: f64,
    /// `η_a / β_a`, radiated wave amplitude per unit roll angle, m/rad.
    pub roll_wave_amplitude_ratio: f64,
    /// Relative departure from the sway energy identity, `M_0 P_0 - N_0 Q_0 = π²/2`.
    pub sway_energy_residual: f64,
    /// Relative departure from the roll energy identity, `Y_R P_0 - X_R Q_0 = π²/8`.
    pub roll_energy_residual: f64,
    /// Relative disagreement between `M'_42` and `M'_24`.
    ///
    /// Potential flow makes the added-mass matrix symmetric, so the sway solve's
    /// roll moment and the roll solve's lateral force must report the same
    /// number. They are computed from different solutions by different formulae
    /// with different constant factors, and nothing in the arithmetic forces them
    /// to agree — which is exactly why their agreement is worth measuring. It
    /// tests both solves, both pressure integrals and the whole shared basis at
    /// once, against no stored answer at all.
    pub reciprocity_residual: f64,
}

/// A reusable solver for the section radiation problem.
///
/// Holds the Gauss-Legendre rule, which depends only on
/// [`TasaiOptions::quadrature`] and not on the section or the frequency. That
/// sounds like a detail and is not: building the rule means Newton's method on a
/// polynomial of the rule's own degree, and measurement puts it at roughly
/// nine-tenths of the cost of a solve. A frequency sweep over a hull is
/// thousands of solves, so the rule is built once here and the solves that
/// follow are cheap.
///
/// This is why there is no free function taking [`TasaiOptions`]: it would have
/// exactly the shape of the mistake.
#[derive(Debug, Clone)]
pub struct SectionSolver {
    nodes: Vec<f64>,
    weights: Vec<f64>,
    multipoles: usize,
}

impl SectionSolver {
    /// Builds the quadrature rule for these options.
    #[must_use]
    pub fn new(options: TasaiOptions) -> Self {
        let (nodes, weights) = gauss_legendre(options.quadrature, 0.0, PI / 2.0);
        Self {
            nodes,
            weights,
            multipoles: options.multipoles,
        }
    }

    /// Number of multipoles this solver was built for.
    #[must_use]
    pub fn multipoles(&self) -> usize {
        self.multipoles
    }

    /// Solves the heave radiation problem for a Lewis section.
    ///
    /// `omega` is the radian frequency of oscillation, `density` the water density
    /// and `gravity` the acceleration of gravity. Returns `None` for a
    /// non-positive frequency, where the radiation problem is not posed: the
    /// progressive wave system that carries the energy away does not exist, and the
    /// wave integral's argument collapses onto the origin.
    #[must_use]
    pub fn heave(
        &self,
        form: &LewisForm,
        omega: f64,
        density: f64,
        gravity: f64,
    ) -> Option<HeaveCoefficients> {
        if omega <= 0.0 || self.multipoles == 0 {
            return None;
        }
        let multipoles = self.multipoles;
        // The mapping coefficients, indexed as the source indexes them: `a[n]` is
        // `a_{2n-1}`, so `a[0]` is `a_-1`, which is always one.
        let a = [1.0, form.a1, form.a3];
        const N: usize = 2;

        // σ_a is the value the half-breadth series takes at the waterline, so that
        // `h(π/2) = 1`; `b_0` is the full beam; `ξ_b` is the squared
        // non-dimensional frequency, `ω² b_0 / 2g`.
        let sigma_a = 1.0 + form.a1 + form.a3;
        let beam = 2.0 * form.scale * sigma_a;
        let nu = omega * omega / gravity;
        let xi_b = nu * beam / 2.0;
        let frequency_ratio = xi_b / sigma_a;

        // `h(θ) = 2 x_0 / b_0`: the half-breadth normalised so that it is one at
        // the waterline.
        let h = |theta: f64| form.contour(theta).0 / (form.scale * sigma_a);

        // The standing-wave stream function `ψ_A0_2m`, decomposed once into the
        // harmonics it is made of.
        //
        // `ψ_A0_2m(θ) = sin(2mθ) - (ξ_b/σ_a) Σ_n (-1)ⁿ (2n-1)/(2m+2n-1) a_{2n-1}
        // sin((2m+2n-1)θ)`, so for each `m` it is a fixed short list of
        // `(harmonic index, weight)` pairs, independent of `θ`. Building that list
        // here means the basis assembly below is multiply-adds over a table of sines
        // rather than three transcendental calls per multipole per node — which was
        // the dominant cost of a solve, ahead even of the exponential integral.
        let mut harmonics: Vec<[(usize, f64); N + 2]> = Vec::with_capacity(multipoles);
        let mut highest = 0;
        for m in 1..=multipoles {
            let mut terms = [(0_usize, 0.0_f64); N + 2];
            terms[0] = (2 * m, 1.0);
            for (n, &coefficient) in a.iter().enumerate() {
                let order = 2.0 * n as f64 - 1.0;
                let shifted = 2 * m + 2 * n - 1;
                let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
                terms[n + 1] = (
                    shifted,
                    -frequency_ratio * sign * (order / shifted as f64) * coefficient,
                );
            }
            highest = highest.max(terms.iter().map(|&(index, _)| index).max().unwrap_or(0));
            harmonics.push(terms);
        }

        // `sin(kθ)` for every `k` the harmonics ask for, by the Chebyshev
        // recurrence `sin((k+1)θ) = 2 cos θ sin(kθ) - sin((k-1)θ)`. Two
        // transcendental calls per node instead of one per harmonic.
        let sines_at = |theta: f64| {
            let (sine, cosine) = theta.sin_cos();
            let mut table = vec![0.0; highest + 1];
            if highest >= 1 {
                table[1] = sine;
            }
            for k in 2..=highest {
                table[k] = 2.0 * cosine * table[k - 1] - table[k - 2];
            }
            table
        };

        // `ψ_A0_2m(π/2)`, which the source gives in closed form. Deriving it from
        // the general expression above reproduces this exactly, which is worth
        // knowing: it is an independent check that the index conventions here match
        // the source's.
        let standing_at_waterline = |m: usize| {
            let mut series = 0.0;
            for (n, coefficient) in a.iter().enumerate() {
                let order = 2.0 * n as f64 - 1.0;
                let shifted = 2.0 * m as f64 + 2.0 * n as f64 - 1.0;
                series += (order / shifted) * coefficient;
            }
            let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
            frequency_ratio * sign * series
        };

        // The four progressive-wave functions at one point of the surface. They
        // share a single exponential integral, which is the expensive part: an
        // earlier arrangement asked for them one at a time and paid for the same
        // integral four times at every quadrature node.
        let progressive = |theta: f64| {
            let (x, y) = form.contour(theta);
            let decay = PI * (-nu * y).exp();
            let (sine, cosine) = (nu * x).sin_cos();
            let wave = progressive_wave(nu, x, y);
            ProgressiveWaves {
                stream_cos: decay * sine,
                stream_sin: -decay * cosine + wave.re,
                potential_cos: decay * cosine,
                potential_sin: decay * sine + wave.im,
            }
        };

        // Everything that depends on the quadrature node but not on the multipole
        // index, tabulated once.
        let sampled: Vec<(f64, f64, ProgressiveWaves, f64)> = self
            .nodes
            .iter()
            .zip(self.weights.iter())
            .map(|(&theta, &weight)| {
                // The half-breadth slope the pressure integral weights by: the same
                // series as `h`, differentiated and unnormalised.
                let mut shape = 0.0;
                for (n, coefficient) in a.iter().enumerate() {
                    let order = 2.0 * n as f64 - 1.0;
                    let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
                    shape += sign * order * coefficient * (order * theta).cos();
                }
                (weight, h(theta), progressive(theta), shape)
            })
            .collect();

        // `f_2m(θ) = -ψ_A0_2m(θ) + h(θ) ψ_A0_2m(π/2)`: the basis the boundary
        // condition is projected onto.
        let waterline_standing: Vec<f64> = (1..=multipoles).map(standing_at_waterline).collect();
        let basis: Vec<Vec<f64>> = self
            .nodes
            .iter()
            .zip(sampled.iter())
            .map(|(&theta, (_, h_here, _, _))| {
                let sines = sines_at(theta);
                harmonics
                    .iter()
                    .zip(waterline_standing.iter())
                    .map(|(terms, &at_waterline)| {
                        let standing: f64 = terms
                            .iter()
                            .map(|&(index, weight)| weight * sines[index])
                            .sum();
                        h_here * at_waterline - standing
                    })
                    .collect()
            })
            .collect();

        // Galerkin projection. The matrix is the Gram matrix of that basis and is
        // therefore symmetric; the two right hand sides differ only in which
        // progressive-wave function they carry, so one factorisation serves both.
        let mut gram = DMatrix::zeros(multipoles, multipoles);
        let mut rhs_cos = DVector::zeros(multipoles);
        let mut rhs_sin = DVector::zeros(multipoles);
        let waterline = progressive(PI / 2.0);

        for (i, (weight, h_here, waves, _)) in sampled.iter().enumerate() {
            let forcing_cos = waves.stream_cos - h_here * waterline.stream_cos;
            let forcing_sin = waves.stream_sin - h_here * waterline.stream_sin;
            for n in 0..multipoles {
                let f_n = weight * basis[i][n];
                rhs_cos[n] += f_n * forcing_cos;
                rhs_sin[n] += f_n * forcing_sin;
                for m in 0..multipoles {
                    gram[(n, m)] += f_n * basis[i][m];
                }
            }
        }

        let factored = gram.lu();
        let p = factored.solve(&rhs_cos)?;
        let q = factored.solve(&rhs_sin)?;

        // `A_0` and `B_0`: the boundary condition evaluated at the waterline, which
        // is where the free surface meets the hull and `h` is one.
        let mut a0 = waterline.stream_cos;
        let mut b0 = waterline.stream_sin;
        for (m, &at_waterline) in waterline_standing.iter().enumerate() {
            a0 += p[m] * at_waterline;
            b0 += q[m] * at_waterline;
        }

        // `M_0` and `N_0`: the pressure integrated over the surface, in the two
        // phases. The three terms are the source's, in its order: the progressive
        // wave's own contribution, the standing waves', and a term the free surface
        // condition contributes at the waterline.
        let pressure = |multipole: &DVector<f64>, sine: bool| {
            let mut integral = 0.0;
            for (weight, _, waves, shape) in &sampled {
                let potential = if sine {
                    waves.potential_sin
                } else {
                    waves.potential_cos
                };
                integral += weight * potential * shape;
            }
            let mut standing_term = 0.0;
            for m in 1..=multipoles {
                let mut inner = 0.0;
                for (n, coefficient) in a.iter().enumerate() {
                    let order = 2.0 * n as f64 - 1.0;
                    let even = 2.0 * m as f64;
                    inner += (order * order / (even * even - order * order)) * coefficient;
                }
                let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
                standing_term += sign * multipole[m - 1] * inner;
            }
            // The waterline term reaches only as far as the mapping has
            // coefficients to pair up: `a_{2m+2n-1}` must exist, so `m + n ≤ N`.
            let mut waterline_term = multipole[0];
            for m in 1..=N.min(multipoles) {
                let mut inner = 0.0;
                for n in 0..=(N - m) {
                    let order = 2.0 * n as f64 - 1.0;
                    inner += order * a[n] * a[m + n];
                }
                let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
                waterline_term += sign * multipole[m - 1] * inner;
            }
            -integral / sigma_a - standing_term / sigma_a
                + PI * xi_b / (4.0 * sigma_a * sigma_a) * waterline_term
        };

        let m0 = pressure(&q, true);
        let n0 = pressure(&p, false);

        let denominator = a0 * a0 + b0 * b0;
        let energy = m0 * a0 - n0 * b0;

        Some(HeaveCoefficients {
            added_mass: density * beam * beam / 2.0 * (m0 * b0 + n0 * a0) / denominator,
            damping: density * beam * beam / 2.0 * energy / denominator * omega,
            wave_amplitude_ratio: PI * xi_b / denominator.sqrt(),
            energy_residual: energy / (PI * PI / 2.0) - 1.0,
        })
    }

    /// Solves the sway and roll radiation problems for a Lewis section.
    ///
    /// The antisymmetric half of §4.1, and one solve rather than two. Sway and
    /// roll share the standing-wave basis, the progressive-wave system and
    /// therefore the Gram matrix; only the shape that forces the free surface
    /// differs. Sway is driven by `g(θ) = 2y_0/b_0` and roll by `μ(θ) - 1`, where
    /// `μ` is the squared radius over the squared half beam.
    ///
    /// Both of those shapes vanish at the waterline, and that is not decoration:
    /// the boundary condition on the hull determines the stream function only up
    /// to a function of time, and it is evaluating the condition where the
    /// forcing shape is zero that eliminates the constant. The symmetric problem
    /// cannot do this — heave's `f(π/2)` is one — which is why the heave solve
    /// carries `h(θ)` through its basis and this one does not.
    ///
    /// Returns `None` for a non-positive frequency, where the radiation problem is
    /// not posed.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn lateral(
        &self,
        form: &LewisForm,
        omega: f64,
        density: f64,
        gravity: f64,
    ) -> Option<LateralCoefficients> {
        if omega <= 0.0 || self.multipoles == 0 {
            return None;
        }
        let multipoles = self.multipoles;
        let a = [1.0, form.a1, form.a3];
        const N: usize = 2;

        let sigma_a = 1.0 + form.a1 + form.a3;
        let half_beam = form.scale * sigma_a;
        let beam = 2.0 * half_beam;
        let nu = omega * omega / gravity;
        let xi_b = nu * beam / 2.0;
        let frequency_ratio = xi_b / sigma_a;

        // `ψ_A0_2m(θ) = -cos((2m+1)θ)
        //   + (ξ_b/σ_a) Σ_n (-1)ⁿ (2n-1)/(2m+2n) a_{2n-1} cos((2m+2n)θ)`,
        // decomposed once into the harmonics it is made of, for the same reason
        // the heave solve decomposes its own: the basis assembly below becomes
        // multiply-adds over a table rather than transcendental calls per
        // multipole per node.
        let mut harmonics: Vec<[(usize, f64); N + 2]> = Vec::with_capacity(multipoles);
        let mut highest = 0;
        for m in 1..=multipoles {
            let mut terms = [(0_usize, 0.0_f64); N + 2];
            terms[0] = (2 * m + 1, -1.0);
            for (n, &coefficient) in a.iter().enumerate() {
                let order = 2.0 * n as f64 - 1.0;
                let shifted = 2 * m + 2 * n;
                let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
                terms[n + 1] = (
                    shifted,
                    frequency_ratio * sign * (order / shifted as f64) * coefficient,
                );
            }
            highest = highest.max(terms.iter().map(|&(index, _)| index).max().unwrap_or(0));
            harmonics.push(terms);
        }

        // `cos(kθ)` for every `k` asked for, by the Chebyshev recurrence. The
        // antisymmetric basis is built of cosines where the symmetric one is built
        // of sines — the whole difference between the two problems, in one word.
        let cosines_at = |theta: f64| {
            let cosine = theta.cos();
            let mut table = vec![0.0; highest + 1];
            table[0] = 1.0;
            if highest >= 1 {
                table[1] = cosine;
            }
            for k in 2..=highest {
                table[k] = 2.0 * cosine * table[k - 1] - table[k - 2];
            }
            table
        };

        // `ψ_A0_2m(π/2)`, in the closed form the source gives: the odd harmonic
        // vanishes there and `cos((2m+2n)π/2) = (-1)^(m+n)`.
        let standing_at_waterline = |m: usize| {
            let mut series = 0.0;
            for (n, &coefficient) in a.iter().enumerate() {
                let order = 2.0 * n as f64 - 1.0;
                series += (order / (2 * m + 2 * n) as f64) * coefficient;
            }
            let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
            frequency_ratio * sign * series
        };

        // The progressive wave system, which here is a horizontal doublet at the
        // origin where the symmetric problem has a pulsating source. The source
        // reaches these through Porter's power series; `progressive_wave` returns
        // the two integrals Porter approximates as the real and imaginary parts of
        // one exponential integral, which is closed form and costs the same as the
        // heave solve's.
        let progressive = |theta: f64| {
            let (x, y) = form.contour(theta);
            let decay = PI * (-nu * y).exp();
            let (sine, cosine) = (nu * x).sin_cos();
            let wave = progressive_wave(nu, x, y);
            let radius = nu * (x * x + y * y);
            ProgressiveWaves {
                stream_cos: decay * cosine,
                stream_sin: decay * sine + wave.im - y / radius,
                potential_cos: -decay * sine,
                potential_sin: decay * cosine - wave.re + x / radius,
            }
        };

        // Everything that depends on the quadrature node but not the multipole
        // index, tabulated once.
        struct Sampled {
            weight: f64,
            /// `g(θ)`, the shape that forces sway.
            sway_shape: f64,
            /// `μ(θ) - 1`, the shape that forces roll.
            roll_shape: f64,
            waves: ProgressiveWaves,
            /// `Σ_n (-1)ⁿ (2n-1) a_{2n-1} sin((2n-1)θ)`, the weight the lateral
            /// pressure integral carries. It is `-(1/M_s) dy_0/dθ`, which is the
            /// projection of the surface normal onto the lateral direction.
            lateral_weight: f64,
            /// `Σ_n Σ_i (-1)^(n+i) (2i-1) a_{2n-1} a_{2i-1} sin((2n-2i)θ)`, the
            /// weight the roll moment integral carries.
            moment_weight: f64,
        }

        let sampled: Vec<Sampled> = self
            .nodes
            .iter()
            .zip(self.weights.iter())
            .map(|(&theta, &weight)| {
                let (x, y) = form.contour(theta);
                let mut lateral_weight = 0.0;
                for (n, &coefficient) in a.iter().enumerate() {
                    let order = 2.0 * n as f64 - 1.0;
                    let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
                    lateral_weight += sign * order * coefficient * (order * theta).sin();
                }
                let mut moment_weight = 0.0;
                for (n, &an) in a.iter().enumerate() {
                    for (i, &ai) in a.iter().enumerate() {
                        let odd = 2.0 * i as f64 - 1.0;
                        let sign = if (n + i) % 2 == 0 { 1.0 } else { -1.0 };
                        let difference = 2.0 * n as f64 - 2.0 * i as f64;
                        moment_weight += sign * odd * an * ai * (difference * theta).sin();
                    }
                }
                Sampled {
                    weight,
                    sway_shape: y / half_beam,
                    roll_shape: (x * x + y * y) / (half_beam * half_beam) - 1.0,
                    waves: progressive(theta),
                    lateral_weight,
                    moment_weight,
                }
            })
            .collect();

        // `f_2m(θ) = -ψ_A0_2m(θ) + ψ_A0_2m(π/2)` for `m ≥ 1`, with the mode's own
        // forcing shape taking the `m = 0` slot. That is the source's device for
        // making `P_0` and `Q_0` fall out of the same least-squares solve as the
        // multipole strengths, instead of being recovered separately: they are the
        // coefficients of the forcing shape in the same expansion.
        let unknowns = multipoles + 1;
        let waterline_standing: Vec<f64> = (1..=multipoles).map(standing_at_waterline).collect();
        let basis: Vec<Vec<f64>> = self
            .nodes
            .iter()
            .map(|&theta| {
                let cosines = cosines_at(theta);
                let mut row = Vec::with_capacity(unknowns);
                // The `m = 0` slot is the mode's own forcing shape, which is what
                // sway and roll disagree about; the solve below supplies it. Left
                // as a signalling value rather than zero so that reading it by
                // mistake is loud instead of quietly wrong.
                row.push(f64::NAN);
                row.extend(harmonics.iter().zip(waterline_standing.iter()).map(
                    |(terms, &at_waterline)| {
                        let standing: f64 = terms
                            .iter()
                            .map(|&(index, weight)| weight * cosines[index])
                            .sum();
                        at_waterline - standing
                    },
                ));
                row
            })
            .collect();

        // The Gram matrix differs between the modes only in its first row and
        // column, so it is assembled twice — but the right-hand sides, the
        // progressive-wave forcing, are shared. Assembling both here keeps the
        // expensive part, the exponential integrals in `sampled`, paid once.
        let waterline = progressive(PI / 2.0);
        let solve = |shape: fn(&Sampled) -> f64| -> Option<(DVector<f64>, DVector<f64>)> {
            let mut gram = DMatrix::zeros(unknowns, unknowns);
            let mut rhs_cos = DVector::zeros(unknowns);
            let mut rhs_sin = DVector::zeros(unknowns);
            for (sample, row) in sampled.iter().zip(basis.iter()) {
                let forcing_cos = sample.waves.stream_cos - waterline.stream_cos;
                let forcing_sin = sample.waves.stream_sin - waterline.stream_sin;
                let value = |k: usize| if k == 0 { shape(sample) } else { row[k] };
                for n in 0..unknowns {
                    let f_n = sample.weight * value(n);
                    rhs_cos[n] += f_n * forcing_cos;
                    rhs_sin[n] += f_n * forcing_sin;
                    for m in 0..unknowns {
                        gram[(n, m)] += f_n * value(m);
                    }
                }
            }
            let factored = gram.lu();
            Some((factored.solve(&rhs_cos)?, factored.solve(&rhs_sin)?))
        };

        let (p_sway, q_sway) = solve(|s| s.sway_shape)?;
        let (p_roll, q_roll) = solve(|s| s.roll_shape)?;

        // `M_0` and `N_0`: the lateral pressure integral of §4.1.2, three terms in
        // the source's order — the progressive wave's own contribution, a term the
        // free surface condition leaves at the waterline, and the standing waves'.
        let lateral_force = |multipole: &DVector<f64>, sine: bool| {
            let mut integral = 0.0;
            for sample in &sampled {
                let potential = if sine {
                    sample.waves.potential_sin
                } else {
                    sample.waves.potential_cos
                };
                integral += sample.weight * potential * sample.lateral_weight;
            }
            // Reaches only as far as the mapping has coefficients to offer:
            // `a_{2m+1}` is `a[m + 1]`, so this stops at `m = N - 1`.
            let mut at_waterline = 0.0;
            for m in 1..N.min(multipoles + 1) {
                let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
                at_waterline += sign * multipole[m] * (2 * m + 1) as f64 * a[m + 1];
            }
            let mut standing = 0.0;
            for m in 1..=multipoles {
                let mut inner = 0.0;
                for (n, &an) in a.iter().enumerate() {
                    let order = 2.0 * n as f64 - 1.0;
                    for (i, &ai) in a.iter().enumerate() {
                        let odd = 2.0 * i as f64 - 1.0;
                        let even = (2 * m + 2 * i) as f64;
                        inner += order * odd / (even * even - order * order) * an * ai;
                    }
                }
                let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
                standing += sign * multipole[m] * inner;
            }
            -integral / sigma_a
                + PI / (4.0 * sigma_a) * at_waterline
                + xi_b / (sigma_a * sigma_a) * standing
        };

        // `Y_R` and `X_R`: the roll moment integral, shared verbatim between
        // §4.1.2 and §4.1.3 — the source says so, and it is the reason one pair of
        // routines serves both couplings.
        let roll_moment = |multipole: &DVector<f64>, sine: bool| {
            let mut integral = 0.0;
            for sample in &sampled {
                let potential = if sine {
                    sample.waves.potential_sin
                } else {
                    sample.waves.potential_cos
                };
                integral += sample.weight * potential * sample.moment_weight;
            }
            let mut standing = 0.0;
            for m in 1..=multipoles {
                let mut inner = 0.0;
                let odd_squared = (2 * m + 1) as f64 * (2 * m + 1) as f64;
                for (n, &an) in a.iter().enumerate() {
                    for (i, &ai) in a.iter().enumerate() {
                        let odd = 2.0 * i as f64 - 1.0;
                        let difference = 2.0 * n as f64 - 2.0 * i as f64;
                        inner +=
                            odd * difference / (odd_squared - difference * difference) * an * ai;
                    }
                }
                let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
                standing += sign * multipole[m] * inner;
            }
            // The triple-product term, whose third factor is a mapping coefficient
            // at a shifted index: `a_{-2m+2n-2i-1}` is `a[n - i - m]`, and the
            // summation limits are exactly the ones that keep that index inside the
            // mapping — and, incidentally, keep `2n - 2i` away from zero.
            let mut triple = 0.0;
            for m in 1..=N.min(multipoles) {
                let mut inner = 0.0;
                for n in m..=N {
                    for i in 0..=(n - m) {
                        let shifted = -2.0 * m as f64 + 2.0 * n as f64 - 2.0 * i as f64 - 1.0;
                        let odd = 2.0 * i as f64 - 1.0;
                        let difference = 2.0 * n as f64 - 2.0 * i as f64;
                        inner += shifted * odd / difference * a[n] * a[i] * a[n - i - m];
                    }
                }
                for n in 0..=N {
                    for i in (m + n)..=N {
                        let shifted = -2.0 * m as f64 - 2.0 * n as f64 + 2.0 * i as f64 - 1.0;
                        let odd = 2.0 * i as f64 - 1.0;
                        let difference = 2.0 * n as f64 - 2.0 * i as f64;
                        inner += shifted * odd / difference * a[n] * a[i] * a[i - n - m];
                    }
                }
                let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
                triple += sign * multipole[m] * inner;
            }
            let two_sigma_squared = 2.0 * sigma_a * sigma_a;
            integral / two_sigma_squared + standing / two_sigma_squared
                - PI * xi_b / (8.0 * sigma_a * sigma_a * sigma_a) * triple
        };

        let sway_lateral = (lateral_force(&q_sway, true), lateral_force(&p_sway, false));
        let sway_moment = (roll_moment(&q_sway, true), roll_moment(&p_sway, false));
        let roll_lateral = (lateral_force(&q_roll, true), lateral_force(&p_roll, false));
        let roll_moment_pair = (roll_moment(&q_roll, true), roll_moment(&p_roll, false));

        let sway_denominator = p_sway[0] * p_sway[0] + q_sway[0] * q_sway[0];
        let roll_denominator = p_roll[0] * p_roll[0] + q_roll[0] * q_roll[0];

        // In phase with the displacement, and out of phase with it: added mass and
        // damping respectively, for whichever force this pair describes.
        let in_phase = |(m, n): (f64, f64), p: f64, q: f64| m * q + n * p;
        let out_of_phase = |(m, n): (f64, f64), p: f64, q: f64| m * p - n * q;

        let sway_energy = out_of_phase(sway_lateral, p_sway[0], q_sway[0]);
        let roll_energy = out_of_phase(roll_moment_pair, p_roll[0], q_roll[0]);

        let sway_scale = density * beam * beam / 2.0 / sway_denominator;
        let coupling_scale = -density * beam.powi(3) / 2.0 / sway_denominator;
        let roll_scale = density * beam.powi(4) / 8.0 / roll_denominator;
        let reciprocity_scale = -density * beam.powi(3) / 8.0 / roll_denominator;

        let coupling_added_mass = coupling_scale * in_phase(sway_moment, p_sway[0], q_sway[0]);
        let reciprocity_added_mass =
            reciprocity_scale * in_phase(roll_lateral, p_roll[0], q_roll[0]);
        let average = 0.5 * (coupling_added_mass.abs() + reciprocity_added_mass.abs());

        Some(LateralCoefficients {
            sway_added_mass: sway_scale * in_phase(sway_lateral, p_sway[0], q_sway[0]),
            sway_damping: sway_scale * sway_energy * omega,
            roll_added_inertia: roll_scale * in_phase(roll_moment_pair, p_roll[0], q_roll[0]),
            roll_damping: roll_scale * roll_energy * omega,
            coupling_added_mass,
            coupling_damping: coupling_scale
                * out_of_phase(sway_moment, p_sway[0], q_sway[0])
                * omega,
            sway_wave_amplitude_ratio: PI * xi_b / sway_denominator.sqrt(),
            roll_wave_amplitude_ratio: PI * xi_b * beam / (4.0 * roll_denominator.sqrt()),
            sway_energy_residual: sway_energy / (PI * PI / 2.0) - 1.0,
            roll_energy_residual: roll_energy / (PI * PI / 8.0) - 1.0,
            reciprocity_residual: if average > 0.0 {
                (coupling_added_mass - reciprocity_added_mass) / average
            } else {
                0.0
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boat::{Offset, Station};
    use crate::lewis::{station_geometry, SectionGeometry};
    use approx::assert_relative_eq;

    const WATER: f64 = 1025.0;
    const GRAVITY: f64 = 9.81;

    /// The section the source itself works through, from its §3.4: the amidships
    /// section of a container vessel, 25.40 m broad and 9.00 m deep. Its offsets
    /// are published, so this is the section whose Lewis fit
    /// `lewis::tests::the_fit_matches_an_independently_published_computation`
    /// already checks, and the section Figure 4.12 plots coefficients for.
    fn containership_midship() -> LewisForm {
        let offsets = [
            (0.000, 0.000),
            (0.135, 4.950),
            (0.270, 9.900),
            (0.500, 10.960),
            (1.000, 11.740),
            (2.000, 12.440),
            (3.050, 12.700),
            (6.000, 12.700),
            (9.000, 12.700),
        ];
        let station = Station {
            x: 0.0,
            points: offsets.into_iter().map(|(z, y)| Offset { y, z }).collect(),
        };
        LewisForm::fit(&station_geometry(&station, 9.0).expect("the section is immersed"))
    }

    fn solve(form: &LewisForm, omega: f64) -> HeaveCoefficients {
        SectionSolver::new(TasaiOptions::default())
            .heave(form, omega, WATER, GRAVITY)
            .expect("a positive frequency has a solution")
    }

    /// The closed form is the source's own expansion, not a substitute for it.
    ///
    /// The source avoids the wave integral by expanding it after
    /// [[Porter, 1960]], because it calls the integral's numerical convergence
    /// *"very slowly"*. Its §4.1.2 prints that expansion in full, and it turns
    /// out to be the same function: with `ζ = ν(y + i x)`, the source's `Q` and
    /// `S` satisfy `Q + iS = γ + ln ζ + Σ ζⁿ/(n·n!) = -E_1(-ζ) + iπ`, and
    /// substituting that into the source's own relation between `Q`, `S` and the
    /// two integrals collapses to `-i e^w E_1(w)` exactly.
    ///
    /// So this test transcribes `Q` and `S` as printed and checks the identity
    /// against [`progressive_wave`]. It is the strongest of the checks on this
    /// part, because it does not merely agree with the source numerically — it
    /// shows the code is evaluating the source's method rather than one of its
    /// own choosing.
    #[test]
    fn the_closed_form_is_the_sources_porter_expansion() {
        for &(nu, x, y) in &[
            (1.0_f64, 1.0_f64, 1.0_f64),
            (0.4, 3.0, 6.0),
            (2.0, 0.3, 1.5),
            (0.2, 12.7, 4.0),
            (1.0, 0.05, 9.0),
        ] {
            // Source §4.1.2, transcribed: β = arctan(x/y), pₙ = (ν r)ⁿ/(n·n!),
            // Q = γ + ln(ν r) + Σ pₙ cos(nβ), S = β + Σ pₙ sin(nβ).
            let radius = (x * x + y * y).sqrt();
            let beta = x.atan2(y);
            let mut q = EULER_GAMMA + (nu * radius).ln();
            let mut s = beta;
            let mut p = 1.0;
            for n in 1..=200 {
                p *= nu * radius / n as f64;
                let term = p / n as f64;
                q += term * (n as f64 * beta).cos();
                s += term * (n as f64 * beta).sin();
                if term < 1e-18 {
                    break;
                }
            }

            // And the source's relation between those and the two integrals.
            let decay = (-nu * y).exp();
            let cosine = (q * (nu * x).sin() - (s - PI) * (nu * x).cos()) * decay;
            let sine = (q * (nu * x).cos() + (s - PI) * (nu * x).sin()) * decay;

            let closed = progressive_wave(nu, x, y);
            assert_relative_eq!(closed.re, cosine, max_relative = 1e-11);
            assert_relative_eq!(closed.im, sine, max_relative = 1e-11);
        }
    }

    /// The same integral against brute-force quadrature.
    ///
    /// A second, blunter opinion than the expansion above: the integral as
    /// printed, integrated numerically, with no shared algebra to share an error
    /// with.
    ///
    /// The quadrature is deliberately crude — a fixed panel sweep out to a large
    /// wavenumber — because a crude method that agrees is stronger evidence than
    /// a clever one that might be wrong the same way.
    #[test]
    fn the_wave_integral_matches_direct_quadrature() {
        // The last three reach past `SERIES_CANCELLATION_LIMIT` and so exercise
        // the continued fraction; the fourth sits a thousandth of a degree off
        // the branch cut, which is the keel of a real section at a real
        // frequency and the case a naive `|w|` cutoff gets wrong.
        for &(nu, x, y) in &[
            (1.0, 1.0, 1.0),
            (0.5, 2.0, 0.7),
            (2.0, 0.3, 1.5),
            (0.2, 12.7, 4.0),
            (1.6, 12.7, 0.0),
            (1.6, 12.7, 4.0),
            (3.0, 12.7, 2.0),
            (2.0, 0.009, 9.0),
        ] {
            let (nodes, weights) = gauss_legendre(64, 0.0, 1.0);
            let mut cosine = 0.0;
            let mut sine = 0.0;
            let panels = 3000;
            let ceiling = 3000.0;
            for panel in 0..panels {
                let lo = ceiling * panel as f64 / panels as f64;
                let hi = ceiling * (panel + 1) as f64 / panels as f64;
                for (&node, &weight) in nodes.iter().zip(weights.iter()) {
                    let k = lo + (hi - lo) * node;
                    let scale = weight * (hi - lo) * (-k * x).exp() / (k * k + nu * nu);
                    cosine += scale * (nu * (k * y).cos() + k * (k * y).sin());
                    sine += scale * (nu * (k * y).sin() - k * (k * y).cos());
                }
            }
            let closed = progressive_wave(nu, x, y);
            assert_relative_eq!(closed.re, cosine, max_relative = 1e-9);
            assert_relative_eq!(closed.im, sine, max_relative = 1e-9);
        }
    }

    /// The two evaluations of the scaled integral have to agree where they meet.
    ///
    /// Otherwise a coefficient would step discontinuously as frequency swept a
    /// section across the crossover. Both are evaluated at the *same* argument,
    /// which is the only comparison that means anything — the function varies
    /// along the ray by more than either method's error, so straddling the
    /// boundary with two nearby points measures the slope, not the agreement.
    ///
    /// The arguments chosen sit away from the branch cut, where the continued
    /// fraction converges in a dozen steps and the series has not yet spent its
    /// digits, because agreement is only evidence when both methods are entitled
    /// to be believed.
    #[test]
    fn the_two_evaluations_agree_at_their_crossover() {
        for &degrees in &[-100.0_f64, -110.0, -125.0, -150.0] {
            let direction = Complex::new(degrees.to_radians().cos(), degrees.to_radians().sin());
            // `|w| + Re w` is linear in `|w|` along a fixed ray, so this radius
            // puts the criterion exactly at its limit.
            let w = direction * (SERIES_CANCELLATION_LIMIT / (1.0 + direction.re));
            assert_relative_eq!(
                scaled_exp1_series(w).re,
                scaled_exp1_fraction(w).re,
                max_relative = 1e-9
            );
            assert_relative_eq!(
                scaled_exp1_series(w).im,
                scaled_exp1_fraction(w).im,
                max_relative = 1e-9
            );
        }
    }

    /// The series is the accurate one beside the branch cut.
    ///
    /// This is why the criterion is on `|w| + Re w` rather than on `|w|`. At the
    /// argument below — an ordinary hull's keel at an ordinary frequency, where
    /// the half-breadth has gone almost to zero — a cutoff on `|w|` alone would
    /// have chosen the continued fraction, which here exhausts two thousand
    /// iterations without meeting its tolerance and lands nearly six orders of
    /// magnitude further from the answer than the series does.
    ///
    /// The answer it is measured against is the direct quadrature in
    /// `the_wave_integral_matches_direct_quadrature`, which covers this same
    /// region; what is pinned here is only that the two evaluations are *not*
    /// interchangeable, so that nobody simplifies the criterion back to `|w|`.
    #[test]
    fn the_series_and_the_fraction_are_not_interchangeable_near_the_cut() {
        let w = Complex::new(-12.19, -0.4876);
        assert!(
            w.norm() > SERIES_CANCELLATION_LIMIT,
            "a cutoff on |w| would send this to the fraction"
        );
        assert!(
            w.norm() + w.re < SERIES_CANCELLATION_LIMIT,
            "the cancellation criterion keeps it in the series"
        );
        let series = scaled_exp1_series(w);
        let fraction = scaled_exp1_fraction(w);
        let separation = (series - fraction).norm() / series.norm();
        assert!(
            separation > 1e-10,
            "the two evaluations agreed to {separation} here, so the criterion is moot"
        );
    }

    /// The source's energy identity, which is the real test of the transcription.
    ///
    /// The pressure integral and the radiated-energy argument are two
    /// independent routes to the damping, and equating them gives
    /// `M_0 A_0 - N_0 B_0 = π²/2` at every frequency. Every quantity in the
    /// method appears in it, so a dropped factor or a slipped index cannot
    /// survive it.
    ///
    /// The frequencies run to `ω = 1.5`, which for this 25.4 m section is
    /// `ξ_b = 2.9` — the top of the non-dimensional range
    /// [`TasaiOptions::multipoles`] claims. What happens above it is not left to
    /// inference; `the_identity_degrades_at_high_reduced_frequency` states it.
    #[test]
    fn the_energy_identity_holds_at_every_frequency() {
        let form = containership_midship();
        for &omega in &[0.25, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5] {
            let solved = solve(&form, omega);
            assert!(
                solved.energy_residual.abs() < 1e-4,
                "energy identity at omega = {omega}: residual {}",
                solved.energy_residual
            );
        }
    }

    /// The known limit of the method, recorded rather than avoided.
    ///
    /// Short waves concentrate the flow near the waterline, and a multipole
    /// series about the origin needs many terms to describe that. At `ξ_b ≈ 16`
    /// the default truncation is off by percent, and a caller who cares has to
    /// raise it — which is exactly why the residual is published alongside the
    /// coefficients instead of being checked once here and forgotten.
    #[test]
    fn the_identity_degrades_at_high_reduced_frequency() {
        let form = containership_midship();
        let loose = solve(&form, 3.5).energy_residual.abs();
        assert!(
            loose > 1e-3,
            "the high-frequency limit is real; if this got better, the note above is stale"
        );
        let tightened = SectionSolver::new(TasaiOptions {
            multipoles: 48,
            quadrature: 96,
        })
        .heave(&form, 3.5, WATER, GRAVITY)
        .expect("a positive frequency has a solution");
        assert!(
            tightened.energy_residual.abs() < 0.3 * loose,
            "more multipoles must buy accuracy at high frequency"
        );
    }

    /// Truncating the multipole series has to converge, not wander.
    #[test]
    fn more_multipoles_close_the_energy_identity() {
        let form = containership_midship();
        let mut previous = f64::INFINITY;
        for &multipoles in &[4, 6, 8, 12, 16] {
            let options = TasaiOptions {
                multipoles,
                quadrature: 192,
            };
            let solved = SectionSolver::new(options)
                .heave(&form, 2.5, WATER, GRAVITY)
                .expect("a positive frequency has a solution");
            let residual = solved.energy_residual.abs();
            assert!(
                residual < previous,
                "{multipoles} multipoles gave residual {residual}, worse than {previous}"
            );
            previous = residual;
        }
    }

    /// Damping is what the radiated waves carry away, so it cannot be negative,
    /// and it has to vanish at high frequency where a section makes no waves.
    #[test]
    fn damping_is_positive_and_dies_at_high_frequency() {
        let form = containership_midship();
        let mut last = f64::INFINITY;
        for &omega in &[1.5, 2.0, 2.5, 3.0, 3.5] {
            let solved = solve(&form, omega);
            assert!(
                solved.damping > 0.0,
                "damping at omega = {omega} went negative"
            );
            assert!(
                solved.damping < last,
                "damping at omega = {omega} rose again"
            );
            last = solved.damping;
        }
        assert!(
            solve(&form, 3.5).damping < 0.01 * solve(&form, 0.5).damping,
            "damping must all but vanish at high frequency"
        );
    }

    /// Damping and the radiated wave amplitude are the same statement.
    ///
    /// The source derives `N'_33 = ρ g² / ω³ · (η_a / y_a)²` from the energy the
    /// waves remove. It is a different arrangement of the same quantities than
    /// the one computed, so it pins the wave amplitude ratio against the damping
    /// rather than leaving it as an unchecked output.
    ///
    /// The two differ by exactly the energy residual, which is why the tolerance
    /// is taken from it rather than picked: these agree to whatever the
    /// multipole truncation allows, and no better.
    #[test]
    fn the_radiated_wave_amplitude_accounts_for_the_damping() {
        let form = containership_midship();
        for &omega in &[0.4, 0.75, 1.0, 1.5] {
            let solved = solve(&form, omega);
            let from_waves = WATER * GRAVITY * GRAVITY / omega.powi(3)
                * solved.wave_amplitude_ratio
                * solved.wave_amplitude_ratio;
            let tolerance = 2.0 * solved.energy_residual.abs().max(1e-12);
            assert_relative_eq!(solved.damping, from_waves, max_relative = tolerance);
        }
    }

    /// The published figure, which is the external oracle.
    ///
    /// Figure 4.12 of the source plots these coefficients against frequency for
    /// this section, computed by the same method. Read off the curves, heave mass
    /// falls from around 500 ton/m at the left edge to a minimum near 210 around
    /// `ω = 0.8`, then climbs to about 320 by `ω = 2.5`; damping peaks near
    /// 130 ton/m/s at about `ω = 0.6` and is gone by `ω = 2`.
    ///
    /// The tolerance is 8 % and it is a figure-reading tolerance, not a physics
    /// one: what this test defends is that the shape, the location of the
    /// minimum and the magnitudes are right, which is what catches a
    /// transcription that is wrong by a factor or a sign. The energy identity
    /// above is what defends the precision.
    #[test]
    fn the_coefficients_match_the_published_figure() {
        let form = containership_midship();
        let tonnes = 1000.0;

        let minimum = solve(&form, 0.8).added_mass / tonnes;
        assert_relative_eq!(minimum, 215.0, max_relative = 0.08);

        let high = solve(&form, 2.5).added_mass / tonnes;
        assert_relative_eq!(high, 320.0, max_relative = 0.08);

        let peak = solve(&form, 0.55).damping / tonnes;
        assert_relative_eq!(peak, 132.0, max_relative = 0.08);

        // The minimum is a minimum: the curve is above it either side.
        assert!(solve(&form, 0.4).added_mass / tonnes > minimum);
        assert!(solve(&form, 1.5).added_mass / tonnes > minimum);
    }

    /// A frequency of zero has no radiation problem to solve.
    #[test]
    fn a_still_section_has_no_radiation_solution() {
        let form = containership_midship();
        let solver = SectionSolver::new(TasaiOptions::default());
        assert!(solver.heave(&form, 0.0, WATER, GRAVITY).is_none());
        assert!(solver.heave(&form, -1.0, WATER, GRAVITY).is_none());
    }

    /// Added mass scales with the square of a section's size.
    ///
    /// Two geometrically similar sections at the same non-dimensional frequency
    /// must give the same coefficient once the length scale is divided out. The
    /// non-dimensional frequency is `ω² b_0 / 2g`, so doubling the section means
    /// dividing `ω` by `√2`. This is the one property of the answer that follows
    /// from dimensional analysis alone, independent of the source.
    #[test]
    fn geometrically_similar_sections_scale_by_area() {
        let small = SectionGeometry {
            beam: 10.0,
            draft: 4.0,
            area: 0.85 * 10.0 * 4.0,
        };
        let large = SectionGeometry {
            beam: 20.0,
            draft: 8.0,
            area: 0.85 * 20.0 * 8.0,
        };
        let small_form = LewisForm::fit(&small);
        let large_form = LewisForm::fit(&large);

        let omega = 1.2;
        let a = solve(&small_form, omega).added_mass;
        let b = solve(&large_form, omega / 2.0_f64.sqrt()).added_mass;
        assert_relative_eq!(b / a, 4.0, max_relative = 1e-9);
    }

    fn solve_lateral(form: &LewisForm, omega: f64) -> LateralCoefficients {
        SectionSolver::new(TasaiOptions::default())
            .lateral(form, omega, WATER, GRAVITY)
            .expect("a positive frequency has a solution")
    }

    /// A half circle, whose Lewis mapping is exact: `a_1` and `a_3` are zero and
    /// the mapping is the identity on the unit semicircle. The one section shape
    /// with closed-form answers to compare against.
    fn half_circle(radius: f64) -> LewisForm {
        LewisForm::fit(&SectionGeometry {
            beam: 2.0 * radius,
            draft: radius,
            area: PI * radius * radius / 2.0,
        })
    }

    /// Sway added mass reaches its closed-form value as the frequency vanishes.
    ///
    /// The free-surface condition is `(ω²/g) Φ + ∂Φ/∂y = 0`, so as `ω → 0` it
    /// degenerates into `∂Φ/∂y = 0` — a rigid wall. Reflecting the half circle in
    /// that wall gives a whole circle in unbounded fluid, whose lateral added mass
    /// is the mass of the fluid it displaces, and the half gets half of it:
    /// `ρ π r² / 2`.
    ///
    /// This is the only genuinely external check in this module — an answer from
    /// classical hydrodynamics, owing nothing to Tasai's formulation, the mapping
    /// or this implementation. Everything else here is an internal identity.
    #[test]
    fn sway_added_mass_reaches_the_analytic_value_of_a_circle() {
        let radius = 1.0;
        let form = half_circle(radius);
        let analytic = WATER * PI * radius * radius / 2.0;

        // ξ_b = ω² b_0 / 2g, and the approach is first order in it, so a
        // ten-thousandth buys four digits.
        let mut previous = f64::INFINITY;
        for xi in [1e-2_f64, 1e-3, 1e-4] {
            let omega = (2.0 * GRAVITY * xi / form.beam()).sqrt();
            let error = (solve_lateral(&form, omega).sway_added_mass / analytic - 1.0).abs();
            assert!(
                error < previous,
                "the limit must be approached, not straddled: {error:.2e} after {previous:.2e}"
            );
            previous = error;
        }
        assert!(
            previous < 3e-4,
            "a ten-thousandth of ξ_b left {previous:.2e} of relative error"
        );
    }

    /// A circle rolling about its own centre does not move the water.
    ///
    /// Rotation leaves the boundary invariant, so there is no normal velocity
    /// anywhere on it, no disturbance, no radiated wave and no reaction: added
    /// inertia and damping are both exactly zero. The value of the test is that
    /// it is exactly zero rather than approximately anything — it catches a
    /// spurious term in the roll moment integral that a non-degenerate section
    /// would bury in a plausible-looking number.
    ///
    /// It also says why the circle cannot validate roll anywhere else, which is
    /// what the containership section below is for.
    #[test]
    fn a_circle_radiates_nothing_when_it_rolls() {
        let form = half_circle(1.0);
        for omega in [0.2_f64, 0.8, 1.5, 3.0] {
            let c = solve_lateral(&form, omega);
            // Scaled against the roll inertia of the displaced fluid, which is what
            // a section this size would have if it did radiate.
            let scale = WATER * form.beam().powi(4) / 8.0;
            assert!(
                c.roll_added_inertia.abs() / scale < 1e-12,
                "at {omega} rad/s a circle claimed {} kg·m of roll added inertia",
                c.roll_added_inertia
            );
            assert!(
                c.roll_damping.abs() / (scale * omega) < 1e-12,
                "at {omega} rad/s a circle claimed {} of roll damping",
                c.roll_damping
            );
        }
    }

    /// Both energy identities hold, and tighten when the series is taken further.
    ///
    /// The exciting force's work must equal the energy the radiated wave carries
    /// away, which pins `M_0 P_0 - N_0 Q_0 = π²/2` for sway and
    /// `Y_R P_0 - X_R Q_0 = π²/8` for roll — exactly, in exact arithmetic, at
    /// every frequency. What the residual measures in practice is the multipole
    /// truncation, which is why the test asserts the trend as well as the size: a
    /// residual that did not fall with `M` would be a bug wearing the costume of a
    /// convergence error.
    ///
    /// The trend is asserted between the ends and not step by step, deliberately.
    /// The sway residual at low `ξ_b` is already down at a few parts in a million
    /// with six multipoles, which is the floor the quadrature and the Gram matrix's
    /// conditioning set rather than the truncation — and at the floor the sequence
    /// wanders instead of descending. Demanding monotonicity there would be
    /// testing the noise.
    #[test]
    fn the_lateral_energy_identities_hold_and_tighten_with_the_series() {
        let form = containership_midship();
        let residuals = |multipoles: usize, omega: f64| {
            let c = SectionSolver::new(TasaiOptions {
                multipoles,
                quadrature: 96,
            })
            .lateral(&form, omega, WATER, GRAVITY)
            .expect("a positive frequency has a solution");
            (c.sway_energy_residual.abs(), c.roll_energy_residual.abs())
        };

        for xi in [0.3_f64, 1.0, 2.0] {
            let omega = (2.0 * GRAVITY * xi / form.beam()).sqrt();
            let coarse = residuals(6, omega);
            let fine = residuals(20, omega);
            assert!(
                fine.0 < coarse.0 && fine.1 < coarse.1,
                "at ξ_b {xi} twenty multipoles gave {fine:?}, no better than six at {coarse:?}"
            );
            assert!(
                fine.0 < 1e-4 && fine.1 < 1e-3,
                "at ξ_b {xi} twenty multipoles left residuals of {fine:?}"
            );
        }
    }

    /// The added-mass coupling agrees by two routes that share no arithmetic.
    ///
    /// Potential flow makes the added-mass matrix symmetric, so the roll moment
    /// caused by swaying and the lateral force caused by rolling are the same
    /// number. They come from different solves — different forcing shapes, so
    /// different `P` and `Q` — through different pressure integrals with constant
    /// factors differing by a factor of four. Nothing in the algebra forces them
    /// together, so their agreement exercises both solves, both integrals and the
    /// shared basis at once, against no stored value at all.
    ///
    /// This is the strongest check in the module, and the reason the two modes are
    /// solved in one call rather than two.
    #[test]
    fn the_coupling_agrees_by_two_independent_routes() {
        let form = containership_midship();
        for xi in [0.1_f64, 0.3, 0.6, 1.0, 1.5, 2.0] {
            let omega = (2.0 * GRAVITY * xi / form.beam()).sqrt();
            let residual = solve_lateral(&form, omega).reciprocity_residual;
            assert!(
                residual.abs() < 2e-3,
                "at ξ_b {xi} the two routes to the coupling disagreed by {residual:.2e}"
            );
        }
    }

    /// Sway damping vanishes like the fifth power of frequency.
    ///
    /// A section damps sway exactly to the extent that it makes waves, and the
    /// amplitude it radiates is first order in `ξ_b`. Damping goes as `ω` times
    /// the square of that ratio, so `B₂₂ ∝ ω · ω⁴`. Worth pinning because it is
    /// the asymptote the whole low-frequency end of the spectrum rests on, and
    /// because the frequency grid now reaches far enough down to sit in it.
    #[test]
    fn sway_damping_vanishes_like_the_fifth_power_of_frequency() {
        let form = half_circle(1.0);
        let at = |xi: f64| {
            let omega = (2.0 * GRAVITY * xi / form.beam()).sqrt();
            solve_lateral(&form, omega).sway_damping
        };
        // A decade in ξ_b is half a decade in ω, so five powers of ω is two and a
        // half powers of ξ_b: a factor of 10^2.5 per decade.
        let ratio = at(1e-3) / at(1e-4);
        assert_relative_eq!(ratio, 10.0_f64.powf(2.5), max_relative = 0.02);
    }

    /// No frequency, no radiation problem — the same contract heave keeps.
    #[test]
    fn a_still_section_has_no_lateral_solution() {
        let form = half_circle(1.0);
        let solver = SectionSolver::new(TasaiOptions::default());
        assert!(solver.lateral(&form, 0.0, WATER, GRAVITY).is_none());
        assert!(solver.lateral(&form, -1.0, WATER, GRAVITY).is_none());
    }
}
