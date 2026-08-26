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

/// Solves the heave radiation problem for a Lewis section.
///
/// `omega` is the radian frequency of oscillation, `density` the water density
/// and `gravity` the acceleration of gravity. Returns `None` for a
/// non-positive frequency, where the radiation problem is not posed: the
/// progressive wave system that carries the energy away does not exist, and the
/// wave integral's argument collapses onto the origin.
#[must_use]
pub fn heave_coefficients(
    form: &LewisForm,
    omega: f64,
    density: f64,
    gravity: f64,
    options: TasaiOptions,
) -> Option<HeaveCoefficients> {
    if omega <= 0.0 || options.multipoles == 0 {
        return None;
    }

    // The mapping coefficients, indexed as the source indexes them: `a[n]` is
    // `a_{2n-1}`, so `a[0]` is `a_-1`, which is always one.
    let a = [1.0, form.a1, form.a3];
    const N: usize = 2;
    let multipoles = options.multipoles;

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

    // The standing-wave stream and potential functions, `ψ_A0_2m` and `φ_A0_2m`.
    // The two differ only in sine against cosine, so they share one body.
    let standing = |m: usize, theta: f64, sine: bool| {
        let mut series = 0.0;
        for (n, coefficient) in a.iter().enumerate() {
            let order = 2.0 * n as f64 - 1.0;
            let shifted = 2.0 * m as f64 + 2.0 * n as f64 - 1.0;
            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
            let harmonic = if sine {
                (shifted * theta).sin()
            } else {
                (shifted * theta).cos()
            };
            series += sign * (order / shifted) * coefficient * harmonic;
        }
        let principal = if sine {
            (2.0 * m as f64 * theta).sin()
        } else {
            (2.0 * m as f64 * theta).cos()
        };
        principal - frequency_ratio * series
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

    // The progressive-wave functions on the section surface. `stream` selects
    // the stream function `ψ_B0` over the potential `φ_B0`, `sine` the
    // component in phase with `sin(ωt)`.
    let progressive = |theta: f64, stream: bool, sine: bool| {
        let (x, y) = form.contour(theta);
        let decay = PI * (-nu * y).exp();
        match (stream, sine) {
            (true, false) => decay * (nu * x).sin(),
            (true, true) => -decay * (nu * x).cos() + progressive_wave(nu, x, y).re,
            (false, false) => decay * (nu * x).cos(),
            (false, true) => decay * (nu * x).sin() + progressive_wave(nu, x, y).im,
        }
    };

    // `f_2m(θ) = -ψ_A0_2m(θ) + h(θ) ψ_A0_2m(π/2)`: the basis the boundary
    // condition is projected onto.
    let (nodes, weights) = gauss_legendre(options.quadrature, 0.0, PI / 2.0);
    let basis: Vec<Vec<f64>> = nodes
        .iter()
        .map(|&theta| {
            (1..=multipoles)
                .map(|m| -standing(m, theta, true) + h(theta) * standing_at_waterline(m))
                .collect()
        })
        .collect();

    // Galerkin projection. The matrix is the Gram matrix of that basis and is
    // therefore symmetric; the two right hand sides differ only in which
    // progressive-wave function they carry, so one factorisation serves both.
    let mut gram = DMatrix::zeros(multipoles, multipoles);
    let mut rhs_cos = DVector::zeros(multipoles);
    let mut rhs_sin = DVector::zeros(multipoles);
    let waterline_cos = progressive(PI / 2.0, true, false);
    let waterline_sin = progressive(PI / 2.0, true, true);

    for (i, (&theta, &weight)) in nodes.iter().zip(weights.iter()).enumerate() {
        let forcing_cos = progressive(theta, true, false) - h(theta) * waterline_cos;
        let forcing_sin = progressive(theta, true, true) - h(theta) * waterline_sin;
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
    let mut a0 = waterline_cos;
    let mut b0 = waterline_sin;
    for m in 1..=multipoles {
        a0 += p[m - 1] * standing_at_waterline(m);
        b0 += q[m - 1] * standing_at_waterline(m);
    }

    // `M_0` and `N_0`: the pressure integrated over the surface, in the two
    // phases. The three terms are the source's, in its order: the progressive
    // wave's own contribution, the standing waves', and a term the free surface
    // condition contributes at the waterline.
    let pressure = |multipole: &DVector<f64>, sine: bool| {
        let mut integral = 0.0;
        for (&theta, &weight) in nodes.iter().zip(weights.iter()) {
            let mut shape = 0.0;
            for (n, coefficient) in a.iter().enumerate() {
                let order = 2.0 * n as f64 - 1.0;
                let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
                shape += sign * order * coefficient * (order * theta).cos();
            }
            integral += weight * progressive(theta, false, sine) * shape;
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
        heave_coefficients(form, omega, WATER, GRAVITY, TasaiOptions::default())
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
        let tightened = heave_coefficients(
            &form,
            3.5,
            WATER,
            GRAVITY,
            TasaiOptions {
                multipoles: 48,
                quadrature: 96,
            },
        )
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
            let solved = heave_coefficients(&form, 2.5, WATER, GRAVITY, options)
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
        assert!(heave_coefficients(&form, 0.0, WATER, GRAVITY, TasaiOptions::default()).is_none());
        assert!(heave_coefficients(&form, -1.0, WATER, GRAVITY, TasaiOptions::default()).is_none());
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
}
