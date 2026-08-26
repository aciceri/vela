//! The fluid memory of radiation, moved from frequency to time.
//!
//! # Why this exists
//!
//! [`crate::tasai`] and [`crate::strip`] give added mass and damping as
//! functions of frequency. A time-domain simulation cannot use them: at any
//! instant a boat is not oscillating at a frequency, it is doing whatever the
//! waves and the rig have just done to it. The bridge is Cummins,
//! [[Cummins, 1962]], who showed that the frequency-dependence is equivalent to
//! a constant added mass plus a convolution over the history of the velocity:
//!
//! ```text
//! (M + A_∞) ν̇ + ∫₀^t K(t - τ) ν(τ) dτ + G η = τ_ext
//! ```
//!
//! The kernel `K` is the *retardation function*, and [[Ogilvie, 1964]] gives
//! both directions between it and the frequency-domain pair:
//!
//! ```text
//! K(t) = (2/π) ∫₀^∞ B(ω) cos(ωt) dω
//! A(ω) = A_∞ - (1/ω) ∫₀^∞ K(t) sin(ωt) dt
//! B(ω) =        ∫₀^∞ K(t) cos(ωt) dt
//! ```
//!
//! Those relations are also the verification. `A_∞` is a *constant*, so
//! computing it from the second relation at many different frequencies must give
//! many copies of the same number. It involves the whole chain — the sectional
//! solve, the strip integration, the cosine transform, the truncation of both
//! grids — so agreement across frequency is a strong statement that all of it is
//! right, and [`InfiniteAddedMass::spread`] reports it rather than a test
//! asserting it once in private.
//!
//! # Why the convolution is not evaluated
//!
//! A convolution over all history is not a real-time operation, and storing the
//! history is not a real-time memory budget. Standard practice replaces it with
//! a low-order linear system whose impulse response *is* `K`
//! [[Perez & Fossen, 2008]](#bib-pf2008): fit a rational transfer function to
//!
//! ```text
//! K̂(jω) = B(ω) + jω [A(ω) - A_∞]
//! ```
//!
//! and realise it as a handful of states. Then the cost per step is a small
//! matrix-vector product — four or five states for one degree of freedom — and
//! the history lives in the states rather than in a buffer.
//!
//! Three things have to be true of that fit, and all three are checked here
//! rather than hoped for: it must reproduce `K̂` ([`FluidMemory::worst_error`]),
//! its poles must lie in the left half plane or the simulation diverges
//! ([`FluidMemory::slowest_pole`]), and it must be passive — `Re K̂(jω) ≥ 0`,
//! since a fluid that pumped energy into the boat would be a perpetual motion
//! machine ([`FluidMemory::passivity_violation`]).

use nalgebra::{Complex, DMatrix, DVector};
use std::f64::consts::PI;

/// A sampled radiation spectrum for one coefficient of the motion.
///
/// Frequencies must be positive and strictly increasing. The spectrum is the
/// only input to everything below, so its extent decides the accuracy: it has to
/// reach high enough that `B` has genuinely died away, because the cosine
/// transform integrates to infinity and simply stops where the samples do.
///
/// For a yacht that is further out than ship experience suggests. Radiation is
/// governed by the reduced frequency `ω²B/2g`, so a narrow section reaches a
/// given reduced frequency only at a high `ω`: the damping of a 3.2 m section is
/// still a quarter of its peak at 6 rad/s and needs sampling to something like
/// 30 before the tail is negligible.
#[derive(Debug, Clone, PartialEq)]
pub struct Spectrum {
    frequencies: Vec<f64>,
    added_mass: Vec<f64>,
    damping: Vec<f64>,
}

/// Why a spectrum could not be accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectrumError {
    /// Fewer than three samples: nothing to transform.
    TooFewSamples,
    /// The three columns are not the same length.
    Ragged,
    /// A frequency was not positive, or the sequence did not increase.
    NotIncreasing,
}

impl Spectrum {
    /// Builds a spectrum from matching frequency, added-mass and damping columns.
    pub fn new(
        frequencies: Vec<f64>,
        added_mass: Vec<f64>,
        damping: Vec<f64>,
    ) -> Result<Self, SpectrumError> {
        if frequencies.len() != added_mass.len() || frequencies.len() != damping.len() {
            return Err(SpectrumError::Ragged);
        }
        if frequencies.len() < 3 {
            return Err(SpectrumError::TooFewSamples);
        }
        if frequencies[0] <= 0.0 || frequencies.windows(2).any(|pair| pair[1] <= pair[0]) {
            return Err(SpectrumError::NotIncreasing);
        }
        Ok(Self {
            frequencies,
            added_mass,
            damping,
        })
    }

    /// The sampled frequencies, rad/s.
    #[must_use]
    pub fn frequencies(&self) -> &[f64] {
        &self.frequencies
    }

    /// The sampled added mass, in the caller's units.
    #[must_use]
    pub fn added_mass(&self) -> &[f64] {
        &self.added_mass
    }

    /// The sampled damping, in the caller's units.
    #[must_use]
    pub fn damping(&self) -> &[f64] {
        &self.damping
    }

    /// The frequency at which damping is greatest, rad/s.
    ///
    /// Used as the natural scale of the problem: the fit below is done in
    /// frequency normalised by this, because a sixth-degree polynomial in a
    /// variable that ranges to 30 has a condition number that ruins a least
    /// squares, and one in a variable of order 1 does not.
    ///
    /// Taken on the *magnitude* of the damping, because a coupling coefficient
    /// is legitimately negative throughout — `B₃₅` for a hull whose added mass
    /// sits aft of the origin is negative at every frequency — and the largest
    /// signed value of such a spectrum is wherever it is closest to zero, which
    /// is the opposite of the scale wanted.
    #[must_use]
    pub fn peak_frequency(&self) -> f64 {
        let mut best = self.frequencies[0];
        let mut largest = f64::NEG_INFINITY;
        for (&frequency, &damping) in self.frequencies.iter().zip(self.damping.iter()) {
            if damping.abs() > largest {
                largest = damping.abs();
                best = frequency;
            }
        }
        best
    }

    /// The retardation function `K(t)`, by cosine transform of the damping.
    ///
    /// Trapezoidal over the sampled frequencies, with `B(0) = 0` supplied: a body
    /// that is not moving radiates nothing, so the transform starts from the
    /// origin whatever the first sample happens to be.
    #[must_use]
    pub fn retardation(&self, time: f64) -> f64 {
        let mut integral = 0.0;
        let mut previous_frequency = 0.0;
        let mut previous_value = 0.0;
        for (&frequency, &damping) in self.frequencies.iter().zip(self.damping.iter()) {
            let value = damping * (frequency * time).cos();
            integral += 0.5 * (previous_value + value) * (frequency - previous_frequency);
            previous_frequency = frequency;
            previous_value = value;
        }
        2.0 / PI * integral
    }

    /// Added mass at infinite frequency, by two independent routes.
    ///
    /// The first is Ogilvie's relation rearranged,
    /// `A_∞ = A(ω) + (1/ω) ∫ K(t) sin(ωt) dt`, evaluated at every sampled
    /// frequency at or above the damping peak. Below the peak the `1/ω` magnifies
    /// whatever error the truncated time grid carries, so those frequencies are
    /// left out — not because their answer is inconvenient but because the factor
    /// in front of the error is large and known.
    ///
    /// The second needs no time grid at all. Integrating that same relation by
    /// parts gives the asymptote
    ///
    /// ```text
    /// A(ω) = A_∞ - K(0)/ω² + O(ω⁻⁴)
    /// ```
    ///
    /// and `K(0) = (2/π)∫B dω` is a single integral over the spectrum. So
    /// `A(ω) + K(0)/ω²` is a second estimate of the same constant, sharing the
    /// spectrum with the first but nothing else: not the transform, not the time
    /// horizon, not the trapezoid over it.
    ///
    /// Both are reduced by the median rather than the mean, because the estimate
    /// at the very top of the grid inherits the truncation of the frequency axis
    /// and one such sample should not be able to move the answer.
    /// [`InfiniteAddedMass::disagreement`] between them is the diagnostic that
    /// matters, and it does not depend on tuning a horizon.
    #[must_use]
    pub fn infinite_added_mass(&self, options: TransformOptions) -> InfiniteAddedMass {
        let step = options.horizon / options.steps as f64;
        let kernel: Vec<f64> = (0..=options.steps)
            .map(|i| self.retardation(i as f64 * step))
            .collect();
        let at_rest = kernel[0];

        let peak = self.peak_frequency();
        let mut ogilvie = Vec::new();
        let mut asymptotic = Vec::new();
        for (&frequency, &added_mass) in self.frequencies.iter().zip(self.added_mass.iter()) {
            if frequency < peak {
                continue;
            }
            let mut integral = 0.0;
            for i in 1..kernel.len() {
                let before = (i - 1) as f64 * step;
                let after = i as f64 * step;
                integral += 0.5
                    * (kernel[i - 1] * (frequency * before).sin()
                        + kernel[i] * (frequency * after).sin())
                    * step;
            }
            ogilvie.push(added_mass + integral / frequency);
            // The asymptote is only an asymptote: it needs to be well past the
            // peak before the neglected fourth-order term stops mattering.
            if frequency >= 3.0 * peak {
                asymptotic.push(added_mass + at_rest / (frequency * frequency));
            }
        }

        let median = |mut values: Vec<f64>| -> Option<f64> {
            if values.is_empty() {
                return None;
            }
            values.sort_by(f64::total_cmp);
            Some(values[values.len() / 2])
        };
        let value =
            median(ogilvie.clone()).unwrap_or_else(|| self.added_mass[self.added_mass.len() - 1]);
        let cross_check = median(asymptotic).unwrap_or(value);
        let disagreement = if value.abs() > 0.0 {
            (value - cross_check).abs() / value.abs()
        } else {
            f64::INFINITY
        };
        InfiniteAddedMass {
            value,
            cross_check,
            disagreement,
            samples: ogilvie.len(),
        }
    }
}

/// How far and how finely the time integral of Ogilvie's relation is taken.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransformOptions {
    /// How long the retardation function is followed, s.
    ///
    /// It has to outlast the memory, and a yacht's memory is long. Truncating
    /// early leaves a residue that the `1/ω` in Ogilvie's relation magnifies, and
    /// the residue shows up directly as [`InfiniteAddedMass::spread`]: on the
    /// YD-41 hull, thirty seconds gives 14 %, sixty gives 6 %, a hundred and
    /// twenty gives 2 % and this default gives half of one.
    ///
    /// The reason is the same thing that makes a yacht radiate over a wide band.
    /// A shallow, wide canoe body reaches a given reduced frequency only at a
    /// high `ω`, so its memory reaches correspondingly far in time — much further
    /// than a ship section's, and much further than the thirty seconds this
    /// default first carried.
    pub horizon: f64,
    /// Number of steps across that horizon.
    ///
    /// Has to resolve the fastest oscillation in the integrand, which is set by
    /// the top of the spectrum: `dt ≲ π / 5ω_max`. For a grid reaching 30 rad/s
    /// that is about a fiftieth of a second.
    pub steps: usize,
}

impl Default for TransformOptions {
    fn default() -> Self {
        Self {
            horizon: 120.0,
            steps: 12_000,
        }
    }
}

/// How a memory model is fitted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryOptions {
    /// Number of states.
    ///
    /// Four is the usual choice and reproduces the memory function of a yacht
    /// hull to a couple of per cent; five reaches a few parts in a thousand.
    /// Going higher buys less than it looks: the extra poles arrive slow, with
    /// time constants of tens of seconds, which is the fit describing the tail of
    /// a transform rather than the physics of a boat.
    pub order: usize,
    /// Highest frequency the fit must match, as a multiple of the damping peak.
    ///
    /// The transform needs the whole spectrum; the fit does not. Beyond ten times
    /// the peak, a yacht's memory function is orders below its largest value, and
    /// a fit told to match it there spends poles on nothing and comes back very
    /// slightly *active* — a negative real part where the true one is almost
    /// zero. Ten keeps the fit honest where the boat actually moves.
    pub fit_ceiling: f64,
}

impl Default for MemoryOptions {
    fn default() -> Self {
        Self {
            order: 4,
            fit_ceiling: 10.0,
        }
    }
}

/// Added mass at infinite frequency, and how well it deserves the name.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InfiniteAddedMass {
    /// The constant, in the units of the spectrum's added mass.
    pub value: f64,
    /// The same constant from the asymptote of `A(ω)`, which shares no machinery
    /// with `value` beyond the spectrum itself.
    pub cross_check: f64,
    /// Relative disagreement between the two, dimensionless.
    ///
    /// Zero in exact arithmetic. This is the diagnostic worth reading: the two
    /// routes have different error sources — one a truncated time grid, the other
    /// a neglected fourth-order term — so agreement between them is evidence
    /// about the pipeline rather than about either estimator's own bias.
    ///
    /// A per-frequency spread was tried here first and made a worse diagnostic:
    /// it is dominated by whichever estimate sits at the edge of the band, so it
    /// moved by a factor of three when the time horizon changed while the value
    /// itself moved in the fifth digit. The median is what makes the value
    /// robust; comparing two medians is what makes the check meaningful.
    pub disagreement: f64,
    /// How many frequencies contributed.
    pub samples: usize,
}

/// Why a memory model could not be fitted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryError {
    /// An order below two cannot represent a memory at all.
    OrderTooLow,
    /// The least squares was singular — the spectrum carries no information.
    Singular,
    /// The fitted denominator has a root in the right half plane, so the model
    /// would grow without bound. Reported rather than silently stabilised:
    /// flipping a pole changes the model into one that was not fitted.
    Unstable {
        /// Real part of the offending pole.
        real_part: f64,
    },
}

/// A low-order linear system whose impulse response is the retardation function.
///
/// Owns its states and its own time stepping. The step is the exact solution of
/// the linear system over the interval, not a Runge-Kutta approximation to it:
/// the states of a memory model span a wide range of time constants — commonly
/// a decade and a half — and an explicit scheme on the fast ones dictates the
/// step size for the whole simulation. Exactness here costs one matrix
/// exponential per distinct `dt`.
#[derive(Debug, Clone)]
pub struct FluidMemory {
    transition: DMatrix<f64>,
    input: DVector<f64>,
    output: DVector<f64>,
    states: DVector<f64>,
    infinite_added_mass: f64,
    worst_error: f64,
    slowest_pole: f64,
    passivity_violation: f64,
    /// Cached exact step for the last `dt` asked for.
    stepper: Option<(f64, DMatrix<f64>, DVector<f64>)>,
}

impl FluidMemory {
    /// Fits a memory model of the given order to a spectrum.
    ///
    /// `order` is the number of states. Four is the usual choice and reproduces
    /// the memory function of a yacht section to a few per cent; five reaches a
    /// few parts in a thousand. Going higher buys less than it looks: the extra
    /// poles come in slow, with time constants of tens of seconds, which is the
    /// fit describing the tail of a transform rather than the physics of a boat.
    /// `infinite` comes from [`Spectrum::infinite_added_mass`] and is a
    /// parameter rather than something computed here on purpose: it carries
    /// [`InfiniteAddedMass::spread`], which is the one number that says whether
    /// the pipeline behind the spectrum resolved anything, and a caller who never
    /// has it in their hand will never look at it. It is also the expensive part,
    /// and refitting at several orders should not pay for it several times.
    pub fn fit(
        spectrum: &Spectrum,
        infinite: InfiniteAddedMass,
        options: MemoryOptions,
    ) -> Result<Self, MemoryError> {
        if options.order < 2 {
            return Err(MemoryError::OrderTooLow);
        }
        let order = options.order;
        let scale = spectrum.peak_frequency();
        let ceiling = options.fit_ceiling * scale;

        // The target: K̂(jω) = B(ω) + jω [A(ω) - A_∞], in normalised frequency.
        //
        // Only up to the ceiling. The transform above needs the whole spectrum —
        // it integrates to infinity — but the fit does not, and asking it to
        // chase a tail that carries no energy spends its few poles badly. On a
        // yacht section the damping at thirty times the peak frequency is four
        // orders below it, and a fit told to match that comes back very slightly
        // *active*: a negative real part where the true one is nearly zero.
        let targets: Vec<(f64, Complex<f64>)> = spectrum
            .frequencies
            .iter()
            .zip(spectrum.added_mass.iter())
            .zip(spectrum.damping.iter())
            .filter(|((&frequency, _), _)| frequency <= ceiling)
            .map(|((&frequency, &added_mass), &damping)| {
                (
                    frequency / scale,
                    Complex::new(damping, frequency * (added_mass - infinite.value)),
                )
            })
            .collect();
        if targets.len() < 2 * order {
            return Err(MemoryError::Singular);
        }

        let unknowns = 2 * order - 1;
        let mut weights = vec![1.0; targets.len()];
        let mut solution = DVector::zeros(unknowns);

        // Sanathanan-Koerner: the linearised least squares below minimises the
        // numerator residual, which over-weights the frequencies where the
        // denominator is large. Dividing by the previous denominator undoes that,
        // and iterating converges on the error that was actually wanted.
        for _ in 0..8 {
            let mut rows = DMatrix::zeros(2 * targets.len(), unknowns);
            let mut right = DVector::zeros(2 * targets.len());
            for (k, &(frequency, target)) in targets.iter().enumerate() {
                let weight = weights[k];
                for power in 1..order {
                    let term = imaginary_power(frequency, power) * weight;
                    rows[(2 * k, power - 1)] = term.re;
                    rows[(2 * k + 1, power - 1)] = term.im;
                }
                for power in 0..order {
                    let term = -target * imaginary_power(frequency, power) * weight;
                    rows[(2 * k, order - 1 + power)] = term.re;
                    rows[(2 * k + 1, order - 1 + power)] = term.im;
                }
                let leading = target * imaginary_power(frequency, order) * weight;
                right[2 * k] = leading.re;
                right[2 * k + 1] = leading.im;
            }

            solution = rows
                .svd(true, true)
                .solve(&right, 1e-12)
                .map_err(|_| MemoryError::Singular)?;

            for (k, &(frequency, _)) in targets.iter().enumerate() {
                let denominator = evaluate_denominator(&solution, order, frequency);
                weights[k] = 1.0 / denominator.norm().max(1e-30);
            }
        }

        // Un-normalise: with ŝ = s/ω₀, p_j = p̂_j ω₀^(n-j) and q_i = q̂_i ω₀^(n-i).
        let mut numerator = vec![0.0; order];
        for power in 1..order {
            numerator[power] = solution[power - 1] * scale.powi((order - power) as i32);
        }
        let mut denominator = vec![0.0; order];
        for power in 0..order {
            denominator[power] = solution[order - 1 + power] * scale.powi((order - power) as i32);
        }

        // Controllable canonical form: the companion matrix of the denominator,
        // driven through its last row, read out through the numerator.
        let mut transition = DMatrix::zeros(order, order);
        for i in 0..order - 1 {
            transition[(i, i + 1)] = 1.0;
        }
        for (i, &coefficient) in denominator.iter().enumerate() {
            transition[(order - 1, i)] = -coefficient;
        }
        let mut input = DVector::zeros(order);
        input[order - 1] = 1.0;
        let output = DVector::from_vec(numerator);

        let poles = transition.clone().complex_eigenvalues();
        let slowest_pole = poles
            .iter()
            .map(|pole| pole.re)
            .fold(f64::NEG_INFINITY, f64::max);
        if slowest_pole >= 0.0 {
            return Err(MemoryError::Unstable {
                real_part: slowest_pole,
            });
        }

        let mut model = Self {
            transition,
            input,
            output,
            states: DVector::zeros(order),
            infinite_added_mass: infinite.value,
            worst_error: 0.0,
            slowest_pole,
            passivity_violation: 0.0,
            stepper: None,
        };

        // How well it did, over the band it was asked to match. Reported against
        // the largest magnitude in that band, so the number means "a per cent of
        // the memory function at its strongest" rather than a per cent of
        // whatever happens to be nearly zero.
        let magnitude = targets
            .iter()
            .map(|&(_, target)| target.norm())
            .fold(0.0_f64, f64::max);
        for &(normalised, wanted) in &targets {
            let got = model.response(normalised * scale);
            model.worst_error = model.worst_error.max((got - wanted).norm() / magnitude);
            // Passivity: the real part is the damping, and damping cannot be
            // negative without the water driving the boat.
            model.passivity_violation = model.passivity_violation.max(-got.re / magnitude);
        }

        Ok(model)
    }

    /// The fitted transfer function at a frequency, `K̂(jω)`.
    #[must_use]
    pub fn response(&self, frequency: f64) -> Complex<f64> {
        let order = self.output.len();
        let mut numerator = Complex::new(0.0, 0.0);
        for (power, &coefficient) in self.output.iter().enumerate() {
            numerator += coefficient * imaginary_power(frequency, power);
        }
        let mut denominator = imaginary_power(frequency, order);
        for (power, &coefficient) in self.transition.row(order - 1).iter().enumerate() {
            denominator += -coefficient * imaginary_power(frequency, power);
        }
        numerator / denominator
    }

    /// Added mass at infinite frequency, which belongs in the mass matrix.
    #[must_use]
    pub fn infinite_added_mass(&self) -> f64 {
        self.infinite_added_mass
    }

    /// Worst relative departure of the fit from the sampled memory function.
    #[must_use]
    pub fn worst_error(&self) -> f64 {
        self.worst_error
    }

    /// Real part of the least-damped pole, 1/s. Negative for a stable model.
    #[must_use]
    pub fn slowest_pole(&self) -> f64 {
        self.slowest_pole
    }

    /// Worst negative real part of the fitted response, relative — zero for a
    /// passive model.
    ///
    /// Meaningful only for a **diagonal** coefficient. Damping cannot be negative
    /// in a mode taken by itself, so a negative real part there means the fitted
    /// water would drive the boat. A *coupling* coefficient carries no such
    /// requirement: `B₃₅` is negative at every frequency for a hull whose added
    /// mass sits aft of the origin, and what has to be positive semi-definite is
    /// the damping matrix, not its entries. Reading this on a coupling fit and
    /// concluding something is wrong would be a mistake.
    #[must_use]
    pub fn passivity_violation(&self) -> f64 {
        self.passivity_violation
    }

    /// The memory force for the current states.
    ///
    /// Opposes the motion, so it is returned with the sign the equation of motion
    /// wants: a force to be *subtracted* from the applied load, as the integral in
    /// Cummins' equation is.
    #[must_use]
    pub fn force(&self) -> f64 {
        self.output.dot(&self.states)
    }

    /// Advances the memory by one step, holding the velocity constant across it.
    ///
    /// Returns the memory force after the step. Exact for constant velocity, and
    /// therefore stable at any step size — which matters because the fitted poles
    /// routinely span more than a decade in time constant, and the fastest of them
    /// would otherwise set the step for the whole simulation.
    pub fn advance(&mut self, velocity: f64, dt: f64) -> f64 {
        if dt <= 0.0 {
            return self.force();
        }
        let reusable = matches!(self.stepper, Some((cached, _, _)) if cached == dt);
        if !reusable {
            let exponential = (&self.transition * dt).exp();
            let order = self.transition.nrows();
            // x(t+dt) = E x + A⁻¹(E - I) b u, the exact response to a held input.
            let shift = &exponential - DMatrix::identity(order, order);
            let forced = self
                .transition
                .clone()
                .lu()
                .solve(&(&shift * &self.input))
                .unwrap_or_else(|| DVector::zeros(order));
            self.stepper = Some((dt, exponential, forced));
        }
        let Some((_, exponential, forced)) = &self.stepper else {
            return self.force();
        };
        self.states = exponential * &self.states + forced * velocity;
        self.force()
    }

    /// Clears the memory, as if the boat had always been still.
    pub fn reset(&mut self) {
        self.states.fill(0.0);
    }
}

/// `(jω)^power`.
fn imaginary_power(frequency: f64, power: usize) -> Complex<f64> {
    let magnitude =
        frequency.powi(i32::try_from(power).expect("polynomial orders here are single digits"));
    match power % 4 {
        0 => Complex::new(magnitude, 0.0),
        1 => Complex::new(0.0, magnitude),
        2 => Complex::new(-magnitude, 0.0),
        _ => Complex::new(0.0, -magnitude),
    }
}

/// `Q(jω)` for the denominator packed in the solution vector.
fn evaluate_denominator(solution: &DVector<f64>, order: usize, frequency: f64) -> Complex<f64> {
    let mut total = imaginary_power(frequency, order);
    for power in 0..order {
        total += solution[order - 1 + power] * imaginary_power(frequency, power);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lewis::{LewisForm, SectionGeometry};
    use crate::tasai::{SectionSolver, TasaiOptions};
    use approx::assert_relative_eq;

    const WATER: f64 = 1025.0;
    const GRAVITY: f64 = 9.81;

    /// A yacht-like section, sampled far enough out that the tail has died.
    ///
    /// Beam 3.2 m on 0.6 m of draft, which is a mid-hull section of a boat this
    /// size. The grid reaches 30 rad/s because this section is still radiating
    /// at 6 — see [`Spectrum`] on why a narrow section needs that.
    fn section_spectrum() -> Spectrum {
        let form = LewisForm::fit(&SectionGeometry {
            beam: 3.2,
            draft: 0.6,
            area: 0.8 * 3.2 * 0.6,
        });
        let solver = SectionSolver::new(TasaiOptions {
            multipoles: 24,
            quadrature: 128,
        });
        let points = 300;
        let mut frequencies = Vec::with_capacity(points);
        let mut added_mass = Vec::with_capacity(points);
        let mut damping = Vec::with_capacity(points);
        for i in 1..=points {
            let omega = 30.0 * i as f64 / points as f64;
            let solved = solver
                .heave(&form, omega, WATER, GRAVITY)
                .expect("a positive frequency has a solution");
            frequencies.push(omega);
            added_mass.push(solved.added_mass);
            damping.push(solved.damping);
        }
        Spectrum::new(frequencies, added_mass, damping).expect("a well-formed spectrum")
    }

    /// Ogilvie's relation says `A_∞` is a constant, so it had better be one.
    ///
    /// This is the test that covers everything upstream at once. `A_∞` is
    /// recovered from `A(ω)` and the cosine transform of `B(ω)` separately at
    /// every frequency above the damping peak, and those are different
    /// computations of the same number: they share the spectrum but not the
    /// frequency, so an error in the sectional solve, the transform, or either
    /// grid's extent shows up as disagreement.
    #[test]
    fn the_infinite_frequency_added_mass_is_the_same_from_every_frequency() {
        let spectrum = section_spectrum();
        let infinite = spectrum.infinite_added_mass(TransformOptions::default());
        assert!(
            infinite.samples > 50,
            "only {} frequencies contributed",
            infinite.samples
        );
        assert!(
            infinite.disagreement < 0.01,
            "the two routes to A_inf differed by {:.3} %: {:?}",
            100.0 * infinite.disagreement,
            infinite
        );
    }

    /// The value does not depend on the time horizon, and that is the point.
    ///
    /// An earlier version of this module reported the spread of the per-frequency
    /// estimates and tuned the horizon to make it small. That was chasing the
    /// wrong number: the spread is dominated by whichever estimate sits at the
    /// edge of the band, so it moved by a factor of three between a
    /// thirty-second and a two-hundred-and-forty-second horizon while the value
    /// moved in the fifth digit. Worse, the two directions disagreed — a longer
    /// horizon improved a hull and degraded a section, because past the point
    /// where the frequency grid resolves the memory there is nothing left to
    /// integrate but ringing.
    ///
    /// The median is what makes the value robust to all of that, and this pins
    /// it: eight times the horizon must not move the answer.
    #[test]
    fn the_value_survives_a_horizon_eight_times_too_short() {
        let spectrum = section_spectrum();
        let generous = spectrum.infinite_added_mass(TransformOptions::default());
        let meagre = spectrum.infinite_added_mass(TransformOptions {
            horizon: 15.0,
            steps: 1500,
        });
        assert_relative_eq!(meagre.value, generous.value, max_relative = 0.005);
    }

    /// And it agrees with a value computed by other means entirely.
    ///
    /// 4164 kg/m came out of an independent prototype of this whole chain,
    /// written in a different language while working the algebra out. Keeping it
    /// as an assertion is worth more than it looks: it is the only number in this
    /// module that did not come from this module.
    #[test]
    fn the_infinite_frequency_added_mass_matches_an_independent_computation() {
        let infinite = section_spectrum().infinite_added_mass(TransformOptions::default());
        assert_relative_eq!(infinite.value, 4164.0, max_relative = 0.01);
    }

    /// The retardation function starts positive and decays.
    ///
    /// `K(0) = (2/π)∫B dω > 0`, and a memory that did not fade would mean a boat
    /// still feeling water it displaced minutes ago. It is not monotone — it goes
    /// negative on the way — so what is pinned is the decay of the envelope.
    #[test]
    fn the_retardation_function_decays() {
        let spectrum = section_spectrum();
        let start = spectrum.retardation(0.0);
        assert!(start > 0.0, "K(0) must be positive, got {start}");
        for &(early, late) in &[(0.5, 5.0), (1.0, 10.0), (2.0, 20.0)] {
            assert!(
                spectrum.retardation(late).abs() < spectrum.retardation(early).abs(),
                "memory at {late} s was not weaker than at {early} s"
            );
        }
    }

    /// Higher order fits better, and every order that fits is stable and passive.
    #[test]
    fn the_fit_improves_with_order_and_never_goes_unstable() {
        let spectrum = section_spectrum();
        let infinite = spectrum.infinite_added_mass(TransformOptions::default());
        let mut previous = f64::INFINITY;
        for order in [3, 4, 5] {
            let model = FluidMemory::fit(
                &spectrum,
                infinite,
                MemoryOptions {
                    order,
                    ..MemoryOptions::default()
                },
            )
            .expect("a yacht section admits a memory model");
            assert!(
                model.worst_error() < previous,
                "order {order} fitted worse ({}) than the order below ({previous})",
                model.worst_error()
            );
            previous = model.worst_error();
            assert!(
                model.slowest_pole() < 0.0,
                "order {order} produced a pole at {}",
                model.slowest_pole()
            );
            assert!(
                model.passivity_violation() < 1e-3,
                "order {order} radiates energy into the boat: {}",
                model.passivity_violation()
            );
        }
        assert!(
            previous < 0.02,
            "the order-5 fit should be within a couple of per cent, got {previous}"
        );
    }

    /// An order of one is not a memory.
    #[test]
    fn an_order_below_two_is_refused() {
        let spectrum = section_spectrum();
        assert_eq!(
            FluidMemory::fit(
                &spectrum,
                spectrum.infinite_added_mass(TransformOptions::default()),
                MemoryOptions {
                    order: 1,
                    ..MemoryOptions::default()
                }
            )
            .err(),
            Some(MemoryError::OrderTooLow)
        );
    }

    /// A spectrum has to be a spectrum.
    #[test]
    fn a_malformed_spectrum_is_refused() {
        assert_eq!(
            Spectrum::new(vec![1.0, 2.0], vec![1.0, 2.0], vec![1.0, 2.0]),
            Err(SpectrumError::TooFewSamples)
        );
        assert_eq!(
            Spectrum::new(vec![1.0, 2.0, 3.0], vec![1.0, 2.0], vec![1.0, 2.0, 3.0]),
            Err(SpectrumError::Ragged)
        );
        assert_eq!(
            Spectrum::new(vec![1.0, 3.0, 2.0], vec![1.0; 3], vec![1.0; 3]),
            Err(SpectrumError::NotIncreasing)
        );
        assert_eq!(
            Spectrum::new(vec![0.0, 1.0, 2.0], vec![1.0; 3], vec![1.0; 3]),
            Err(SpectrumError::NotIncreasing)
        );
    }

    /// A held velocity produces no lasting memory force.
    ///
    /// The transfer function is zero at zero frequency because `B(0) = 0`: a body
    /// moving steadily makes no waves and so radiates nothing. Worth pinning
    /// because it is counter-intuitive — the memory term looks like damping and is
    /// not, and a model with a spurious constant term would drag a boat moving at
    /// a steady speed.
    #[test]
    fn a_steady_velocity_leaves_no_memory_force() {
        let spectrum = section_spectrum();
        let mut model = FluidMemory::fit(
            &spectrum,
            spectrum.infinite_added_mass(TransformOptions::default()),
            MemoryOptions::default(),
        )
        .expect("a yacht section admits a memory model");
        let mut force = 0.0;
        for _ in 0..20_000 {
            force = model.advance(1.0, 0.01);
        }
        let reference = spectrum.retardation(0.0);
        assert!(
            force.abs() < 1e-6 * reference,
            "a steady velocity settled to a memory force of {force}"
        );
    }

    /// The exact step is exact: halving it changes nothing.
    ///
    /// The states are advanced by the matrix exponential rather than by a
    /// Runge-Kutta scheme, so a held velocity over a given interval must give the
    /// same answer however that interval is subdivided. This is what buys freedom
    /// from the fastest pole dictating the simulation's step size.
    #[test]
    fn the_step_is_exact_under_refinement() {
        let spectrum = section_spectrum();
        let build = || {
            FluidMemory::fit(
                &spectrum,
                spectrum.infinite_added_mass(TransformOptions::default()),
                MemoryOptions::default(),
            )
            .expect("a yacht section admits a memory model")
        };
        let mut coarse = build();
        let mut fine = build();
        for _ in 0..40 {
            coarse.advance(1.0, 0.05);
        }
        for _ in 0..400 {
            fine.advance(1.0, 0.005);
        }
        assert_relative_eq!(coarse.force(), fine.force(), max_relative = 1e-9);
    }

    /// The payoff: the time domain gives back the frequency domain it came from.
    ///
    /// This closes the loop the whole module exists for. The memory model is
    /// driven at a single frequency in the *time* domain, its force is correlated
    /// against the drive to recover the real and imaginary parts of `K̂(jω)`, and
    /// those are compared against the damping and added mass that the sectional
    /// solve produced in the *frequency* domain.
    ///
    /// Everything is in the path: the cosine transform, `A_∞`, the rational fit,
    /// the companion realisation and the exponential stepper. A sign error, a
    /// scale error, or a states-driven-by-the-wrong-quantity error all show up
    /// here, and none of them can hide, because the comparison is against
    /// numbers in physical units rather than against a stored answer.
    ///
    /// A free decay was tried first and makes a worse test: with a damping ratio
    /// near a third there are only two clean peaks before the amplitude drops
    /// into the slow residual mode of the fit, and recovering a damping ratio
    /// from two peaks needs the damped period, the added mass at the frequency
    /// that period implies, and a linear-damping formula that is itself only
    /// approximate when added mass varies this fast with frequency. Three
    /// approximations to test one thing. `a_free_oscillation_decays` keeps what
    /// that test was actually good for.
    #[test]
    fn the_time_domain_reproduces_the_frequency_domain() {
        let spectrum = section_spectrum();
        let mut model = FluidMemory::fit(
            &spectrum,
            spectrum.infinite_added_mass(TransformOptions::default()),
            MemoryOptions {
                order: 5,
                ..MemoryOptions::default()
            },
        )
        .expect("a yacht section admits a memory model");
        let infinite = model.infinite_added_mass();

        for &target in &[0.6_f64, 1.2, 2.0, 3.0, 5.0] {
            let index = spectrum
                .frequencies()
                .iter()
                .enumerate()
                .min_by(|a, b| (a.1 - target).abs().total_cmp(&(b.1 - target).abs()))
                .map_or(0, |(i, _)| i);
            let omega = spectrum.frequencies()[index];
            let wanted = Complex::new(
                spectrum.damping[index],
                omega * (spectrum.added_mass[index] - infinite),
            );

            // Drive at `omega`, discard the transient, then correlate.
            model.reset();
            let dt = 0.001;
            let period = std::f64::consts::TAU / omega;
            let settle = (12.0 * period / dt) as usize;
            let cycles = (8.0 * period / dt) as usize;
            let velocity_at = |step: usize| omega * (omega * step as f64 * dt).cos();
            for step in 0..settle {
                model.advance(velocity_at(step), dt);
            }
            let mut in_phase = 0.0;
            let mut quadrature = 0.0;
            for step in 0..cycles {
                let time = (settle + step) as f64 * dt;
                let force = model.advance(velocity_at(settle + step), dt);
                in_phase += force * (omega * time).cos() * dt;
                quadrature += force * (omega * time).sin() * dt;
            }
            let span = cycles as f64 * dt;
            let got = Complex::new(
                2.0 * in_phase / span / omega,
                -2.0 * quadrature / span / omega,
            );

            assert_relative_eq!(got.re, wanted.re, max_relative = 0.02);
            assert_relative_eq!(got.im, wanted.im, max_relative = 0.02);
        }
    }

    /// A released section settles instead of running away.
    ///
    /// What the free decay is genuinely good for: the sign of the memory force.
    /// Get it backwards and this grows, which no amount of agreement in the
    /// frequency domain would catch if the force were handed to the integrator
    /// with the wrong sign.
    #[test]
    fn a_free_oscillation_decays() {
        let spectrum = section_spectrum();
        let mut model = FluidMemory::fit(
            &spectrum,
            spectrum.infinite_added_mass(TransformOptions::default()),
            MemoryOptions::default(),
        )
        .expect("a yacht section admits a memory model");

        let structural = 500.0;
        let stiffness = WATER * GRAVITY * 3.2;
        let virtual_mass = structural + model.infinite_added_mass();

        let dt = 0.0005;
        let mut position = 0.05;
        let mut velocity = 0.0;
        let mut largest_late = 0.0_f64;
        for step in 1..20_000 {
            let memory = model.advance(velocity, dt);
            let acceleration = -(stiffness * position + memory) / virtual_mass;
            velocity += acceleration * dt;
            position += velocity * dt;
            // After four seconds — about two periods — nothing should be left.
            if f64::from(step) * dt > 4.0 {
                largest_late = largest_late.max(position.abs());
            }
        }
        assert!(
            largest_late < 0.05 * 0.05,
            "after two periods the motion still reached {largest_late} m of an initial 0.05"
        );
    }
}
