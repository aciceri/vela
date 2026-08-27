//! An irregular sea, as a sum of linear waves.
//!
//! # What this is
//!
//! A *realisation*: a spectrum, a seed, and a stated synthesis convention. Not a
//! height field and not a texture. The distinction is the one thing in this
//! module worth getting right, because the same sea has to be evaluated twice —
//! by the physics on the CPU, wherever a hull triangle happens to be, and by the
//! renderer on the GPU, on whatever grid it likes. If the two synthesised
//! independently from the same spectrum they would produce different water, and
//! a boat would float on a surface nobody could see.
//!
//! So the convention below is part of the public contract, not an implementation
//! detail: given the same [`Seaway::seed`] and the same spectrum, the elevation
//! at a point and time is a stated function, and anything that evaluates it the
//! same way gets the same water.
//!
//! # The spectrum
//!
//! The ITTC/ISSC form of Pierson-Moskowitz, parameterised by significant wave
//! height and peak period rather than by wind speed, because those are what a
//! sailor and a test both want to set:
//!
//! ```text
//! S(ω) = (5/16) H_s² ω_p⁴ ω⁻⁵ exp(-(5/4)(ω_p/ω)⁴)
//! ```
//!
//! Its zeroth moment is `m₀ = (H_s/4)²` by construction, which is the check the
//! synthesis is verified against: the discrete components must carry the variance
//! the continuous spectrum says they should.
//!
//! # The two things a hull needs
//!
//! [`Seaway::elevation`] says where the surface is, which decides what is wet.
//! [`Seaway::pressure_head`] says what the pressure is, which decides the force,
//! and it is *not* the same question: the dynamic part of a wave's pressure dies
//! away with depth as `e^(-k z)`, so a deeply immersed keel feels far less of a
//! passing wave than its depth below the instantaneous surface would suggest.
//!
//! Treating the wavy surface as a datum and integrating `ρ g d` below it — the
//! usual shortcut in game physics — gets the surface exactly right and the depths
//! progressively wrong, over-predicting the excitation of anything deep in short
//! waves. Both are here, and the decay is not optional.

use std::f64::consts::PI;

/// One linear wave component of a realisation.
///
/// Public because a renderer needs exactly these numbers to draw the water the
/// physics is using. Named `Wave` rather than `Component` deliberately: the
/// frontend this is shared with calls something else a component, and a type
/// that means two things across a boundary is a bug waiting for a busy day.
///
/// A realisation is long-crested, so every wave shares one direction; it is
/// carried per wave anyway rather than alongside, because that is the form the
/// synthesis below reads and a directional spectrum would need it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wave {
    /// Amplitude, m.
    pub amplitude: f64,
    /// Wavenumber, rad/m.
    pub wavenumber: f64,
    /// Radian frequency, rad/s.
    pub frequency: f64,
    /// Phase at the origin at `t = 0`, rad.
    pub phase: f64,
    /// Unit direction of travel in the world plane, as `(north, east)`.
    pub direction: (f64, f64),
}

/// How a sea is specified.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeaState {
    /// Significant wave height, m — the traditional `H_s`, four times the
    /// standard deviation of the surface.
    pub significant_height: f64,
    /// Peak period, s.
    pub peak_period: f64,
    /// Direction the waves travel *towards*, radians, as a compass bearing.
    ///
    /// Towards, not from: the opposite of the wind convention in
    /// [`crate::env::UniformWind`], and stated here because the two sit next to
    /// each other in every call site.
    pub heading: f64,
    /// Number of components the spectrum is discretised into.
    ///
    /// More is smoother in time but no more accurate in any particular instant.
    /// Sixty is enough that the surface does not visibly repeat over a few
    /// minutes, which is the only thing a low count actually costs.
    pub components: usize,
    /// Seed for the random phases.
    ///
    /// Part of the realisation, not a nuisance parameter: two runs with the same
    /// seed are the same sea, which is what makes a wave test reproducible and
    /// what lets a renderer draw the water the physics is using.
    pub seed: u64,
    /// Directional spreading exponent at the spectral peak, `s_p`.
    ///
    /// A real sea is short-crested: its components do not all travel the same
    /// way, and the ones that carry the least energy are the least organised.
    /// This is the `s` of the Longuet-Higgins spreading function
    /// `D(θ) ∝ cos^(2s)((θ - θ₀)/2)`, quoted at the peak frequency, from which
    /// [`Seaway`] derives the exponent at every other frequency by Mitsuyasu's
    /// law. Ten is the usual wind-sea value; a long swell is nearer twenty-five.
    ///
    /// **Zero means unidirectional** — every component on `heading`, a sea whose
    /// crests are parallel and infinitely long. That is a real sea state only for
    /// a pure distant swell, and it is worth knowing that it is the degenerate
    /// case rather than the simple one: a long-crested sea excites no roll from
    /// a head sea and looks, from a boat, like corrugated iron.
    pub spreading: f64,
}

impl Default for SeaState {
    fn default() -> Self {
        Self {
            significant_height: 1.0,
            peak_period: 5.0,
            heading: 0.0,
            components: 60,
            seed: 1,
            // A wind sea, not a swell. The old default was implicitly zero and
            // that was the wrong way round: unidirectional is the special case.
            spreading: 10.0,
        }
    }
}

impl SeaState {
    /// The ITTC/ISSC spectral density at a frequency, m²·s/rad.
    ///
    /// Zero at and below zero frequency, where the expression has an essential
    /// singularity and no energy.
    #[must_use]
    pub fn density(&self, frequency: f64) -> f64 {
        if frequency <= 0.0 || self.peak_period <= 0.0 {
            return 0.0;
        }
        let peak = 2.0 * PI / self.peak_period;
        let ratio = peak / frequency;
        5.0 / 16.0 * self.significant_height * self.significant_height * peak.powi(4)
            / frequency.powi(5)
            * (-1.25 * ratio.powi(4)).exp()
    }

    /// The variance the spectrum carries, m².
    ///
    /// `m₀ = (H_s/4)²` exactly, by the definition of significant height. Given
    /// rather than integrated because it is the *target* the discretisation is
    /// checked against.
    #[must_use]
    pub fn variance(&self) -> f64 {
        let quarter = self.significant_height / 4.0;
        quarter * quarter
    }
}

/// Mitsuyasu's directional spreading exponent at a frequency.
///
/// ```text
/// s(ω) = s_p (ω/ω_p)^5      below the peak
/// s(ω) = s_p (ω/ω_p)^-2.5   above it
/// ```
///
/// The asymmetry is the observation the law exists for, and it is what makes a
/// drawn sea look like water: energy at the peak arrives nearly all from one
/// bearing, while the short waves are scattered across a wide fan. A single
/// exponent for the whole spectrum gets both ends wrong at once — either the
/// swell wanders or the ripples line up.
///
/// Clamped below at a value near unity because `cos²ˢ` with `s` under one is
/// nearly uniform and the exponent is then doing no work, and above at a value
/// where the fan is already narrower than the direction resolution sixty
/// components can carry.
fn mitsuyasu_exponent(peak_exponent: f64, frequency: f64, peak: f64) -> f64 {
    if peak_exponent <= 0.0 || peak <= 0.0 || frequency <= 0.0 {
        return 0.0;
    }
    let ratio = frequency / peak;
    let exponent = if ratio < 1.0 {
        peak_exponent * ratio.powi(5)
    } else {
        peak_exponent * ratio.powf(-2.5)
    };
    exponent.clamp(0.6, 80.0)
}

/// Angle off the mean direction at a given cumulative probability, radians.
///
/// Inverts the Longuet-Higgins spreading `D(θ) ∝ cos^(2s)(θ/2)` on `[-π, π]`.
/// There is no closed form for the inverse, so the CDF is integrated on a fixed
/// grid and inverted by linear interpolation between the two straddling nodes.
///
/// A table rather than a Newton solve because this runs at most a few hundred
/// times per boat load and the grid makes the result a deterministic function of
/// `(quantile, spread)` on every platform — which the shared-realisation
/// contract needs, since a renderer and the physics must agree on the sea to the
/// last bit. Sixty-four intervals of Simpson's rule resolve even the narrowest
/// admitted fan to well under a degree.
///
/// `spread` of zero is the unidirectional case and returns zero: every component
/// on the mean heading.
fn spreading_quantile(quantile: f64, spread: f64) -> f64 {
    if spread <= 0.0 {
        return 0.0;
    }
    const NODES: usize = 64;

    // Unnormalised density, and its running integral from -π.
    let density = |angle: f64| (0.5 * angle).cos().abs().powf(2.0 * spread);
    let step = 2.0 * PI / NODES as f64;

    let mut cumulative = [0.0; NODES + 1];
    for node in 0..NODES {
        let left = -PI + node as f64 * step;
        // Simpson on the interval: the density is smooth and this keeps the
        // narrow-fan case from being under-integrated by the trapezoid rule.
        let middle = left + 0.5 * step;
        let right = left + step;
        let slab = step / 6.0 * (density(left) + 4.0 * density(middle) + density(right));
        cumulative[node + 1] = cumulative[node] + slab;
    }

    let total = cumulative[NODES];
    if total <= 0.0 {
        return 0.0;
    }
    let target = quantile * total;

    // The CDF is monotone, so a linear scan is both correct and, at sixty-four
    // nodes, faster than the bisection that would replace it.
    for node in 0..NODES {
        if cumulative[node + 1] >= target {
            let slab = cumulative[node + 1] - cumulative[node];
            let within = if slab > 0.0 {
                (target - cumulative[node]) / slab
            } else {
                0.0
            };
            return -PI + (node as f64 + within) * step;
        }
    }
    PI
}

/// A realised sea: a fixed set of components with fixed phases.
#[derive(Debug, Clone, PartialEq)]
pub struct Seaway {
    state: SeaState,
    components: Vec<Wave>,
    gravity: f64,
}

impl Seaway {
    /// Realises a sea state into components.
    ///
    /// The frequency band runs from a quarter of the peak to four times it, split
    /// into equal intervals, with each component taking the variance of its own
    /// interval: `a = √(2 S(ω) Δω)`. That is the standard synthesis, and the
    /// band is wide enough to carry essentially all of the variance — which
    /// `the_components_carry_the_spectrum's_variance` measures rather than
    /// assumes.
    ///
    /// Phases come from a small deterministic generator seeded by
    /// [`SeaState::seed`], so a realisation is reproducible across runs,
    /// machines and — the point of it — between the physics and a renderer.
    ///
    /// # Why each component gets its own direction
    ///
    /// A sea is a function of two horizontal coordinates. Give every component
    /// the same heading and it collapses to a function of one: crests become
    /// parallel ridges, identical to the horizon, and the surface has no
    /// short-crestedness for a hull to feel. That is not a discretisation
    /// artefact to be lived with, it is the wrong sea.
    ///
    /// The textbook fix is a double summation over frequency *and* direction,
    /// which costs `N × M` components for `N` frequencies. This uses the
    /// **single-summation** method instead: each frequency band keeps its whole
    /// variance and is assigned one direction drawn from the spreading function.
    /// The component count, and so the per-step cost, is unchanged — see
    /// `examples/frame_cost`, where the component count is nearly the entire
    /// step — while the surface becomes genuinely two-dimensional.
    ///
    /// What single summation gives up is stated plainly in the literature: the
    /// realisation is not spatially homogeneous, because a single direction per
    /// frequency cannot represent the directional variance *within* a band. Over
    /// a hull twelve metres long against wavelengths of tens of metres that is
    /// far below the errors already accepted in §5, and it buys the difference
    /// between a sea and a corrugated roof.
    #[must_use]
    pub fn new(state: SeaState, gravity: f64) -> Self {
        let mut components = Vec::with_capacity(state.components);
        if state.components > 0 && state.peak_period > 0.0 && state.significant_height > 0.0 {
            let peak = 2.0 * PI / state.peak_period;
            let lowest = 0.25 * peak;
            let highest = 4.0 * peak;
            let step = (highest - lowest) / state.components as f64;
            let mut phases = Phases::new(state.seed);

            // Directions come from their own generator, not from the phase
            // stream. Sharing one would mean that turning spreading on moved
            // every crest in the sea as well as fanning it, and that a swell
            // written with `spreading: 0.0` was no longer the sea it used to be.
            // One stream per question keeps `spreading: 0.0` bit-identical to a
            // unidirectional realisation.
            //
            // Drawn stratified rather than independently: stratum `i` of `N`
            // covers cumulative probability `[i/N, (i+1)/N)` and the draw lands
            // inside it. That guarantees the realised directions cover the
            // spreading function instead of clumping by luck, which for sixty
            // components matters — an unlucky independent draw can leave a whole
            // flank of the distribution empty and put a false ridge in the sea.
            //
            // The strata are then shuffled, because assigning stratum `i` to
            // frequency band `i` would tie direction to frequency: the sea would
            // fan out monotonically from the longest wave to the shortest, which
            // is a pattern no wind ever made.
            let mut bearings = Phases::new(state.seed ^ 0x9E37_79B9_7F4A_7C15);
            let mut strata: Vec<usize> = (0..state.components).collect();
            for index in (1..strata.len()).rev() {
                // `Phases::next` returns a phase in `[0, 2π)`; scaled to a unit
                // fraction it is a perfectly good uniform for a Fisher-Yates.
                let unit = bearings.next() / (2.0 * PI);
                let pick = ((unit * (index + 1) as f64) as usize).min(index);
                strata.swap(index, pick);
            }

            for index in 0..state.components {
                // Mid-interval, so that no component sits on the band edges.
                let frequency = lowest + (index as f64 + 0.5) * step;
                let amplitude = (2.0 * state.density(frequency) * step).sqrt();
                let spread = mitsuyasu_exponent(state.spreading, frequency, peak);
                // Jittered inside the stratum, so the fan is not a fixed comb of
                // sixty bearings that would repeat between realisations.
                let jitter = bearings.next() / (2.0 * PI);
                let quantile = (strata[index] as f64 + jitter) / strata.len() as f64;
                let offset = spreading_quantile(quantile, spread);
                let (sine, cosine) = (state.heading + offset).sin_cos();
                components.push(Wave {
                    amplitude,
                    wavenumber: frequency * frequency / gravity,
                    frequency,
                    phase: phases.next(),
                    // A compass bearing: x is north, y is east.
                    direction: (cosine, sine),
                });
            }
        }
        Self {
            state,
            components,
            gravity,
        }
    }

    /// The sea state this realises.
    #[must_use]
    pub fn state(&self) -> SeaState {
        self.state
    }

    /// The waves this realisation is made of.
    ///
    /// This is the shared-realisation contract in its concrete form: a renderer
    /// that evaluates the closed form of [`Seaway::elevation`] over these waves
    /// draws exactly the water the physics is standing on, on whatever grid it
    /// likes, with no per-frame traffic between them. Changing the synthesis
    /// convention is therefore a breaking change to this crate's public API and
    /// not an implementation detail, which is what
    /// `the_synthesis_convention_is_pinned` exists to enforce.
    #[must_use]
    pub fn waves(&self) -> &[Wave] {
        &self.components
    }

    /// Surface elevation above the mean level at a world point and time, m,
    /// positive **up**.
    ///
    /// The synthesis convention, which is public contract:
    ///
    /// ```text
    /// ζ(x, y, t) = Σ aᵢ cos(kᵢ (x dᵢx + y dᵢy) - ωᵢ t + φᵢ)
    /// ```
    ///
    /// Positive up while the world frame's `z` is down, because an elevation that
    /// went negative when the water rose would be read wrong by every caller
    /// exactly once.
    #[must_use]
    pub fn elevation(&self, north: f64, east: f64, time: f64) -> f64 {
        self.components
            .iter()
            .map(|it| {
                let along = north * it.direction.0 + east * it.direction.1;
                it.amplitude * (it.wavenumber * along - it.frequency * time + it.phase).cos()
            })
            .sum()
    }

    /// Depth of a world point below the instantaneous surface, m, positive below.
    ///
    /// `z` is measured down from the mean level, so a point at `z` sits
    /// `z + ζ` below a surface that has risen by `ζ`.
    #[must_use]
    pub fn depth(&self, north: f64, east: f64, down: f64, time: f64) -> f64 {
        down + self.elevation(north, east, time)
    }

    /// Pressure at a world point divided by `ρ g`, m.
    ///
    /// The hydrostatic head plus the wave's dynamic head, each component decaying
    /// with its own wavenumber:
    ///
    /// ```text
    /// p / ρg = z + Σ aᵢ e^(-kᵢ max(z + ζ, 0)) cos(...)
    /// ```
    ///
    /// The exponential is evaluated at the depth below the **instantaneous**
    /// surface rather than below the mean level. That is Wheeler stretching, and
    /// it is here for a specific reason rather than for tidiness: with the decay
    /// taken from the mean level, the head does not vanish on the surface. It
    /// misses by a second-order amount — order `k ζ a`, which for a 1.5 m sea
    /// came to two centimetres of head — because linear theory imposes its
    /// boundary condition at `z = 0` and the surface is not there.
    ///
    /// Two centimetres sounds ignorable and is not, because it breaks the one
    /// invariant these two functions have to share: [`Seaway::depth`] decides
    /// which triangles of a hull are wet, and this decides what they carry, and a
    /// waterline where one says "just submerged" and the other says "already
    /// loaded" leaks force in proportion to wave height. Stretching makes the
    /// agreement exact by construction — a point at `z = -ζ` gets `-ζ + ζ = 0`
    /// however steep the wave — and reduces to the unstretched form at depth,
    /// where `ζ` is negligible against `z`.
    ///
    /// The `max(·, 0)` is the remaining edge of linear theory: inside a crest,
    /// above the mean level, the exponential would grow, and holding it at its
    /// surface value is the standard treatment.
    #[must_use]
    pub fn pressure_head(&self, north: f64, east: f64, down: f64, time: f64) -> f64 {
        self.depth_and_pressure_head(north, east, down, time).1
    }

    /// Both of the surface's questions at one point, sharing their expensive half.
    ///
    /// Callers that ask "is this wet?" almost always go on to ask "what does it
    /// carry?", and both answers are built from the same sum over components. A
    /// hull clip asks the pair at every vertex of a thousand-triangle mesh two
    /// hundred times a second, so evaluating the elevation once instead of twice
    /// is a third of the step rather than a micro-optimisation.
    ///
    /// Above the surface the second sum is skipped rather than approximated. It
    /// is not an optimisation with an error: with the decay clamped at the surface
    /// the dynamic part there is exactly the elevation, so the head is exactly the
    /// depth, which the first sum already gave. Dry vertices are most of a hull
    /// above the waterline and the clip discards them, so this is the common case.
    #[must_use]
    pub fn depth_and_pressure_head(
        &self,
        north: f64,
        east: f64,
        down: f64,
        time: f64,
    ) -> (f64, f64) {
        let elevation = self.elevation(north, east, time);
        let depth = down + elevation;
        if depth <= 0.0 {
            return (depth, depth);
        }

        // Wheeler stretching: the decay is taken from the instantaneous surface,
        // which is what makes the head vanish there exactly. See the note above.
        let attenuation_depth = depth;
        let dynamic: f64 = self
            .components
            .iter()
            .map(|it| {
                let along = north * it.direction.0 + east * it.direction.1;
                it.amplitude
                    * (-it.wavenumber * attenuation_depth).exp()
                    * (it.wavenumber * along - it.frequency * time + it.phase).cos()
            })
            .sum();
        (depth, down + dynamic)
    }

    /// The variance the realised components actually carry, m².
    ///
    /// `Σ aᵢ²/2`. Compared against [`SeaState::variance`] it says whether the
    /// discretisation captured the spectrum, and it is public so a caller can ask
    /// before trusting a sea rather than after.
    #[must_use]
    pub fn realised_variance(&self) -> f64 {
        self.components
            .iter()
            .map(|it| 0.5 * it.amplitude * it.amplitude)
            .sum()
    }

    /// Significant height of the realised sea, m: four standard deviations.
    #[must_use]
    pub fn realised_height(&self) -> f64 {
        4.0 * self.realised_variance().sqrt()
    }

    /// Gravity this realisation was built with, m/s².
    ///
    /// Held because the dispersion relation `k = ω²/g` is baked into the
    /// components: a realisation is only valid for the gravity it was made with.
    #[must_use]
    pub fn gravity(&self) -> f64 {
        self.gravity
    }
}

/// A small deterministic phase generator.
///
/// SplitMix64. Chosen because it is four lines, has no state to get wrong, and
/// is stable across platforms and compilers — which a realisation shared with a
/// renderer needs and a library random number generator does not promise.
struct Phases(u64);

impl Phases {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }

    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // The top 53 bits give a uniform in [0, 1) exactly representable.
        (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 * PI
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const GRAVITY: f64 = 9.81;

    fn moderate() -> SeaState {
        SeaState {
            significant_height: 1.5,
            peak_period: 6.0,
            ..SeaState::default()
        }
    }

    /// The discrete components carry the variance the spectrum says they should.
    ///
    /// The check that the synthesis is the spectrum and not merely inspired by
    /// it. `m₀ = (H_s/4)²` is exact by definition, and the sum of `aᵢ²/2` over a
    /// band from a quarter to four times the peak has to come close to it. It
    /// cannot be exact — the band is finite and the tails are real — so what is
    /// asserted is that the shortfall is a couple of per cent, which is the
    /// spectrum's own tail rather than an error.
    #[test]
    fn the_components_carry_the_spectrums_variance() {
        for height in [0.5_f64, 1.5, 4.0] {
            let state = SeaState {
                significant_height: height,
                ..moderate()
            };
            let sea = Seaway::new(state, GRAVITY);
            let captured = sea.realised_variance() / state.variance();
            assert!(
                (0.97..=1.0).contains(&captured),
                "H_s {height}: components carried {:.4} of the variance",
                captured
            );
            assert_relative_eq!(sea.realised_height(), height, max_relative = 0.02);
        }
    }

    /// More components do not change the variance, only the smoothness.
    ///
    /// The amplitude of each is `√(2 S Δω)`, so halving `Δω` halves each
    /// component's variance and doubles their number. If refining the
    /// discretisation changed the total, the synthesis would be scaling wrongly
    /// and every sea would have the wrong height at some resolution.
    #[test]
    fn refining_the_discretisation_preserves_the_height() {
        let coarse = Seaway::new(
            SeaState {
                components: 20,
                ..moderate()
            },
            GRAVITY,
        );
        let fine = Seaway::new(
            SeaState {
                components: 400,
                ..moderate()
            },
            GRAVITY,
        );
        assert_relative_eq!(
            coarse.realised_height(),
            fine.realised_height(),
            max_relative = 0.02
        );
    }

    /// A one-component sea is a regular wave, with the dispersion it should have.
    ///
    /// Reduced to a single component the synthesis has to be a plain travelling
    /// cosine: the elevation at a fixed point must repeat with the component's
    /// period, and the pattern must repeat in space with the wavelength that
    /// `k = ω²/g` implies. This is the deep-water dispersion relation, and
    /// getting it wrong would make every wave the wrong length for its period —
    /// which is invisible in a still picture and obvious in motion.
    #[test]
    fn a_single_component_travels_with_deep_water_dispersion() {
        let state = SeaState {
            components: 1,
            peak_period: 6.0,
            heading: 0.0,
            // Unidirectional, so that "a wavelength north" is a wavelength along
            // the wave and not a slanted section through it. Dispersion is what
            // this measures; the fan has its own test.
            spreading: 0.0,
            ..SeaState::default()
        };
        let sea = Seaway::new(state, GRAVITY);
        let component = sea.components[0];
        let period = 2.0 * PI / component.frequency;
        let wavelength = 2.0 * PI / component.wavenumber;

        // The dispersion relation itself.
        assert_relative_eq!(
            component.wavenumber,
            component.frequency * component.frequency / GRAVITY,
            max_relative = 1e-15
        );
        // And so the wavelength for this period, which for deep water is
        // g T² / 2π — about 56 m for six seconds.
        assert_relative_eq!(
            wavelength,
            GRAVITY * period * period / (2.0 * PI),
            max_relative = 1e-12
        );

        // Periodic in time at a point, and in space at an instant.
        let here = sea.elevation(0.0, 0.0, 0.0);
        assert_relative_eq!(sea.elevation(0.0, 0.0, period), here, epsilon = 1e-9);
        assert_relative_eq!(sea.elevation(wavelength, 0.0, 0.0), here, epsilon = 1e-9);
        // Half a wavelength away it is the opposite.
        assert_relative_eq!(
            sea.elevation(0.5 * wavelength, 0.0, 0.0),
            -here,
            epsilon = 1e-9
        );
    }

    /// The dynamic pressure dies away with depth, at the rate theory says.
    ///
    /// `e^(-k z)` per component. For a single wave the ratio of dynamic head at
    /// two depths is a pure exponential, and that is what is checked — not a
    /// stored number but the law. A model that skipped the decay would over-drive
    /// a deep keel in short waves, which is the failure the module documentation
    /// warns about.
    #[test]
    fn the_dynamic_pressure_decays_with_depth() {
        let sea = Seaway::new(
            SeaState {
                components: 1,
                ..moderate()
            },
            GRAVITY,
        );
        let wavenumber = sea.components[0].wavenumber;

        // The decay is over the depth below the INSTANTANEOUS surface, so the
        // elevation enters the law and the test states it that way rather than
        // evaluating at a convenient instant.
        let elevation = sea.elevation(0.0, 0.0, 0.0);
        for &depth in &[0.5_f64, 2.0, 5.0] {
            let dynamic = sea.pressure_head(0.0, 0.0, depth, 0.0) - depth;
            assert_relative_eq!(
                dynamic,
                elevation * (-wavenumber * (depth + elevation).max(0.0)).exp(),
                max_relative = 1e-12
            );
        }
        // And it really does decay: five metres down is a fraction of the surface.
        let deep = (sea.pressure_head(0.0, 0.0, 5.0, 0.0) - 5.0).abs();
        assert!(
            deep < 0.7 * elevation.abs(),
            "no measurable decay: {deep} against {elevation}"
        );
    }

    /// The pressure is exactly zero on the surface, wherever the surface is.
    ///
    /// The consistency condition between the two accessors: a point the depth
    /// function calls "on the surface" must be one the pressure function calls
    /// "at zero gauge pressure". If they disagreed, a hull would be clipped at
    /// one waterline and loaded as though it were at another, and the error would
    /// grow with wave height rather than announcing itself.
    #[test]
    fn the_two_accessors_agree_on_where_the_surface_is() {
        let sea = Seaway::new(moderate(), GRAVITY);
        for &(north, east, time) in &[(0.0, 0.0, 0.0), (37.0, -12.0, 4.5), (-100.0, 250.0, 61.25)] {
            let elevation = sea.elevation(north, east, time);
            // The surface sits at z = -ζ in a z-down world.
            let on_surface = -elevation;
            assert_relative_eq!(
                sea.depth(north, east, on_surface, time),
                0.0,
                epsilon = 1e-12
            );
            assert_relative_eq!(
                sea.pressure_head(north, east, on_surface, time),
                0.0,
                epsilon = 1e-12
            );
        }
    }

    /// A calm is still water, exactly.
    ///
    /// Zero height must not merely be small: every wave path has to collapse to
    /// the flat-water one, or a test written in a calm would be testing the wave
    /// code's rounding.
    #[test]
    fn a_calm_is_exactly_still_water() {
        let sea = Seaway::new(
            SeaState {
                significant_height: 0.0,
                ..SeaState::default()
            },
            GRAVITY,
        );
        for &down in &[-1.0_f64, 0.0, 3.5] {
            assert_relative_eq!(sea.elevation(10.0, 20.0, 3.0), 0.0, epsilon = 0.0);
            assert_relative_eq!(sea.depth(10.0, 20.0, down, 3.0), down, epsilon = 0.0);
            assert_relative_eq!(
                sea.pressure_head(10.0, 20.0, down, 3.0),
                down,
                epsilon = 0.0
            );
        }
    }

    /// The same seed is the same sea.
    ///
    /// What makes a wave test reproducible and what lets a renderer draw the
    /// water the physics is standing on. A different seed has to give a different
    /// sea, or the seed is decoration.
    #[test]
    fn a_realisation_is_reproducible_and_the_seed_matters() {
        let one = Seaway::new(moderate(), GRAVITY);
        let same = Seaway::new(moderate(), GRAVITY);
        let other = Seaway::new(
            SeaState {
                seed: 2,
                ..moderate()
            },
            GRAVITY,
        );

        assert_relative_eq!(
            one.elevation(12.0, 3.0, 7.0),
            same.elevation(12.0, 3.0, 7.0),
            epsilon = 0.0
        );
        assert!(
            (one.elevation(12.0, 3.0, 7.0) - other.elevation(12.0, 3.0, 7.0)).abs() > 1e-6,
            "two seeds produced the same water"
        );
        // But the same statistics: a different seed is the same sea state.
        assert_relative_eq!(
            one.realised_height(),
            other.realised_height(),
            max_relative = 1e-12
        );
    }

    /// Waves run the way they are pointed.
    ///
    /// A heading of zero travels north, and a heading of a right angle travels
    /// east — which is worth a test because the sea's convention is *towards*
    /// while the wind's is *from*, and the two sit next to each other in every
    /// call site.
    #[test]
    fn the_heading_points_where_the_waves_go() {
        // Unidirectional, because the claim is about the *mean* heading and a fan
        // around it would blur exactly the "across it, nothing varies" half of
        // the check. The fan's own centring is asserted in
        // `the_realised_directions_follow_the_spread`.
        let northward = Seaway::new(
            SeaState {
                components: 1,
                heading: 0.0,
                spreading: 0.0,
                ..moderate()
            },
            GRAVITY,
        );
        let eastward = Seaway::new(
            SeaState {
                components: 1,
                heading: 0.5 * PI,
                spreading: 0.0,
                ..moderate()
            },
            GRAVITY,
        );

        // Along its own direction the pattern varies; across it, it does not.
        let reference = northward.elevation(0.0, 0.0, 0.0);
        assert_relative_eq!(
            northward.elevation(0.0, 500.0, 0.0),
            reference,
            epsilon = 1e-9
        );
        assert!((northward.elevation(20.0, 0.0, 0.0) - reference).abs() > 1e-6);

        let reference = eastward.elevation(0.0, 0.0, 0.0);
        assert_relative_eq!(
            eastward.elevation(500.0, 0.0, 0.0),
            reference,
            epsilon = 1e-9
        );
        assert!((eastward.elevation(0.0, 20.0, 0.0) - reference).abs() > 1e-6);
    }

    /// The synthesis convention is public API, and this is what says so.
    ///
    /// A renderer resynthesises the same water from [`Seaway::waves`] on its own
    /// grid, so the convention — the sign of the phase, the direction of the `ω t`
    /// term, the dispersion relation, the order the waves come out in, and the
    /// bit pattern of the phase generator — is a contract rather than an
    /// implementation choice. Any change to it moves the water under a boat that
    /// is already floating on it.
    ///
    /// So this pins the realisation rather than a property of it: a fixed seed
    /// against elevations recorded from this implementation. It is deliberately a
    /// stored-number test, which the rest of this file avoids on principle. The
    /// numbers are not a physical oracle and are not claimed to be one — they are
    /// the CPU side of an agreement with a shader, and a test that let them drift
    /// silently would be no agreement at all.
    ///
    /// If this fails after an intentional change, the shader is now wrong too.
    #[test]
    fn the_synthesis_convention_is_pinned() {
        // Unidirectional on purpose. This test pins the frequency, amplitude and
        // phase conventions, and mixing the direction machinery into it would
        // mean a change to the spreading law showed up as a failure here, in a
        // test about something else. `the_realised_directions_follow_the_spread`
        // pins the other half.
        let sea = Seaway::new(
            SeaState {
                significant_height: 2.0,
                peak_period: 6.0,
                heading: 0.5,
                components: 8,
                seed: 12345,
                spreading: 0.0,
            },
            9.81,
        );

        // Dispersion, applied to every wave: a renderer that recomputed `k` from
        // `ω` with a different gravity would draw waves of the wrong length.
        for wave in sea.waves() {
            assert_relative_eq!(
                wave.wavenumber,
                wave.frequency * wave.frequency / 9.81,
                max_relative = 1e-15
            );
            assert_relative_eq!(wave.direction.0, 0.5_f64.cos(), max_relative = 1e-15);
            assert_relative_eq!(wave.direction.1, 0.5_f64.sin(), max_relative = 1e-15);
        }

        // The first wave, whole: amplitude and phase together, because a phase
        // generator that changed its stream would keep the amplitudes and move
        // every crest.
        let first = sea.waves()[0];
        assert_relative_eq!(
            first.amplitude,
            7.770_671_769_839_567e-5,
            max_relative = 1e-12
        );
        assert_relative_eq!(first.phase, 1.589_529_189_573_176_7, max_relative = 1e-12);

        // And the sum, at points and times chosen so that no term is stationary:
        // this is what a vertex shader has to reproduce.
        for (north, east, time, expected) in [
            (0.0, 0.0, 0.0, 6.609_954_400_768_778e-1),
            (17.0, -9.0, 3.5, -3.359_566_594_173_393e-1),
            (-40.0, 25.0, 11.25, 6.993_477_678_858_909e-2),
        ] {
            assert_relative_eq!(
                sea.elevation(north, east, time),
                expected,
                max_relative = 1e-12
            );
        }
    }

    /// The pressure head against the closed form in the documentation, not
    /// against the other accessor: `pressure_head` delegates to the pair, so
    /// comparing the two would be a test that cannot fail. The formula is
    /// transcribed here from [`Seaway::pressure_head`]'s own doc comment, which
    /// makes this the oracle for the branch at the surface — where a plausible
    /// "a dry point carries nothing" simplification would return zero instead of
    /// the depth and silently move the waterline.
    #[test]
    fn the_pressure_head_follows_the_stretched_closed_form() {
        let sea = Seaway::new(
            SeaState {
                significant_height: 2.5,
                peak_period: 6.0,
                components: 24,
                seed: 7,
                ..SeaState::default()
            },
            9.81,
        );

        let expected = |north: f64, east: f64, down: f64, time: f64| {
            let elevation = sea.elevation(north, east, time);
            let attenuation = (down + elevation).max(0.0);
            let dynamic: f64 = sea
                .waves()
                .iter()
                .map(|wave| {
                    let along = north * wave.direction.0 + east * wave.direction.1;
                    wave.amplitude
                        * (-wave.wavenumber * attenuation).exp()
                        * (wave.wavenumber * along - wave.frequency * time + wave.phase).cos()
                })
                .sum();
            down + dynamic
        };

        for &time in &[0.0, 1.3, 7.75] {
            for &north in &[-30.0, 0.0, 12.5] {
                for &east in &[-5.0, 0.0, 21.0] {
                    // Straddle the *local* surface, not the mean level: the branch
                    // is on depth, and depth is measured from the wave.
                    let surface = -sea.elevation(north, east, time);
                    for offset in [-4.0, -0.05, 0.0, 0.05, 4.0] {
                        let down = surface + offset;
                        let (depth, head) = sea.depth_and_pressure_head(north, east, down, time);
                        assert_relative_eq!(depth, down + sea.elevation(north, east, time));
                        assert_relative_eq!(
                            head,
                            expected(north, east, down, time),
                            epsilon = 1e-12
                        );
                    }
                }
            }
        }
    }

    /// The property the whole spreading change exists for.
    ///
    /// A unidirectional sea is a function of one horizontal coordinate: walk
    /// along a crest, perpendicular to the heading, and the surface does not
    /// move. That is what makes such a sea look like corrugated iron out to the
    /// horizon, and it is a physical claim about the water rather than a
    /// complaint about the picture — a hull in a long-crested head sea feels no
    /// asymmetry across its beam and so no wave-induced roll.
    ///
    /// Measured along the crest direction, at a spacing well over a wavelength so
    /// that a small residue could not pass for a wave.
    #[test]
    fn a_spread_sea_is_not_a_function_of_one_coordinate() {
        let along_crest = |spreading: f64| {
            let heading = 0.0;
            let sea = Seaway::new(
                SeaState {
                    significant_height: 2.0,
                    peak_period: 6.0,
                    heading,
                    components: 60,
                    seed: 4,
                    spreading,
                },
                9.81,
            );
            // Heading is north, so the crests run east: sample along east.
            let samples: Vec<f64> = (0..48)
                .map(|step| sea.elevation(0.0, f64::from(step) * 37.0, 0.0))
                .collect();
            let mean = samples.iter().sum::<f64>() / samples.len() as f64;
            (samples.iter().map(|it| (it - mean).powi(2)).sum::<f64>() / samples.len() as f64)
                .sqrt()
        };

        // Long-crested: the surface is constant along the crest, to the bit.
        assert!(
            along_crest(0.0) < 1e-12,
            "a unidirectional sea varied along its own crests"
        );

        // Short-crested: the variation along the crest is a real sea's worth. Not
        // compared against a published number — there is no measurement campaign
        // here — but against the sea's own scale: a spread sea should vary along
        // the crest by a decent fraction of its standard deviation.
        let deviation = 2.0 / 4.0;
        let spread = along_crest(10.0);
        assert!(
            spread > 0.3 * deviation,
            "a spread sea barely varied along the crest: {spread:.3} m against {deviation:.3} m"
        );
    }

    /// The directions themselves, against the law they are drawn from.
    ///
    /// Two claims worth pinning, both of which a plausible sign or normalisation
    /// error breaks: the fan is centred on the sea's heading, and it is narrower
    /// for a larger exponent. Checked with circular statistics, because a mean of
    /// bearings taken as plain numbers is meaningless.
    #[test]
    fn the_realised_directions_follow_the_spread() {
        let circular = |spreading: f64| {
            let heading = 1.1;
            let sea = Seaway::new(
                SeaState {
                    significant_height: 2.0,
                    peak_period: 6.0,
                    heading,
                    components: 200,
                    seed: 9,
                    spreading,
                },
                9.81,
            );
            // Weighted by variance, which is what the spreading function is a
            // distribution of: an unweighted mean would be dominated by the
            // near-zero-amplitude tail of the band.
            let mut north = 0.0;
            let mut east = 0.0;
            let mut total = 0.0;
            for wave in sea.waves() {
                let weight = wave.amplitude * wave.amplitude;
                north += weight * wave.direction.0;
                east += weight * wave.direction.1;
                total += weight;
            }
            let resultant = (north * north + east * east).sqrt() / total;
            let mean = east.atan2(north);
            // Circular standard deviation, radians.
            (mean, (-2.0 * resultant.ln()).sqrt())
        };

        let (narrow_mean, narrow) = circular(25.0);
        let (wide_mean, wide) = circular(3.0);

        assert_relative_eq!(narrow_mean, 1.1, epsilon = 0.05);
        assert_relative_eq!(wide_mean, 1.1, epsilon = 0.12);
        assert!(
            narrow < wide,
            "a larger exponent gave a wider fan: {:.3} against {:.3} rad",
            narrow,
            wide
        );
        // And the fan is a fan, not a line and not a puddle.
        assert!(
            (0.05..1.4).contains(&narrow) && (0.1..1.6).contains(&wide),
            "spreads out of any physical range: {narrow:.3}, {wide:.3} rad"
        );
    }
}
