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
#[derive(Debug, Clone, Copy, PartialEq)]
struct Component {
    /// Amplitude, m.
    amplitude: f64,
    /// Wavenumber, rad/m.
    wavenumber: f64,
    /// Radian frequency, rad/s.
    frequency: f64,
    /// Phase at the origin at `t = 0`, rad.
    phase: f64,
    /// Unit direction of travel in the world plane.
    direction: (f64, f64),
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
}

impl Default for SeaState {
    fn default() -> Self {
        Self {
            significant_height: 1.0,
            peak_period: 5.0,
            heading: 0.0,
            components: 60,
            seed: 1,
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

/// A realised sea: a fixed set of components with fixed phases.
#[derive(Debug, Clone, PartialEq)]
pub struct Seaway {
    state: SeaState,
    components: Vec<Component>,
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
    #[must_use]
    pub fn new(state: SeaState, gravity: f64) -> Self {
        let mut components = Vec::with_capacity(state.components);
        if state.components > 0 && state.peak_period > 0.0 && state.significant_height > 0.0 {
            let peak = 2.0 * PI / state.peak_period;
            let lowest = 0.25 * peak;
            let highest = 4.0 * peak;
            let step = (highest - lowest) / state.components as f64;
            let mut phases = Phases::new(state.seed);
            let (sin_heading, cos_heading) = state.heading.sin_cos();
            for index in 0..state.components {
                // Mid-interval, so that no component sits on the band edges.
                let frequency = lowest + (index as f64 + 0.5) * step;
                let amplitude = (2.0 * state.density(frequency) * step).sqrt();
                components.push(Component {
                    amplitude,
                    wavenumber: frequency * frequency / gravity,
                    frequency,
                    phase: phases.next(),
                    // A compass bearing: x is north, y is east.
                    direction: (cos_heading, sin_heading),
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
        let mut dynamic = 0.0;
        let mut elevation = 0.0;
        for component in &self.components {
            let along = north * component.direction.0 + east * component.direction.1;
            elevation += component.amplitude
                * (component.wavenumber * along - component.frequency * time + component.phase)
                    .cos();
        }
        let attenuation_depth = (down + elevation).max(0.0);
        for component in &self.components {
            let along = north * component.direction.0 + east * component.direction.1;
            dynamic += component.amplitude
                * (-component.wavenumber * attenuation_depth).exp()
                * (component.wavenumber * along - component.frequency * time + component.phase)
                    .cos();
        }
        down + dynamic
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
        let northward = Seaway::new(
            SeaState {
                components: 1,
                heading: 0.0,
                ..moderate()
            },
            GRAVITY,
        );
        let eastward = Seaway::new(
            SeaState {
                components: 1,
                heading: 0.5 * PI,
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
}
