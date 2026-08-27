//! The sea, drawn by resynthesising the realisation the physics is standing on.
//!
//! # The contract this implements
//!
//! `vela_core::seaway::Seaway` is a *realisation*, not a height field: a fixed
//! set of waves with fixed phases, and a stated closed form
//!
//! ```text
//! ζ(north, east, t) = Σ aᵢ cos(kᵢ (north dᵢx + east dᵢy) - ωᵢ t + φᵢ)
//! ```
//!
//! The physics evaluates that on the CPU wherever a hull triangle happens to be.
//! This module evaluates the *same sum* in a vertex shader, on whatever grid the
//! renderer likes. Nothing is transferred per frame and no height field is
//! shared — the two sides agree because they compute the same function from the
//! same numbers, and `vela_core`'s `the_synthesis_convention_is_pinned` is what
//! keeps them from drifting apart.
//!
//! Getting this wrong has a very specific and very silly failure mode: a boat
//! floating on water nobody can see, sinking into crests that are not where they
//! are drawn.
//!
//! # Why the time comes from the engine and not from the shader
//!
//! Bevy offers `globals.time` to any material shader, and it is the obvious
//! thing to use. It is wrong here twice. It wraps to zero every hour, which
//! would make the sea jump; and more importantly it is the *render* clock, while
//! the physics runs on a fixed-step accumulator that deliberately does not track
//! it. Two clocks means two seas. So the engine's own `Sim::time` is pushed into
//! the uniform every frame, and the shader has no clock of its own.
//!
//! # Why a uniform array and not a storage buffer
//!
//! Storage buffers would take the wave count at runtime and are the tidier tool.
//! They do not exist under WebGL2, which is the browser target this frontend is
//! for, so the waves travel as a fixed-length uniform array padded to WGSL's
//! 16-byte stride. [`MAX_WAVES`] is the cap that follows, and a realisation
//! carrying more is refused loudly rather than silently truncated — a sea drawn
//! from the first 64 of 96 waves is a different sea.
//!
//! # Why the shader is embedded
//!
//! `embedded_asset!` bakes the WGSL into the binary rather than loading it from
//! an assets directory. On the web that removes a runtime fetch and a
//! deployment step; on the desktop it removes a whole class of "works from the
//! repository root, fails from anywhere else" bug, which is what a path relative
//! to the executable produces the first time someone installs the thing.
//!
//! The cost is no hot reload, which for a shader that transcribes a pinned
//! contract is not a cost.

use bevy::asset::{embedded_asset, embedded_path, AssetPath};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;
use vela_core::seaway::Seaway;

/// Waves the shader can carry.
///
/// Sized by WebGL2's guaranteed 16 KiB uniform binding: 64 waves at 32 bytes is
/// 2 KiB, comfortably inside it. `SeaState::components` defaults to 60, so the
/// default sea fits and there is room to raise it a little without a redesign.
pub const MAX_WAVES: usize = 64;

/// One linear wave, in the layout WGSL wants.
///
/// `vela_core::seaway::Wave` carries the same five numbers in `f64` with the
/// direction as a tuple. This is that, narrowed to `f32` and padded: WGSL's
/// std140 rules give every element of a uniform array a 16-byte aligned stride,
/// and a struct that does not respect it silently mismatches the Rust side
/// element by element — the failure is a sea that looks plausible and is not the
/// one the boat is floating on.
#[derive(ShaderType, Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct ShaderWave {
    /// Wave vector `k d`, rad/m, in the render plane as `(north, east)`.
    pub wave_vector: Vec2,
    /// Radian frequency, rad/s.
    pub frequency: f32,
    /// Phase at the origin at `t = 0`, rad.
    pub phase: f32,
    /// Amplitude, m.
    pub amplitude: f32,
    /// Padding to the 16-byte stride. Never read.
    pub padding: Vec3,
}

/// The scalars every wave shares.
#[derive(ShaderType, Clone, Debug)]
pub struct SeaUniform {
    /// How many entries of the array are real.
    pub count: u32,
    /// The **engine's** time, s. See the module documentation.
    pub time: f32,
    /// Colour of deep water, linear RGB.
    pub deep: Vec3,
    /// Colour where the surface faces the sky, linear RGB.
    pub shallow: Vec3,
}

/// The ocean surface material.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
pub struct OceanMaterial {
    #[uniform(0)]
    pub sea: SeaUniform,
    #[uniform(1)]
    pub waves: [ShaderWave; MAX_WAVES],
}

impl OceanMaterial {
    /// Loads a realisation into the material.
    ///
    /// # Panics
    ///
    /// If the realisation carries more than [`MAX_WAVES`]. That is a
    /// configuration error and not a runtime condition — the alternative is to
    /// draw a different sea from the one the physics is using, quietly, which is
    /// the failure this whole module is arranged to prevent.
    #[must_use]
    pub fn realising(sea: &Seaway, time: f64) -> Self {
        let source = sea.waves();
        assert!(
            source.len() <= MAX_WAVES,
            "a realisation of {} waves cannot be drawn by a shader that carries {MAX_WAVES}; \
             lower SeaState::components or raise MAX_WAVES together with the WGSL array",
            source.len()
        );

        let mut waves = [ShaderWave::default(); MAX_WAVES];
        for (slot, wave) in waves.iter_mut().zip(source) {
            *slot = ShaderWave {
                wave_vector: Vec2::new(
                    (wave.wavenumber * wave.direction.0) as f32,
                    (wave.wavenumber * wave.direction.1) as f32,
                ),
                frequency: wave.frequency as f32,
                phase: wave.phase as f32,
                amplitude: wave.amplitude as f32,
                padding: Vec3::ZERO,
            };
        }

        Self {
            sea: SeaUniform {
                count: source.len() as u32,
                time: time as f32,
                deep: Vec3::new(0.008, 0.035, 0.070),
                shallow: Vec3::new(0.060, 0.180, 0.230),
            },
            waves,
        }
    }

    /// Advances the sea's clock without rebuilding the waves.
    ///
    /// The waves are a property of the realisation and never change; only the
    /// time does. Rewriting the array every frame would work and would upload
    /// two kilobytes for no reason.
    pub fn set_time(&mut self, time: f64) {
        self.sea.time = time as f32;
    }
}

/// The embedded WGSL, as an asset path.
///
/// `embedded_path!` gives the path the `embedded_asset!` in [`OceanPlugin`]
/// registered it under; the `embedded` source has to be named explicitly because
/// the default source is the filesystem. This is the same two-step
/// `StandardMaterial` uses for its own shader.
fn shader() -> ShaderRef {
    ShaderRef::Path(
        AssetPath::from_path_buf(embedded_path!("shaders/ocean.wgsl")).with_source("embedded"),
    )
}

impl Material for OceanMaterial {
    fn vertex_shader() -> ShaderRef {
        shader()
    }

    fn fragment_shader() -> ShaderRef {
        shader()
    }
}

/// Registers the ocean material and its shader.
///
/// Both together on purpose: a material whose shader was not embedded fails at
/// the first frame with a missing-asset error, and the two lines belong in one
/// place so that cannot happen.
pub struct OceanPlugin;

impl Plugin for OceanPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/ocean.wgsl");
        app.add_plugins(MaterialPlugin::<OceanMaterial>::default());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vela_core::seaway::SeaState;

    fn sea(components: usize) -> Seaway {
        Seaway::new(
            SeaState {
                significant_height: 1.5,
                peak_period: 6.0,
                heading: 0.7,
                components,
                seed: 9,
            },
            9.81,
        )
    }

    /// The wave vector is `k` times the direction, and the shader gets it
    /// pre-multiplied so that it can take one dot product per wave.
    ///
    /// Asserted against the core's own numbers rather than against a recorded
    /// value: this is the join, and the only thing worth checking is that nothing
    /// was dropped or transposed crossing it.
    #[test]
    fn every_wave_crosses_intact() {
        let sea = sea(12);
        let material = OceanMaterial::realising(&sea, 3.25);

        assert_eq!(material.sea.count, 12);
        assert!((material.sea.time - 3.25).abs() < 1e-6);

        for (shader, wave) in material.waves.iter().zip(sea.waves()) {
            let expected = (
                (wave.wavenumber * wave.direction.0) as f32,
                (wave.wavenumber * wave.direction.1) as f32,
            );
            assert!((shader.wave_vector.x - expected.0).abs() < 1e-6);
            assert!((shader.wave_vector.y - expected.1).abs() < 1e-6);
            assert!((shader.frequency - wave.frequency as f32).abs() < 1e-6);
            assert!((shader.phase - wave.phase as f32).abs() < 1e-6);
            assert!((shader.amplitude - wave.amplitude as f32).abs() < 1e-9);
        }
    }

    /// Unused slots are zero, so a shader that ignored `count` would draw flat
    /// water rather than garbage.
    ///
    /// Belt and braces on top of the count: an amplitude of zero contributes
    /// nothing whatever the loop bound turns out to be.
    #[test]
    fn the_unused_slots_are_calm() {
        let material = OceanMaterial::realising(&sea(4), 0.0);
        for slot in &material.waves[4..] {
            assert_eq!(slot.amplitude, 0.0);
        }
    }

    /// The Rust struct must occupy the stride the WGSL array assumes.
    ///
    /// If this ever fails, the uniform is being read element by element at the
    /// wrong offset and the drawn sea has nothing to do with the computed one.
    /// Cheaper to assert here than to recognise on screen.
    #[test]
    fn the_wave_layout_matches_the_shader_stride() {
        assert_eq!(std::mem::size_of::<ShaderWave>(), 32);
        assert_eq!(std::mem::align_of::<ShaderWave>() % 4, 0);
    }

    /// A realisation too large to draw is refused rather than truncated.
    #[test]
    #[should_panic(expected = "cannot be drawn by a shader")]
    fn too_many_waves_is_an_error_and_not_a_crop() {
        let _ = OceanMaterial::realising(&sea(MAX_WAVES + 1), 0.0);
    }
}
