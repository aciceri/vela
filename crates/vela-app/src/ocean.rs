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
//! # What is physical and what is cosmetic
//!
//! Near the hull, geometry is the exact physical realisation. Farther out,
//! wavelengths smaller than the polar mesh can resolve are filtered before
//! sampling; their slope variance survives as optical roughness. The renderer
//! never feeds this level of detail back into the physics. Detail below the
//! realisation's shortest wave is added only as slope, foam and light:
//!
//! - a wind sea of four Gerstner components between one and three metres,
//!   running downwind in independently modulated wave packets, contributing slope
//!   and the whitecaps its own Jacobian says it would have broken;
//! - GGX sunlight with dielectric Fresnel and unresolved slope variance, rather
//!   than an artificially capped reflection or magnified distant noise;
//! - persistent whitecaps injected by swell compression and the bow/stern into
//!   a scrolling parcel-space history texture; transport and decay follow the
//!   engine clock, while cellular pores stretch with the physical wave orbit;
//! - fixed-scale ripple bands and foam detail filtered by pixel footprint,
//!   rather than rescaling their coordinates as the camera moves;
//! - the statistical self-shadowing of a rough surface for a low sun;
//! - the boat's planar reflection, distorted along the reflected ray and blurred
//!   by roughness, composited over the same animated sky drawn above the sea.
//!
//! Each of these is a known lie stated as one, in the shader, next to the
//! thing it fakes. None of them is fed back into the physics.
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
//! for, so the waves travel as a fixed-length uniform array whose element
//! stride is a multiple of WGSL's sixteen bytes. [`MAX_WAVES`] is the cap that
//! follows, and a realisation carrying more is refused loudly rather than
//! silently truncated — a sea drawn from the first 64 of 96 waves is a different
//! sea.
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

use bevy::asset::embedded_asset;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::shader::{ShaderDefVal, ShaderRef};
use vela_core::seaway::Seaway;

/// Waves the shader can carry.
///
/// Sized by WebGL2's guaranteed 16 KiB uniform binding: 64 waves at 32 bytes is
/// 2 KiB, comfortably inside it. `SeaState::components` defaults to 60, so the
/// default sea fits and there is room to raise it a little without a redesign.
///
/// The WGSL array is sized from this constant through a shader def — see
/// [`Material::specialize`] — so there is one number, not two that have to agree.
pub const MAX_WAVES: usize = 64;

/// One linear wave, in the layout WGSL wants.
///
/// `vela_core::seaway::Wave` carries the same five numbers in `f64` with the
/// direction as a tuple. This is that, narrowed to `f32` and padded out to 32
/// bytes: a uniform array's stride must be a multiple of 16, and the five
/// payload floats plus the two-float alignment of the vector come to 24. Three
/// scalar pads rather than a `Vec3`, which was what this had first — a `vec3` is
/// 16-aligned in WGSL and `encase` follows suit, so that "padding" started at
/// offset 32 and made the element 48 bytes, while the shader read it at 32. The
/// picture was still a sea, and not the one the boat was floating on, which is
/// the failure `the_wave_layout_matches_the_shader_stride` now pins.
#[derive(ShaderType, Clone, Copy, Default, Debug)]
pub struct ShaderWave {
    /// Wave vector `k d`, rad/m, in the render plane as `(north, east)`.
    pub wave_vector: Vec2,
    /// Radian frequency, rad/s.
    pub frequency: f32,
    /// Phase at the origin at `t = 0`, rad.
    pub phase: f32,
    /// Amplitude, m.
    pub amplitude: f32,
    /// Scalar padding preserves the 32-byte uniform-array stride.
    pub pad0: f32,
    pub pad1: f32,
    pub pad2: f32,
}

/// The scalars every wave shares.
#[derive(ShaderType, Clone, Debug)]
pub struct SeaUniform {
    /// How many entries of the wave array are real.
    pub count: u32,
    /// The **engine's** time, s. See the module documentation.
    pub time: f32,
    /// Significant wave height of the realisation, m.
    ///
    /// The shader needs a scale to judge a crest against: "steep and high" is
    /// meaningless without one, and the foam threshold has to mean the same thing
    /// in a half-metre chop as in a four-metre sea. Sits here rather than being
    /// recomputed in the shader because it is a property of the realisation.
    pub significant_height: f32,
    /// How many entries of the trail array are real. Packs into the same
    /// sixteen-byte slot as the three fields above it.
    pub trail_count: u32,
    /// Colour of deep water, linear RGB.
    pub deep: Vec3,
    /// Colour where the surface faces the sky, linear RGB.
    pub shallow: Vec3,
    /// Colour of light scattered *through* a thin crest lit from behind,
    /// linear RGB.
    pub scatter: Vec3,
    /// Direction towards the sun, `.w` unused.
    ///
    /// Taken from [`crate::sky::SUN`] rather than chosen here: the same constant
    /// orients the `DirectionalLight`, and a highlight that did not line up with
    /// the sun lighting the boat is the kind of wrongness that is obvious on
    /// screen and invisible in the code.
    pub sun: Vec4,
    /// The boat's stern and its velocity over the ground in the render plane:
    /// `(x, z)` of the body origin, `(vx, vz)` of the horizontal world velocity,
    /// m and m/s. The newest point of the wake, ahead of the recorded trail —
    /// see `wake` in the shader. Zero until [`OceanMaterial::record`] has been
    /// called, which draws no wake, and that is the right picture of a boat that
    /// has not moved yet.
    pub motion: Vec4,
    /// The boat's heading and the bow's motion, for the bow wave: `(x, z)` of
    /// the unit heading in the render plane — the hull's axis, which is not
    /// the track over the ground because of leeway — the bow's vertical
    /// velocity in m/s, positive downward into the water, and in `.w` whether
    /// the reflection texture is live (1) or the material has none yet (0).
    pub heading: Vec4,
    /// Downwind `(x, z)` in the render plane, unit: the realisation's mean
    /// direction of travel, amplitude-weighted, which is where the wind is
    /// blowing to and what the shader's wind-sea layer runs along. A property
    /// of the realisation, computed once from it. `.zw` carry the hull's
    /// length and greatest half-breadth, m, for the bow wave.
    pub wind: Vec4,
    /// Choppiness `λ` of the realisation, `SeaState::choppiness`: the number
    /// the shader moves its parcels sideways by, which is the number the
    /// physics clips the hull with.
    pub choppiness: f32,
    /// The wake's reach in the render plane, `(min x, min z, max x, max z)`:
    /// the recorded track grown by the widest the wake gets, kept by
    /// [`OceanMaterial::record`] so the shader can skip the track's loop
    /// for every fragment that is not near it, which is nearly all of them.
    pub trail_bounds: Vec4,
    /// Parcel-space history bounds: `(origin.x, origin.z, span, enabled)`.
    /// The origin follows the boat on a downwind-advected texel lattice.
    pub foam_region: Vec4,
}

/// Samples of the stern's track the shader can carry.
///
/// One kilobyte at sixteen bytes a sample: sixty-four points a metre or so
/// apart is sixty metres of track, and the wake is invisible long before that.
/// Sized through the same shader def mechanism as [`MAX_WAVES`].
pub const MAX_TRAIL: usize = 64;

/// Where the stern was, and when: one point of the wake's spine.
///
/// `(x, z)` in the render plane, the engine's time, and a pad — a `vec4` in
/// WGSL, which is what keeps the array's stride at sixteen.
#[derive(ShaderType, Clone, Copy, Default, Debug)]
pub struct TrailPoint {
    pub point: Vec4,
}

/// The ocean surface material.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
pub struct OceanMaterial {
    #[uniform(0)]
    pub sea: SeaUniform,
    #[uniform(1)]
    pub waves: [ShaderWave; MAX_WAVES],
    /// The stern's track, oldest first, `sea.trail_count` entries real.
    #[uniform(2)]
    pub trail: [TrailPoint; MAX_TRAIL],
    /// The above-water scene seen in a mirror at the mean sea level, rendered
    /// by `crate::reflection`'s camera into an image the size of the window
    /// (or a fraction of it), sampled at the fragment's own screen position.
    /// `None` until that camera exists, which the shader is told through
    /// `sea.heading.w`.
    #[texture(3)]
    #[sampler(4)]
    pub reflection: Option<Handle<Image>>,
    /// Completed persistent foam, packed as `r + g / 255` in linear UNORM.
    #[texture(5)]
    #[sampler(6)]
    pub foam_history: Option<Handle<Image>>,
    /// A new physical realisation invalidates accumulated foam, even at rest.
    pub(crate) foam_epoch: u64,
}

impl OceanMaterial {
    /// A material drawing a realisation - or still water, for `None`.
    ///
    /// # Panics
    ///
    /// If the realisation carries more than [`MAX_WAVES`]. That is a
    /// configuration error and not a runtime condition — the alternative is to
    /// draw a different sea from the one the physics is using, quietly, which is
    /// the failure this whole module is arranged to prevent.
    #[must_use]
    pub fn realising(sea: Option<&Seaway>, time: f64) -> Self {
        // The palette is Sea of Thieves' (Ang, SIGGRAPH 2018), by request: a
        // saturated deep blue, a turquoise where the water is thin or faces the
        // sky, and a green for light scattered *through* a crest. These are
        // the *transmitted* colours, seen only where the Fresnel term lets
        // them through; the blue a viewer reads at a distance is mostly the
        // reflected sky, which is how water works and why picking water colours
        // by eye without the reflection in place produces paint. The first
        // palette was a North Atlantic in winter - grey-steel, desaturated -
        // and read as lifeless next to a boat in sunshine.
        let mut material = Self {
            sea: SeaUniform {
                count: 0,
                time: time as f32,
                significant_height: 0.0,
                trail_count: 0,
                deep: Vec3::new(0.005, 0.038, 0.095),
                shallow: Vec3::new(0.030, 0.290, 0.330),
                scatter: Vec3::new(0.040, 0.520, 0.400),
                sun: crate::sky::SUN.normalize().extend(0.0),
                motion: Vec4::ZERO,
                heading: Vec4::new(1.0, 0.0, 0.0, 0.0),
                wind: Vec4::new(1.0, 0.0, 0.0, 0.0),
                choppiness: 0.0,
                trail_bounds: Vec4::ZERO,
                foam_region: Vec4::ZERO,
            },
            waves: [ShaderWave::default(); MAX_WAVES],
            trail: [TrailPoint::default(); MAX_TRAIL],
            reflection: None,
            foam_history: None,
            foam_epoch: 0,
        };
        material.realise(sea);
        material
    }

    /// Loads a realisation into the material - or still water, for `None` -
    /// keeping everything that is not the sea's: the clock, the wake's track,
    /// the boat's motion, the mirror. What a change of weather is to the
    /// water: the waves are new, and the boat has still been where it has been.
    ///
    /// # Panics
    ///
    /// As [`OceanMaterial::realising`].
    pub fn realise(&mut self, sea: Option<&Seaway>) {
        let source = sea.map_or(&[][..], Seaway::waves);
        assert!(
            source.len() <= MAX_WAVES,
            "a realisation of {} waves cannot be drawn by a shader that carries {MAX_WAVES}; \
             lower SeaState::components or raise MAX_WAVES together with the WGSL array",
            source.len()
        );

        self.waves = [ShaderWave::default(); MAX_WAVES];
        for (slot, wave) in self.waves.iter_mut().zip(source) {
            *slot = ShaderWave {
                wave_vector: Vec2::new(
                    (wave.wavenumber * wave.direction.0) as f32,
                    (wave.wavenumber * wave.direction.1) as f32,
                ),
                frequency: wave.frequency as f32,
                phase: wave.phase as f32,
                amplitude: wave.amplitude as f32,
                pad0: 0.0,
                pad1: 0.0,
                pad2: 0.0,
            };
        }

        // Where the sea is going, amplitude-weighted: the realisation travels
        // from the wind's direction, so this points downwind. Render x is north
        // and render z is east, the same map the wave vectors use above. Still
        // water keeps whatever direction it had; nothing reads it there.
        let mut downwind = Vec2::ZERO;
        for wave in source {
            downwind +=
                Vec2::new(wave.direction.0 as f32, wave.direction.1 as f32) * wave.amplitude as f32;
        }
        let downwind = downwind.normalize_or(self.sea.wind.xy());

        self.sea.count = source.len() as u32;
        self.sea.significant_height = sea.map_or(0.0, |sea| sea.state().significant_height as f32);
        self.sea.wind.x = downwind.x;
        self.sea.wind.y = downwind.y;
        self.sea.choppiness = sea.map_or(0.0, |sea| sea.state().choppiness as f32);
        self.foam_epoch = self.foam_epoch.wrapping_add(1);
        self.sea.foam_region = Vec4::ZERO;
    }

    /// Tells the water which way the hull points and how the bow is moving
    /// through the surface, for the bow wave.
    ///
    /// `heading` is the unit hull axis in the render plane and `bow_plunge`
    /// the bow's vertical velocity relative to the surface, m/s, positive
    /// downward — a bow driving into a wave throws spray, one lifting out of it
    /// does not. `hull` is the hull's length and greatest half-breadth, m:
    /// where the bow is along that axis, and how far outboard the bow wave
    /// clears the hull.
    pub fn set_heading(&mut self, heading: Vec2, bow_plunge: f32, hull: Vec2) {
        self.sea.heading.x = heading.x;
        self.sea.heading.y = heading.y;
        self.sea.heading.z = bow_plunge;
        self.sea.wind.z = hull.x;
        self.sea.wind.w = hull.y;
    }

    /// Gives the water the mirror image to reflect the boat from, or takes it
    /// away. The shader reads `sea.heading.w` to know which.
    pub fn set_reflection(&mut self, image: Option<Handle<Image>>) {
        self.sea.heading.w = if image.is_some() { 1.0 } else { 0.0 };
        self.reflection = image;
    }

    /// Advances the sea's clock without rebuilding the waves.
    ///
    /// The waves are a property of the realisation and never change; only the
    /// time does. This is not an upload optimisation and should not be read as
    /// one: touching the asset at all marks it modified, and Bevy then rebuilds
    /// the whole bind group — both uniforms, waves included — so the traffic is
    /// the same whether one float or the whole array is written. What it keeps is
    /// the statement that the realisation is fixed, which is the contract the
    /// physics and the picture agree by.
    pub fn set_time(&mut self, time: f64) {
        self.sea.time = time as f32;
    }

    /// Tells the water where the stern is now, how fast it is going over the
    /// ground, and — when it has moved far enough — adds where it was to the
    /// trail the wake is drawn along.
    ///
    /// The wake has to stay on the water. Drawing it from the boat's *current*
    /// position and velocity, which is what this did first, made a straight
    /// band that swung with every yaw of the hull like a searchlight, and a
    /// viewer saw at once that it was attached to the boat rather than left
    /// behind on the sea. So the stern's track is recorded, a point every metre
    /// of travel, oldest first, and the shader draws foam along the recorded
    /// polyline with the age of each point taken from the clock it was recorded
    /// at. The buffer is a queue: when it is full the oldest point falls off,
    /// which is also the one the wake has long since faded at.
    ///
    /// Render-plane coordinates: `stern` is the body origin, which the engine
    /// puts on the aft perpendicular, and `velocity` the horizontal part of the
    /// world velocity. Both come through `frame::to_render`, so the wake cannot
    /// lie on the mirror image of the track.
    pub fn record(&mut self, stern: Vec3, velocity: Vec3, time: f64) {
        /// Metres of travel between recorded points: fine enough that a turn
        /// is a curve and not a corner, coarse enough that sixty-four points
        /// outlast the wake.
        const SPACING: f32 = 1.0;

        self.sea.motion = Vec4::new(stern.x, stern.z, velocity.x, velocity.z);
        let here = Vec2::new(stern.x, stern.z);
        let count = self.sea.trail_count as usize;
        let newest = (count > 0).then(|| self.trail[count - 1].point.xy());
        if newest.is_none_or(|last| last.distance(here) >= SPACING) {
            if count == MAX_TRAIL {
                self.trail.copy_within(1.., 0);
            } else {
                self.sea.trail_count += 1;
            }
            let slot = self.sea.trail_count as usize - 1;
            self.trail[slot] = TrailPoint {
                point: Vec4::new(here.x, here.y, time as f32, 0.0),
            };
        }

        // The wake's reach: the points still young enough to show, and the
        // stern, grown by the widest the wake gets. `wake` in the shader
        // widens by a tenth of a metre a second and shows nothing past
        // fifteen seconds, so the margin is that width with the ragged edge on
        // top; a point older than that is inside the bounds only by accident,
        // and the shader stops at it anyway.
        let count = self.sea.trail_count as usize;
        let young = self.trail[..count]
            .iter()
            .map(|point| point.point)
            .filter(|point| time as f32 - point.z <= 15.0);
        let (min, max) = young.fold((here, here), |(min, max), point| {
            (min.min(point.xy()), max.max(point.xy()))
        });
        const MARGIN: f32 = (0.9 + 0.1 * 15.0) * 1.1 + 0.5;
        self.sea.trail_bounds = Vec4::new(
            min.x - MARGIN,
            min.y - MARGIN,
            max.x + MARGIN,
            max.y + MARGIN,
        );
    }
}

/// An embedded shader of this crate, as an asset path.
///
/// `embedded_path!` gives the path the `embedded_asset!` in the owning plugin
/// registered it under; the `embedded` source has to be named explicitly because
/// the default source is the filesystem. This is the same two-step
/// `StandardMaterial` uses for its own shader, and it is a macro rather than a
/// function because `embedded_path!` reads `file!()` at its call site. Shared
/// with [`crate::sky`] so the two materials cannot spell the path differently.
macro_rules! embedded_shader {
    ($file:literal) => {
        ::bevy::shader::ShaderRef::Path(
            ::bevy::asset::AssetPath::from_path_buf(::bevy::asset::embedded_path!($file))
                .with_source("embedded"),
        )
    };
}
pub(crate) use embedded_shader;

impl Material for OceanMaterial {
    fn vertex_shader() -> ShaderRef {
        embedded_shader!("shaders/ocean.wgsl")
    }

    fn fragment_shader() -> ShaderRef {
        embedded_shader!("shaders/ocean.wgsl")
    }

    /// Sizes the shader's wave and trail arrays from [`MAX_WAVES`] and
    /// [`MAX_TRAIL`].
    ///
    /// The array lengths are compile-time constants in WGSL, and writing them
    /// as literals in the shader would leave two copies of one number that only
    /// a mismatched picture could tell apart. Shader defs pushed here reach
    /// `ocean.wgsl` as `#{MAX_WAVES}` and `#{MAX_TRAIL}`, the same mechanism
    /// Bevy uses for `MATERIAL_BIND_GROUP`, and both stages get them because
    /// both declare the bindings.
    ///
    /// No prepass override, on purpose. The prepass and shadow passes would need
    /// the same displacement as the visible pass, but neither runs for this
    /// material: the camera carries no `DepthPrepass`, and both sea meshes are
    /// `NotShadowCaster`. The boat's shadow still lands on the displaced water,
    /// because the shadow map is the boat's own and the fragment stage samples it
    /// at the displaced position — see `fetch_directional_shadow` in
    /// `ocean.wgsl`. Nothing about the sea has to be rasterised for that.
    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let waves = ShaderDefVal::UInt("MAX_WAVES".into(), MAX_WAVES as u32);
        let trail = ShaderDefVal::UInt("MAX_TRAIL".into(), MAX_TRAIL as u32);
        descriptor.vertex.shader_defs.push(waves.clone());
        descriptor.vertex.shader_defs.push(trail.clone());
        if let Some(fragment) = &mut descriptor.fragment {
            fragment.shader_defs.push(waves);
            fragment.shader_defs.push(trail);
        }
        Ok(())
    }

    /// Opaque, and therefore depth-writing.
    ///
    /// The transparency is faked by the Fresnel mix rather than by blending. Real
    /// alpha would put the sea in the transparent phase, where it stops writing
    /// depth and stops occluding the hull below the waterline — a boat seen
    /// through its own water, keel and all.
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }
}

/// Registers the ocean material and its shader.
///
/// Together on purpose: a material whose shader was not embedded fails at the
/// first frame with a missing-asset error, and the lines belong in one place so
/// that cannot happen.
pub struct OceanPlugin;

impl Plugin for OceanPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/ocean.wgsl");
        app.add_plugins((
            MaterialPlugin::<OceanMaterial>::default(),
            crate::foam::FoamPlugin,
        ));
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
                spreading: 8.0,
                choppiness: 0.8,
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
        let material = OceanMaterial::realising(Some(&sea), 3.25);

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
        let material = OceanMaterial::realising(Some(&sea(4)), 0.0);
        for slot in &material.waves[4..] {
            assert_eq!(slot.amplitude, 0.0);
        }
    }

    /// The Rust struct must occupy the stride the WGSL array assumes.
    ///
    /// Asked of `encase`, which is what actually lays the uniform out, rather
    /// than of `size_of`: the Rust size of a struct says nothing about its GPU
    /// size, and the previous version of this test passed at 32 bytes while the
    /// uniform was being written at 48. If this ever fails, the uniform is being
    /// read element by element at the wrong offset and the drawn sea has nothing
    /// to do with the computed one. Cheaper to assert here than to recognise on
    /// screen.
    #[test]
    fn the_wave_layout_matches_the_shader_stride() {
        assert_eq!(<ShaderWave as ShaderType>::min_size().get(), 32);
        assert_eq!(
            <[ShaderWave; MAX_WAVES] as ShaderType>::min_size().get(),
            32 * MAX_WAVES as u64
        );
    }

    /// A realisation too large to draw is refused rather than truncated.
    #[test]
    #[should_panic(expected = "cannot be drawn by a shader")]
    fn too_many_waves_is_an_error_and_not_a_crop() {
        let _ = OceanMaterial::realising(Some(&sea(MAX_WAVES + 1)), 0.0);
    }
}
