//! The sky, and the sun that is the one source of both.
//!
//! # Why the sun lives here
//!
//! Three things need the sun's direction: the `DirectionalLight` that shades the
//! boat, the sky shader that draws the disc, and the ocean shader that puts a
//! highlight on the water. Two copies of a direction is two answers to one
//! question, and the failure is quiet — a specular streak that does not line up
//! with the sun, which reads as "the water looks wrong" and takes an afternoon
//! to trace. So [`SUN`] is declared once and everything else is fed from it.
//!
//! # Why a dome and not a clear colour
//!
//! The clear colour was a flat dark blue, and it made the sea look like a lit
//! object floating in a void: there was no horizon, and a mirror surface had
//! nothing to be a mirror of. The dome is the cheapest way to have both. It is a
//! sphere drawn from the inside, unlit, whose fragment shader evaluates the same
//! [`vela::atmosphere::sky_colour`] the ocean reflects — so the reflection is of
//! the sky that is actually there.

use bevy::asset::embedded_asset;
use bevy::mesh::{Mesh, Mesh3d, MeshBuilder, SphereKind, SphereMeshBuilder};
use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::load_shader_library;
use bevy::shader::ShaderRef;

use crate::ocean::embedded_shader;
use crate::view::Chase;

/// Direction *towards* the sun, in the render frame, normalised on use.
///
/// Towards, not from: it is what a shader dots a normal against.
///
/// Two things decide this vector and neither is aesthetic. It is **low**, because
/// a sun overhead lays its glitter directly beneath the viewer where none of it
/// is visible, and because light across a sea rather than down onto it is what
/// makes wave faces legible at all. And it is **ahead of the default camera**,
/// which orbits from astern: the specular path from a sun behind the viewer falls
/// on the far side of every wave and cannot be seen, so a physically fine sun in
/// the wrong half of the sky produces water with no sparkle in it — which was
/// exactly the first attempt.
pub const SUN: Vec3 = Vec3::new(0.46, 0.30, -0.62);

/// How fast the cloud layer drifts, in shader units per second.
///
/// Slow enough to be a sky rather than a weather report. Driven by the engine's
/// clock, not the render clock, for the same reason the sea is: two clocks would
/// let the clouds and the water disagree about how long the run has been going.
const CLOUD_DRIFT: Vec2 = Vec2::new(0.014, 0.006);

/// Radius of the dome, m.
///
/// Inside the camera's far plane and outside everything else. It does not need
/// to be large — the shader ignores position and uses only direction — but it
/// must comfortably contain the sea, or the sea's far edge would poke through the
/// sky and cut a straight line across the horizon.
const DOME: f32 = 12_000.0;

/// Marks the sky dome.
#[derive(Component)]
pub struct Sky;

/// The sky's material: nothing but a sun and a clock.
///
/// No textures, no cubemap, no environment map to download. Everything is a
/// closed form of the view direction, which is what makes the sky and the water's
/// reflection of it the same arithmetic rather than two approximations of each
/// other.
#[derive(Asset, AsBindGroup, TypePath, Debug, Clone)]
pub struct SkyMaterial {
    /// Direction towards the sun, and the cloud drift packed into `.w`-adjacent
    /// slots: a single `vec4` pair, because a uniform of loose floats wastes a
    /// binding on padding under WebGL2's alignment rules.
    #[uniform(0)]
    pub sun: Vec4,
    #[uniform(1)]
    pub drift: Vec4,
}

/// Where the cloud layer has drifted to by a given time.
///
/// Public because the ocean reflects these clouds and has to reflect the ones
/// that are there. Same function, one definition — the sea and the sky are the
/// same arithmetic for the same reason the sea and the physics are.
#[must_use]
pub fn cloud_drift(time: f64) -> Vec2 {
    CLOUD_DRIFT * time as f32
}

impl SkyMaterial {
    /// The sky at a given moment.
    #[must_use]
    pub fn at(time: f64) -> Self {
        let drift = cloud_drift(time);
        Self {
            sun: SUN.normalize().extend(0.0),
            drift: Vec4::new(drift.x, drift.y, 0.0, 0.0),
        }
    }

    /// Moves the clouds to a new time.
    pub fn set_time(&mut self, time: f64) {
        let drift = cloud_drift(time);
        self.drift = Vec4::new(drift.x, drift.y, 0.0, 0.0);
    }
}

/// The embedded WGSL, as an asset path — the same way [`crate::ocean`] names
/// its own, so that neither spells the `embedded://` path by hand.
fn shader() -> ShaderRef {
    embedded_shader!("shaders/sky.wgsl")
}

impl Material for SkyMaterial {
    fn fragment_shader() -> ShaderRef {
        shader()
    }

    /// The dome is drawn from the inside, so the faces that matter are the ones
    /// pointing away from the camera. Culling them is what an unmodified sphere
    /// would do.
    fn specialize(
        _pipeline: &bevy::pbr::MaterialPipeline,
        descriptor: &mut bevy::render::render_resource::RenderPipelineDescriptor,
        _layout: &bevy::mesh::MeshVertexBufferLayoutRef,
        _key: bevy::pbr::MaterialPipelineKey<Self>,
    ) -> Result<(), bevy::render::render_resource::SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None;
        Ok(())
    }

    /// Blended, so that the dome is drawn *after* the opaque sea and
    /// depth-tested against it — not because it is transparent.
    ///
    /// Bevy bins opaque draws by pipeline, not by distance, and the dome —
    /// whose bounding sphere is centred on the camera — came out first: every
    /// pixel of the screen ran the sky's cloud noise, and the sea then painted
    /// over two thirds of them. The GPU counter said 4.9 million fragment
    /// invocations for a 1.8 million pixel window. The transparent pass runs
    /// after the opaque one with the depth test on, so a dome drawn there costs
    /// exactly the pixels the sea leaves it, and the dome writes an alpha of one
    /// so nothing about its colour changes. It writes no depth, and nothing is
    /// ever behind it.
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }
}

/// Spawns the dome, parented to nothing and moved with the camera.
pub fn spawn(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<SkyMaterial>>,
) {
    // Enough segments that the dome's silhouette is not a polygon against the
    // horizon, which is the only place its tessellation is visible.
    let mesh = SphereMeshBuilder::new(
        DOME,
        SphereKind::Uv {
            sectors: 64,
            stacks: 32,
        },
    )
    .build();

    commands.spawn((
        Sky,
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(SkyMaterial::at(0.0))),
        Transform::default(),
    ));
}

/// Keeps the dome centred on the camera and its clouds on the engine's clock.
///
/// Centred on the *camera*, not the boat: a dome centred on the boat would slide
/// under the camera as it orbits, and the horizon would tilt.
pub fn follow(
    engine: Res<crate::sim::Engine>,
    cameras: Query<&Transform, (With<Chase>, Without<Sky>)>,
    mut skies: Query<&mut Transform, With<Sky>>,
    mut materials: ResMut<Assets<SkyMaterial>>,
) {
    let Ok(camera) = cameras.single() else {
        return;
    };
    for mut transform in &mut skies {
        transform.translation = camera.translation;
    }
    let time = engine.sim.time();
    for (_, material) in materials.iter_mut() {
        material.set_time(time);
    }
}

/// Registers the sky, its shader, and the shared atmosphere library.
///
/// The library is registered here rather than in a plugin of its own because
/// this is the module that owns the sun: whoever loads the atmosphere has to
/// agree with whoever decides where the light comes from.
pub struct SkyPlugin;

impl Plugin for SkyPlugin {
    fn build(&self, app: &mut App) {
        // The shared library the sky and the ocean both import. `load_shader_library!`
        // embeds it and holds the handle forever, which is what keeps the compiled
        // module alive for naga_oil to resolve `#import vela::atmosphere::...`.
        load_shader_library!(app, "shaders/atmosphere.wgsl");
        embedded_asset!(app, "shaders/sky.wgsl");
        app.add_plugins(MaterialPlugin::<SkyMaterial>::default())
            .add_systems(Startup, spawn)
            .add_systems(Update, follow);
    }
}
