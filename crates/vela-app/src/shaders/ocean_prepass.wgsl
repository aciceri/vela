// The sea's depth and shadow pass: the same displacement as `ocean.wgsl`.
//
// # Why this file has to exist
//
// The shadow map and the depth prepass are rendered with their own vertex shader,
// not the material's. Left at Bevy's default, they would see the *undisplaced*
// grid — a flat plane at the mean level — and the consequences are both visible
// and confusing: the boat's shadow lands on a sheet that is not where the water
// is, and the sea occludes itself against a depth buffer describing a different
// surface.
//
// So the displacement is duplicated here. Duplication is the wrong instinct
// almost everywhere, and it is the right one at a shader-stage boundary: the two
// stages have different I/O structs and different bindings, and the only thing
// they can actually share is the arithmetic. That arithmetic is the pinned
// contract in `vela_core::seaway`, transcribed identically in both files. If one
// changes and the other does not, the boat's shadow slides off the wave it
// belongs to, which is at least a symptom someone will notice.

#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_world}
#import bevy_pbr::prepass_io::{Vertex, VertexOutput}
#import bevy_pbr::view_transformations::position_world_to_clip

struct ShaderWave {
    wave_vector: vec2<f32>,
    frequency: f32,
    phase: f32,
    amplitude: f32,
    padding: vec3<f32>,
};

struct SeaUniform {
    count: u32,
    time: f32,
    significant_height: f32,
    deep: vec3<f32>,
    shallow: vec3<f32>,
    sun: vec4<f32>,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> sea: SeaUniform;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> waves: array<ShaderWave, 64>;

/// Elevation only. The prepass has no use for the slopes, and asking for them
/// would cost a second multiply per wave in the one place where nothing reads it.
fn elevation(plane: vec2<f32>) -> f32 {
    var height: f32 = 0.0;
    for (var index: u32 = 0u; index < sea.count; index = index + 1u) {
        let wave = waves[index];
        let angle = dot(wave.wave_vector, plane) - wave.frequency * sea.time + wave.phase;
        height = height + wave.amplitude * cos(angle);
    }
    return height;
}

/// The distance fade, identical to `ocean.wgsl`'s `resolved`.
fn resolved(distance: f32) -> f32 {
    return 1.0 - smoothstep(130.0, 200.0, distance);
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let world_from_local = get_world_from_local(vertex.instance_index);

    // World position for the phase, local radius for the fade. Same reasoning,
    // and the same wording, as `ocean.wgsl`: read the note there. If these two
    // files ever disagree about which space the phase lives in, the shadow map
    // describes a sea that is not on the screen.
    let base = mesh_position_local_to_world(world_from_local, vec4<f32>(vertex.position, 1.0));
    let fade = resolved(length(vertex.position.xz));
    let world = vec3<f32>(base.x, base.y + elevation(base.xz) * fade, base.z);

    var out: VertexOutput;
    out.position = position_world_to_clip(world);
#ifdef VERTEX_UVS
    out.uv = vertex.uv;
#endif
#ifdef NORMAL_PREPASS
    // Flat up rather than the analytic normal: nothing in this project reads the
    // normal prepass, and computing the slopes here to satisfy an unused output
    // would put a second copy of the derivative in a third file.
    out.world_normal = vec3<f32>(0.0, 1.0, 0.0);
#endif
#ifdef MOTION_VECTOR_PREPASS
    out.world_position = vec4<f32>(world, 1.0);
    out.previous_world_position = vec4<f32>(world, 1.0);
#endif
    return out;
}
