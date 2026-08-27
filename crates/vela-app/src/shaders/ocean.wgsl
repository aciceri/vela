// The sea surface: the same sum vela_core::seaway::Seaway::elevation computes on
// the CPU, evaluated per vertex.
//
// The convention below is a transcription of a public contract, not a choice
// made here. `vela_core`'s the_synthesis_convention_is_pinned holds the CPU side
// to it; if that test ever fails after an intentional change, this file is wrong
// too and the water will be drawn somewhere the boat is not floating.
//
//   zeta(north, east, t) = sum_i a_i * cos(k_i . (north, east) - w_i t + phase_i)
//
// Positive up. The engine's z axis points down, and the Rust side has already
// mapped the plane so that render x is north and render z is east; the sum is
// therefore taken over (position.x, position.z) and added to render y.

#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_world}
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
    deep: vec3<f32>,
    shallow: vec3<f32>,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> sea: SeaUniform;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> waves: array<ShaderWave, 64>;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
};

// Elevation and its two surface slopes, from one pass over the waves.
//
// The slopes come out of the same loop as the height because they are the same
// derivative: d/dx of a cos is -k_x a sin, so a second pass would cost another
// sixty-four sines to learn nothing new. The normal is then exact for the
// surface actually drawn rather than reconstructed from neighbouring vertices,
// which is what keeps the lighting from showing the grid.
fn surface(plane: vec2<f32>) -> vec3<f32> {
    var height: f32 = 0.0;
    var slope: vec2<f32> = vec2<f32>(0.0, 0.0);
    for (var index: u32 = 0u; index < sea.count; index = index + 1u) {
        let wave = waves[index];
        let angle = dot(wave.wave_vector, plane) - wave.frequency * sea.time + wave.phase;
        height = height + wave.amplitude * cos(angle);
        slope = slope - wave.wave_vector * wave.amplitude * sin(angle);
    }
    return vec3<f32>(height, slope.x, slope.y);
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    // The grid is laid out in the render plane: x is north, z is east.
    let plane = vec2<f32>(vertex.position.x, vertex.position.z);
    let evaluated = surface(plane);

    var local = vec4<f32>(vertex.position, 1.0);
    local.y = local.y + evaluated.x;

    let world_from_local = get_world_from_local(vertex.instance_index);
    let world = mesh_position_local_to_world(world_from_local, local);

    // The surface is y = zeta(x, z), so its normal is (-dzeta/dx, 1, -dzeta/dz).
    let normal = normalize(vec3<f32>(-evaluated.y, 1.0, -evaluated.z));

    var out: VertexOutput;
    out.world_position = world.xyz;
    out.world_normal = normal;
    out.clip_position = position_world_to_clip(world.xyz);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Deliberately not a PBR water shader. Two colours mixed by how much of the
    // sky a facet faces, which is enough to read the shape of the sea and cheap
    // enough to leave the frame budget to the physics. Anything more convincing
    // would be a graphics project rather than this one, and §5.5a's warning
    // applies: the sea drawn here must not look better resolved than the sea the
    // engine computes.
    let up = clamp(in.world_normal.y, 0.0, 1.0);
    let colour = mix(sea.deep, sea.shallow, pow(up, 8.0));

    // A single hard light from above and abaft, so that crests catch it and
    // troughs do not. No shadow map: an ocean plane is the worst possible
    // shadow caster and the worst possible receiver.
    let sun = normalize(vec3<f32>(-0.4, 0.8, 0.45));
    let lit = 0.35 + 0.65 * clamp(dot(in.world_normal, sun), 0.0, 1.0);

    return vec4<f32>(colour * lit, 1.0);
}
