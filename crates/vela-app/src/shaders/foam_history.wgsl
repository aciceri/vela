#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import vela::ocean_surface::{sea, surface, hull_foam}

// All positions are parcel/rest coordinates. The physical orbit enters only
// when locating a hull source, and the ocean supplies it again when drawing.
struct FoamPass {
    // Previous and destination origins, span, and one texel in metres.
    previous_region: vec4<f32>,
    region: vec4<f32>,
    // Engine delta, downwind speed (m/s), decay rate, and unused padding.
    step: vec4<f32>,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(3) var previous: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var previous_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(5) var<uniform> history: FoamPass;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let parcel = history.region.xy + in.uv * history.region.z;
    let dt = history.step.x;
    let previous_parcel = parcel - sea.wind.xy * (history.step.y * dt);
    let previous_uv = (previous_parcel - history.previous_region.xy) / history.previous_region.z;
    // Clamp-to-edge is a sampler implementation detail, not an inflow
    // condition. A newly exposed texel has no previous foam, including after
    // a teleport farther than the whole region.
    let texel = history.previous_region.w / history.previous_region.z;
    let valid = all(previous_uv >= vec2<f32>(0.5 * texel))
        && all(previous_uv <= vec2<f32>(1.0 - 0.5 * texel));
    let packed = textureSampleLevel(previous, previous_sampler, previous_uv, 0.0);
    let old = select(0.0, packed.r + packed.g / 255.0, valid);

    let wave = surface(parcel, sea.time, 0.0);
    // Ambient breaking is evaluated everywhere by the optical field. This
    // bounded history stores only aeration injected by the physical hull.
    let rate = 5.0 * hull_foam(parcel + wave.shift);
    let total_rate = rate + history.step.z;
    let equilibrium = rate / total_rate;
    // Exact solution of dF/dt = source*(1-F) - decay*F for this source sample.
    // Source saturation and decay are both scaled by simulation time; paused
    // frames never enter this pass in the first place.
    let density = clamp(equilibrium + (old - equilibrium) * exp(-total_rate * dt), 0.0, 1.0);

    // Two UNORM channels provide ~16-bit precision without a WebGL float RT
    // extension. Single-channel8-bit decay would round into permanent foam at
    // small dt. Decode after linear filtering as r + g/255, also in the ocean.
    let scaled = density * 255.0;
    return vec4<f32>(floor(scaled) / 255.0, fract(scaled), 0.0, 1.0);
}
