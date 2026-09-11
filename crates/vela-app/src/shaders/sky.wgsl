// The sky dome: the shared atmosphere, evaluated along the view ray.
//
// Position is used only to get a direction. The dome's radius therefore sets
// nothing but where it sits in the depth buffer, which is why `sky.rs` is free to
// pick it to clear the sea rather than to model a distance.

#import vela::atmosphere::sky_colour
#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::mesh_view_bindings::view

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> sun: vec4<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> drift: vec4<f32>;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // From the camera to this fragment. The dome is centred on the camera, so
    // this is the same as the normalised dome position, but taking the difference
    // costs nothing and does not depend on that staying true.
    let direction = normalize(in.world_position.xyz - view.world_position);
    // The zero-roughness atmosphere used by water, with only the solar disc
    // added. `drift.xy` is sky::cloud_drift at the ocean's engine time.
    return vec4<f32>(sky_colour(direction, sun.xyz, drift.xy), 1.0);
}
