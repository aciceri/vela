#import bevy_pbr::{
    prepass_io::{Vertex, FragmentOutput},
    prepass_bindings::previous_view_uniforms,
    mesh_view_bindings::view,
    view_transformations::position_world_to_clip,
}
#import vela::ocean_geometry::projected_vertex
#import vela::ocean_surface::{sea, surface}

// Match the main pass's invariant position. Different native compiler
// optimizations must not move a displaced vertex behind its own prepass depth.
struct VertexOutput {
    @builtin(position) @invariant position: vec4<f32>,
    @location(4) world_position: vec4<f32>,
#ifdef MOTION_VECTOR_PREPASS
    @location(5) previous_world_position: vec4<f32>,
#endif
#ifdef NORMAL_PREPASS_OR_DEFERRED_PREPASS
    @location(2) world_normal: vec3<f32>,
#endif
#ifdef UNCLIPPED_DEPTH_ORTHO_EMULATION
    @location(6) unclipped_depth: f32,
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    @location(7) instance_index: u32,
#endif
};

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let projected = projected_vertex(in.position);
    out.world_position = vec4<f32>(projected.world, 1.0);
    out.position = position_world_to_clip(projected.world);
#ifdef MOTION_VECTOR_PREPASS
    // The projected grid is only a sampling window. Its old screen-space
    // vertex is NOT the old water parcel: advect this same parcel in time.
    out.previous_world_position = out.world_position;
    if sea.temporal.x != sea.time {
        let previous = surface(projected.parcel, sea.temporal.x, projected.spacing);
        out.previous_world_position = vec4<f32>(
            projected.parcel.x + previous.shift.x, previous.height,
            projected.parcel.y + previous.shift.y, 1.0);
    }
#endif
#ifdef NORMAL_PREPASS_OR_DEFERRED_PREPASS
    let wave = projected.wave;
    let along_x = vec3<f32>(1.0 + wave.strain.x, wave.slope.x, wave.strain.z);
    let along_z = vec3<f32>(wave.strain.z, wave.slope.y, 1.0 + wave.strain.y);
    out.world_normal = normalize(cross(along_z, along_x));
#endif
#ifdef UNCLIPPED_DEPTH_ORTHO_EMULATION
    out.unclipped_depth = out.position.z;
    out.position.z = min(out.position.z, 1.0);
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = in.instance_index;
#endif
    return out;
}

#ifdef PREPASS_FRAGMENT
@fragment
fn fragment(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
#ifdef MOTION_VECTOR_PREPASS
    let current = view.unjittered_clip_from_world * in.world_position;
    let previous = previous_view_uniforms.clip_from_world * in.previous_world_position;
    // Reject crossings behind the camera instead of dividing by a tiny/negative w.
    out.motion_vector = vec2<f32>(0.0);
    if current.w > 0.0001 && previous.w > 0.0001 {
        let velocity = (current.xy / current.w - previous.xy / previous.w) * vec2<f32>(0.5, -0.5);
        if all(abs(velocity) < vec2<f32>(0.25)) {
            out.motion_vector = velocity;
        }
    }
#endif
#ifdef NORMAL_PREPASS
    out.normal = vec4<f32>(in.world_normal * 0.5 + vec3<f32>(0.5), 1.0);
#endif
#ifdef UNCLIPPED_DEPTH_ORTHO_EMULATION
    out.frag_depth = in.unclipped_depth;
#endif
    return out;
}
#endif
