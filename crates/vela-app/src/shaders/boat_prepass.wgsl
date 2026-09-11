// The boat is rigid geometry plus CPU-deformed cloth, not skinned or morphed.
// Retain StandardMaterial's fragment stage for alpha, normals and motion output.
#import bevy_pbr::{
    prepass_io::{Vertex, VertexOutput},
    mesh_functions,
    view_transformations::position_world_to_clip,
}

@vertex
fn vertex(
    in: Vertex,
#ifdef SAIL_PREVIOUS_POSITION
    @location(8) previous_position: vec3<f32>,
#endif
) -> VertexOutput {
    var out: VertexOutput;
    let model = mesh_functions::get_world_from_local(in.instance_index);
    out.world_position = mesh_functions::mesh_position_local_to_world(model, vec4<f32>(in.position, 1.0));
    out.position = position_world_to_clip(out.world_position.xyz);
#ifdef UNCLIPPED_DEPTH_ORTHO_EMULATION
    out.unclipped_depth = out.position.z;
    out.position.z = min(out.position.z, 1.0);
#endif
#ifdef VERTEX_UVS_A
    out.uv = in.uv;
#endif
#ifdef VERTEX_UVS_B
    out.uv_b = in.uv_b;
#endif
#ifdef NORMAL_PREPASS_OR_DEFERRED_PREPASS
#ifdef VERTEX_NORMALS
    out.world_normal = mesh_functions::mesh_normal_local_to_world(in.normal, in.instance_index);
#endif
#ifdef VERTEX_TANGENTS
    out.world_tangent = mesh_functions::mesh_tangent_local_to_world(model, in.tangent, in.instance_index);
#endif
#endif
#ifdef VERTEX_COLORS
    out.color = in.color;
#endif
#ifdef MOTION_VECTOR_PREPASS
    let previous_model = mesh_functions::get_previous_world_from_local(in.instance_index);
#ifdef SAIL_PREVIOUS_POSITION
    let previous_local = previous_position;
#else
    let previous_local = in.position;
#endif
    out.previous_world_position = mesh_functions::mesh_position_local_to_world(
        previous_model, vec4<f32>(previous_local, 1.0));
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = in.instance_index;
#endif
#ifdef VISIBILITY_RANGE_DITHER
    out.visibility_range_dither = mesh_functions::get_visibility_range_dither_level(in.instance_index, model[3]);
#endif
    return out;
}
