// Finish changes are applied to Bevy's fully sampled material, not a replacement
// texture: the model's colour, packed AO/roughness/metalness and normal maps stay
// authoritative, and the normal PBR lighting/shadow/fog path remains intact.
#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
}

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> finish: vec4<f32>;

// Pixel coverage of a crosscut seam. Under minification it converges to the
// seam's area fraction rather than turning into thick bands or crawling lines.
fn seam_coverage(coordinate: f32, footprint: f32) -> f32 {
    let period = 0.62;
    let half_width = 0.006;
    let distance = (fract(coordinate / period + 0.5) - 0.5) * period;
    let pixel = max(footprint, 0.0001);
    let coverage = clamp((distance + half_width) / pixel + 0.5, 0.0, 1.0)
        - clamp((distance - half_width) / pixel + 0.5, 0.0, 1.0);
    return mix(2.0 * half_width / period, coverage,
        1.0 - smoothstep(0.12, 0.36, pixel));
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr = pbr_input_from_standard_material(in, is_front);
    pbr.material.base_color = alpha_discard(pbr.material, pbr.material.base_color);

    if finish.x < 0.5 {
        // The actual GLB has ONE atlas shared by hull, rails, mast and boom.
        // Blue ORM texels already mark metal, so never promote a whole mesh.
        // Its teak/varnished-wood islands are yellow/brown; gelcoat and fittings
        // are neutral. Classify that baked chroma before lighting, not from the
        // sun/view direction, leaving all original colour and normal detail.
        let color = pbr.material.base_color.rgb;
        let wood = smoothstep(0.025, 0.09, color.r - color.b)
            * smoothstep(0.01, 0.055, color.g - color.b);
        let metal = clamp(pbr.material.metallic, 0.0, 1.0);
        let baked_roughness = pbr.material.perceptual_roughness;
        let gelcoat_roughness = clamp(baked_roughness * 0.60, 0.22, 0.50);
        let wood_roughness = clamp(baked_roughness * 0.90 + 0.13, 0.42, 0.78);
        let metal_roughness = clamp(baked_roughness + 0.13, 0.16, 0.45);
        pbr.material.perceptual_roughness = mix(
            mix(gelcoat_roughness, wood_roughness, wood), metal_roughness, metal);
        pbr.material.clearcoat = (1.0 - metal) * mix(0.32, 0.20, wood);
        pbr.material.clearcoat_perceptual_roughness = mix(0.17, 0.28, wood);
    } else {
#ifdef VERTEX_UVS_A
        // Sail UVs are unrolled metres and follow the same cloth parcel through
        // camber, twist and a tack change. Threads are 1.25 mm, not a huge noise
        // texture. Fade their normal/colour contrast before the Nyquist limit.
        let uv = in.uv;
        let footprint = fwidth(uv);
        let thread_visibility = vec2<f32>(1.0) - smoothstep(
            vec2<f32>(0.00022), vec2<f32>(0.00063), footprint);
        let phase = uv * (6.28318530718 / 0.00125);
        let thread = sin(phase) * thread_visibility;
        let weave = dot(cos(phase) * thread_visibility, vec2<f32>(0.5));
        let seam = seam_coverage(uv.y, footprint.y);

        pbr.material.base_color = vec4<f32>(
            pbr.material.base_color.rgb * (1.0 + 0.008 * weave - 0.025 * seam),
            pbr.material.base_color.a);
        pbr.material.perceptual_roughness = clamp(
            pbr.material.perceptual_roughness + 0.025 * weave + 0.04 * seam, 0.72, 0.94);
        // Bevy's diffuse-transmission lobe supplies backlighting from the real
        // lights. The doubled cloth at a seam transmits less; no emission.
        pbr.material.diffuse_transmission *= 1.0 - 0.68 * seam;

        // A small lap-seam ridge, with an analytic slope and pixel filtering.
        let seam_distance = (fract(uv.y / 0.62 + 0.5) - 0.5) * 0.62;
        let seam_slope = -27.5 * seam_distance
            * exp(-seam_distance * seam_distance / 0.000016)
            * (1.0 - smoothstep(0.003, 0.016, footprint.y));
        let slope = 0.045 * thread + vec2<f32>(0.0, seam_slope);

        // Derivative tangents need no added mesh geometry and stay correct on
        // both opposed sail windings and under the reflected camera.
        let dx_uv = dpdx(uv);
        let dy_uv = dpdy(uv);
        let dx_position = dpdx(in.world_position.xyz);
        let dy_position = dpdy(in.world_position.xyz);
        let handedness = sign(dx_uv.x * dy_uv.y - dx_uv.y * dy_uv.x);
        let tangent = (dx_position * dy_uv.y - dy_position * dx_uv.y) * handedness;
        let bitangent = (dy_position * dx_uv.x - dx_position * dy_uv.x) * handedness;
        let T = tangent / max(length(tangent), 0.000001);
        let B = bitangent / max(length(bitangent), 0.000001);
        pbr.N = normalize(pbr.N - slope.x * T - slope.y * B);
#endif
    }

#ifdef PREPASS_PIPELINE
    return deferred_output(in, pbr);
#else
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr);
    out.color = main_pass_post_lighting_processing(pbr, out.color);
    return out;
#endif
}
