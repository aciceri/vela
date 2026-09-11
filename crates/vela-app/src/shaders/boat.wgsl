// Preserve the embedded atlas and Bevy's real direct lights, shadows and fog.
// Procedural relief is in boat/cloth metres, never atlas pixels or screen space.
#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
}
#import vela::atmosphere::{sky_irradiance, sky_reflection, cloud_offset, SKY_LUMINANCE}

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    mesh_view_bindings::view,
}
#endif

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> finish: vec4<f32>;
struct BoatEnvironment {
    boat_from_world: mat4x4<f32>,
    grid: vec4<f32>,
    sun_time: vec4<f32>,
    contact: array<vec4<f32>, 85>,
}
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var<uniform> environment: BoatEnvironment;

// Bilinear interpolation of actual CPU sea intersections and their draining
// film envelope. Both are stored in body space, including heel and pitch.
fn hull_wetness(local: vec3<f32>) -> f32 {
    let grid = environment.grid;
    let cell = clamp((local.xz - grid.xy) / grid.zw, vec2<f32>(0.0), vec2<f32>(16.0, 4.0));
    let base = vec2<u32>(min(floor(cell), vec2<f32>(15.0, 3.0)));
    let fraction = cell - vec2<f32>(base);
    let index = base.y * 17u + base.x;
    let height = mix(
        mix(environment.contact[index].xy, environment.contact[index + 1u].xy, fraction.x),
        mix(environment.contact[index + 17u].xy, environment.contact[index + 18u].xy, fraction.x),
        fraction.y);
    let pixel = max(fwidth(local.y), 0.012);
    let contact = 1.0 - smoothstep(height.x - pixel, height.x + pixel, local.y);
    let film = 1.0 - smoothstep(height.y - 0.18 - pixel, height.y + pixel, local.y);
    return max(contact, film * 0.88);
}

// Pixel coverage converges to the 12 mm seam's area fraction under minification.
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

fn metric_ripple(coordinate: f32, spacing: f32) -> f32 {
    let visibility = 1.0 - smoothstep(spacing * 0.18, spacing * 0.50, fwidth(coordinate));
    return sin(coordinate * (6.28318530718 / spacing)) * visibility;
}

// Integrated GGX environment BRDF fit; bounded Fresnel keeps polished metal
// and the separate dielectric coat finite, including at grazing incidence.
fn environment_brdf(roughness: f32, ndotv: f32) -> vec2<f32> {
    let r = roughness * vec4<f32>(-1.0, -0.0275, -0.572, 0.022)
        + vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let a004 = min(r.x * r.x, exp2(-9.28 * ndotv)) * r.x + r.y;
    return vec2<f32>(-1.04, 1.04) * a004 + r.zw;
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr = pbr_input_from_standard_material(in, is_front);
    pbr.material.base_color = alpha_discard(pbr.material, pbr.material.base_color);
    let local = (environment.boat_from_world * in.world_position).xyz;

    if finish.x < 0.5 || finish.x > 1.5 {
        // The original blue ORM channel classifies metal. Teak islands are
        // identified by their baked chroma; all original texture detail stays.
        let color = pbr.material.base_color.rgb;
        let wood = smoothstep(0.025, 0.09, color.r - color.b)
            * smoothstep(0.01, 0.055, color.g - color.b);
        let metal = clamp(pbr.material.metallic, 0.0, 1.0);
        let baked_roughness = pbr.material.perceptual_roughness;
        let gelcoat_roughness = clamp(baked_roughness * 0.60, 0.22, 0.50);
        let wood_roughness = clamp(baked_roughness * 0.90 + 0.13, 0.42, 0.78);
        let metal_roughness = clamp(baked_roughness + 0.08, 0.16, 0.45);
        // 28 cm polishing variation, 3.8 mm grain, 2.8 mm orange-peel and
        // 0.7 mm brushed finish. Fine structure fades before it can shimmer.
        let polish = metric_ripple(local.x + local.y * 0.31 + local.z * 0.47, 0.28);
        let grain = metric_ripple(local.z + 0.002 * sin(local.x * 8.0), 0.0038);
        let peel = vec3<f32>(metric_ripple(local.x, 0.0028),
            metric_ripple(local.y, 0.0031), metric_ripple(local.z, 0.0026));
        let brushing = metric_ripple(local.x + 0.13 * local.z, 0.0007);
        let wet = hull_wetness(local);
        let dry_roughness = mix(mix(gelcoat_roughness, wood_roughness, wood), metal_roughness, metal)
            + polish * 0.035 + wood * grain * 0.045 + metal * brushing * 0.022;
        pbr.material.perceptual_roughness = clamp(mix(dry_roughness,
            mix(0.18, 0.25, wood), wet * (1.0 - metal)), 0.12, 0.88);
        pbr.material.base_color = vec4<f32>(color
            * (1.0 + (1.0 - metal) * (0.012 * polish + 0.018 * wood * grain))
            * (1.0 - wet * (1.0 - metal) * mix(0.10, 0.28, wood)),
            pbr.material.base_color.a);
        pbr.material.clearcoat = (1.0 - metal) * mix(mix(0.32, 0.20, wood), 0.70, wet);
        pbr.material.clearcoat_perceptual_roughness = mix(mix(0.17, 0.28, wood), 0.10, wet);
        let local_slope = (peel * 0.018 * (1.0 - wood)
            + vec3<f32>(0.002 * grain, 0.0, 0.045 * grain) * wood
            + vec3<f32>(0.012 * brushing, 0.0, 0.0) * metal) * (1.0 - 0.75 * wet);
        let world_slope = transpose(mat3x3<f32>(environment.boat_from_world[0].xyz,
            environment.boat_from_world[1].xyz, environment.boat_from_world[2].xyz)) * local_slope;
        pbr.N = normalize(pbr.N - (world_slope - pbr.N * dot(pbr.N, world_slope)));
    } else {
#ifdef VERTEX_UVS_A
        let uv = in.uv;
        let footprint = fwidth(uv);
        let thread_visibility = vec2<f32>(1.0) - smoothstep(
            vec2<f32>(0.00022), vec2<f32>(0.00063), footprint);
        let phase = uv * (6.28318530718 / 0.00125);
        let thread = sin(phase) * thread_visibility;
        let weave = dot(cos(phase) * thread_visibility, vec2<f32>(0.5));
        let seam = seam_coverage(uv.y, footprint.y);
        // Concentric layers around the actual tack, clew and head. They move
        // with the unrolled cut, not world coordinates or screen-space noise.
        let corner_distance = min(min(length(uv), length(uv - vec2<f32>(finish.z, 0.0))),
            length(uv - vec2<f32>(0.0, finish.w)));
        let corner_pixel = max(fwidth(corner_distance), 0.0005);
        let layers = (vec4<f32>(1.0) - smoothstep(
            vec4<f32>(0.18, 0.30, 0.44, 0.62) - corner_pixel,
            vec4<f32>(0.18, 0.30, 0.44, 0.62) + corner_pixel,
            vec4<f32>(corner_distance)));
        let reinforcement = dot(layers, vec4<f32>(0.25));
        let patch_stitch = max(max(
            1.0 - smoothstep(0.0015, 0.0015 + corner_pixel, abs(corner_distance - 0.18)),
            1.0 - smoothstep(0.0015, 0.0015 + corner_pixel, abs(corner_distance - 0.30))), max(
            1.0 - smoothstep(0.0015, 0.0015 + corner_pixel, abs(corner_distance - 0.44)),
            1.0 - smoothstep(0.0015, 0.0015 + corner_pixel, abs(corner_distance - 0.62))))
            * min(1.0, 0.003 / corner_pixel);
        var edge = 0.0;
#ifdef VERTEX_UVS_B
        // UV1 is the cut's normalized coordinate. The derivative ratio gives
        // local chord metres even with the main's roach and the jib's rake.
        let chord_metres = max(uv.x / max(in.uv_b.x, 0.0001), 0.001);
        let edge_distance = min(min(uv.x, (1.0 - in.uv_b.x) * chord_metres),
            min(uv.y, finish.w - uv.y));
        let edge_pixel = max(fwidth(edge_distance), 0.0005);
        edge = 1.0 - smoothstep(0.014 - edge_pixel, 0.014 + edge_pixel, edge_distance);
#endif
        pbr.material.base_color = vec4<f32>(pbr.material.base_color.rgb
            * (1.0 + 0.008 * weave - 0.025 * seam - 0.055 * reinforcement
                - 0.026 * patch_stitch - 0.02 * edge), pbr.material.base_color.a);
        pbr.material.perceptual_roughness = clamp(pbr.material.perceptual_roughness
            + 0.025 * weave + 0.04 * seam + 0.035 * reinforcement, 0.72, 0.94);
        // Additional cloth plies attenuate real transmitted light, not emission.
        pbr.material.diffuse_transmission *= (1.0 - 0.68 * seam)
            * (1.0 - 0.92 * reinforcement) * (1.0 - 0.72 * edge);
        let seam_distance = (fract(uv.y / 0.62 + 0.5) - 0.5) * 0.62;
        let seam_slope = -27.5 * seam_distance
            * exp(-seam_distance * seam_distance / 0.000016)
            * (1.0 - smoothstep(0.003, 0.016, footprint.y));
        let slope = 0.045 * thread + vec2<f32>(0.0, seam_slope);
        let dx_uv = dpdx(uv);
        let dy_uv = dpdy(uv);
        let dx_position = dpdx(in.world_position.xyz);
        let dy_position = dpdy(in.world_position.xyz);
        let handedness = sign(dx_uv.x * dy_uv.y - dx_uv.y * dy_uv.x);
        let tangent = (dx_position * dy_uv.y - dy_position * dx_uv.y) * handedness;
        let bitangent = (dy_position * dx_uv.x - dx_position * dy_uv.x) * handedness;
        let T = tangent / max(length(tangent), 0.000001);
        let B = bitangent / max(length(bitangent), 0.000001);
        let face_sign = sign(dot(cross(T, B), pbr.N));
        pbr.N = normalize(pbr.N - face_sign * (slope.x * T + slope.y * B));
#endif
    }

#ifdef PREPASS_PIPELINE
    return deferred_output(in, pbr);
#else
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr);
    // Bevy returns exposed direct lighting. Add physical outgoing sky radiance
    // at the same point, before fog/tonemapping, and expose it exactly once.
    let sun = environment.sun_time.xyz;
    let drift = cloud_offset(environment.sun_time.w);
    let metallic = clamp(pbr.material.metallic, 0.0, 1.0);
    let ndotv = max(dot(pbr.N, pbr.V), 0.0001);
    let f0 = mix(0.16 * pbr.material.reflectance * pbr.material.reflectance,
        pbr.material.base_color.rgb, metallic);
    let brdf = environment_brdf(pbr.material.perceptual_roughness, ndotv);
    let specular_energy = clamp(f0 * brdf.x + brdf.y, vec3<f32>(0.0), vec3<f32>(1.0));
    let transmission = clamp(pbr.material.diffuse_transmission, 0.0, 1.0);
    let diffuse_color = pbr.material.base_color.rgb * (1.0 - metallic)
        * (1.0 - pbr.material.specular_transmission) * (vec3<f32>(1.0) - specular_energy);
    var irradiance = sky_irradiance(pbr.N, sun, drift);
    if transmission > 0.0 {
        irradiance = mix(irradiance, sky_irradiance(-pbr.N, sun, drift), transmission);
    }
    var diffuse = diffuse_color * irradiance * pbr.diffuse_occlusion;
    let ambient_occlusion = min(pbr.specular_occlusion,
        dot(pbr.diffuse_occlusion, vec3<f32>(0.2126, 0.7152, 0.0722)));
    var specular = sky_reflection(reflect(-pbr.V, pbr.N), sun, drift,
        pbr.material.perceptual_roughness) * specular_energy * ambient_occlusion;
#ifdef STANDARD_MATERIAL_CLEARCOAT
    let coat = clamp(pbr.material.clearcoat, 0.0, 1.0);
    let coat_ndotv = max(dot(pbr.clearcoat_N, pbr.V), 0.0001);
    let coat_brdf = environment_brdf(pbr.material.clearcoat_perceptual_roughness, coat_ndotv);
    let coat_energy = clamp(0.04 * coat_brdf.x + coat_brdf.y, 0.0, 1.0) * coat;
    // Incoming and outgoing transmission through the dielectric layer.
    let coat_transmission = (1.0 - 0.04 * coat) * (1.0 - coat_energy);
    diffuse *= coat_transmission;
    specular = specular * coat_transmission
        + sky_reflection(reflect(-pbr.V, pbr.clearcoat_N), sun, drift,
            pbr.material.clearcoat_perceptual_roughness) * coat_energy * ambient_occlusion;
#endif
    out.color = vec4<f32>(out.color.rgb
        + (diffuse + specular) * (view.exposure * SKY_LUMINANCE), out.color.a);
    out.color = main_pass_post_lighting_processing(pbr, out.color);
    return out;
#endif
}
