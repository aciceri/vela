// ABI-compatible replacement for Bevy 0.19's built-in MotionBlur fragment.
// Current-frame reconstruction only: no color history, accumulation or HUD.
#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_render::globals::Globals

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
#ifdef MULTISAMPLED
@group(0) @binding(1) var motion_vectors: texture_multisampled_2d<f32>;
@group(0) @binding(2) var depth: texture_multisampled_2d<f32>;
#else
@group(0) @binding(1) var motion_vectors: texture_2d<f32>;
@group(0) @binding(2) var depth: texture_2d<f32>;
#endif
@group(0) @binding(3) var texture_sampler: sampler;
struct MotionBlur {
    shutter_angle: f32,
    samples: u32,
#ifdef SIXTEEN_BYTE_ALIGNMENT
    _webgl2_padding: vec2<f32>,
#endif
}
@group(0) @binding(4) var<uniform> settings: MotionBlur;
@group(0) @binding(5) var<uniform> globals: Globals;

fn velocity_at(pixel: vec2<i32>) -> vec2<f32> {
    // Sample zero is also valid for multisampled prepasses; main camera is 1x.
    return textureLoad(motion_vectors, pixel, 0).rg;
}
fn depth_at(pixel: vec2<i32>) -> f32 {
    return textureLoad(depth, pixel, 0).x;
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(screen_texture));
    let pixel = clamp(vec2<i32>(in.uv * size), vec2<i32>(0), vec2<i32>(size) - vec2<i32>(1));
    let base = textureLoad(screen_texture, pixel, 0);
    let velocity = velocity_at(pixel);
    let center_depth = depth_at(pixel);
    // Positive comparisons also reject NaNs. A cut/teleport is not an exposure.
    if !(settings.shutter_angle > 0.0) || settings.samples == 0u
        || !all(abs(velocity) < vec2<f32>(0.25))
        || !(globals.delta_time > 0.0) || globals.delta_time > 0.25 {
        return base;
    }
    // At most 1/120 second exposure, irrespective of a slow render frame.
    let shutter = min(clamp(settings.shutter_angle, 0.0, 1.0),
        (1.0 / 120.0) / globals.delta_time);
    var exposure_pixels = velocity * size * shutter;
    let distance = length(exposure_pixels);
    if !(distance > 0.75) {
        return base;
    }
    // Full streak at most eight pixels (four either side); no long smears.
    exposure_pixels *= min(1.0, 8.0 / distance);
    let exposure = exposure_pixels / size;
    let direction = exposure_pixels / max(length(exposure_pixels), 0.0001);
    let samples = i32(min(settings.samples, 4u));
    var sum = base;
    var total = 1.0;
    for (var index = -samples; index <= samples; index += 1) {
        if index == 0 {
            continue;
        }
        let fraction = 0.5 * f32(index) / f32(samples);
        let sample_uv = in.uv + exposure * fraction;
        if any(sample_uv <= vec2<f32>(0.0)) || any(sample_uv >= vec2<f32>(1.0)) {
            continue;
        }
        let sample_pixel = vec2<i32>(sample_uv * size);
        let sample_depth = depth_at(sample_pixel);
        // Reverse-Z perspective depth: relative depth is scale-independent.
        // Reject silhouette crossings both ways, including sky/water borders.
        let relative_depth = abs(sample_depth - center_depth)
            / max(max(sample_depth, center_depth), 0.000001);
        if relative_depth > 0.035 || (sample_depth == 0.0) != (center_depth == 0.0) {
            continue;
        }
        let sample_velocity = velocity_at(sample_pixel);
        if !all(abs(sample_velocity) < vec2<f32>(0.25)) {
            continue;
        }
        var sample_exposure = sample_velocity * size * shutter;
        sample_exposure *= min(1.0, 8.0 / max(length(sample_exposure), 0.0001));
        // A tracked foreground hull cannot bleed into moving water. Avoid
        // cosine normalization: zero-speed samples must never divide by zero.
        let travel = abs(dot(sample_exposure, direction)) * 0.5;
        let needed = length(exposure_pixels) * abs(fraction);
        if travel + 0.35 < needed {
            continue;
        }
        let mismatch = length(sample_exposure - exposure_pixels);
        let agreement = 1.0 - smoothstep(1.5, 4.0, mismatch);
        let weight = (1.0 - abs(fraction)) * agreement;
        sum += textureSampleLevel(screen_texture, texture_sampler, sample_uv, 0.0) * weight;
        total += weight;
    }
    return sum / total;
}
