#define_import_path vela::atmosphere

// The sky, as a closed form of direction.
//
// # Why this is a shared library and not two copies
//
// Two shaders need the same answer to "what colour is the sky this way": the sky
// dome, which draws it, and the ocean, which reflects it. If they disagreed the
// horizon would show a seam and the reflection would be of a sky that is not
// there — the same class of error the engine avoids by sharing one sea
// realisation between the physics and the renderer, and avoided the same way.
//
// The sun direction is *not* defined here. It comes in as a parameter, because
// the one place it is decided is Rust: the same constant orients the
// `DirectionalLight` that lights the boat. A sun baked into this file would be a
// second answer, and the two would drift the first time anyone moved the light.
//
// # What this is not
//
// Not a physical atmosphere. No Rayleigh or Mie integral, no ozone, no
// multiple scattering. It is a gradient chosen to look like a clear day over
// open water, with a sun and a haze band at the horizon, because the ocean needs
// something plausible to reflect and this frame budget belongs to the physics.
// A real sky model would be a Hosek-Wilkie fit or a Bruneton precompute, both of
// which are larger than everything else in this frontend put together.

/// Zenith colour of the clear sky.
const ZENITH: vec3<f32> = vec3<f32>(0.10, 0.26, 0.56);

/// Colour the sky fades to at the horizon: hazier, warmer, much paler.
const HAZE: vec3<f32> = vec3<f32>(0.56, 0.66, 0.74);

/// Colour below the horizon.
///
/// Two things see it, and both want it close to the haze. A wave face steep
/// enough to reflect downwards should read as a dull patch of sky rather than as a
/// hole; and the dome shows a thin sliver of it *below* the far sea, because a sea
/// cut off at eight kilometres has its edge a fifth of a degree under the true
/// horizon. A dark nadir drew that sliver as a dark line along the horizon, which
/// is the second half of "something odd out there". Only a shade below the haze
/// now, so the sliver is invisible whether or not it is covered.
const NADIR: vec3<f32> = vec3<f32>(0.50, 0.60, 0.69);

/// Colour and strength of the sun itself.
const SUN_COLOUR: vec3<f32> = vec3<f32>(1.0, 0.93, 0.80);

// Scene luminance scale, cd/m² per unit of the analytic sky. Every procedural
// surface applies the camera's exposure once, just like Bevy's PBR materials.
const SKY_LUMINANCE: f32 = 1000.0;

/// A cheap 2D value hash, in `[0, 1)`.
///
/// Dave Hoskins' `hash12`, and specifically *not* the `fract(sin(dot(...)))` one
/// everybody reaches for first. That one was here and cost ten frames a second:
/// four hashes per noise octave, four octaves, one `sin` each, evaluated over
/// most of the screen. This is the same quality of nothing-in-particular with
/// three multiplies and three `fract`s and no transcendental at all.
fn hash(cell: vec2<f32>) -> f32 {
    var scattered = fract(vec3<f32>(cell.xyx) * 0.1031);
    scattered = scattered + dot(scattered, scattered.yzx + 33.33);
    return fract((scattered.x + scattered.y) * scattered.z);
}

/// Value and analytic gradient of a smooth lattice, using the same four hashes.
/// Quintic interpolation keeps the cloud-lighting normal smooth at cell edges.
fn value_noise(at: vec2<f32>) -> vec3<f32> {
    let cell = floor(at);
    let offset = fract(at);
    let weight = offset * offset * offset * (offset * (offset * 6.0 - 15.0) + 10.0);
    let derivative = 30.0 * offset * offset * (offset - 1.0) * (offset - 1.0);
    let a = hash(cell);
    let b = hash(cell + vec2<f32>(1.0, 0.0));
    let c = hash(cell + vec2<f32>(0.0, 1.0));
    let d = hash(cell + vec2<f32>(1.0, 1.0));
    return vec3<f32>(
        mix(mix(a, b, weight.x), mix(c, d, weight.x), weight.y),
        mix(b - a, d - c, weight.y) * derivative.x,
        mix(c - a, d - b, weight.x) * derivative.y,
    );
}

/// The cloud field and its slope, low-passed to a footprint in cloud coordinates.
/// Removed octaves contribute their mean, not zero: filtering must not change
/// the weather. Coarse octaves are shared with the dome, never independent noise.
fn cloud_field(at: vec2<f32>, footprint: f32) -> vec3<f32> {
    var field = vec3<f32>(0.5, 0.0, 0.0);
    var amplitude = 8.0 / 15.0;
    var frequency = 1.0;
    var scaled = at;
    var basis_x = vec2<f32>(1.0, 0.0);
    var basis_z = vec2<f32>(0.0, 1.0);
    let octave_transform = mat2x2<f32>(1.6, 1.2, -1.2, 1.6);
    for (var octave = 0; octave < 4; octave = octave + 1) {
        let visible = 1.0 - smoothstep(0.25, 0.75, footprint * frequency);
        if (visible <= 0.0) {
            break;
        }
        let sample = value_noise(scaled);
        field = field + amplitude * visible * vec3<f32>(
            sample.x - 0.5,
            dot(sample.yz, basis_x),
            dot(sample.yz, basis_z),
        );
        scaled = octave_transform * scaled + vec2<f32>(7.1, 3.7);
        basis_x = octave_transform * basis_x;
        basis_z = octave_transform * basis_z;
        frequency = frequency * 2.0;
        amplitude = amplitude * 0.5;
    }
    return field;
}

/// Engine seconds to layer coordinates. Keep this velocity in agreement with
/// `sky::cloud_drift`; converting the engine time to f32 happens before multiply
/// on both CPU and GPU, so the dome and ocean sample the same moving field.
fn cloud_offset(time: f32) -> vec2<f32> {
    return vec2<f32>(0.014, 0.006) * time;
}

/// One atmosphere for the dome and every reflected/ambient lookup.
/// `roughness` is perceptual GGX roughness; zero reproduces the visible clouds.
/// Only the dome requests a solar disc: direct ocean GGX accounts for that light.
fn sky_radiance(
    direction: vec3<f32>,
    sun: vec3<f32>,
    drift: vec2<f32>,
    roughness: f32,
    solar_disc: bool,
) -> vec3<f32> {
    let up = direction.y;
    let above = clamp(up, 0.0, 1.0);
    var colour = mix(HAZE, ZENITH, pow(above, 0.42));
    colour = mix(colour, NADIR, smoothstep(0.0, 0.08, -up));

    let towards_sun = clamp(dot(direction, sun), 0.0, 1.0);
    let daylight = smoothstep(-0.04, 0.03, sun.y);
    // Broad atmospheric forward scattering belongs in reflections too.
    colour = colour + SUN_COLOUR * (0.30 * daylight) * pow(towards_sun, 8.0);
    if (solar_disc) {
        colour = colour + SUN_COLOUR * (9.0 * daylight)
            * smoothstep(0.99992, 0.99999, towards_sun);
    }

    // A shared flat layer naturally crowds clouds towards the horizon. Haze
    // hides the projection's singularity rather than wrapping clouds below it.
    let horizon_visibility = smoothstep(0.025, 0.12, up);
    if (horizon_visibility <= 0.0) {
        return colour;
    }
    let layer_height = max(up, 0.04);
    let plane = (direction.xz / layer_height + drift) * 0.32 + vec2<f32>(17.1, 4.7);
    let perceptual = clamp(roughness, 0.0, 1.0);
    // A GGX cone grows with alpha = roughness²; planar projection magnifies it
    // at grazing angles. The tiny common floor also filters distant dome detail.
    let cone = max(0.0005, 0.8 * perceptual * perceptual);
    let footprint = cone * 0.32 / (layer_height * layer_height);
    let field = cloud_field(plane, footprint);
    // Filtering the coverage threshold as well as the noise preserves soft
    // cloud masses instead of erasing them as the fine octaves disappear.
    let edge_width = 0.13 * smoothstep(0.03, 0.75, footprint);
    let cover = smoothstep(0.46 - edge_width, 0.68 + edge_width, field.x);
    let opacity = cover * 0.96 * horizon_visibility;
    if (opacity <= 0.0) {
        return colour;
    }

    // Density gradients give rounded sunward lobes without extra fBm probes.
    // Dense undersides remain cool; thin sunward edges forward-scatter warm light.
    let cloud_normal = normalize(vec3<f32>(-field.y * 0.85, 1.0, -field.z * 0.85));
    let sunlit = max(dot(cloud_normal, sun), 0.0) * daylight;
    let thickness = smoothstep(0.46, 0.80, field.x);
    let underside = vec3<f32>(0.50, 0.58, 0.69) * (1.0 - 0.18 * thickness);
    let edge_light = (0.22 + 0.55 * (1.0 - thickness)) * pow(towards_sun, 12.0);
    let cloud = underside + SUN_COLOUR * (0.16 + 0.62 * sunlit + edge_light * daylight);
    return mix(colour, cloud, opacity);
}

/// Visible sky, including the solar disc, in the render frame (`y` is up).
/// Direction and authoritative sun direction must be normalised.
fn sky_colour(direction: vec3<f32>, sun: vec3<f32>, drift: vec2<f32>) -> vec3<f32> {
    return sky_radiance(direction, sun, drift, 0.0, true);
}

/// The same animated sky, excluding only the solar disc owned by ocean GGX.
/// Roughness filters coherent cloud detail and lighting, not a separate sky.
fn sky_reflection(
    direction: vec3<f32>,
    sun: vec3<f32>,
    drift: vec2<f32>,
    roughness: f32,
) -> vec3<f32> {
    return sky_radiance(direction, sun, drift, roughness, false);
}

// Cosine-weighted low-frequency environment quadrature. Normalizing the weights
// preserves constant radiance while letting shaded faces see different sky.
// The downward hemisphere is dark ocean bounce, not the horizon's pale haze.
fn sky_irradiance(normal: vec3<f32>, sun: vec3<f32>, drift: vec2<f32>) -> vec3<f32> {
    let weights = abs(normal);
    let x = sky_reflection(vec3<f32>(select(-1.0, 1.0, normal.x >= 0.0), 0.35, 0.0) / 1.059481,
        sun, drift, 1.0);
    let z = sky_reflection(vec3<f32>(0.0, 0.35, select(-1.0, 1.0, normal.z >= 0.0)) / 1.059481,
        sun, drift, 1.0);
    let y = select(vec3<f32>(0.025, 0.045, 0.060),
        sky_reflection(vec3<f32>(0.0, 1.0, 0.0), sun, drift, 1.0), normal.y >= 0.0);
    return (x * weights.x + y * weights.y + z * weights.z)
        / max(weights.x + weights.y + weights.z, 0.001);
}
