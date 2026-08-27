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

const PI: f32 = 3.141592653589793;

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

/// Value noise with a smooth interpolant, in `[0, 1]`.
fn value_noise(at: vec2<f32>) -> f32 {
    let cell = floor(at);
    let offset = fract(at);
    // Smoothstep weights: linear interpolation of a value lattice shows the
    // lattice, because the derivative jumps at every cell edge.
    let weight = offset * offset * (3.0 - 2.0 * offset);
    let a = hash(cell);
    let b = hash(cell + vec2<f32>(1.0, 0.0));
    let c = hash(cell + vec2<f32>(0.0, 1.0));
    let d = hash(cell + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, weight.x), mix(c, d, weight.x), weight.y);
}

/// Four octaves of value noise, in `[0, 1]`.
///
/// Four rather than more because these clouds are seen at a distance and through
/// a haze; the fifth octave costs four hashes and lands under a pixel.
fn fbm(at: vec2<f32>) -> f32 {
    var total: f32 = 0.0;
    var amplitude: f32 = 0.5;
    var scaled: vec2<f32> = at;
    for (var octave: i32 = 0; octave < 4; octave = octave + 1) {
        total = total + amplitude * value_noise(scaled);
        // Rotate as well as scale, so the octaves do not line up into a grid.
        scaled = mat2x2<f32>(1.6, 1.2, -1.2, 1.6) * scaled;
        amplitude = amplitude * 0.5;
    }
    return total;
}

/// Cloud cover along a view direction, in `[0, 1]`.
///
/// The clouds live on a flat layer at a fixed height, and the direction is
/// projected onto it — the standard trick, and the reason clouds crowd towards
/// the horizon exactly as real ones do: the projection stretches without limit as
/// the ray flattens. Looking down, or level, there are none.
///
/// `drift` moves the layer with time. It is passed in rather than read from a
/// clock so that this file has no hidden state.
fn cloud_cover(direction: vec3<f32>, drift: vec2<f32>) -> f32 {
    // Below this the ray never reaches the layer within any useful distance, and
    // the projection blows up.
    if (direction.y < 0.02) {
        return 0.0;
    }
    // Layer at unit height: the constant only sets the apparent scale, which the
    // frequency below absorbs.
    let plane = direction.xz / direction.y + drift;

    let cover = fbm(plane * 0.09);
    // Two thresholds rather than one: the lower cuts the flat overcast out of the
    // noise so there is open sky, the upper keeps the tops from saturating into
    // white paper.
    let shaped = smoothstep(0.52, 0.78, cover);

    // Fade out towards the zenith so the layer reads as a ceiling seen obliquely
    // rather than as a texture wrapped over the whole dome.
    let towards_horizon = 1.0 - smoothstep(0.0, 0.55, direction.y);
    return shaped * mix(0.35, 1.0, towards_horizon);
}

/// The sky colour along a unit direction, with the sun at `sun`.
///
/// `drift` is the cloud layer's offset; pass `vec2(0.0)` for a still sky.
///
/// Directions are in the render frame: `y` is up.
fn sky_colour(direction: vec3<f32>, sun: vec3<f32>, drift: vec2<f32>) -> vec3<f32> {
    let up = direction.y;

    // Gradient. The exponent is what puts the haze in a band near the horizon
    // instead of smeared over the whole dome.
    let above = clamp(up, 0.0, 1.0);
    var colour = mix(HAZE, ZENITH, pow(above, 0.42));
    // Below the horizon, fade to the nadir colour over a few degrees so a
    // reflecting wave face does not step abruptly.
    colour = mix(colour, NADIR, smoothstep(0.0, -0.08, up));

    let towards_sun = clamp(dot(direction, sun), 0.0, 1.0);

    // Aureole: the bright wash around the sun, which is most of what makes a sky
    // read as having a sun in it at all.
    colour = colour + SUN_COLOUR * 0.30 * pow(towards_sun, 8.0);
    // And the disc. Half a degree of arc is a dot product of about 0.99996; this
    // is deliberately softer and larger than the real thing, because a
    // half-degree disc is a couple of pixels and aliases into a flicker.
    colour = colour + SUN_COLOUR * 9.0 * smoothstep(0.9993, 0.99975, towards_sun);

    // Clouds last, lit by how much of the sun they face. A cloud in front of the
    // sun is brighter, not darker: these are thin enough to be translucent, and
    // the alternative reads as a hole punched in the sky.
    let cover = cloud_cover(direction, drift);
    if (cover > 0.0) {
        let shade = mix(vec3<f32>(0.52, 0.56, 0.62), vec3<f32>(1.0, 0.98, 0.95), 0.35 + 0.65 * towards_sun);
        let bright = shade + SUN_COLOUR * 0.55 * pow(towards_sun, 5.0);
        colour = mix(colour, bright, cover);
    }

    return colour;
}

/// The sky as seen by a reflecting surface: gradient and aureole, no sun disc and
/// no clouds.
///
/// Two omissions, both measured rather than assumed.
///
/// The **disc** is left out because a mirror-sharp sun reflected off an
/// interpolated normal lands on whichever triangles happen to face it and
/// flickers as the mesh slides under the water. The ocean adds its own specular
/// lobe instead, computed from the analytic normal and therefore stable.
///
/// The **clouds** are left out because they cost four octaves of value noise —
/// sixteen hashes, each a `sin` — and this function runs once per *sea* fragment,
/// which is most of the screen. Including them cost ten frames a second, and
/// bought an effect that a wavy surface scatters into an even grey anyway: a
/// cloud reflected in chop is not a cloud, it is a slightly duller patch of sky.
/// The dome still draws them at full detail, where they are actually legible.
fn sky_reflection(direction: vec3<f32>, sun: vec3<f32>) -> vec3<f32> {
    let up = direction.y;
    let above = clamp(up, 0.0, 1.0);
    var colour = mix(HAZE, ZENITH, pow(above, 0.42));
    colour = mix(colour, NADIR, smoothstep(0.0, -0.08, up));

    let towards_sun = clamp(dot(direction, sun), 0.0, 1.0);
    return colour + SUN_COLOUR * 0.30 * pow(towards_sun, 8.0);
}
