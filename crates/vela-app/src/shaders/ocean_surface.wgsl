#define_import_path vela::ocean_surface

// The same Lagrangian wave sum as vela_core::seaway::Seaway. The plane is
// (north, east), mapped to render (x, z), and height is positive upward.
// Scalar rotation/padding fields preserve the Rust array's 32-byte stride.
struct ShaderWave {
    wave_vector: vec2<f32>,
    frequency: f32,
    phase: f32,
    amplitude: f32,
    // Phase rotation to sample the same parcel 0.3 engine seconds earlier.
    age_rotation_cos: f32,
    age_rotation_sin: f32,
    padding_z: f32,
};

struct SeaUniform {
    count: u32,
    time: f32,
    significant_height: f32,
    trail_count: u32,
    deep: vec3<f32>,
    shallow: vec3<f32>,
    scatter: vec3<f32>,
    sun: vec4<f32>,
    motion: vec4<f32>,
    heading: vec4<f32>,
    wind: vec4<f32>,
    choppiness: f32,
    trail_bounds: vec4<f32>,
    // Rest-plane origin, square span in metres, and whether history is live.
    foam_region: vec4<f32>,
    // Previous rendered simulation time, reset on a new realisation/rewind.
    temporal: vec4<f32>,
};

struct TrailPoint {
    point: vec4<f32>,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> sea: SeaUniform;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> waves: array<ShaderWave, #{MAX_WAVES}>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<uniform> trail: array<TrailPoint, #{MAX_TRAIL}>;

struct Surface {
    height: f32,
    shift: vec2<f32>,
    slope: vec2<f32>,
    // (dDx/dx, dDz/dz, dDx/dz).
    strain: vec3<f32>,
    // Slope energy filtered out of explicit waves, retained by optical roughness.
    variance: f32,
};

fn jacobian(strain: vec3<f32>) -> f32 {
    return (1.0 + strain.x) * (1.0 + strain.y) - strain.z * strain.z;
}

// Spacing zero is the exact physical sum. A positive mesh/pixel spacing
// suppresses each wavelength separately before Nyquist: long swell survives
// after shorter waves cease to be representable, without a radial cutoff.
fn surface(plane: vec2<f32>, time: f32, spacing: f32) -> Surface {
    var height = 0.0;
    var shift = vec2<f32>(0.0);
    var slope = vec2<f32>(0.0);
    var strain = vec3<f32>(0.0);
    var variance = 0.0;
    let lambda = sea.choppiness;
    for (var index = 0u; index < sea.count; index = index + 1u) {
        let wave = waves[index];
        let k = wave.wave_vector;
        let magnitude = max(length(k), 1e-6);
        let weight = 1.0 - smoothstep(0.8, 2.8, magnitude * max(spacing, 0.0));
        let steepness = wave.amplitude * magnitude;
        variance = variance + 0.5 * steepness * steepness * (1.0 - weight * weight);
        if (weight <= 0.0) {
            continue;
        }
        let angle = dot(k, plane) - wave.frequency * time + wave.phase;
        let amplitude = wave.amplitude * weight;
        let c = amplitude * cos(angle);
        let s = amplitude * sin(angle);
        height = height + c;
        slope = slope - k * s;
        shift = shift - lambda * k * s / magnitude;
        let kk = lambda * vec3<f32>(k.x * k.x, k.y * k.y, k.x * k.y) / magnitude;
        strain = strain - kk * c;
    }
    return Surface(height, shift, slope, strain, variance);
}

fn folding(strain: vec3<f32>) -> f32 {
    return smoothstep(0.06, 0.32, 1.0 - jacobian(strain));
}

// Smooth world-space value noise: fixed cells, no per-frame random phases.
// Contact crosses the same eddies continuously; asymmetric cells keep port
// and starboard from becoming two identical painted lines.
fn contact_hash(cell: vec2<f32>) -> f32 {
    var p = fract(vec3<f32>(cell.x, cell.y, cell.x) * 0.1031);
    p = p + dot(p, p.yzx + vec3<f32>(33.33));
    return fract((p.x + p.y) * p.z);
}

fn contact_noise(point: vec2<f32>) -> f32 {
    let cell = floor(point);
    let f = fract(point);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(contact_hash(cell), contact_hash(cell + vec2<f32>(1.0, 0.0)), u.x),
        mix(contact_hash(cell + vec2<f32>(0.0, 1.0)), contact_hash(cell + vec2<f32>(1.0)), u.x),
        u.y
    );
}

// Current bow/stern aeration only. The history pass stores what is left after
// the hull moves on; the old track is not traversed for every history texel.
// Input is displaced world position, so injection lands next to the hull,
// while storage remains on the parcel that actually occupies that position.
fn hull_foam(world_plane: vec2<f32>) -> f32 {
    let speed = length(sea.motion.zw);
    let hull = sea.wind.zw;
    let entry = clamp(sea.heading.z - 0.12, 0.0, 3.0);
    // sun.w is the physical bow-waterline immersion; heading.w remains the
    // reflection-valid flag. A dry, rising or deeply buried stem cannot
    // continuously aerate a sheet on the free surface.
    let wet = step(0.0, sea.sun.w) * (1.0 - smoothstep(0.0, 0.55, sea.sun.w));
    if (hull.x <= 0.0 || (speed < 0.5 && entry * wet <= 0.0)) {
        return 0.0;
    }
    let heading = sea.heading.xy;
    let offset = world_plane - sea.motion.xy;
    let along = dot(offset, heading);
    let across = abs(offset.x * heading.y - offset.y * heading.x);
    let speed_gate = smoothstep(0.5, 2.5, speed);
    let aft = (hull.x - along) / (0.5 * hull.x);
    let run = smoothstep(-0.04, 0.08, aft) * (1.0 - smoothstep(0.35, 0.85, aft));
    let crest_at = hull.y * sqrt(max(aft, 0.0)) + 0.18 + 0.22 * entry;
    let lobe = 1.0 - smoothstep(0.0, 0.45 + 0.22 * entry, abs(across - crest_at));
    let bow = lobe * run * wet * smoothstep(0.0, 1.0, entry);
    let stern_run = smoothstep(-3.2, -1.5, along) * (1.0 - smoothstep(0.0, 0.8, along));
    let stern_width = max(0.75, hull.y * 0.65);
    let stern = stern_run * (1.0 - smoothstep(0.3 * stern_width, stern_width, across));
    let envelope = max(bow, speed_gate * stern);
    if (envelope <= 0.0) {
        return 0.0;
    }

    // Metre-scale rafts with smaller holes, resolvable by the 0.625m history
    // texels rather than aliased subpixel noise. The physical clock evolves
    // the source; history carries the separated parcels away after contact.
    let eddy = world_plane - sea.wind.xy * (sea.time * 0.12);
    let warp = vec2<f32>(
        contact_noise(eddy * 0.28),
        contact_noise(eddy * 0.28 + vec2<f32>(17.0, 5.0))
    ) - vec2<f32>(0.5);
    let patches = contact_noise(eddy * 0.85 + warp * 1.5);
    let bubbles = contact_noise(eddy * 1.4 - warp);
    let broken = smoothstep(0.32, 0.66, patches * 0.72 + bubbles * 0.28);
    return clamp(envelope * broken, 0.0, 1.0);
}
