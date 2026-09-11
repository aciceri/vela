#define_import_path vela::ocean_optics

#import vela::ocean_surface::{sea, waves, jacobian}

struct OpticalSurface {
    // Upward world-space normal is normalize(vec3(-slope.x, 1, -slope.y)).
    slope: vec2<f32>,
    variance: f32,
    foam: f32,
};

fn breaking_rate(compression: f32) -> f32 {
    let fold = smoothstep(0.06, 0.32, compression);
    return 2.4 * fold * fold * fold;
}

// Filtering strain before a nonlinear breaking threshold otherwise makes
// unresolved whitecaps disappear. Project the omitted physical strain's
// covariance through 1-J, then integrate its source with a three-point Gaussian
// quadrature. This converges to the real compression criterion when resolved,
// and to stable subpixel coverage, not randomly sampled crests, at the horizon.
fn filtered_breaking(strain: vec3<f32>, diagonal: vec3<f32>, cross_terms: vec3<f32>) -> f32 {
    let gradient = vec3<f32>(-1.0 - strain.y, -1.0 - strain.x, 2.0 * strain.z);
    let variance = dot(gradient * gradient, diagonal)
        + 2.0 * dot(vec3<f32>(gradient.x * gradient.y, gradient.x * gradient.z, gradient.y * gradient.z), cross_terms);
    let spread = sqrt(3.0 * max(variance, 0.0));
    let compression = 1.0 - jacobian(strain);
    return (breaking_rate(compression - spread) + 4.0 * breaking_rate(compression)
        + breaking_rate(compression + spread)) / 6.0;
}

fn previous_phase(phase: vec2<f32>, rotation: vec2<f32>) -> vec2<f32> {
    // theta(t-age) = theta(t) + frequency*age.
    return vec2<f32>(phase.x * rotation.x - phase.y * rotation.y,
        phase.y * rotation.x + phase.x * rotation.y);
}

// The optical cache supplies its world-space pixel footprint, not a mesh cell
// or distance to the boat. Every visible parcel uses this same realisation.
fn optical_surface(plane: vec2<f32>, time: f32, spacing: f32) -> OpticalSurface {
    var slope = vec2<f32>(0.0);
    var strain_now = vec3<f32>(0.0);
    var strain_03 = vec3<f32>(0.0);
    var strain_06 = vec3<f32>(0.0);
    var strain_09 = vec3<f32>(0.0);
    var strain_12 = vec3<f32>(0.0);
    // Slope covariance (xx, zz, xz); strain covariance diagonal and
    // off-diagonal (xx*zz, xx*xz, zz*xz). No surrogate optical wave spectrum.
    var slope_covariance = vec3<f32>(0.0);
    var strain_diagonal = vec3<f32>(0.0);
    var strain_cross = vec3<f32>(0.0);
    let lambda = sea.choppiness;
    for (var index = 0u; index < sea.count; index = index + 1u) {
        let wave = waves[index];
        let k = wave.wave_vector;
        let magnitude = max(length(k), 1e-6);
        let direction = k / magnitude;
        // Leave room below Nyquist for the nonlinear normal/breaking response.
        // No frequencies are stretched or enlarged as they become unresolved.
        let weight = 1.0 - smoothstep(0.6, 1.8, magnitude * max(spacing, 0.0));
        let steepness = wave.amplitude * magnitude;
        let omitted = 0.5 * steepness * steepness * (1.0 - weight * weight);
        let orientation = vec3<f32>(direction.x * direction.x,
            direction.y * direction.y, direction.x * direction.y);
        slope_covariance += omitted * orientation;
        let strain_energy = lambda * lambda * omitted;
        strain_diagonal += strain_energy * orientation * orientation;
        strain_cross += strain_energy * vec3<f32>(orientation.x * orientation.y,
            orientation.x * orientation.z, orientation.y * orientation.z);
        if (weight <= 0.0) {
            continue;
        }
        let angle = dot(k, plane) - wave.frequency * time + wave.phase;
        var phase = vec2<f32>(cos(angle), sin(angle));
        let amplitude = wave.amplitude * weight;
        slope -= k * (amplitude * phase.y);
        let strain_amplitude = lambda * magnitude * amplitude * orientation;
        strain_now -= strain_amplitude * phase.x;
        // The CPU packs this one fixed age step once per spectrum. Four cheap
        // rotations reuse the current sin/cos; no trigonometry per past age.
        let rotation = vec2<f32>(wave.age_rotation_cos, wave.age_rotation_sin);
        phase = previous_phase(phase, rotation);
        strain_03 -= strain_amplitude * phase.x;
        phase = previous_phase(phase, rotation);
        strain_06 -= strain_amplitude * phase.x;
        phase = previous_phase(phase, rotation);
        strain_09 -= strain_amplitude * phase.x;
        phase = previous_phase(phase, rotation);
        strain_12 -= strain_amplitude * phase.x;
    }

    // Inverse transpose of the horizontal strain Jacobian converts the parcel
    // gradient into a choppy world gradient. Regularise an overturning fold to
    // keep a finite upward-facing optical surface at the singularity.
    let xx = 1.0 + strain_now.x;
    let zz = 1.0 + strain_now.y;
    let xz = strain_now.z;
    let determinant = max(jacobian(strain_now), 0.2);
    let choppy_slope = vec2<f32>(zz * slope.x - xz * slope.y,
        xx * slope.y - xz * slope.x) / determinant;
    let variance = max(((zz * zz + xz * xz) * slope_covariance.x
        + (xx * xx + xz * xz) * slope_covariance.y
        - 2.0 * xz * (xx + zz) * slope_covariance.z) / (determinant * determinant), 0.0);

    // A short causal convolution at this fixed parcel: trapezoidal integration
    // over ages 0,.3,.6,.9,1.2s with exp(-1.5*age) decay. The constants are the
    // integration weights, including half-weight endpoints. Engine time alone
    // determines the result, so pause/rewind/reset require no cache history.
    let recent = 0.15 * filtered_breaking(strain_now, strain_diagonal, strain_cross)
        + 0.19128845 * filtered_breaking(strain_03, strain_diagonal, strain_cross)
        + 0.12197090 * filtered_breaking(strain_06, strain_diagonal, strain_cross)
        + 0.07777208 * filtered_breaking(strain_09, strain_diagonal, strain_cross)
        + 0.02479483 * filtered_breaking(strain_12, strain_diagonal, strain_cross);
    return OpticalSurface(choppy_slope, variance, 1.0 - exp(-recent));
}
