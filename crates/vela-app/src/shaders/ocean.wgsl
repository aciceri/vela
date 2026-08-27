// The sea surface: the same sum vela_core::seaway::Seaway::elevation computes on
// the CPU, evaluated per vertex, and shaded as water.
//
// The convention below is a transcription of a public contract, not a choice
// made here. `vela_core`'s the_synthesis_convention_is_pinned holds the CPU side
// to it; if that test ever fails after an intentional change, this file is wrong
// too and the water will be drawn somewhere the boat is not floating.
//
//   zeta(north, east, t) = sum_i a_i * cos(k_i . (north, east) - w_i t + phase_i)
//
// Positive up. The engine's z axis points down, and the Rust side has already
// mapped the plane so that render x is north and render z is east; the sum is
// therefore taken over (position.x, position.z) and added to render y.
//
// # What the shading is, and what it is borrowed from
//
// The geometry is the engine's. The *shading* follows Alexander Alekseev's
// "Seascape" (shadertoy Ms2SD1, 2014), which is the clearest small statement of
// what makes water read as water: a Fresnel mix between a reflection of the sky
// and a dim transmitted colour, a height tint so crests look thinner than
// troughs, and a specular lobe that tightens with distance so the far sea
// glitters while the near sea shines.
//
// What is deliberately *not* borrowed is Seascape's surface. Its waves are a
// procedural noise fBm raymarched in screen space — beautiful, and unrelated to
// any sea a hull could float on. Taking it would mean the renderer drawing
// different water from the one the physics clips against, which is the single
// thing this frontend exists not to do.

#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_world}
#import bevy_pbr::mesh_view_bindings::view
#import bevy_pbr::shadows::fetch_directional_shadow
#import bevy_pbr::view_transformations::position_world_to_clip
#import vela::atmosphere::sky_reflection

/// Local rather than imported: naga_oil resolves function imports reliably and
/// constants less so, and this is four characters.
const PI: f32 = 3.141592653589793;

struct ShaderWave {
    wave_vector: vec2<f32>,
    frequency: f32,
    phase: f32,
    amplitude: f32,
    padding: vec3<f32>,
};

struct SeaUniform {
    count: u32,
    time: f32,
    significant_height: f32,
    deep: vec3<f32>,
    shallow: vec3<f32>,
    /// Direction towards the sun, `.w` unused. From `sky::SUN`, so the highlight
    /// and the light on the boat cannot disagree.
    sun: vec4<f32>,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> sea: SeaUniform;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> waves: array<ShaderWave, 64>;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    /// Elevation of this vertex above the mean level, m. Carried rather than
    /// recovered from `world_position.y` because the mesh follows the boat and
    /// the boat is not at `y = 0`.
    @location(2) elevation: f32,
    /// Magnitude of the surface slope. The steepness a crest reaches is what
    /// decides where foam goes, and it is already computed in the height loop.
    @location(3) steepness: f32,
};

/// Elevation and its two surface slopes, from one pass over the waves.
///
/// The slopes come out of the same loop as the height because they are the same
/// derivative: d/dx of a cos is -k_x a sin, so a second pass would cost another
/// sixty-four sines to learn nothing new. The normal is then exact for the
/// surface actually drawn rather than reconstructed from neighbouring vertices,
/// which is what keeps the lighting from showing the grid.
fn surface(plane: vec2<f32>) -> vec3<f32> {
    var height: f32 = 0.0;
    var slope: vec2<f32> = vec2<f32>(0.0, 0.0);
    for (var index: u32 = 0u; index < sea.count; index = index + 1u) {
        let wave = waves[index];
        let angle = dot(wave.wave_vector, plane) - wave.frequency * sea.time + wave.phase;
        height = height + wave.amplitude * cos(angle);
        slope = slope - wave.wave_vector * wave.amplitude * sin(angle);
    }
    return vec3<f32>(height, slope.x, slope.y);
}

/// How much of the analytic displacement survives at a given distance.
///
/// The grid that carries the near sea is fine enough to resolve the swell; the
/// ring that carries it out to the horizon is not, and cannot be — a triangle
/// tens of metres across samples a twenty metre wave as noise, and the result is
/// a band of flicker at the horizon that is entirely a sampling artefact.
///
/// So the displacement is faded out over the outer part of the ring and the
/// surface goes flat, which is also what a real sea does to the eye: past a
/// kilometre or so, wave faces are below the angular resolution of anything and
/// the sea is a plane with a texture. The shading keeps working there because it
/// depends on the normal, which stays up, and the Fresnel term, which at grazing
/// incidence is nearly one — a distant sea is a mirror, and that is correct.
fn resolved(distance: f32) -> f32 {
    return 1.0 - smoothstep(130.0, 200.0, distance);
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let world_from_local = get_world_from_local(vertex.instance_index);

    // Undisplaced world position first, because the wave phase is a function of
    // *world* position and nothing else.
    //
    // This is the one line in the file that has to be right. The mesh is a window
    // that follows the boat and snaps to its own cell size, so its local
    // coordinates slide continuously and jump by a cell every time the snap
    // advances. Take the phase from the local position and the whole sea is
    // nailed to the mesh: it travels with the boat and lurches sideways by a cell
    // width once a second — which is exactly the symptom that found this — and,
    // worse, it is then a different sea from the one `vela_core` clips the hull
    // against, which is the failure this module exists to prevent.
    //
    // Render x is north and render z is east, matching the CPU's
    // `Seaway::elevation(north, east, t)`.
    let base = mesh_position_local_to_world(world_from_local, vec4<f32>(vertex.position, 1.0));
    let plane = base.xz;

    // The level-of-detail fade, on the other hand, *is* a local question: it asks
    // how far this vertex is from the middle of the window, because that is what
    // decides how coarsely the mesh samples there.
    let evaluated = surface(plane) * resolved(length(vertex.position.xz));

    let world = vec3<f32>(base.x, base.y + evaluated.x, base.z);

    // The surface is y = zeta(x, z), so its normal is (-dzeta/dx, 1, -dzeta/dz).
    let normal = normalize(vec3<f32>(-evaluated.y, 1.0, -evaluated.z));

    var out: VertexOutput;
    out.world_position = world;
    out.world_normal = normal;
    out.elevation = evaluated.x;
    out.steepness = length(evaluated.yz);
    out.clip_position = position_world_to_clip(world);
    return out;
}

/// A cheap 2D hash, in `[0, 1)`. Dave Hoskins' `hash12`: no transcendentals, and
/// the same one `vela::atmosphere` uses so the two noises are of a piece.
fn hash(cell: vec2<f32>) -> f32 {
    var scattered = fract(vec3<f32>(cell.xyx) * 0.1031);
    scattered = scattered + dot(scattered, scattered.yzx + 33.33);
    return fract((scattered.x + scattered.y) * scattered.z);
}

/// Gradient of a smooth value-noise field, for perturbing a normal.
///
/// Returns the two partial derivatives rather than the value: nothing here needs
/// the height, only the slope it would have had. Differentiating the interpolant
/// analytically is exact and costs one extra multiply over evaluating it.
fn ripple_slope(at: vec2<f32>) -> vec2<f32> {
    let cell = floor(at);
    let offset = fract(at);
    let a = hash(cell);
    let b = hash(cell + vec2<f32>(1.0, 0.0));
    let c = hash(cell + vec2<f32>(0.0, 1.0));
    let d = hash(cell + vec2<f32>(1.0, 1.0));

    // Smoothstep weights and their derivative.
    let weight = offset * offset * (3.0 - 2.0 * offset);
    let slope = 6.0 * offset * (1.0 - offset);

    let bottom = b - a;
    let top = d - c;
    return vec2<f32>(
        slope.x * mix(bottom, top, weight.y),
        slope.y * ((c - a) + weight.x * (top - bottom)),
    );
}

/// Sub-metre surface texture, as a slope to add to the physical one.
///
/// # Why this is here and not in the engine
///
/// The realisation carries sixty spectral components from a quarter of the peak
/// frequency to four times it. For a five second sea that is a shortest
/// wavelength near three and a half metres, and the mesh samples it at one and a
/// half — so everything below a metre is absent from the physics *and* from the
/// geometry, and it is absent for good reasons in both. Extending the spectrum
/// down to capillary waves would multiply the component count, and so the step
/// cost, to model waves that a twelve metre hull integrates to nothing.
///
/// But that missing scale is most of what the eye uses to identify water. A
/// surface with only metre-scale structure reads as painted plaster, which is
/// exactly what the sea looked like before this function existed. So the fine
/// scale is added here, in the fragment stage, as a perturbation of the normal
/// with no displacement and no force behind it: a purely visual layer over a
/// physical surface.
///
/// This is a knowing dishonesty and it is the only one in the frontend, so it is
/// worth being exact about what it is not. It does not move a vertex, it does not
/// reach the buoyancy integral, and a hull sails as if it were not there. What it
/// changes is the normal a fragment reflects with — which is also, physically,
/// almost all that a centimetre-scale ripple does.
///
/// Two octaves travelling in different directions at different speeds, because
/// one octave drifting one way reads as a moving texture rather than as water.
/// Faded out with distance: beyond a hundred metres a ripple is far below a pixel
/// and keeping it only produces aliasing that no amount of sampling removes.
fn ripples(plane: vec2<f32>, time: f32, distance: f32) -> vec2<f32> {
    let fade = 1.0 - smoothstep(30.0, 160.0, distance);
    if (fade <= 0.0) {
        return vec2<f32>(0.0, 0.0);
    }

    // Roughly 0.6 m and 0.22 m features. Chosen to sit below the shortest wave
    // the realisation carries, so this adds to the spectrum rather than competing
    // with it.
    let first = ripple_slope(plane * 1.7 + vec2<f32>(0.31, -0.18) * time) * 0.055;
    let second = ripple_slope(plane * 4.6 + vec2<f32>(-0.24, 0.37) * time) * 0.026;
    return (first + second) * fade;
}

/// A specular lobe that sharpens with distance.
///
/// Seascape's trick, and the reason its water looks photographed rather than
/// rendered. The exponent is not a material property here but a stand-in for
/// what a pixel covers: a pixel of near water is a patch of one facet and shows
/// a broad sheen, while a pixel of far water averages a great many facets whose
/// normals scatter, and the average of many mirrors is a hard sparkle. Making the
/// lobe narrow with distance is a cheap way of getting the statistics right
/// without supersampling anything.
fn glitter(normal: vec3<f32>, sun: vec3<f32>, eye: vec3<f32>, distance: f32) -> f32 {
    let sharpness = mix(60.0, 2200.0, smoothstep(20.0, 500.0, distance));
    // Capped. The physical normalisation of a Phong lobe grows without bound as it
    // narrows, and at the far end of the range above that is a factor of ninety on
    // a term that bloom then spreads: a single aligned facet would white out a
    // patch of sea. Twenty is bright enough to read as a glint through the display
    // transform and not bright enough to be a hole.
    let normalisation = min((sharpness + 8.0) / (PI * 8.0), 20.0);
    return pow(max(dot(reflect(eye, normal), sun), 0.0), sharpness) * normalisation;
}

/// Foam coverage in `[0, 1]`, from the surface's own steepness and height.
///
/// Not a model of breaking. Real whitecapping starts when a crest's particle
/// velocity approaches the phase speed, which linear superposition cannot reach —
/// this surface never breaks, however hard it blows. What it does have is
/// steepness, and steepness is where a real sea breaks, so putting foam on the
/// steepest crests puts it in the right places for the right reason even though
/// the mechanism is missing.
///
/// Gated on being near a crest as well as steep, because the steepest points of a
/// linear sea are on the flanks, and foam on the flank of a wave with a bare crest
/// above it looks like a mistake. Broken up by the ripple slope the caller has
/// already computed, so the coverage has an edge rather than a gradient — foam is
/// a patch of bubbles, not an airbrush — and so that a third noise lookup is not
/// paid for the privilege.
fn foam(detail: vec2<f32>, steepness: f32, elevation: f32, significant_height: f32) -> f32 {
    if (significant_height <= 0.0) {
        return 0.0;
    }
    let crest = smoothstep(0.18 * significant_height, 0.48 * significant_height, elevation);
    let sharp = smoothstep(0.13, 0.38, steepness);
    let mottle = 0.45 + 12.0 * length(detail);
    return clamp(crest * sharp * mottle, 0.0, 1.0);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let sun = normalize(sea.sun.xyz);
    let significant_height = sea.significant_height;
    let plane = vec2<f32>(in.world_position.x, in.world_position.z);

    let to_eye = view.world_position - in.world_position;
    let distance = length(to_eye);
    let eye = -to_eye / max(distance, 1e-4);

    // The physical slope, recovered from the interpolated normal, plus the
    // cosmetic one. Renormalising matters: the interpolated normal is a mix of
    // unit vectors and is not one, and skipping it bands the gentle slopes.
    let physical = normalize(in.world_normal);
    let detail = ripples(plane, sea.time, distance);
    let normal = normalize(vec3<f32>(
        physical.x - detail.x,
        physical.y,
        physical.z - detail.y,
    ));

    // Fresnel, Schlick-shaped and capped. The cap is Seascape's and it is a
    // deliberate lie: a real air-water interface goes to a reflectance of one at
    // grazing incidence, which turns the middle distance into a mirror and hides
    // the sea's shape entirely. Holding it below three quarters keeps some
    // transmitted colour everywhere.
    let facing = clamp(1.0 - dot(normal, -eye), 0.0, 1.0);
    let fresnel = min(0.02 + 0.98 * facing * facing * facing * facing * facing, 0.72);

    // What the sky does, in the direction this facet points.
    let reflected = sky_reflection(reflect(eye, normal), sun);

    // What the water does. A wrapped diffuse rather than a Lambertian one: the
    // term stands for light that entered the surface, scattered, and came back
    // out, and that does not switch off where the geometric normal turns away.
    let wrapped = pow(dot(normal, sun) * 0.4 + 0.6, 6.0);
    var transmitted = sea.deep + sea.shallow * wrapped * 0.22;

    // Height tint: a crest carries less water above the trough line than a trough
    // carries below it, so it transmits more of the light that got in. This is the
    // green-in-the-crest cue, and it is most of what makes a sea look like a
    // volume rather than a painted sheet. Attenuated with distance because at
    // range the effect is below what the haze leaves visible.
    let attenuation = max(1.0 - distance * distance * 2.5e-6, 0.0);
    transmitted = transmitted
        + sea.shallow * clamp(in.elevation / max(significant_height, 0.2), -1.0, 1.0)
            * 0.22 * attenuation;

    var colour = mix(transmitted, reflected, fresnel);

    // The boat's shadow, sampled by hand.
    //
    // A custom material gets Bevy's view and light bind groups but none of its
    // shading, so the shadow has to be fetched explicitly. It is applied only to
    // the terms that come *from the sun* — the glitter and the transmitted
    // scatter — and not to the reflected sky, which is what a shadowed patch of
    // water actually looks like: still a mirror, just with no highlight in it.
    // Multiplying the whole colour instead would paint a grey blot on the sea.
    let view_z = dot(
        vec4<f32>(view.world_from_view[2].xyz, 0.0),
        vec4<f32>(in.world_position, 1.0) - view.world_from_view[3],
    );
    let lit = fetch_directional_shadow(
        0u,
        vec4<f32>(in.world_position, 1.0),
        normal,
        view_z,
        in.clip_position.xy,
    );

    // Glitter, gated on the sun being above the horizon rather than on nothing:
    // `reflect` happily produces a downward ray that still dots well against a
    // sun below the water, and the result is a sea that sparkles at night.
    let sun_up = clamp(sun.y * 8.0, 0.0, 1.0);
    colour = colour
        + glitter(normal, sun, eye, distance) * vec3<f32>(1.0, 0.96, 0.86) * sun_up * lit;
    // And take the sun's share out of the transmitted colour where it is blocked.
    colour = colour - sea.shallow * wrapped * 0.22 * (1.0 - fresnel) * (1.0 - lit);

    // Foam sits on top of everything: it is a surface of bubbles, not a property
    // of the water under it, and it neither reflects the sky nor transmits.
    let whitecap = foam(detail, in.steepness, in.elevation, significant_height);
    colour = mix(colour, vec3<f32>(0.92, 0.95, 0.97), whitecap * 0.9);

    // Aerial perspective, blended towards the sky *at the horizon* rather than
    // along the view ray.
    //
    // The difference is the whole horizon. The ray to a distant patch of sea
    // points slightly downward, and the atmosphere function answers a downward
    // direction with its below-horizon colour — so fading the sea along its own
    // view ray drove it towards a dark grey while the dome a pixel higher was
    // pale haze, and the two met in a hard line. What is physically wanted is the
    // light scattered into a nearly horizontal path, which is the sky colour at
    // the horizon in that bearing. Flattening the ray to `y = 0` asks for exactly
    // that, and the seam closes because both sides are then evaluating the same
    // function at the same place.
    let along_horizon = normalize(vec3<f32>(eye.x, 0.0, eye.z));
    let haze = smoothstep(600.0, 6000.0, distance);
    colour = mix(colour, sky_reflection(along_horizon, sun), haze);

    return vec4<f32>(colour, 1.0);
}
