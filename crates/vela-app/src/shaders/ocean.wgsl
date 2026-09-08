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
//
// On top of Seascape's core, each stated where it lives: a wind sea of four
// Gerstner components as slope only (`wind_sea`), crest sharpening of the
// physical slope (Horvath 2015, in `fragment`), whitecaps from the swell's
// curvature and the wind sea's Jacobian (`foam`), the bow wave (`bow_wave`) and
// the wake along the stern's recorded track (`wake`), Smith self-shadowing for
// a low sun (`sun_visible`), the boat's shadow fetched from Bevy's cascades and
// its mirror image from `crate::reflection`'s camera. None of it moves a
// vertex: the geometry is the engine's and stays so.

#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_world}
#import bevy_pbr::mesh_view_bindings::view
#import bevy_pbr::shadows::fetch_directional_shadow
#import bevy_pbr::view_transformations::position_world_to_clip
#import vela::atmosphere::sky_reflection

/// Local rather than imported: naga_oil resolves function imports reliably and
/// constants less so, and this is four characters.
const PI: f32 = 3.141592653589793;

// Thirty-two bytes an element, and the Rust side pins that with
// `the_wave_layout_matches_the_shader_stride`. Three scalar pads and not a
// `vec3`, whose sixteen-byte alignment would push the element to forty-eight.
struct ShaderWave {
    wave_vector: vec2<f32>,
    frequency: f32,
    phase: f32,
    amplitude: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
};

struct SeaUniform {
    count: u32,
    time: f32,
    significant_height: f32,
    trail_count: u32,
    deep: vec3<f32>,
    shallow: vec3<f32>,
    /// Direction towards the sun, `.w` unused. From `sky::SUN`, so the highlight
    /// and the light on the boat cannot disagree.
    sun: vec4<f32>,
    /// The boat's stern and its velocity over the ground, in the render plane:
    /// `(x, z)` of the body origin — which sits on the aft perpendicular — and
    /// `(vx, vz)` of its horizontal world velocity, m/s. The newest point of the
    /// wake, ahead of the recorded trail; see `wake`.
    motion: vec4<f32>,
    /// The hull's heading `(x, z)` in the render plane, the bow's plunge rate
    /// (m/s, positive into the water) and, in `.w`, whether the reflection
    /// texture is live.
    heading: vec4<f32>,
    /// Downwind `(x, z)`: the realisation's mean direction of travel, unit.
    /// What the wind sea runs along.
    wind: vec4<f32>,
};

/// One recorded point of the stern's track: `(x, z)` in the render plane, the
/// engine's time it was passed, and a pad.
struct TrailPoint {
    point: vec4<f32>,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> sea: SeaUniform;
// The lengths are `ocean::MAX_WAVES` and `ocean::MAX_TRAIL`, pushed as shader
// defs by the material's `specialize`, so the arrays cannot be a different size
// from the uniforms.
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> waves: array<ShaderWave, #{MAX_WAVES}>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<uniform> trail: array<TrailPoint, #{MAX_TRAIL}>;

/// The above-water scene in a mirror at the mean sea level, rendered by the
/// reflection camera into a window-shaped image over transparency; see
/// `crate::reflection`. Live only when `sea.heading.w` says so.
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var reflection: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var reflection_sampler: sampler;

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
    /// The surface's second derivatives `(zeta_xx, zeta_zz, zeta_xz)`, 1/m:
    /// its curvature, which peaks *at* a crest where the slope is zero, and is
    /// what the crest foam is gated on. See `Surface`.
    @location(4) curvature: vec3<f32>,
};

/// Everything one pass over the waves knows about the surface at a point.
struct Surface {
    height: f32,
    slope: vec2<f32>,
    /// `(zeta_xx, zeta_zz, zeta_xz)`.
    curvature: vec3<f32>,
};

/// Elevation, its two surface slopes and its three second derivatives, from
/// one pass over the waves.
///
/// The slopes come out of the same loop as the height because they are the same
/// derivative: d/dx of a cos is -k_x a sin, so a second pass would cost another
/// sixty-four sines to learn nothing new. The normal is then exact for the
/// surface actually drawn rather than reconstructed from neighbouring vertices,
/// which is what keeps the lighting from showing the grid.
///
/// The curvature is the same cosine again, times `-k_i k_j`: three multiplies
/// per wave on a value already computed. It is what Tessendorf's foam
/// criterion reads — the Jacobian of a virtual horizontal displacement
/// `lambda * grad(zeta)` is `det(I + lambda * H)` with `H` this Hessian, and
/// where it drops below one the crest would be folding over if the water were
/// allowed to move sideways (Tessendorf 2001 §4.4; Dupuy & Bruneton 2012).
/// The water here is not allowed to move sideways, because the hull is
/// clipped against the sum as it stands; but the *criterion* is still the
/// right one for where a linear crest is steepest at its top, which the
/// slope, largest on the flank, is not.
fn surface(plane: vec2<f32>) -> Surface {
    var height: f32 = 0.0;
    var slope: vec2<f32> = vec2<f32>(0.0, 0.0);
    var curvature: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    for (var index: u32 = 0u; index < sea.count; index = index + 1u) {
        let wave = waves[index];
        let k = wave.wave_vector;
        let angle = dot(k, plane) - wave.frequency * sea.time + wave.phase;
        let rise = wave.amplitude * cos(angle);
        height = height + rise;
        slope = slope - k * wave.amplitude * sin(angle);
        curvature = curvature - vec3<f32>(k.x * k.x, k.y * k.y, k.x * k.y) * rise;
    }
    return Surface(height, slope, curvature);
}

/// How much of the analytic displacement survives at a given distance.
///
/// The disc that carries the near sea is fine enough to resolve the swell; the
/// ring that carries it out to the horizon is not, and cannot be — a triangle
/// tens of metres across samples a twenty metre wave as noise, and the result is
/// a band of flicker at the horizon that is entirely a sampling artefact.
///
/// So the displacement is faded out over the outer third of the disc and the
/// surface goes flat, which is also what a real sea does to the eye: past a
/// kilometre or so, wave faces are below the angular resolution of anything and
/// the sea is a plane with a texture.
fn resolved(distance: f32) -> f32 {
    return 1.0 - smoothstep(400.0, 600.0, distance);
}

/// How much of the analytic slope survives at a given distance.
///
/// Separate from `resolved`, and further out, and the reason is what the sea
/// looked like when the two were one: a disc of shaded, moving crests inside a
/// plane of flat ones, with a visible circle where the shading stopped. The
/// shading only needs a *normal*, and a normal does not need the geometry to
/// move — the far ring's twenty-five metre cells cannot displace a wave without
/// flicker, but they can carry its slope for a while yet, because a normal that
/// aliases shades a plane a little wrong rather than tearing it. So the slope
/// runs out a kilometre and a half further than the displacement, over a band
/// wide enough that no edge is drawn, and the Fresnel term takes over from
/// there: at grazing incidence it is nearly one, a distant sea is a mirror, and
/// that is correct.
fn shaded(distance: f32) -> f32 {
    return 1.0 - smoothstep(600.0, 2000.0, distance);
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let world_from_local = get_world_from_local(vertex.instance_index);

    // Undisplaced world position first, because the wave phase is a function of
    // *world* position and nothing else.
    //
    // This is the one line in the file that has to be right. The mesh is a window
    // that follows the boat, so its local coordinates slide continuously as the
    // boat sails. Take the phase from the local position and the whole sea is
    // nailed to the mesh: it travels with the boat — which is exactly the symptom
    // that found this — and, worse, it is then a different sea from the one
    // `vela_core` clips the hull against, which is the failure this module exists
    // to prevent.
    //
    // Render x is north and render z is east, matching the CPU's
    // `Seaway::elevation(north, east, t)`.
    let base = mesh_position_local_to_world(world_from_local, vec4<f32>(vertex.position, 1.0));
    let plane = base.xz;

    // The level-of-detail fades, on the other hand, *are* a local question: they
    // ask how far this vertex is from the middle of the window, because that is
    // what decides how coarsely the mesh samples there.
    let distance = length(vertex.position.xz);
    let wave = surface(plane);
    let displacement = wave.height * resolved(distance);
    let slope = wave.slope * shaded(distance);

    let world = vec3<f32>(base.x, base.y + displacement, base.z);

    // The surface is y = zeta(x, z), so its normal is (-dzeta/dx, 1, -dzeta/dz).
    let normal = normalize(vec3<f32>(-slope.x, 1.0, -slope.y));

    var out: VertexOutput;
    out.world_position = world;
    out.world_normal = normal;
    out.elevation = displacement;
    out.steepness = length(slope);
    out.curvature = wave.curvature * shaded(distance);
    out.clip_position = position_world_to_clip(world);
    return out;
}

/// A cheap 2D vector hash, roughly uniform in `[-1, 1]²`. Dave Hoskins' `hash22`.
///
/// Two outputs rather than one because what this feeds is a *gradient* per lattice
/// point, and no transcendentals because a `cos`/`sin` pair per corner per octave
/// is four sines a fragment for a direction nobody can identify.
fn hash_gradient(cell: vec2<f32>) -> vec2<f32> {
    var scattered = fract(vec3<f32>(cell.xyx) * vec3<f32>(0.1031, 0.1030, 0.0973));
    scattered = scattered + dot(scattered, scattered.yzx + 33.33);
    return fract((scattered.xx + scattered.yz) * scattered.zy) * 2.0 - 1.0;
}

/// Slope of a smooth gradient-noise field, for perturbing a normal.
///
/// # Why gradient noise and not the derivative of value noise
///
/// This was the derivative of value noise, and it drew a grid. The reason is
/// arithmetic rather than bad luck: value noise interpolates hashed *heights* with
/// `f²(3 - 2f)`, whose derivative is `6f(1 - f)`, and that vanishes at `f = 0` and
/// `f = 1` — on every cell boundary. The slope field therefore has a line of
/// exactly zero slope along every lattice edge, and a sea lit through those lines
/// shows a tiling of squares, most visible at a grazing angle where a single cell
/// spans many pixels of screen.
///
/// Gradient noise interpolates hashed *gradients* instead, so the derivative on a
/// cell boundary is the lattice gradient there rather than zero, and there is no
/// structure to see. The quintic weight is Perlin's improved one: its second
/// derivative also vanishes at the boundaries, which the cubic's does not, and a
/// discontinuous curvature in a slope field is visible as faint creases along the
/// same lines this is meant to remove.
///
/// Returns only the two partials. Nothing here wants the height — the wave sum
/// already owns the surface — and the value would cost an extra dot product.
fn ripple_slope(at: vec2<f32>) -> vec2<f32> {
    let cell = floor(at);
    let offset = fract(at);

    let g00 = hash_gradient(cell);
    let g10 = hash_gradient(cell + vec2<f32>(1.0, 0.0));
    let g01 = hash_gradient(cell + vec2<f32>(0.0, 1.0));
    let g11 = hash_gradient(cell + vec2<f32>(1.0, 1.0));

    // Each corner's contribution is its gradient dotted with the offset from it.
    let v00 = dot(g00, offset);
    let v10 = dot(g10, offset - vec2<f32>(1.0, 0.0));
    let v01 = dot(g01, offset - vec2<f32>(0.0, 1.0));
    let v11 = dot(g11, offset - vec2<f32>(1.0, 1.0));

    // Quintic weight and its derivative.
    let weight = offset * offset * offset * (offset * (offset * 6.0 - 15.0) + 10.0);
    let slope = 30.0 * offset * offset * (offset * (offset - 2.0) + 1.0);

    // Bilinear blend of the four contributions, differentiated by hand: the weights
    // move as well as the values, and dropping the second term is what makes a
    // hand-rolled gradient noise look subtly wrong.
    let bottom = mix(v00, v10, weight.x);
    let top = mix(v01, v11, weight.x);
    let d_bottom_dx = mix(g00.x, g10.x, weight.x) + slope.x * (v10 - v00);
    let d_top_dx = mix(g01.x, g11.x, weight.x) + slope.x * (v11 - v01);
    let d_bottom_dy = mix(g00.y, g10.y, weight.x);
    let d_top_dy = mix(g01.y, g11.y, weight.x);

    return vec2<f32>(
        mix(d_bottom_dx, d_top_dx, weight.y),
        mix(d_bottom_dy, d_top_dy, weight.y) + slope.y * (top - bottom),
    );
}

/// Height of the same gradient-noise field, in about `[-0.7, 0.7]`.
///
/// For modulating something slowly across the sea — where the wind sea is
/// blowing and where it is not — rather than for a normal, which is why this
/// one returns the value and `ripple_slope` the partials: each caller wants
/// one of them and neither wants to pay for both.
fn ripple_value(at: vec2<f32>) -> f32 {
    let cell = floor(at);
    let offset = fract(at);
    let v00 = dot(hash_gradient(cell), offset);
    let v10 = dot(hash_gradient(cell + vec2<f32>(1.0, 0.0)), offset - vec2<f32>(1.0, 0.0));
    let v01 = dot(hash_gradient(cell + vec2<f32>(0.0, 1.0)), offset - vec2<f32>(0.0, 1.0));
    let v11 = dot(hash_gradient(cell + vec2<f32>(1.0, 1.0)), offset - vec2<f32>(1.0, 1.0));
    let weight = offset * offset * offset * (offset * (offset * 6.0 - 15.0) + 10.0);
    return mix(mix(v00, v10, weight.x), mix(v01, v11, weight.x), weight.y);
}

/// Sub-metre surface texture, as a slope to add to the physical one.
///
/// # Why this is here and not in the engine
///
/// The realisation carries sixty spectral components from a quarter of the peak
/// frequency to four times it. For a five second sea that is a shortest
/// wavelength near three and a half metres, and the mesh's cells are a metre
/// across near the boat — so everything below a metre is absent from the physics
/// *and* from the geometry, and it is absent for good reasons in both. Extending
/// the spectrum down to capillary waves would multiply the component count, and
/// so the step cost, to model waves that a twelve metre hull integrates to
/// nothing.
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
/// # Why the scale follows the distance
///
/// The obvious thing is to fade this out with range, and that was the first
/// attempt: beyond a hundred metres a ripple is below a pixel, and asking for it
/// there is asking for aliasing. But the displacement fades out by six hundred
/// metres too — a mesh with metre-scale triangles cannot carry a twenty metre wave
/// without shimmering — so fading both left the sea a *perfectly flat mirror* from
/// there to the haze. That is most of the screen, and it is the
/// strange smooth band a viewer notices immediately: water does not stop having
/// texture because it is far away.
///
/// So the feature size grows with distance instead, keeping the ripple roughly
/// constant in *screen* space. That is what a mip level does, done by hand:
/// aliasing comes from detail below a pixel, and the cure is to stop asking for
/// detail below a pixel rather than to stop asking for detail.
///
/// Two octaves travelling in different directions at different speeds, because one
/// octave drifting one way reads as a moving texture rather than as water.
fn ripples(plane: vec2<f32>, time: f32, distance: f32) -> vec2<f32> {
    // One at the reference range, coarsening beyond it. Never below: the near
    // water is where the detail belongs at its true size.
    let coarsen = max(distance / 45.0, 1.0);

    // Roughly 0.6 m and 0.22 m features at the reference range, both below the
    // shortest wave the realisation carries, so this adds to the spectrum rather
    // than competing with it.
    //
    // The amplitudes are deliberately *not* rescaled with `coarsen`. What
    // `ripple_slope` returns is a derivative with respect to its own argument, and
    // the magnitude of that does not depend on how the argument was scaled — so
    // widening the feature while keeping the amplitude keeps the slope the light
    // sees. Compensating "to preserve the slope", which this did first,
    // multiplies it by a hundred and seventy at the horizon and turns the far sea
    // into razor wire.
    let first = ripple_slope((plane * 1.7 + vec2<f32>(0.31, -0.18) * time) / coarsen) * 0.016;
    let second = ripple_slope((plane * 4.6 + vec2<f32>(-0.24, 0.37) * time) / coarsen) * 0.008;
    return first + second;
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

/// The wind sea: what a fragment adds to the surface's slope, and how close
/// that surface is to folding.
struct WindSea {
    slope: vec2<f32>,
    /// `1 - J` of the layer's own virtual displacement, clamped to `[0, 1]`:
    /// zero on a smooth surface, rising to one where a Gerstner crest of this
    /// steepness would be cusping. Where the whitecaps are.
    fold: f32,
};

/// The short wind-driven waves the realisation does not carry, as a slope.
///
/// The physical sea is a six second swell: its shortest component is three and
/// a half metres long, and a twelve metre hull integrates anything shorter to
/// nothing, which is why the physics stops there. The eye does not. A sea in
/// ten knots of wind is covered in half-metre to three-metre wind waves with
/// short steep crests, and it is those — not the swell — that carry the
/// whitecaps and the texture a viewer reads as "water". Without them the
/// surface here read as rolling plaster, and that was the complaint.
///
/// So four Gerstner components, spread thirty degrees either side of the
/// wind, contribute their **slope** to the normal and nothing to the geometry:
/// the same knowing dishonesty as the ripples, one scale up, stated the same
/// way. Gerstner rather than sine because the wind sea's crests are sharp and
/// its troughs flat, and a trochoid's slope has exactly that asymmetry.
/// Their steepness `kA` scales with the sea state, and where two components
/// add they pass the cusping limit — which is where the Jacobian
/// `J = (1 - sum k Dx² A cos)(1 - sum k Dz² A cos) - (sum k Dx Dz A cos)²`
/// of the layer's own (virtual) horizontal displacement goes below one,
/// Tessendorf's criterion for a crest breaking and the gate the whitecaps use.
///
/// Four coherent components interfere, and a sea that interferes coherently
/// over a kilometre is corduroy - the first version was, and a viewer read
/// it as a pattern rather than as water. A real wind sea is patchy on the
/// scale of tens of metres: gusts and lulls lay cat's paws on it, and the
/// wind waves under one gust are not in phase with those under the next. A
/// slow noise across the plane, drifting downwind, sets each patch's
/// amplitude, and that is what breaks the bands up.
///
/// Faded with distance the way the ripples are, and for the same reason: past
/// a couple of hundred metres a two metre wave is a pixel, and asking for its
/// slope is asking for shimmer. Wavelengths and phases are co-prime-ish so the
/// four do not beat visibly.
fn wind_sea(plane: vec2<f32>, time: f32, distance: f32) -> WindSea {
    let fade = 1.0 - smoothstep(60.0, 350.0, distance);
    if (fade <= 0.0) {
        return WindSea(vec2<f32>(0.0, 0.0), 0.0);
    }
    // The wind is where the physical sea's mean heading comes from: the
    // realisation travels *towards* the boat from the wind's direction, so its
    // mean wave vector points downwind, and the wind sea runs the same way.
    // Computed once on the CPU from the realisation and carried in the
    // uniform; a sum over sixty components per fragment is not a direction.
    let downwind = sea.wind.xy;
    let across = vec2<f32>(-downwind.y, downwind.x);

    // Cat's paws: patches thirty metres or so across, drifting downwind at a
    // walking pace, between a quarter and full strength.
    let paw = ripple_value(plane / 32.0 + downwind * time * 0.04);
    let gust = mix(0.25, 1.0, smoothstep(-0.4, 0.4, paw));

    // Four components between one and three metres, on bearings within
    // thirty degrees of the wind: the short end of a fetch-limited spectrum,
    // below what the realisation carries and above the ripple noise. Their
    // steepness `ak` is where the layer's whole look lives - at 0.03 they are
    // barely there, at 0.15 they are near the breaking limit where they add
    // and the Jacobian caps them; the sea's state sets it in between.
    var wavelengths = array<f32, 4>(1.3, 1.9, 2.4, 3.1);
    var bearings = array<f32, 4>(-0.45, 0.2, -0.15, 0.5);
    var offsets = array<f32, 4>(0.0, 1.9, 4.1, 2.7);
    let steepness = mix(0.03, 0.15, smoothstep(0.0, 1.5, sea.significant_height)) * gust;

    var slope = vec2<f32>(0.0, 0.0);
    var jxx = 1.0;
    var jzz = 1.0;
    var jxz = 0.0;
    for (var index: u32 = 0u; index < 4u; index = index + 1u) {
        let wavelength = wavelengths[index];
        let k = 2.0 * PI / wavelength;
        // Deep-water dispersion, the same law the physical sea obeys.
        let omega = sqrt(9.81 * k);
        let amplitude = steepness / k;
        let direction = downwind * cos(bearings[index]) + across * sin(bearings[index]);
        let phase = k * dot(direction, plane) - omega * time + offsets[index];
        let s = sin(phase);
        let c = cos(phase);
        // Height `A cos(phase)`, the same form as the physical sea so the two
        // layers move alike; its slope is `-k D A sin(phase)`.
        slope = slope - direction * k * amplitude * s;
        // The trochoid's horizontal displacement is `-D A sin(phase)`, so the
        // Jacobian of the map it would have made, with unit choppiness, has
        // these derivatives.
        jxx = jxx - k * direction.x * direction.x * amplitude * c;
        jzz = jzz - k * direction.y * direction.y * amplitude * c;
        jxz = jxz - k * direction.x * direction.y * amplitude * c;
    }
    let jacobian = jxx * jzz - jxz * jxz;
    // Gated where the fold is real rather than where it is merely positive:
    // caps on the sharpest crests, none on the rest, and the whole thing
    // scaled with the sea so a calm has none.
    let fold = smoothstep(0.4, 0.85, 1.0 - jacobian) * fade;
    return WindSea(slope * fade, fold);
}

/// Foam coverage in `[0, 1]`: whitecaps on the wind sea's folding crests, and
/// on the swell's crests where the swell itself is steep enough to.
///
/// Not a model of breaking. Real whitecapping starts when a crest's particle
/// velocity approaches the phase speed, which linear superposition cannot reach —
/// this surface never breaks, however hard it blows. What it does have is the
/// *criterion* for breaking, on two scales.
///
/// The swell's is its curvature: Tessendorf's `J = det(I + lambda H)` with `H`
/// the Hessian the vertex loop carries and `lambda` the virtual choppiness a
/// Gerstner sea of the same spectrum would have. It goes below one where a
/// crest would be folding, which for a half-metre swell is nowhere and for a
/// three-metre one is on every steep crest — the earlier gate on slope
/// magnitude fired on the flanks instead, where a linear wave is steepest,
/// and needed the height to say "no, the crest", which the curvature says by
/// itself. The wind sea's is its own Jacobian, computed with it. Both are
/// broken up by the ripple slope the caller has already computed, so the
/// coverage has an edge rather than a gradient — foam is a patch of bubbles,
/// not an airbrush — and so that a third noise lookup is not paid for the
/// privilege.
fn foam(detail: vec2<f32>, curvature: vec3<f32>, wind_fold: f32, significant_height: f32) -> f32 {
    if (significant_height <= 0.0) {
        return 0.0;
    }
    // Virtual choppiness, m: what a trochoidal sea of this height would move
    // its water sideways by. Scaled with the sea so the criterion is the
    // same shape in a chop and in a gale.
    let lambda = 1.6 * max(significant_height, 0.5);
    let jxx = 1.0 + lambda * curvature.x;
    let jzz = 1.0 + lambda * curvature.y;
    let jxz = lambda * curvature.z;
    let swell_fold = smoothstep(0.35, 0.8, 1.0 - (jxx * jzz - jxz * jxz));
    let mottle = 0.45 + 12.0 * length(detail);
    return clamp(max(swell_fold, wind_fold) * mottle, 0.0, 1.0);
}

/// Foam of the bow wave in `[0, 1]`.
///
/// A hull under way pushes a wave up at its stem and the crest of it breaks
/// along the forward third of the waterline, brighter the faster the boat goes
/// and much brighter when the bow drives down into an oncoming crest — which
/// is the spray a viewer asked for, drawn as a surface of bubbles because a
/// fragment shader has no particles to throw. Placed from the stern along the
/// hull's *heading*, not its track, because the bow wave belongs to the hull.
/// Two lobes, one each side of the stem, widening aft and fading by the
/// midship section.
fn bow_wave(plane: vec2<f32>, detail: vec2<f32>, hull: vec2<f32>) -> f32 {
    let speed = length(sea.motion.zw);
    if (speed < 0.5) {
        return 0.0;
    }
    let heading = sea.heading.xy;
    let offset = plane - sea.motion.xy;
    let along = dot(offset, heading);
    let across = abs(offset.x * heading.y - offset.y * heading.x);
    // From the stem back to amidships, as a fraction.
    let aft = (hull.x - along) / (0.5 * hull.x);
    if (aft < 0.0 || aft > 1.0) {
        return 0.0;
    }
    // Where the crest sits: just outboard of the waterline, which flares
    // from nothing at the stem to the full half-beam amidships roughly as a
    // square root, plus a margin the plunge widens. Inboard of that the water
    // is under the hull, where foam drawn on the sea is foam nobody sees -
    // which is where the first version put it.
    let plunge = clamp(sea.heading.z, 0.0, 3.0);
    let half_beam = hull.y * sqrt(aft);
    let crest_at = half_beam + 0.6 + 0.6 * plunge;
    let band = 1.0 + 0.5 * plunge;
    let lobe = 1.0 - smoothstep(0.0, band, abs(across - crest_at));
    // Brighter with speed, and along the forward half of the run; the plunge
    // throws it well past that.
    let strength = smoothstep(0.5, 2.5, speed) * (1.0 - aft * (0.8 - 0.3 * plunge));
    let mottle = 0.5 + 8.0 * length(detail);
    return clamp(lobe * strength * mottle, 0.0, 1.0);
}

/// How much of the sun a facet sees past its neighbours, in `[0, 1]`.
///
/// Waves shadow each other, and a shadow map of the sea cannot afford to say
/// so: it would put forty thousand vertices of sixty waves each through a
/// second pass per cascade. The statistics say it instead. Smith's shadowing
/// term for a Gaussian slope distribution (Bruneton, Neyret & Holzschuch
/// 2010, §4; Ross, Dion & St-Germain 2005) gives the fraction of a surface of
/// slope variance `sigma²` that is lit from an elevation `mu`:
/// `1 / (1 + Lambda)`, with
/// `Lambda = (sqrt(2 sigma² / pi) / mu * exp(-mu² / 2 sigma²) - erfc(mu / sqrt(2 sigma²))) / 2`.
/// Cox and Munk's variance for the wind here, plus the swell's own slope
/// squared. Low sun over a rough sea goes dim in the troughs, which is the
/// self-shadowing a viewer was missing, for one exponential a fragment.
fn sun_visible(sun: vec3<f32>, steepness: f32) -> f32 {
    let mu = max(sun.y, 0.02);
    // Cox & Munk, ten knots of wind, plus what the swell adds.
    let variance = 0.003 + 0.00512 * 5.0 + steepness * steepness;
    let s = sqrt(variance);
    let x = mu / (s * 1.4142135);
    // Abramowitz-Stegun 7.1.26 for erfc, good to 1.5e-7.
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let erfc = poly * exp(-x * x);
    let lambda = 0.5 * (s * 0.7978846 / mu * exp(-mu * mu / (2.0 * variance)) - erfc);
    return 1.0 / (1.0 + max(lambda, 0.0));
}

/// Foam coverage of the boat's wake in `[0, 1]`.
///
/// This is what tells the eye the boat is moving. A hull with no trail sits on
/// the water like a decal, and a viewer reads it as stationary even when the
/// crests are going past at ten metres a second — which was the complaint that
/// put this here. The wake is the one thing in the scene that is *about* the
/// boat's motion relative to the water, so it carries the whole sense of it.
///
/// # It stays on the water
///
/// The wake is drawn along the stern's **recorded track** — the polyline the
/// material keeps of where the transom has been and when — with one more
/// segment from the newest recorded point to where the stern is now. The first
/// version drew a straight band from the current position along the current
/// velocity, and it swung with every yaw of the hull like a searchlight: a
/// viewer saw at once that it was attached to the boat, not left behind on the
/// sea. Water that has been passed does not move because the boat turns.
///
/// For a fragment, each segment is asked how far the point is across it and
/// how far along, and the age of the water there is interpolated from the
/// times the two ends were passed. The nearest segment wins. The loop is over
/// every recorded point, so it is gated first on being within reach of the
/// newest one at all, which almost all of the sea is not.
///
/// # What it draws
///
/// Not a model of the wake. The hull's turbulent wake is a band of aerated
/// water that widens by entrainment and fades as the bubbles rise; both are the
/// simplest thing that behaves that way — a width growing linearly with age, a
/// strength decaying exponentially with it. The edge is not a line: a real
/// wake's margin is ragged, so the half-width is modulated by a low-frequency
/// slope field along the track and the coverage inside is streaked by the same
/// ripple slope the whitecaps use. A perfect trapezoid, which is what came out
/// before either, reads as a decal however well it is placed.
fn wake(plane: vec2<f32>, detail: vec2<f32>) -> f32 {
    let count = sea.trail_count;
    let stern = sea.motion.xy;
    let speed = length(sea.motion.zw);
    // Nothing recorded, or nothing near: the reach is the longest a wake can
    // be before it has faded, and the newest point is where it is strongest.
    if (count == 0u || distance(plane, stern) > 80.0) {
        return 0.0;
    }

    var best = 0.0;
    for (var index: u32 = 0u; index < count; index = index + 1u) {
        let older = trail[index].point;
        // The segment ends at the next recorded point, or at the stern itself
        // for the newest one, which is where the wake is being made now.
        var newer: vec4<f32>;
        if (index + 1u < count) {
            newer = trail[index + 1u].point;
        } else {
            newer = vec4<f32>(stern, sea.time, 0.0);
        }
        let span = newer.xy - older.xy;
        let span_length = length(span);
        if (span_length < 1e-3) {
            continue;
        }
        let direction = span / span_length;
        let offset = plane - older.xy;
        let along = clamp(dot(offset, direction), 0.0, span_length);
        // Distance to the *segment*, not to its line: measured to the line, a
        // point ahead of the stern on the track's extension is "on" the newest
        // segment, and the wake ran out in front of the bow.
        let across = distance(plane, older.xy + direction * along);
        let age = sea.time - mix(older.z, newer.z, along / span_length);

        // Half a transom's width when it was made, widening by entrainment at a
        // tenth of a metre a second, the margin ragged by a slow noise along
        // the track — clamped, because the slope of a noise field is not
        // bounded and unclamped it threw blobs of foam twenty metres off the
        // track; mostly gone in five seconds, which at five knots is a boat
        // length of clear trail and a faint one for a few more.
        let ragged = 0.8 + 0.3 * clamp(ripple_slope(plane * 0.18).x, -1.0, 1.0);
        let half_width = (0.9 + 0.1 * age) * ragged;
        let strength = 0.55 * exp(-age / 5.0);
        let inside = 1.0 - smoothstep(0.45 * half_width, half_width, across);
        best = max(best, strength * inside);
    }

    // Never solid: a wake is aerated water, not paint, and even at the transom
    // most of what shows is streaks. A boat at rest leaves nothing.
    let mottle = min(0.5 + 6.0 * length(detail), 1.0);
    return best * mottle * smoothstep(0.2, 1.0, speed);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let sun = normalize(sea.sun.xyz);
    let significant_height = sea.significant_height;
    let plane = vec2<f32>(in.world_position.x, in.world_position.z);

    let to_eye = view.world_position - in.world_position;
    let distance = length(to_eye);
    let eye = -to_eye / max(distance, 1e-4);

    // The physical slope, recovered from the interpolated normal, then
    // sharpened, then the two cosmetic layers on top. Renormalising matters:
    // the interpolated normal is a mix of unit vectors and is not one, and
    // skipping it bands the gentle slopes.
    //
    // Sharpening is Horvath's peak enhancement (DigiPro 2015 §5): the slope
    // is scaled up where it is already large, so the shading answers as if the
    // crests were sharper than a sum of cosines can make them — without a
    // vertex moving, which the hull's clip forbids. A linear sea's crests are
    // as round as its troughs; a real sea's are not, and the eye knows.
    let physical = normalize(in.world_normal);
    var slope = vec2<f32>(physical.x, physical.z) / max(physical.y, 1e-3);
    slope = slope * (1.0 + 1.0 * smoothstep(0.10, 0.35, length(slope)));
    let wind = wind_sea(plane, sea.time, distance);
    let detail = ripples(plane, sea.time, distance);
    let normal = normalize(vec3<f32>(
        slope.x - wind.slope.x - detail.x,
        1.0,
        slope.y - wind.slope.y - detail.y,
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

    // And what the boat does. The reflection camera has drawn the above-water
    // scene mirrored in the mean sea level with this camera's own projection,
    // so the mirror image of the boat is at this fragment's own screen
    // position on a flat sea, and on a rough one it is displaced by the slope:
    // a facet tilted by `s` reflects a ray tilted by `2s`, which at the boat's
    // distance is roughly that many metres sideways on the screen. The image
    // is over transparency, so where there is no boat there is nothing to
    // add, and the blend is by its alpha. Faded out past the range at which a
    // hull is a few pixels, where the sky alone is the right answer anyway.
    //
    // Composited below, after the Fresnel mix, at the Fresnel weight and no
    // more. A sunlit sail is ten times brighter than the sky behind it, so
    // even the two or three per cent a viewer looking down at the water gets
    // is plainly visible against the transmitted colour - which is what a
    // real hull's image on dark water is. A floor under the weight was tried
    // and painted the reflection whiter than the boat.
    var boat = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (sea.heading.w > 0.5) {
        let uv = in.clip_position.xy / view.viewport.zw;
        let tilt = vec2<f32>(normal.x, normal.z) * 2.0;
        let shift = tilt * 0.12 * (1.0 - smoothstep(0.0, 400.0, distance));
        // Level zero explicitly: the branch is uniform, but a sampler that
        // wants derivatives inside any branch is a portability argument.
        boat = textureSampleLevel(reflection, reflection_sampler, uv + shift, 0.0);
        boat.a = boat.a * (1.0 - smoothstep(300.0, 600.0, distance));
    }

    // How much of the sun this facet sees past its neighbours: the statistical
    // self-shadowing of a rough surface, which a shadow map of the sea could
    // not afford to compute. See `sun_visible`.
    let seen = sun_visible(sun, in.steepness);

    // What the water does. A wrapped diffuse rather than a Lambertian one: the
    // term stands for light that entered the surface, scattered, and came back
    // out, and that does not switch off where the geometric normal turns away.
    let wrapped = pow(dot(normal, sun) * 0.4 + 0.6, 6.0) * seen;
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
    colour = mix(colour, boat.rgb, boat.a * fresnel);

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
        + glitter(normal, sun, eye, distance) * vec3<f32>(1.0, 0.96, 0.86) * sun_up * lit * seen;
    // And take the sun's share out of the transmitted colour where it is blocked.
    colour = colour - sea.shallow * wrapped * 0.22 * (1.0 - fresnel) * (1.0 - lit);

    // Foam sits on top of everything: it is a surface of bubbles, not a property
    // of the water under it, and it neither reflects the sky nor transmits.
    // Whitecaps, the bow wave and the wake are all foam, and they combine as
    // coverage rather than adding, so a whitecap crossing the trail is not
    // brighter than white. Foam is lit like a diffuse white surface: it takes
    // the shadow the water takes, which is what keeps a wake in the hull's
    // shadow from glowing - but only in part, because foam is lit by the whole
    // sky as much as by the sun, and the bow wave is mostly in the hull's
    // shadow, which at the sun's share painted it grey.
    let whitecap = foam(detail, in.curvature, wind.fold, significant_height);
    let bow = bow_wave(plane, detail, sea.wind.zw);
    let trail = wake(plane, detail);
    let bubbles = max(max(whitecap, bow), trail);
    let foam_colour = vec3<f32>(0.92, 0.95, 0.97) * (0.6 + 0.4 * lit);
    colour = mix(colour, foam_colour, bubbles * 0.9);

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
