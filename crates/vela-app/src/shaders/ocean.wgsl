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
// physical slope (Horvath 2015, in `fragment`), whitecaps where the swell's
// Jacobian says a parcel is folding - remembered for a few seconds on the
// parcel (`vertex`) - and where the wind sea's own says so, the bow wave
// (`bow_wave`) and the wake along the stern's recorded track (`wake`), all
// drawn as a lit, textured layer (`foam`); light through thin crests; Smith
// self-shadowing for a low sun (`sun_visible`); the boat's shadow fetched from
// Bevy's cascades and its mirror image from `crate::reflection`'s camera. The
// only geometry is the engine's: the choppy displacement is the physics' own.

#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_world}
#import bevy_pbr::mesh_view_bindings::view
#import bevy_pbr::shadows::fetch_directional_shadow
#import bevy_pbr::view_transformations::position_world_to_clip
#import vela::atmosphere::sky_reflection

/// Local rather than imported: naga_oil resolves function imports reliably and
/// constants less so, and this is four characters.
const PI: f32 = 3.141592653589793;

// Thirty-two bytes an element, and the Rust side pins that with
// `the_wave_layout_matches_the_shader_stride`. Three scalars after the
// amplitude and not a `vec3`, whose sixteen-byte alignment would push the
// element to forty-eight. Two of them carry the cosine and sine of the phase
// this wave advances by over the foam memory's interval - see `surface` for
// what that buys - and the third is a pad.
struct ShaderWave {
    wave_vector: vec2<f32>,
    frequency: f32,
    phase: f32,
    amplitude: f32,
    /// `cos(omega * MEMORY)` and `sin(omega * MEMORY)`.
    memory_cos: f32,
    memory_sin: f32,
    pad: f32,
};

struct SeaUniform {
    count: u32,
    time: f32,
    significant_height: f32,
    trail_count: u32,
    deep: vec3<f32>,
    shallow: vec3<f32>,
    /// Colour of light scattered *through* a thin crest, linear RGB; see the
    /// sub-surface term in `fragment`.
    scatter: vec3<f32>,
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
    wind: vec4<f32>,
    /// Choppiness `lambda` of the realisation: how far parcels move sideways,
    /// as a fraction of the Gerstner orbit. The physics clips the hull with
    /// the same number; see `surface`.
    choppiness: f32,
    /// The wake's reach in the render plane, `(min x, min z, max x, max z)`:
    /// the recorded track grown by the widest the wake gets. See `wake`.
    trail_bounds: vec4<f32>,
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
    /// Magnitude of the surface slope: what the statistical self-shadowing
    /// reads, and it is already computed in the height loop.
    @location(3) steepness: f32,
    /// How much foam the swell has put on this parcel of water, `[0, 1]`:
    /// the crest is folding now, or folded in the last few seconds and the
    /// foam has not yet dissolved. See `surface` for how it is remembered.
    @location(4) foam: f32,
    /// How much thinner this crest is than the water around it, `[0, 1]`:
    /// `1 - J` clamped, which is where the crest is being squeezed sideways
    /// and light gets through. What the sub-surface term reads.
    @location(5) thinness: f32,
};

/// Everything one pass over the waves knows about the surface at a parcel.
struct Surface {
    height: f32,
    /// Horizontal displacement `D` of the parcel, m, in the render plane.
    shift: vec2<f32>,
    /// `d zeta / d q`.
    slope: vec2<f32>,
    /// The derivatives of the displacement, `(dDx/dx, dDz/dz, dDx/dz)`.
    strain: vec3<f32>,
    /// The same strain one and two memory intervals ago; see `surface`.
    strain_before: vec3<f32>,
    strain_earlier: vec3<f32>,
};

/// The determinant of the map from rest position to displaced position:
/// `J = (1 + dDx/dx)(1 + dDz/dz) - (dDx/dz)^2`. One where the water has not
/// moved sideways, less than one where parcels have been squeezed together -
/// a crest being sharpened - and below zero where they would have crossed,
/// which is a wave folding over. Tessendorf 2001 §4.4.
fn jacobian(strain: vec3<f32>) -> f32 {
    return (1.0 + strain.x) * (1.0 + strain.y) - strain.z * strain.z;
}

/// The parcel at rest position `plane`, now, from one pass over the waves:
/// where it is, how high, and how the water around it is strained - now and
/// at two moments in the past.
///
/// This is `vela_core::seaway::Seaway`'s Lagrangian construction, transcribed:
///
/// ```text
/// X(q, t) = q - lambda * sum a_i d_i sin(theta_i)
/// zeta(q, t) =        sum a_i     cos(theta_i)
/// ```
///
/// with `lambda` the choppiness (`sea.choppiness`), the same number the physics
/// clips the hull with. A sum of cosines has crests as round as its troughs; a
/// real sea does not, because the water in a wave moves in circles, and this
/// horizontal term is that motion. It pinches the crests and flattens the
/// troughs, which is the difference between plaster and water. The physics
/// evaluates the same surface the other way round - which parcel is under a
/// world point - and `the_choppy_surface_is_the_one_the_parcels_draw` in
/// `vela_core` is what keeps the two sides on one surface.
///
/// The slopes and the strain come out of the same loop because they are the
/// same sine and cosine: d/dq of `a cos` is `-a k sin`, and d/dq of
/// `-lambda a d sin` is `-lambda a k d d cos`. The normal is then exact for
/// the surface actually drawn rather than reconstructed from neighbouring
/// vertices, which is what keeps the lighting from showing the grid.
///
/// The past strains are for the foam's memory (see `vertex`), and they come
/// out of the same sine and cosine too: a wave's phase advances by
/// `omega * MEMORY` over the interval, and `cos(theta + a)` is
/// `cos theta cos a - sin theta sin a` with `cos a` and `sin a` carried in the
/// wave's own record, the double angle giving the second interval. Eight
/// multiplies a wave instead of two more sines - the first version ran the
/// whole loop three times, and the vertex stage was a third of the frame.
fn surface(plane: vec2<f32>, time: f32) -> Surface {
    var height: f32 = 0.0;
    var shift: vec2<f32> = vec2<f32>(0.0, 0.0);
    var slope: vec2<f32> = vec2<f32>(0.0, 0.0);
    var strain: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    var before: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    var earlier: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    let lambda = sea.choppiness;
    for (var index: u32 = 0u; index < sea.count; index = index + 1u) {
        let wave = waves[index];
        let k = wave.wave_vector;
        let angle = dot(k, plane) - wave.frequency * time + wave.phase;
        let c = wave.amplitude * cos(angle);
        let s = wave.amplitude * sin(angle);
        height = height + c;
        slope = slope - k * s;
        // `k` is `|k| d`, so `d s = k s / |k|` and `|k| d d c = k k c / |k|`.
        let magnitude = max(length(k), 1e-6);
        shift = shift - lambda * k * s / magnitude;
        let kk = lambda * vec3<f32>(k.x * k.x, k.y * k.y, k.x * k.y) / magnitude;
        strain = strain - kk * c;
        // One interval back the phase was larger by `omega * MEMORY`.
        let ca = wave.memory_cos;
        let sa = wave.memory_sin;
        let c1 = c * ca - s * sa;
        let c2 = c * (2.0 * ca * ca - 1.0) - s * (2.0 * sa * ca);
        before = before - kk * c1;
        earlier = earlier - kk * c2;
    }
    return Surface(height, shift, slope, strain, before, earlier);
}

/// How much a parcel is folding, `[0, 1]`, from its Jacobian.
///
/// Foam appears well before the fold is complete: a crest whose water has
/// been squeezed by a third is already breaking at the top, and the caps on
/// a two-metre sea are on most crests. A calm never gets near the lower
/// edge, so it keeps a clean surface without a gate of its own.
fn folding(strain: vec3<f32>) -> f32 {
    return smoothstep(0.06, 0.32, 1.0 - jacobian(strain));
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
    // `Seaway::elevation(north, east, t)`. The mesh vertex is the parcel's
    // *rest* position; where it is drawn is where the parcel has gone.
    let base = mesh_position_local_to_world(world_from_local, vec4<f32>(vertex.position, 1.0));
    let plane = base.xz;

    // The level-of-detail fades, on the other hand, *are* a local question: they
    // ask how far this vertex is from the middle of the window, because that is
    // what decides how coarsely the mesh samples there.
    let distance = length(vertex.position.xz);
    let resolve = resolved(distance);
    let shade = shaded(distance);
    let wave = surface(plane, sea.time);
    let displacement = wave.height * resolve;
    let shift = wave.shift * resolve;
    let slope = wave.slope * shade;
    let strain = wave.strain * shade;

    let world = vec3<f32>(base.x + shift.x, base.y + displacement, base.z + shift.y);

    // The normal of a displaced surface is the cross product of its two
    // tangents, `(1 + dDx/dx, dzeta/dx, dDz/dx)` and `(dDx/dz, dzeta/dz, 1 + dDz/dz)`,
    // which reduces to `(-dzeta/dx, 1, -dzeta/dz)` when nothing moves sideways.
    let along_x = vec3<f32>(1.0 + strain.x, slope.x, strain.z);
    let along_z = vec3<f32>(strain.z, slope.y, 1.0 + strain.y);
    let normal = normalize(cross(along_z, along_x));

    // Foam is remembered by the water, not by a buffer: the same parcel is
    // asked whether it was folding a moment ago, and a moment before that,
    // and whatever it answers is faded by the time since. The parcel is the
    // right thing to ask because foam rides on the water - it moves with the
    // orbit, and a Lagrangian sample follows the orbit for free. Three samples
    // over four seconds is the persistence a cap has in a fresh breeze: bright
    // as it breaks, a patch for a few seconds, gone in five (Sea of Thieves
    // keeps a simulated accumulation buffer for the same effect; this is the
    // analytic sea's way of having one without a render pass).
    var foam = folding(strain);
    foam = max(foam, 0.65 * folding(wave.strain_before * shade));
    foam = max(foam, 0.35 * folding(wave.strain_earlier * shade));

    var out: VertexOutput;
    out.world_position = world;
    out.world_normal = normal;
    out.elevation = displacement;
    out.steepness = length(slope);
    out.foam = foam;
    out.thinness = clamp(1.0 - jacobian(strain), 0.0, 1.0);
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
/// So four Gerstner components, spread forty-five degrees either side of the
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
    // forty-five degrees of the wind: the short end of a fetch-limited spectrum,
    // below what the realisation carries and above the ripple noise. Their
    // steepness `ak` is where the layer's whole look lives - at 0.03 they are
    // barely there, at 0.15 they are near the breaking limit where they add
    // and the Jacobian caps them; the sea's state sets it in between.
    var wavelengths = array<f32, 4>(1.3, 1.9, 2.4, 3.1);
    var bearings = array<f32, 4>(-0.7, 0.3, -0.2, 0.8);
    var offsets = array<f32, 4>(0.0, 1.9, 4.1, 2.7);
    let steepness = mix(0.02, 0.09, smoothstep(0.0, 1.5, sea.significant_height)) * gust;

    // Short crests. A component with one phase across the whole plane is a
    // crest a kilometre long, and four of them interfere into corduroy. A
    // real wind wave's crest is a wavelength or two long before it hands over
    // to its neighbour, which is a phase that wanders across the plane. One
    // slow noise supplies the wandering, and each component reads it with its
    // own gain so their crests do not wander together.
    let wander = ripple_value(plane / 7.0 + vec2<f32>(3.1, 1.7)) * 8.0;
    var gains = array<f32, 4>(1.0, -0.8, 0.6, -1.2);

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
        let phase = k * dot(direction, plane) - omega * time + offsets[index] + wander * gains[index];
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
    // scaled with the sea so a calm has none. The gate is lower than the
    // swell's because these components are the ones that actually break in a
    // breeze - Beaufort three has scattered caps, and the eye expects them.
    let fold = smoothstep(0.22, 0.7, 1.0 - jacobian) * fade;
    return WindSea(slope * fade, fold);
}

/// A layer of foam on a fragment: how much of it the bubbles cover, and which
/// way the bubbles face.
struct Foam {
    coverage: f32,
    /// Slope of the bubble surface, to tilt the shading normal by.
    bumps: vec2<f32>,
};

/// Foam from an energy in `[0, 1]` - how much foam the water here *should*
/// carry, from whichever source: the swell's folds with their memory, the
/// wind sea's caps, the bow wave, the wake.
///
/// Not a model of breaking. Real whitecapping starts when a crest's particle
/// velocity approaches the phase speed, and what is drawn is where the
/// Jacobians say the water has been squeezed enough to; the energy is that,
/// and this is what foam of that energy looks like.
///
/// # Why a threshold on a texture and not a mix
///
/// Foam is bubbles: a patch is either there or not, its edge is lace, and it
/// has holes. Mixing white in by the energy - the first version - gave an
/// airbrush, and a viewer read it as paint on the water. Sea of Thieves puts
/// its peak mask through a noise threshold instead, so the patches grow from
/// the noise's own peaks as the energy rises and dissolve back into them as it
/// falls, and that is what this does: two octaves of the same gradient noise
/// the ripples use, a coarse one for the patch shapes and a fine one for the
/// lace, drifting slowly so the foam does not sit still on moving water.
///
/// The fine octave's slope is returned as bumps, so the caller can light the
/// foam as a rough white surface rather than as a colour: a cap in the sun
/// is bright on the side facing it and blue-grey behind, and that is where
/// the thickness a flat mask lacks comes from.
fn foam(plane: vec2<f32>, time: f32, energy: f32) -> Foam {
    if (energy <= 0.02) {
        return Foam(0.0, vec2<f32>(0.0, 0.0));
    }
    let drift = vec2<f32>(0.11, 0.07) * time;
    let coarse = ripple_value((plane + drift) / 0.9);
    let fine = ripple_value((plane - drift * 0.6) / 0.22);
    // In [0, 1] roughly; the coarse octave shapes the patch, the fine one
    // frays its edge.
    let texture = 0.5 + 0.55 * coarse + 0.3 * fine;
    // The energy lowers the bar the texture has to clear: at full energy
    // almost everything is foam, at a tenth only the noise's peaks are.
    let bar = 0.92 - 0.85 * energy;
    let coverage = smoothstep(bar, bar + 0.18, texture);
    let bumps = ripple_slope((plane - drift * 0.6) / 0.22) * 0.35;
    return Foam(coverage, bumps);
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
    // From the stem back to amidships, as a fraction. The foam starts a little
    // ahead of the stem - the wave the bow pushes up breaks in front of it -
    // and dies away over the after half of the run rather than stopping on
    // a line: both ends were hard cuts, and a straight edge across the water
    // at midships is the first thing a viewer's eye finds.
    let aft = (hull.x - along) / (0.5 * hull.x);
    let run = smoothstep(-0.12, 0.08, aft) * (1.0 - smoothstep(0.55, 1.15, aft));
    if (run <= 0.0) {
        return 0.0;
    }
    // Where the crest sits: just outboard of the waterline, which flares
    // from nothing at the stem to the full half-beam amidships roughly as a
    // square root, plus a margin the plunge widens. Inboard of that the water
    // is under the hull, where foam drawn on the sea is foam nobody sees -
    // which is where the first version put it.
    let plunge = clamp(sea.heading.z, 0.0, 3.0);
    let half_beam = hull.y * sqrt(max(aft, 0.0));
    let crest_at = half_beam + 0.6 + 0.6 * plunge;
    let band = 1.0 + 0.5 * plunge;
    let lobe = 1.0 - smoothstep(0.0, band, abs(across - crest_at));
    // Brighter with speed; the plunge throws it further aft.
    let strength = smoothstep(0.5, 2.5, speed) * run * (0.7 + 0.3 * plunge);
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
    // Nothing recorded, or nothing near. The bounds are the track's, grown by
    // the widest the wake gets, and the CPU keeps them: with them the loop
    // below runs on the strip of sea the wake is actually in and not on every
    // fragment within eighty metres of the stern, which at the default view
    // is most of the screen. The first version ran there, sixty-four segments
    // and a noise lookup each, and cost fifty milliseconds a frame.
    let bounds = sea.trail_bounds;
    if (count == 0u || plane.x < bounds.x || plane.y < bounds.y || plane.x > bounds.z || plane.y > bounds.w) {
        return 0.0;
    }

    // The margin, ragged by a slow noise along the track — clamped, because
    // the slope of a noise field is not bounded and unclamped it threw blobs of
    // foam twenty metres off the track. A property of the point, not of the
    // segment, so it is computed once.
    let ragged = 0.8 + 0.3 * clamp(ripple_slope(plane * 0.18).x, -1.0, 1.0);

    // Newest segment first, because the wake is strongest there and fades
    // with age: past fifteen seconds it is under three per cent of full and
    // the streaks have dissolved it, so the walk stops there rather than
    // visiting every point the buffer holds.
    var best = 0.0;
    for (var back: u32 = 0u; back < count; back = back + 1u) {
        let index = count - 1u - back;
        let older = trail[index].point;
        // The segment ends at the next recorded point, or at the stern itself
        // for the newest one, which is where the wake is being made now.
        var newer: vec4<f32>;
        if (index + 1u < count) {
            newer = trail[index + 1u].point;
        } else {
            newer = vec4<f32>(stern, sea.time, 0.0);
        }
        if (sea.time - newer.z > 15.0) {
            break;
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
        // tenth of a metre a second; mostly gone in five seconds, which at five
        // knots is a boat length of clear trail and a faint one for a few more.
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
    let crest = clamp(in.elevation / max(significant_height, 0.2), -1.0, 1.0);
    transmitted = transmitted + sea.shallow * crest * 0.22 * attenuation;

    // Light through the crest. A wave seen against the sun is lit from behind,
    // and where the crest is thin the light comes through it green: the
    // turquoise glow on the back of a breaking wave, which is the single most
    // recognisable thing about Sea of Thieves' water and is in every
    // photograph of a sea with the sun low. The term is the usual one (Crest,
    // Sea of Thieves): the view direction against the sun's, raised to a
    // power so it is a lobe and not a wash, times how high the crest is and
    // how much the choppiness has thinned it - `thinness` is `1 - J`, the
    // sideways squeeze, which is exactly where a crest is narrow. Scaled by
    // the sun's visibility so a shadowed crest does not glow.
    let through = pow(max(dot(eye, sun), 0.0), 3.0);
    let thin = max(crest, 0.0) * (0.35 + 2.5 * in.thinness);
    let scatter = sea.scatter * through * thin * attenuation * seen;
    transmitted = transmitted + scatter;

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
    colour = colour - (sea.shallow * wrapped * 0.22 + scatter) * (1.0 - fresnel) * (1.0 - lit);

    // Foam sits on top of everything: it is a surface of bubbles, not a property
    // of the water under it, and it neither reflects the sky nor transmits.
    // The swell's folds with their memory, the wind sea's caps, the bow wave
    // and the wake are all foam, and they combine as an energy rather than
    // adding, so a whitecap crossing the trail is not more than foam.
    //
    // The layer is lit as a rough white surface: the sky from above, and the
    // sun on the side of each bubble that faces it, with the boat's shadow on
    // the sun's share only. That is what gives a cap a lit side and a shaded
    // side, and a wake in the hull's shadow a blue-grey rather than a glow.
    let energy = max(max(in.foam, wind.fold), max(bow_wave(plane, detail, sea.wind.zw), wake(plane, detail)));
    let lace = foam(plane, sea.time, energy);
    if (lace.coverage > 0.0) {
        let bubble_normal = normalize(normal + vec3<f32>(-lace.bumps.x, 0.0, -lace.bumps.y));
        // The sky's light on a white surface is the whole hemisphere's, most
        // of which is the pale band near the horizon, not the deep blue at
        // the zenith - foam lit by the zenith alone came out royal blue.
        let ambient = mix(
            sky_reflection(vec3<f32>(0.0, 1.0, 0.0), sun),
            sky_reflection(normalize(vec3<f32>(eye.x, 0.15, eye.z)), sun),
            0.7,
        );
        let direct = vec3<f32>(1.0, 0.97, 0.9) * max(dot(bubble_normal, sun), 0.0) * 1.8 * sun_up * lit;
        let foam_colour = vec3<f32>(0.93, 0.95, 0.97) * (ambient + direct);
        colour = mix(colour, foam_colour, lace.coverage);
    }
    // Under and around the lace, the water itself is aerated: a paler,
    // milkier body where foam has just been or is about to be.
    let wash = energy * (1.0 - lace.coverage) * 0.2;
    colour = mix(colour, vec3<f32>(0.45, 0.62, 0.68), wash * (1.0 - fresnel));

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
