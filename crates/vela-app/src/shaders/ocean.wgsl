// The physical Gerstner realisation is shared with the foam accumulation pass.
// Near the hull it is exact; farther out each wavelength is filtered against
// the polar mesh's spacing before sampling, not faded as one radial band.
// Water optics use dielectric Fresnel, GGX/Smith and the same cloudy sky as the
// dome. Foam is persistent state in parcel coordinates; the wake also changes
// surface slope and roughness after its visible bubbles have dissolved.

#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_world}
#import bevy_pbr::mesh_view_bindings::view
#import bevy_pbr::shadows::fetch_directional_shadow
#import bevy_pbr::view_transformations::position_world_to_clip
#import vela::atmosphere::{sky_reflection, cloud_offset}
#import vela::ocean_surface::{sea, trail, surface, jacobian, folding, hull_foam}

/// Local rather than imported: naga_oil resolves function imports reliably and
/// constants less so, and this is four characters.
const PI: f32 = 3.141592653589793;


/// The above-water scene in a mirror at the mean sea level, rendered by the
/// reflection camera into a window-shaped image over transparency; see
/// `crate::reflection`. Live only when `sea.heading.w` says so.
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var reflection: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var reflection_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(5) var foam_history: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(6) var foam_sampler: sampler;

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
    /// Fresh breaking outside the bounded persistent-foam region.
    @location(4) foam: f32,
    /// How much thinner this crest is than the water around it, `[0, 1]`:
    /// `1 - J` clamped, which is where the crest is being squeezed sideways
    /// and light gets through. What the sub-surface term reads.
    @location(5) thinness: f32,
    /// Rest coordinate: texture follows the parcel's orbit, not the wave crest.
    @location(6) parcel: vec2<f32>,
    @location(7) variance: f32,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let world_from_local = get_world_from_local(vertex.instance_index);
    let base = mesh_position_local_to_world(world_from_local, vec4<f32>(vertex.position, 1.0));
    let radius = length(vertex.position.xz);
    // Both polar meshes have edges at most ~4.3% of their radius. Feeding all
    // sixty waves to their distant vertices aliased into spokes and blotches.
    // Keep the hull's neighbourhood exact and resolve wavelengths individually.
    let spacing = radius * 0.043 * smoothstep(20.0, 40.0, radius);
    let wave = surface(base.xz, sea.time, spacing);
    let world = base.xyz + vec3<f32>(wave.shift.x, wave.height, wave.shift.y);
    let along_x = vec3<f32>(1.0 + wave.strain.x, wave.slope.x, wave.strain.z);
    let along_z = vec3<f32>(wave.strain.z, wave.slope.y, 1.0 + wave.strain.y);
    var out: VertexOutput;
    out.world_position = world;
    out.world_normal = normalize(cross(along_z, along_x));
    out.elevation = wave.height;
    out.steepness = length(wave.slope);
    out.foam = folding(wave.strain);
    out.thinness = clamp(1.0 - jacobian(wave.strain), 0.0, 1.0);
    out.parcel = base.xz;
    out.variance = wave.variance;
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
/// Sub-metre bands keep fixed world coordinates. Unresolved detail contributes
/// slope variance to GGX, not a magnified low-frequency noise texture: that
/// substitute was visible as cloudy patches over large stretches of water.
fn ripples(plane: vec2<f32>, time: f32, footprint: f32) -> vec3<f32> {
    let downwind = sea.wind.xy;
    let across = vec2<f32>(-downwind.y, downwind.x);
    let uv = vec2<f32>(dot(plane, downwind), dot(plane, across));
    let fine = 1.0 - smoothstep(0.12, 0.45, footprint * 1.7);
    let capillary = 1.0 - smoothstep(0.12, 0.45, footprint * 4.6);
    var local = vec2<f32>(0.0);
    if (fine > 0.0) {
        local += ripple_slope(uv * 1.7 - vec2<f32>(0.75, 0.12) * time) * (0.018 * fine);
    }
    if (capillary > 0.0) {
        let rotation = mat2x2<f32>(0.8, 0.6, -0.6, 0.8);
        local += transpose(rotation) * ripple_slope(rotation * uv * 4.6 - vec2<f32>(0.9, -0.3) * time) * (0.008 * capillary);
    }
    let variance = 0.5 * (0.018 * 0.018 * (1.0 - fine * fine) + 0.008 * 0.008 * (1.0 - capillary * capillary));
    return vec3<f32>(downwind * local.x + across * local.y, variance);
}

/// Air/water IOR 1.333: F0 ~= 0.0204. No artistic grazing-angle cap.
fn fresnel_water(cosine: f32) -> f32 {
    let m = 1.0 - clamp(cosine, 0.0, 1.0);
    return 0.0204 + 0.9796 * m * m * m * m * m;
}

/// GGX normal distribution and height-correlated Smith visibility.
/// `alpha` is RMS microfacet roughness, not the perceptual square root.
fn sun_specular(normal: vec3<f32>, sun: vec3<f32>, viewer: vec3<f32>, alpha: f32) -> f32 {
    let nv = max(dot(normal, viewer), 0.001);
    let nl = max(dot(normal, sun), 0.0);
    let half_vector = (sun + viewer) / max(length(sun + viewer), 1e-4);
    let nh = max(dot(normal, half_vector), 0.0);
    let vh = max(dot(viewer, half_vector), 0.0);
    let a2 = alpha * alpha;
    let denom = nh * nh * (a2 - 1.0) + 1.0;
    let distribution = a2 / max(PI * denom * denom, 1e-7);
    let visibility = 0.5 / max(
        nl * sqrt(nv * nv * (1.0 - a2) + a2)
        + nv * sqrt(nl * nl * (1.0 - a2) + a2), 1e-4);
    return fresnel_water(vh) * distribution * visibility * nl;
}

/// Sampling outside the mirror must reveal sky, not smear its border pixels.
/// Render-target RGB is premultiplied by coverage when blurred over clear.
fn mirror_tap(uv: vec2<f32>) -> vec4<f32> {
    let margin = min(min(uv.x, uv.y), min(1.0 - uv.x, 1.0 - uv.y));
    let valid = smoothstep(0.0, 0.012, margin);
    return textureSampleLevel(reflection, reflection_sampler, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0) * valid;
}

/// Project an actual reflected ray back into the planar camera. The reference
/// height approximates the hull/rig midpoint; unlike a world-XZ UV offset this
/// rotates correctly with the camera and includes the wave's vertical motion.
fn boat_reflection(world: vec3<f32>, ray: vec3<f32>, alpha: f32) -> vec4<f32> {
    if (sea.heading.w < 0.5 || ray.y <= 0.01) {
        return vec4<f32>(0.0);
    }
    let travel = clamp((6.0 - world.y) / max(ray.y, 0.12), 1.0, 60.0);
    let hit = world + ray * travel;
    let clip = position_world_to_clip(vec3<f32>(hit.x, -hit.y, hit.z));
    if (clip.w <= 0.0) {
        return vec4<f32>(0.0);
    }
    let uv = clip.xy / clip.w * vec2<f32>(0.5, -0.5) + 0.5;
    let texel = 1.0 / vec2<f32>(textureDimensions(reflection));
    let radius = texel * (0.75 + 35.0 * alpha);
    var reflected = mirror_tap(uv) * 0.4;
    reflected += mirror_tap(uv + vec2<f32>(radius.x, 0.0)) * 0.15;
    reflected += mirror_tap(uv - vec2<f32>(radius.x, 0.0)) * 0.15;
    reflected += mirror_tap(uv + vec2<f32>(0.0, radius.y)) * 0.15;
    reflected += mirror_tap(uv - vec2<f32>(0.0, radius.y)) * 0.15;
    return reflected;
}

/// The wind sea: what a fragment adds to the surface's slope, and how close
/// that surface is to folding.
struct WindSea {
    slope: vec2<f32>,
    /// `1 - J` of the layer's own virtual displacement, clamped to `[0, 1]`:
    /// zero on a smooth surface, rising to one where a Gerstner crest of this
    /// steepness would be cusping. Where the whitecaps are.
    fold: f32,
    /// Slope variance lost below the pixel footprint becomes microfacet
    /// roughness instead of vanishing into a flat mirror.
    variance: f32,
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
/// Each component is filtered by projected footprint, not camera distance.
fn wind_sea(plane: vec2<f32>, time: f32, footprint: f32) -> WindSea {
    // The wind is where the physical sea's mean heading comes from: the
    // realisation travels *towards* the boat from the wind's direction, so its
    // mean wave vector points downwind, and the wind sea runs the same way.
    // Computed once on the CPU from the realisation and carried in the
    // uniform; a sum over sixty components per fragment is not a direction.
    let downwind = sea.wind.xy;
    let across = vec2<f32>(-downwind.y, downwind.x);

    // Cat's paws: patches thirty metres or so across, drifting downwind at a
    // walking pace, between a quarter and full strength.
    let gust_filter = 1.0 - smoothstep(0.15, 0.5, footprint / 32.0);
    let paw = ripple_value((plane - downwind * time * 1.28) / 32.0) * gust_filter;
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
    let wander = ripple_value((plane - downwind * time * 0.35) / 7.0 + vec2<f32>(3.1, 1.7)) * 5.0;
    var gains = array<f32, 4>(1.0, -0.8, 0.6, -1.2);

    var slope = vec2<f32>(0.0, 0.0);
    var jxx = 1.0;
    var jzz = 1.0;
    var jxz = 0.0;
    var variance = 0.0;
    for (var index: u32 = 0u; index < 4u; index = index + 1u) {
        let wavelength = wavelengths[index];
        let k = 2.0 * PI / wavelength;
        // Deep-water dispersion, the same law the physical sea obeys.
        let omega = sqrt(9.81 * k);
        // A packet moves at half the phase speed in deep water. Each component
        // has its own envelope, so crests grow and subside instead of marching
        // as four equally strong, endless sine trains.
        let direction = downwind * cos(bearings[index]) + across * sin(bearings[index]);
        let group_phase = dot(direction, plane) * k * 0.17 - omega * time * 0.085;
        let group_filter = 1.0 - smoothstep(0.8, 2.8, k * 0.17 * footprint);
        let packet = 0.65 + 0.35 * sin(group_phase + offsets[index] * 2.3) * group_filter;
        let packet_energy = packet * packet + 0.06125 * (1.0 - group_filter * group_filter);
        let filtered = 1.0 - smoothstep(0.8, 2.8, k * footprint);
        variance += 0.5 * steepness * steepness * packet_energy * (1.0 - filtered * filtered);
        if (filtered <= 0.0) {
            continue;
        }
        let amplitude = steepness * packet * filtered / k;
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
    let fold = smoothstep(0.22, 0.7, 1.0 - jacobian);
    // Transform the parametric slope through the horizontal deformation.
    // Without this, the "Gerstner" detail still shades symmetric sine waves.
    let choppy_slope = vec2<f32>(
        jzz * slope.x - jxz * slope.y,
        jxx * slope.y - jxz * slope.x,
    ) / max(jacobian, 0.4);
    return WindSea(choppy_slope, fold, variance);
}

/// A layer of foam on a fragment: how much of it the bubbles cover, and which
/// way the bubbles face.
struct Foam {
    coverage: f32,
    /// Slope of the bubble surface, to tilt the shading normal by.
    bumps: vec2<f32>,
};

/// Cellular lace on the moving parcel, inspired by the broken coverage in
/// Alex Tardif's Water Walkthrough (https://alextardif.com/Water.html).
/// Compression still owns generation; noise only shapes the surviving foam.
/// Fixed cells stretch with the surface, while slow downwind drift moves both
/// scales together. Independent counter-scrolling made the bubbles boil.
fn foam(parcel: vec2<f32>, time: f32, energy: f32, footprint: f32) -> Foam {
    if (energy <= 0.02) {
        return Foam(0.0, vec2<f32>(0.0, 0.0));
    }
    let at = parcel - sea.wind.xy * (0.055 * time);
    let coarse_filter = 1.0 - smoothstep(0.15, 0.5, footprint * 0.65);
    let coarse = ripple_value(at * 0.65) * coarse_filter;
    let coverage_mask = smoothstep(0.28, 0.72, energy + coarse * 0.55);
    let resolved = 1.0 - smoothstep(0.08, 0.24, footprint);
    if (resolved <= 0.0 || coverage_mask <= 0.0) {
        return Foam(coverage_mask * 0.72, vec2<f32>(0.0, 0.0));
    }

    // Rounded, water-filled pores between bubble rafts, not polygon outlines.
    // Nine neighbours avoid discontinuities at cell boundaries. The nearest
    // offset supplies a bubble normal without another noise-derivative lookup.
    // Warp the cellular domain so the lace bends rather than drawing the
    // straight polygon edges of unmodified Voronoi cells in a close view.
    let cells = at / 0.24 + ripple_slope(at * 1.8) * 0.45;
    let cell = floor(cells);
    let offset = fract(cells);
    var nearest = 8.0;
    var bubble = vec2<f32>(0.0, 0.0);
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let neighbour = vec2<f32>(f32(x), f32(y));
            let centre = 0.5 + 0.38 * hash_gradient(cell + neighbour);
            let delta = neighbour + centre - offset;
            let squared = dot(delta, delta);
            if (squared < nearest) {
                nearest = squared;
                bubble = delta;
            }
        }
    }
    let radius = mix(0.48, 0.16, energy);
    let aa = clamp(footprint / 0.24, 0.025, 0.25);
    let lace = smoothstep(radius - aa, radius + aa, sqrt(nearest));
    // Even a fresh cap has pores; old foam erodes into thin connected strands.
    let coverage = coverage_mask * mix(0.72, lace, resolved);
    return Foam(coverage, -bubble * (0.32 * resolved));
}

/// History follows physical parcels; the texture origin follows only the
/// bounded simulation window. The margin hides neither new sea nor old foam:
/// outside the window, the same fresh-breaking criterion remains available.
fn remembered_foam(parcel: vec2<f32>, fresh: f32) -> f32 {
    let region = sea.foam_region;
    if (region.w < 0.5 || region.z <= 0.0) {
        return fresh;
    }
    let uv = (parcel - region.xy) / region.z;
    let edge = min(min(uv.x, uv.y), min(1.0 - uv.x, 1.0 - uv.y));
    let inside = smoothstep(0.0, 0.06, edge);
    let encoded = textureSampleLevel(foam_history, foam_sampler, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rg;
    let history = encoded.x + encoded.y / 255.0;
    return mix(fresh, history, inside);
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

/// A rough turbulent ribbon follows the recorded track even after the boat
/// turns or stops. `xy` is its cosmetic slope, `z` its residual intensity.
/// Foam has its own accumulation; turbulence persists longer than bubbles.
fn wake(plane: vec2<f32>, footprint: f32) -> vec3<f32> {
    let count = sea.trail_count;
    let stern = sea.motion.xy;
    // Nothing recorded, or nothing near. The bounds are the track's, grown by
    // the widest the wake gets, and the CPU keeps them: with them the loop
    // below runs on the strip of sea the wake is actually in and not on every
    // fragment within eighty metres of the stern, which at the default view
    // is most of the screen. The first version ran there, sixty-four segments
    // and a noise lookup each, and cost fifty milliseconds a frame.
    let bounds = sea.trail_bounds;
    if (count == 0u || plane.x < bounds.x || plane.y < bounds.y || plane.x > bounds.z || plane.y > bounds.w) {
        return vec3<f32>(0.0);
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
        let age = max(0.0, sea.time - mix(older.z, newer.z, along / span_length));

        let half_width = (0.9 + 0.18 * age) * ragged;
        let speed = span_length / max(newer.z - older.z, 0.001);
        let strength = exp(-age / 8.0) * smoothstep(0.2, 2.0, speed);
        let inside = 1.0 - smoothstep(0.25 * half_width, half_width, across);
        best = max(best, strength * inside);
    }

    let filtered = 1.0 - smoothstep(0.2, 0.8, footprint);
    let eddies = ripple_slope(plane * 0.85 - sea.wind.xy * sea.time * 0.09);
    return vec3<f32>(eddies * (0.055 * best * filtered), best);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let sun = normalize(sea.sun.xyz);
    let plane = in.world_position.xz;
    let footprint = max(length(dpdx(in.parcel)), length(dpdy(in.parcel)));
    let to_eye = view.world_position - in.world_position;
    let distance = max(length(to_eye), 1e-4);
    let viewer = to_eye / distance;
    let physical = normalize(in.world_normal);
    let slope = physical.xz / max(physical.y, 0.05);
    let wind = wind_sea(in.parcel, sea.time, footprint);
    let detail = ripples(in.parcel, sea.time, footprint);
    let turbulent = wake(plane, footprint);
    let perturbed = slope - wind.slope - detail.xy - turbulent.xy;
    let normal = normalize(vec3<f32>(perturbed.x, 1.0, perturbed.y));

    // Geometric specular antialiasing: unresolved normal variation broadens
    // GGX instead of producing bright isolated pixels or radial mesh patterns.
    let dx = dpdx(normal);
    let dy = dpdy(normal);
    let pixel_variance = 0.25 * (dot(dx, dx) + dot(dy, dy));
    let alpha = clamp(sqrt(0.001 + in.variance + detail.z + wind.variance + pixel_variance + turbulent.z * 0.012), 0.035, 0.35);
    let roughness = sqrt(alpha);
    let fresnel = fresnel_water(dot(normal, viewer));
    let drift = cloud_offset(sea.time);
    let reflected_ray = reflect(-viewer, normal);
    let environment = sky_reflection(reflected_ray, sun, drift, roughness);
    let mirror = boat_reflection(in.world_position, reflected_ray, alpha);
    // Correct premultiplied composition: do not multiply the blurred mirror's
    // coverage twice, and do not replace the transmitted light with a sail.
    let reflected = environment * (1.0 - mirror.a) + mirror.rgb;

    let view_z = dot(
        vec4<f32>(view.world_from_view[2].xyz, 0.0),
        vec4<f32>(in.world_position, 1.0) - view.world_from_view[3]);
    let lit = fetch_directional_shadow(0u, vec4<f32>(in.world_position, 1.0), normal, view_z, in.clip_position.xy);
    let seen = sun_visible(sun, in.steepness);
    let sun_up = smoothstep(0.0, 0.08, sun.y);
    let illumination = lit * seen * sun_up;
    let wrapped = pow(clamp(dot(normal, sun) * 0.4 + 0.6, 0.0, 1.0), 6.0);
    let crest = clamp(in.elevation / max(sea.significant_height, 0.2), -1.0, 1.0);
    let attenuation = 1.0 / (1.0 + distance * distance * 2.5e-6);
    let through = pow(max(dot(-viewer, sun), 0.0), 3.0);
    let thin = max(crest, 0.0) * (0.35 + 2.5 * in.thinness);
    var transmitted = sea.deep + sea.shallow * (wrapped * 0.22 * illumination + crest * 0.12 * attenuation);
    transmitted += sea.scatter * through * thin * attenuation * illumination;
    // Aerated wake water changes absorption without painting an opaque trail.
    transmitted = mix(max(transmitted, vec3<f32>(0.0)), sea.shallow * 0.35, turbulent.z * 0.12);
    var colour = transmitted * (1.0 - fresnel) + reflected * fresnel;
    colour += vec3<f32>(4.2, 3.9, 3.5) * sun_specular(normal, sun, viewer, alpha) * illumination;

    let fresh = max(max(in.foam, wind.fold), hull_foam(plane));
    let energy = remembered_foam(in.parcel, fresh);
    let lace = foam(in.parcel, sea.time, energy, footprint);
    if (lace.coverage > 0.0) {
        let bubble_normal = normalize(normal + vec3<f32>(-lace.bumps.x, 0.0, -lace.bumps.y));
        let ambient = sky_reflection(vec3<f32>(0.0, 1.0, 0.0), sun, drift, 1.0);
        let horizon_light = sky_reflection(normalize(vec3<f32>(viewer.x, 0.2, viewer.z)), sun, drift, 1.0);
        let direct = vec3<f32>(1.0, 0.97, 0.9) * max(dot(bubble_normal, sun), 0.0) * 1.8 * sun_up * lit;
        let foam_colour = vec3<f32>(0.93, 0.95, 0.97) * (mix(ambient, horizon_light, 0.65) + direct);
        colour = mix(colour, foam_colour, lace.coverage);
    }
    let wash = energy * (1.0 - lace.coverage) * 0.12;
    colour = mix(colour, sea.shallow * 0.65, wash * (1.0 - fresnel));
    let horizontal = normalize(vec3<f32>(-viewer.x, 0.001, -viewer.z));
    let haze = 1.0 - exp(-distance / 5500.0);
    colour = mix(colour, sky_reflection(horizontal, sun, drift, 1.0), haze);
    return vec4<f32>(colour, 1.0);
}
