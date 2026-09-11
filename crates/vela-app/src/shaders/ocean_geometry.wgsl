#define_import_path vela::ocean_geometry

#import bevy_pbr::mesh_view_bindings::view
#import vela::ocean_surface::{sea, surface, Surface}

// A screen grid samples the visible water, not concentric rings around the
// boat. Positions remain world-space parcels: orbiting or zooming only changes
// their sampling density. The colour and velocity passes share this projection.
const REACH: f32 = 8000.0;

struct ProjectedVertex {
    parcel: vec2<f32>,
    spacing: f32,
    world: vec3<f32>,
    wave: Surface,
};

// Shared by projected geometry, the optical field and its screen-space lookup.
fn projection_margin() -> f32 {
    return 1.12 + min(0.25, sea.significant_height / max(abs(view.world_position.y), 2.0));
}

fn plane_point(ndc: vec2<f32>) -> vec2<f32> {
    let local = view.view_from_clip * vec4<f32>(ndc, 1.0, 1.0);
    let ray = (view.world_from_view * vec4<f32>(local.xyz, 0.0)).xyz;
    let horizontal = max(length(ray.xz), 1e-6);
    let denominator = select(-1.0, 1.0, ray.y >= 0.0) * max(abs(ray.y), 1e-6);
    let crossing = -view.world_position.y / denominator;
    let range = select(REACH, min(crossing * horizontal, REACH), crossing > 0.0);
    return view.world_position.xz + ray.xz * (range / horizontal);
}

fn projected_vertex(position: vec3<f32>) -> ProjectedVertex {
    let eye = view.world_position;
    // World-up chase camera: the mean-water horizon is horizontal in NDC.
    // Packing rows below it avoids spending most of the grid on empty sky.
    let forward = -view.world_from_view[2].xz;
    let far_plane = eye.xz + forward * (REACH / max(length(forward), 1e-6));
    let horizon_clip = view.clip_from_world * vec4<f32>(far_plane.x, 0.0, far_plane.y, 1.0);
    let horizon = horizon_clip.y / max(horizon_clip.w, 1e-6);
    // Overscan lets displaced crests move into the image from outside it.
    let margin = projection_margin();
    var bottom = -margin;
    var top = margin;
    if (eye.y >= 0.0) {
        top = clamp(horizon, -margin, margin);
    } else {
        bottom = clamp(horizon, -margin, margin);
    }
    let ndc = vec2<f32>(position.x * margin, mix(bottom, top, position.z * 0.5 + 0.5));
    let parcel = plane_point(ndc);
    let step = vec2<f32>(2.0 * margin / f32(#{OCEAN_COLUMNS}),
        (top - bottom) / f32(#{OCEAN_ROWS}));
    let dx = plane_point(ndc + vec2<f32>(select(step.x, -step.x, position.x >= 1.0), 0.0));
    let dy = plane_point(ndc + vec2<f32>(0.0, select(step.y, -step.y, position.z >= 1.0)));
    // Linear interpolation needs more than Nyquist's two vertices per wave.
    // Filter before sampling, retaining the lost slope energy as roughness.
    let spacing = 2.0 * max(length(dx - parcel), length(dy - parcel));
    let wave = surface(parcel, sea.time, spacing);
    let world = vec3<f32>(parcel.x + wave.shift.x, wave.height, parcel.y + wave.shift.y);
    return ProjectedVertex(parcel, spacing, world, wave);
}
