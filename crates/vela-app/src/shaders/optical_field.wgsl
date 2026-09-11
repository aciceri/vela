#import vela::ocean_geometry::{plane_point, projection_margin}
#import vela::ocean_surface::sea
#import vela::ocean_optics::optical_surface

struct FieldVertex {
    @builtin(position) position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

// Two triangles, no mesh allocation, transforms, depth or material lighting.
@vertex
fn vertex(@builtin(vertex_index) index: u32) -> FieldVertex {
    var corner = vec2<f32>(-1.0, -1.0);
    switch index {
        case 1u, 4u: { corner = vec2<f32>(1.0, -1.0); }
        case 2u, 3u: { corner = vec2<f32>(-1.0, 1.0); }
        case 5u: { corner = vec2<f32>(1.0, 1.0); }
        default: {}
    }
    return FieldVertex(vec4<f32>(corner, 0.0, 1.0), corner);
}

@fragment
fn fragment(in: FieldVertex) -> @location(0) vec4<f32> {
    // The view binding is the Chase view itself, not the rounded half-size
    // target's projection. Odd window dimensions therefore cannot skew UVs.
    let plane = plane_point(in.ndc * projection_margin());
    let dx = dpdx(plane);
    let dy = dpdy(plane);
    let xx = dot(dx, dx);
    let yy = dot(dy, dy);
    let xy = dot(dx, dy);
    let discriminant = sqrt(max((xx - yy) * (xx - yy) + 4.0 * xy * xy, 0.0));
    let spacing = sqrt(max(0.5 * (xx + yy + discriminant), 0.0));
    let optics = optical_surface(plane, sea.time, spacing);
    return vec4<f32>(optics.slope, optics.variance, optics.foam);
}
