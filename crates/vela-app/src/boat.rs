//! The boat: the physics hull, drawn, and the rig above it.
//!
//! # There is one hull
//!
//! The mesh rendered here is the mesh `modules::Buoyancy` integrates pressure
//! over — lofted from the boat file's station offsets by `vela_core::loft`, not a
//! visual model that happens to resemble it. The boat format carries no visual
//! mesh on purpose: two hulls that could drift apart is a bug with no symptom
//! until the waterline is somewhere the picture disagrees with.
//!
//! What that costs is honesty rather than beauty. The physics hull is a
//! watertight lofted surface with a flat deck lid and no coachroof, no sheerline
//! detail and no appendages, because none of those are things the pressure
//! integral needs. It looks like what it is.
//!
//! # The rig is a stand-in and says so
//!
//! Mast and sails are drawn as simple geometry placed from the boat file's rig
//! dimensions. They are **not** the flying shape `vela_core::flying` computes:
//! that model is only mounted when a boat asks for the geometric aerodynamics,
//! and the reference boat asks for the tabular one, which has no shape at all —
//! only areas and centres of effort. Drawing a sail whose camber came from
//! nowhere would be exactly the kind of confident fiction this project avoids
//! elsewhere, so the sails here are flat triangles at the sheeting angle the
//! controls ask for, and nothing about their shape should be read as physics.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::Indices;
use bevy::prelude::*;
use bevy::render::render_resource::PrimitiveTopology;
use vela_core::TriMesh;

use crate::frame;
use crate::sim::Engine;

/// Marks the entity that carries the boat's pose.
///
/// Hull and rig are children of it, so the pose is applied once and they cannot
/// come apart.
#[derive(Component)]
pub struct Boat;

/// Converts an engine hull into a render mesh.
///
/// Normals are computed per face and written per vertex of a duplicated triangle
/// list rather than averaged over a shared vertex. That is the right choice for
/// this mesh and not laziness: a lofted hull has a hard chine at the deck edge
/// and a hard stem, and averaging across them would round off exactly the edges
/// that say what shape the boat is.
#[must_use]
pub fn hull_mesh(hull: &TriMesh) -> Mesh {
    let count = hull.triangle_count();
    let mut positions = Vec::with_capacity(count * 3);
    let mut normals = Vec::with_capacity(count * 3);

    for index in 0..count {
        let triangle = hull.triangle(index);
        // The engine's winding is outward by the `TriMesh` contract; the frame
        // map is a proper rotation, so it survives the conversion and the
        // renderer sees front faces from outside.
        let area_normal = triangle.area_normal();
        let normal = frame::to_render(area_normal).normalize_or_zero();
        for vertex in [triangle.a, triangle.b, triangle.c] {
            positions.push(frame::to_render(vertex).to_array());
            normals.push(normal.to_array());
        }
    }

    let indices: Vec<u32> = (0..positions.len() as u32).collect();

    Mesh::new(
        PrimitiveTopology::TriangleList,
        // RENDER_WORLD alone: the hull is rigid, so nothing ever reads it back.
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}

/// A flat triangular sail, in the render frame.
///
/// Two-sided, because a flat sail seen from leeward would otherwise vanish. A
/// real sail has a windward and a leeward side and this does not model that;
/// duplicating the triangle with reversed winding is the honest cheap answer.
fn sail_mesh(tack: Vec3, head: Vec3, clew: Vec3) -> Mesh {
    let front = [tack, head, clew];
    let back = [tack, clew, head];
    let mut positions = Vec::with_capacity(6);
    let mut normals = Vec::with_capacity(6);

    for triangle in [front, back] {
        let normal = (triangle[1] - triangle[0])
            .cross(triangle[2] - triangle[0])
            .normalize_or_zero();
        for vertex in triangle {
            positions.push(vertex.to_array());
            normals.push(normal.to_array());
        }
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32((0..6).collect()))
}

/// Spawns the hull, the mast and two sails under one pose entity.
pub fn spawn(
    mut commands: Commands,
    engine: Res<Engine>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let hull = meshes.add(hull_mesh(&engine.hull));
    let white = materials.add(StandardMaterial {
        base_color: Color::srgb(0.88, 0.88, 0.86),
        perceptual_roughness: 0.6,
        ..default()
    });
    let cloth = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.95, 0.93),
        perceptual_roughness: 0.9,
        double_sided: true,
        cull_mode: None,
        ..default()
    });
    let spar = materials.add(StandardMaterial {
        base_color: Color::srgb(0.25, 0.25, 0.28),
        ..default()
    });

    let rig = Rig::from(&*engine);
    // One line at startup naming the geometry actually built. A frontend whose
    // spars are in the wrong place looks like a rendering bug and is almost
    // always a units or datum mistake, and this is what tells the two apart.
    info!(
        "hull bounds {:?}; sheer {:.2} m, masthead {:.2} m, boom {:.2} m, mast at {:.2} m, \
         P {:.2} m, E {:.2} m, I {:.2} m, J {:.2} m; {} hull triangles",
        engine.hull.bounds(),
        rig.sheer,
        rig.masthead,
        rig.boom,
        rig.mast_at,
        rig.main_hoist,
        rig.main_foot,
        rig.foretriangle_height,
        rig.foretriangle_base,
        engine.hull.triangle_count()
    );

    commands
        .spawn((Boat, Transform::default(), Visibility::default()))
        .with_children(|boat| {
            boat.spawn((Mesh3d(hull), MeshMaterial3d(white)));

            // Mast: a box of the published diameter, from the sheer to the
            // masthead. Round would be prettier and would need a cylinder mesh.
            boat.spawn((
                Mesh3d(meshes.add(Cuboid::new(
                    rig.mast_diameter,
                    rig.masthead - rig.sheer,
                    rig.mast_diameter,
                ))),
                MeshMaterial3d(spar.clone()),
                Transform::from_xyz(rig.mast_at, 0.5 * (rig.masthead + rig.sheer), 0.0),
            ));

            // Boom: from the mast aft along the foot, at the published height.
            boat.spawn((
                Mesh3d(meshes.add(Cuboid::new(rig.main_foot, 0.14, 0.14))),
                MeshMaterial3d(spar),
                Transform::from_xyz(rig.mast_at - 0.5 * rig.main_foot, rig.boom, 0.0),
                Boom,
            ));

            // Mainsail: `P` up the mast from the boom, `E` aft along it. The two
            // IOR letters are exactly the triangle, which is the one place in
            // this file where the drawing and the force model agree by
            // construction rather than by resemblance.
            boat.spawn((
                Mesh3d(meshes.add(sail_mesh(
                    Vec3::new(rig.mast_at, rig.boom, 0.0),
                    Vec3::new(rig.mast_at, rig.boom + rig.main_hoist, 0.0),
                    Vec3::new(rig.mast_at - rig.main_foot, rig.boom, 0.0),
                ))),
                MeshMaterial3d(cloth.clone()),
                Mainsail,
            ));

            // Headsail: the foretriangle, `I` up the mast and `J` forward of it
            // to the stemhead. Tack at the sheer, where a deck-swept jib sets.
            boat.spawn((
                Mesh3d(meshes.add(sail_mesh(
                    Vec3::new(rig.mast_at + rig.foretriangle_base, rig.sheer, 0.0),
                    Vec3::new(rig.mast_at, rig.sheer + rig.foretriangle_height, 0.0),
                    Vec3::new(rig.mast_at, rig.sheer, 0.0),
                ))),
                MeshMaterial3d(cloth),
                Headsail,
            ));
        });
}

/// Marks the mainsail, so the helm can swing it.
#[derive(Component)]
pub struct Mainsail;

/// Marks the headsail.
#[derive(Component)]
pub struct Headsail;

/// Marks the boom, which swings with the mainsail.
#[derive(Component)]
pub struct Boom;

/// The rig's published dimensions, in render coordinates.
///
/// Every field is an IOR letter out of the boat file — `P`, `E`, `I`, `J` — or a
/// height measured from the sheer, which is the top of the lofted hull. Nothing
/// here is chosen to look right: if a boat file declares a rig, this draws that
/// rig, and if the drawing looks wrong the file is wrong.
struct Rig {
    /// Where the mast stands, render `x`, m.
    mast_at: f32,
    /// Sheer height above the waterline, render `y`, m.
    sheer: f32,
    /// Masthead height above the waterline, m.
    masthead: f32,
    /// Boom height above the waterline, m.
    boom: f32,
    /// `P`, the mainsail hoist, m.
    main_hoist: f32,
    /// `E`, the mainsail foot, m.
    main_foot: f32,
    /// `I`, the foretriangle height above the sheer, m.
    foretriangle_height: f32,
    /// `J`, the foretriangle base forward of the mast, m.
    foretriangle_base: f32,
    mast_diameter: f32,
}

impl From<&Engine> for Rig {
    fn from(engine: &Engine) -> Self {
        let rig = &engine.rig;
        // The sheer is the highest point of the lofted hull. In the engine frame
        // `z` points down, so the highest point is the *smallest* `z`, and the
        // frame map turns that into a height. Reading it from the hull rather
        // than from `average_freeboard` keeps the spars standing on the surface
        // that is actually drawn.
        let sheer = engine
            .hull
            .bounds()
            .map_or(0.0, |(min, _)| frame::metres(-min.z));
        // Without a layout the mast has no declared position, and the engine
        // restrains yaw for exactly that boat. Amidships is the honest placeholder
        // and the sails will not steer it either way.
        let mast_at = frame::metres(engine.mast_at.unwrap_or(0.0));

        Self {
            mast_at,
            sheer,
            masthead: sheer + frame::metres(rig.mast_above_sheer),
            boom: sheer + frame::metres(rig.boom_above_sheer),
            main_hoist: frame::metres(rig.main_hoist),
            main_foot: frame::metres(rig.main_foot),
            foretriangle_height: frame::metres(rig.foretriangle_height),
            foretriangle_base: frame::metres(rig.foretriangle_base),
            mast_diameter: frame::metres(rig.mast_diameter).max(0.08),
        }
    }
}

/// Applies the engine's pose to the boat every frame.
///
/// The engine's position is in world metres about the body origin, which sits at
/// the aft perpendicular on the baseline — so the boat is drawn where the physics
/// says it is, and the camera follows rather than the boat being re-centred.
pub fn follow(engine: Res<Engine>, mut boats: Query<&mut Transform, With<Boat>>) {
    let state = engine.sim.state();
    for mut transform in &mut boats {
        transform.translation = frame::to_render(state.position);
        transform.rotation = frame::rotation(&state.attitude);
    }
}

/// Exactly one of the three spars-and-sails, so that the queries are disjoint.
///
/// Bevy refuses a system whose queries could alias the same component mutably,
/// and three `&mut Transform` queries over overlapping sets would. Naming the
/// filter also keeps the signature readable, which the raw tuple did not.
type Only<T, A, B> = (With<T>, Without<A>, Without<B>);

/// Swings the boom and the sails to the angle the controls ask for.
///
/// The angle is real: `flying::Controls::boom_in` is the sheet times the
/// traveller, which is the composition the force model itself uses, and the sail
/// swings to the same fraction of its range that the trimmer set. The sail's
/// *shape* remains unmodelled — see the module documentation — so this is the
/// visible part of the trim and not the whole of it.
pub fn trim(
    engine: Res<Engine>,
    mut booms: Query<&mut Transform, Only<Boom, Mainsail, Headsail>>,
    mut mains: Query<&mut Transform, Only<Mainsail, Boom, Headsail>>,
    mut heads: Query<&mut Transform, Only<Headsail, Boom, Mainsail>>,
) {
    let controls = engine.sim.controls();
    let rig = Rig::from(&*engine);
    // Fully in is on the centreline; fully out is a quarter turn. Beyond that a
    // boom is against the shrouds, which this does not model.
    let out = (1.0 - controls.shape.boom_in()) as f32 * std::f32::consts::FRAC_PI_4;
    // A rotation of `θ` about render `y` takes the boom tip at `(-E, 0, 0)` to
    // `z = E sin θ`, and render `z` is the engine's `y`, which is starboard. So
    // `θ` carries the sign of the leeward direction on the body `y` axis, which
    // is exactly what `leeward_sign` returns.
    let angle = leeward_sign(&engine) * out;

    // The boom pivots at the mast, so the rotation is about a point the child
    // transform does not sit on: rotate, then put the midpoint back where the
    // rotation left it.
    for mut transform in &mut booms {
        let half = 0.5 * rig.main_foot;
        transform.rotation = Quat::from_rotation_y(angle);
        transform.translation = Vec3::new(rig.mast_at, rig.boom, 0.0)
            + Quat::from_rotation_y(angle) * Vec3::new(-half, 0.0, 0.0);
    }
    for mut transform in &mut mains {
        transform.rotation = Quat::from_rotation_y(angle);
    }
    // A headsail sheets closer than a main: it has no boom and its clew comes to
    // a track well inboard. Two thirds is a stand-in for a sheeting-angle model.
    for mut transform in &mut heads {
        transform.rotation = Quat::from_rotation_y(angle * 0.66);
    }
}

/// Which way the sails stream, as `+1` when leeward is to starboard.
///
/// Taken from the engine's own convention rather than re-derived: `leeward_sign`
/// is the single home of the one line of tack logic in this project, and a
/// frontend that picked the other sign would draw a boat with its boom to
/// windward while every number on the HUD stayed right.
///
/// The apparent wind angle comes from the sail module's telemetry, which is
/// where it is published. Before the first step there is none, and a boat
/// released close-hauled on starboard is the honest default for the one frame
/// that needs it.
fn leeward_sign(engine: &Engine) -> f32 {
    let angle = engine
        .sim
        .telemetry()
        .get("aero.apparent_wind.angle")
        .unwrap_or(1.0);
    vela_core::sim::leeward_sign(angle) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A hull mesh must carry three vertices per triangle and nothing else.
    ///
    /// The check that matters is the count: a mesh built from shared vertices
    /// would silently average the normals across the deck edge, and the failure
    /// is a rounded chine nobody notices.
    #[test]
    fn the_hull_mesh_keeps_its_faces_apart() {
        let hull = TriMesh::new(
            vec![
                vela_core::geometry::Point::new(0.0, 0.0, 0.0),
                vela_core::geometry::Point::new(1.0, 0.0, 0.0),
                vela_core::geometry::Point::new(0.0, 1.0, 0.0),
                vela_core::geometry::Point::new(0.0, 0.0, 1.0),
            ],
            vec![[0, 1, 2], [0, 1, 3]],
        );
        let mesh = hull_mesh(&hull);
        assert_eq!(mesh.count_vertices(), 6);
    }
}
