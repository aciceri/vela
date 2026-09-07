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
//! # The spars are the file's and the sails are the engine's
//!
//! Mast and boom are simple geometry placed from the boat file's rig
//! dimensions. The sails are the flying shape [`vela_core::flying`] computes
//! for the control positions the player is holding — camber, draft, twist and
//! the sheeting angle, all from the engine's own statement of what the cloth
//! does — and they are rebuilt whenever that statement changes. What the
//! drawing does **not** claim is that the *forces* come from that shape: the
//! reference boat sails on the tabular aerodynamic model, which reads areas, an
//! aspect ratio and an angle and never asks for a shape. [`sail_mesh`] says why
//! drawing the shape anyway is the honest way round.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::Indices;
use bevy::prelude::*;
use bevy::render::render_resource::PrimitiveTopology;
use nalgebra::Vector3;
use vela_core::aero::Sail;
use vela_core::flying::Shape;
use vela_core::frames::file_to_body;
use vela_core::geometry::Point;
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

/// Panels across a sail's chord.
///
/// Coarse, because the chordwise mean line is a smooth arc that eight panels
/// already round off.
const CHORDS: usize = 8;

/// Panels up a sail's luff.
///
/// Finer than the chord: twist varies continuously with height and is the
/// direction the eye reads.
const PANELS: usize = 14;

/// A sail as a cambered surface, in the render frame.
///
/// # Why this is not a triangle
///
/// It was, and it looked exactly as wrong as it was: a flat sheet stretched on the
/// IOR triangle. A sail is not flat. It takes a camber and a twist from the sheet,
/// the traveller, the outhaul, the cunningham and the leech, and the engine already
/// says what shape a given setting produces — [`vela_core::flying::Shape`] is a
/// planform with camber, draft and twist distributed up the luff, over a NACA-style
/// mean line. `Shape::point` is that surface, parametrically, and this walks it.
///
/// The shape drawn is therefore the engine's own, driven by the control positions
/// the player is holding, and not a decorative bulge invented here. The one thing
/// worth being straight about is that the reference boat sails on the *tabular*
/// aerodynamic model, which consumes area, aspect ratio and angle and never asks
/// for a shape. That does not make this fictional: the flying shape is what the
/// sail does, and the table is a lossy consumer of it chosen because its
/// coefficients are measured. Drawing the truth and approximating the forces is
/// the honest way round.
///
/// # Frame, tack and datum
///
/// `Shape` is in the file frame — x forward, y to **port**, z up from the tack —
/// and is built on starboard tack, its chord swung and its camber bulged to
/// port. `vela_core::frames::file_to_body` takes the surface into the body
/// frame and [`frame::to_render`] takes the body frame to Bevy's: two
/// conversions, each the one its crate owns. The tack goes through the same two,
/// so the foot cannot land on a different datum from the luff — which is what
/// happened when the surface was converted by hand with the y axis read as
/// starboard, and drew every sail mirrored onto the windward side.
///
/// `mirror` is applied to body `y` between the two conversions: `+1` leaves the
/// sail on starboard tack, `-1` mirrors it onto port. That is the mirror
/// `flying` says the caller owes it, rather than carry a sign through every
/// formula in the geometry.
///
/// The tack is the engine's literal body point, `z` being `tack_above_water`
/// negated: the engine applies its sail forces at heights above the water as if
/// the body origin were on the waterline. The body origin is in fact on the
/// baseline, forty centimetres below the design waterline on the reference
/// boat, so the drawn foot sits about half a metre below the boom the rig
/// letters place. Drawn rather than corrected, on purpose: a renderer that
/// quietly offset the engine's datum would be hiding an engine error, and the
/// gap is that error made visible.
///
/// Two-sided, because a sail seen from the leeward side would otherwise vanish.
/// The two windings get opposed normals, so each face is lit as the surface it is.
fn sail_mesh(shape: &Shape, tack: Point, mirror: f64) -> Mesh {
    let tack = file_to_body(tack);
    let station = |along: f64, up: f64| {
        let body = tack + file_to_body(shape.point(along, up));
        frame::to_render(Vector3::new(body.x, mirror * body.y, body.z))
    };

    // Four triangles a quad: two for each face.
    let vertices = CHORDS * PANELS * 12;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(vertices);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(vertices);

    for panel in 0..PANELS {
        let (low, high) = (
            panel as f64 / PANELS as f64,
            (panel + 1) as f64 / PANELS as f64,
        );
        for chord in 0..CHORDS {
            let (aft, forward) = (
                chord as f64 / CHORDS as f64,
                (chord + 1) as f64 / CHORDS as f64,
            );
            let corners = [
                station(aft, low),
                station(forward, low),
                station(forward, high),
                station(aft, high),
            ];
            // Two triangles a quad, then the same two reversed. The normal comes
            // from the quad's own diagonals rather than from a triangle, so both
            // halves of a panel are lit alike and the surface reads as cloth
            // instead of as facets.
            let normal = (corners[2] - corners[0])
                .cross(corners[3] - corners[1])
                .normalize_or_zero();
            for winding in [[0, 1, 2], [0, 2, 3]] {
                for index in winding {
                    positions.push(corners[index].to_array());
                    normals.push(normal.to_array());
                }
            }
            for winding in [[0, 2, 1], [0, 3, 2]] {
                for index in winding {
                    positions.push(corners[index].to_array());
                    normals.push((-normal).to_array());
                }
            }
        }
    }

    let count = positions.len() as u32;
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32((0..count).collect()))
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

            // The sails, as the engine says they are flying.
            //
            // `sail_shapes` derives each planform from the rig's own letters — `P`
            // and `E` for the main, `√(I²+J²)` and `LPG` for the jib — so the
            // drawn sail is still exactly the size the force model is sailing.
            // What it adds is the camber and twist the current control positions
            // produce, which is the difference between a sail and a sheet of
            // plywood. `trim` keeps them that way as the controls move.
            //
            // A boat whose file carries no flying-shape block gets nothing here
            // rather than a fabricated bulge. That is the right failure: the
            // engine has no opinion about that sail's shape, and neither should
            // this.
            let mirror = -leeward_sign(&engine);
            for (sail, tack, shape) in
                vela_core::assembly::sail_shapes(&engine.spec, *engine.sim.controls())
            {
                boat.spawn((
                    Mesh3d(meshes.add(sail_mesh(&shape, tack, mirror))),
                    MeshMaterial3d(cloth.clone()),
                    DrawnSail {
                        sail,
                        shape,
                        mirror,
                    },
                ));
            }
        });
}

/// A drawn sail: which one, and what its mesh was last built from.
///
/// The mesh is a function of the flying shape and of the tack the boat is on,
/// and of nothing else. Holding both is what lets [`trim`] rebuild it exactly
/// when one of them changes and leave it alone otherwise.
#[derive(Component)]
pub struct DrawnSail {
    sail: Sail,
    shape: Shape,
    /// The body-`y` mirror the mesh was built with — see [`sail_mesh`].
    mirror: f64,
}

/// Marks the boom, which swings with the mainsail.
#[derive(Component)]
pub struct Boom;

/// The rig's published dimensions, in render coordinates.
///
/// Every field is an IOR letter out of the boat file — `P`, `E`, `I`, `J` — or a
/// height measured from the sheer, which is the top of the lofted hull. Nothing
/// here is chosen to look right: if a boat file declares a rig, this draws that
/// rig, and if the drawing looks wrong the file is wrong.
///
/// Heights are above the body origin, which is on the baseline — the datum the
/// hull is drawn from — and not above the water, which is the datum the engine
/// quotes the sails' tacks against. See [`sail_mesh`] for what that costs.
struct Rig {
    /// Where the mast stands, render `x`, m.
    mast_at: f32,
    /// Sheer height above the baseline, render `y`, m.
    sheer: f32,
    /// Masthead height above the baseline, m.
    masthead: f32,
    /// Boom height above the baseline, m.
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

/// Keeps the sails on the shape the engine is flying, and the boom under the
/// main.
///
/// The shape carries the sheeting angle itself: `Shape::chord_angle` at the foot
/// is the trim the sheet and traveller set, through the range the boat file
/// declares, and `sail_shapes` has already applied it. So the sails are never
/// rotated as rigid bodies — a rotation on top of the shape sheets them twice,
/// which is what this did first. What changes when a control moves is the shape,
/// and the mesh is rebuilt from it then and only then: a hundred-odd quads a
/// sail is nothing to build, but re-uploading two meshes every frame for a boat
/// nobody is trimming would be. Which tack the boat is on is the other input,
/// and a tack rebuilds the sails onto the new leeward side the same way.
///
/// The boom is the one thing that is rotated, because it is a spar and not
/// cloth: it lies along the foot of the main, so its angle is the main's chord
/// angle at the foot, on whichever side the tack puts it.
pub fn trim(
    engine: Res<Engine>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut sails: Query<(&mut DrawnSail, &Mesh3d)>,
    mut booms: Query<&mut Transform, With<Boom>>,
) {
    let leeward = leeward_sign(&engine);
    let mirror = -leeward;
    let shapes = vela_core::assembly::sail_shapes(&engine.spec, *engine.sim.controls());

    for (sail, tack, shape) in &shapes {
        for (mut drawn, mesh) in &mut sails {
            if drawn.sail != *sail || (drawn.shape == *shape && drawn.mirror == mirror) {
                continue;
            }
            if let Some(mut existing) = meshes.get_mut(mesh) {
                *existing = sail_mesh(shape, *tack, mirror);
                drawn.shape = *shape;
                drawn.mirror = mirror;
            }
        }
    }

    let Some((_, _, main)) = shapes.iter().find(|(sail, _, _)| *sail == Sail::Main) else {
        return;
    };
    let rig = Rig::from(&*engine);
    // A rotation of `θ` about render `y` takes the boom tip at `(-E, 0, 0)` to
    // `z = E sin θ`, and render `z` is the engine's `y`, which is starboard. The
    // foot's chord angle is measured to leeward, so `θ` carries the sign of the
    // leeward direction on the body `y` axis — which is what `leeward_sign` is.
    let rotation = Quat::from_rotation_y((leeward * main.chord_angle(0.0)) as f32);
    // The boom pivots at the mast, so the rotation is about a point the child
    // transform does not sit on: rotate, then put the midpoint back where the
    // rotation left it.
    for mut transform in &mut booms {
        transform.rotation = rotation;
        transform.translation = Vec3::new(rig.mast_at, rig.boom, 0.0)
            + rotation * Vec3::new(-0.5 * rig.main_foot, 0.0, 0.0);
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
fn leeward_sign(engine: &Engine) -> f64 {
    let angle = engine
        .sim
        .telemetry()
        .get("aero.apparent_wind.angle")
        .unwrap_or(1.0);
    vela_core::sim::leeward_sign(angle)
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
