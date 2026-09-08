//! The boat: a visual model worn over the physics hull, and the rig above it.
//!
//! # Two hulls, and which one is true
//!
//! The hull the physics integrates pressure over is lofted from the boat file's
//! station offsets by `vela_core::loft`; it is watertight, has a flat deck lid,
//! no coachroof and no appendages, because the pressure integral needs none of
//! those. It used to be the hull drawn, on the principle that two hulls could
//! drift apart with no symptom until the waterline disagreed with the picture.
//! It looked like what it was, and a viewer said so.
//!
//! What is drawn now is a textured model — `models/sailboat.glb`, "Sailboat"
//! by Sergei (sergeif) on Sketchfab, CC BY 4.0 — prepared in Blender into the
//! render frame at the physics hull's length: aft perpendicular at `x = 0`,
//! stem at the lofted hull's length, canoe body on the baseline, and its own
//! sheer within six centimetres of the lofted one. The rest of its shape is the
//! model's, not the file's: its keel is shallower than the physics' and its
//! beam a few centimetres narrower. Those are the errors accepted, and they
//! are stated here rather than hidden because the principle above was right:
//! the physics hull is still the only one that decides where the waterline
//! is, and the model is a skin that fits it at the stations that matter.
//!
//! The model was stripped of its sails and rigging and split into three
//! meshes — hull, mast, boom — because the spars have to follow the *rig*
//! dimensions of the boat file, not the model's: the mast mesh is a unit-high
//! spar at the origin, scaled to the file's masthead; the boom a unit-long spar
//! pointing aft from the gooseneck, scaled to `E` and swung by the sheeting
//! angle like the box it replaced. The glb is embedded in the binary for the
//! reason the shaders are: nothing to fetch, nothing to fail.
//!
//! # The sails are the engine's
//!
//! The sails are the flying shape [`vela_core::flying`] computes for the
//! control positions the player is holding — camber, draft, twist and the
//! sheeting angle, all from the engine's own statement of what the cloth does
//! — and they are rebuilt whenever that statement changes. What the drawing
//! does **not** claim is that the *forces* come from that shape: the reference
//! boat sails on the tabular aerodynamic model, which reads areas, an aspect
//! ratio and an angle and never asks for a shape. [`sail_mesh`] says why
//! drawing the shape anyway is the honest way round.

use bevy::asset::{embedded_asset, RenderAssetUsages};
use bevy::mesh::Indices;
use bevy::prelude::*;
use bevy::render::render_resource::PrimitiveTopology;
use nalgebra::Vector3;
use vela_core::aero::Sail;
use vela_core::flying::Shape;
use vela_core::frames::file_to_body;
use vela_core::geometry::Point;

use crate::frame;
use crate::reflection;
use crate::sim::Engine;

/// Marks the entity that carries the boat's pose.
///
/// Hull and rig are children of it, so the pose is applied once and they cannot
/// come apart.
#[derive(Component)]
pub struct Boat;

/// The model, embedded; see the module documentation.
const MODEL: &str = "embedded://vela_app/models/sailboat.glb";

/// The three meshes of the model, by the index the glTF exporter gave them:
/// nodes are written in name order, and the file was checked after export.
/// Raise the number here if the model is ever re-exported with more parts.
const MODEL_BOOM: usize = 0;
const MODEL_HULL: usize = 1;
const MODEL_MAST: usize = 2;

/// A mesh of the model, as an asset path the glTF loader resolves to the
/// primitive itself rather than to a scene.
fn model_mesh(index: usize) -> String {
    format!("{MODEL}#Mesh{index}/Primitive0")
}

/// The model's one material — the baked colour, roughness and normal maps —
/// which all three meshes share. `bevy_pbr` registers the `StandardMaterial`
/// it builds from each glTF material under the glTF label with `/std` after
/// it; the bare label is the loader's own `GltfMaterial`.
fn model_material() -> String {
    format!("{MODEL}#Material0/std")
}

/// Embeds the model in the binary.
///
/// Only that: the systems are registered by `main`, where their order against
/// the rest of the frame is stated in one place.
pub struct BoatPlugin;

impl Plugin for BoatPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "models/sailboat.glb");
    }
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
/// # The cut
///
/// The shape's luff runs straight up its own `z` from the tack, which is where
/// `flying` leaves placement to the rig. For a main that *is* the placement:
/// the luff is the mast. For a jib the luff is the forestay, which runs from
/// the tack on the stemhead aft and up to the masthead, so the drawn jib
/// stood on end above the bow with its head five metres ahead of the mast,
/// and its foot ran aft at deck level straight through the coachroof. `Cut`
/// is the two corrections a sailmaker would make: each section is moved aft
/// by its height times the stay's rake, `J / I`, which lays the luff on the
/// stay with the head at the masthead; and the foot is lifted towards the
/// clew, which is the foot angle every headsail is cut with so that the clew
/// clears the deck. Both are shears rather than rotations, so a chord stays a
/// chord and the flying shape's angles are the ones drawn. The area drawn
/// changes by nothing the eye can find and the forces do not read it at all.
///
/// Two-sided, because a sail seen from the leeward side would otherwise vanish.
/// The two windings get opposed normals, so each face is lit as the surface it is.
fn sail_mesh(shape: &Shape, tack: Point, mirror: f64, cut: Cut) -> Mesh {
    let tack = file_to_body(tack);
    let station = |along: f64, up: f64| {
        let mut file = shape.point(along, up);
        // Aft by the rake, and up by the foot's rise: `along` runs from the
        // luff to the leech, `file.z` from the tack to the head.
        file.x -= cut.rake * file.z;
        file.z += cut.foot_rise * along * (1.0 - up);
        let body = tack + file_to_body(file);
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

/// Spawns the model's hull, its mast and boom placed by the rig, and two sails
/// under one pose entity.
pub fn spawn(
    mut commands: Commands,
    engine: Res<Engine>,
    assets: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let skin: Handle<StandardMaterial> = assets.load(model_material());
    let cloth = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.95, 0.93),
        perceptual_roughness: 0.9,
        double_sided: true,
        cull_mode: None,
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

    // Each drawn part is on the mirror's layer as well as the main view's;
    // layers are not inherited, so the pose entity above them carries none.
    let layers = reflection::mirrored();
    commands
        .spawn((Boat, Transform::default(), Visibility::default()))
        .with_children(|boat| {
            // The hull is already in the render frame at the physics hull's
            // length; see the module documentation.
            boat.spawn((
                Mesh3d(assets.load(model_mesh(MODEL_HULL))),
                MeshMaterial3d(skin.clone()),
                layers.clone(),
            ));

            // Mast: the model's, a unit-high spar at its foot, stood on the
            // sheer and stretched to the masthead. Its width is the model's
            // own, scaled with the hull, and not the file's diameter.
            boat.spawn((
                Mesh3d(assets.load(model_mesh(MODEL_MAST))),
                MeshMaterial3d(skin.clone()),
                Transform::from_xyz(rig.mast_at, rig.sheer, 0.0).with_scale(Vec3::new(
                    1.0,
                    rig.masthead - rig.sheer,
                    1.0,
                )),
                layers.clone(),
            ));

            // Boom: the model's, a unit-long spar pointing aft from the
            // gooseneck, stretched to the foot. `trim` swings it.
            boat.spawn((
                Mesh3d(assets.load(model_mesh(MODEL_BOOM))),
                MeshMaterial3d(skin),
                Transform::from_xyz(rig.mast_at, rig.boom, 0.0).with_scale(Vec3::new(
                    rig.main_foot,
                    1.0,
                    1.0,
                )),
                Boom,
                layers.clone(),
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
                let cut = Cut::of(sail, &rig);
                boat.spawn((
                    Mesh3d(meshes.add(sail_mesh(&shape, tack, mirror, cut))),
                    MeshMaterial3d(cloth.clone()),
                    DrawnSail {
                        sail,
                        shape,
                        mirror,
                        cut,
                    },
                    layers.clone(),
                ));
            }
        });
}

/// A drawn sail: which one, and what its mesh was last built from.
///
/// The mesh is a function of the flying shape, of the tack the boat is on and
/// of the cut, and of nothing else. Holding all three is what lets [`trim`]
/// rebuild it exactly when one of them changes and leave it alone otherwise.
#[derive(Component)]
pub struct DrawnSail {
    sail: Sail,
    shape: Shape,
    /// The body-`y` mirror the mesh was built with — see [`sail_mesh`].
    mirror: f64,
    cut: Cut,
}

/// How a sail is cut onto its rig; see [`sail_mesh`].
#[derive(Clone, Copy, Debug, PartialEq)]
struct Cut {
    /// Aft displacement of the luff per metre of height: `J / I` for a jib
    /// on its forestay, zero for a main on its mast.
    rake: f64,
    /// Rise of the foot from tack to clew, m.
    foot_rise: f64,
}

impl Cut {
    /// A sail whose luff is a straight spar: the main.
    const STRAIGHT: Self = Self {
        rake: 0.0,
        foot_rise: 0.0,
    };

    /// The cut for a sail on this rig.
    ///
    /// The jib's foot rise is a metre and a quarter: enough that the clew of a
    /// five-metre foot clears a coachroof of the usual height, and a foot angle
    /// of thirteen degrees, which is what a genoa's is.
    fn of(sail: Sail, rig: &Rig) -> Self {
        match sail {
            Sail::Jib if rig.foretriangle_height > 0.0 => Self {
                rake: f64::from(rig.foretriangle_base / rig.foretriangle_height),
                foot_rise: 1.25,
            },
            _ => Self::STRAIGHT,
        }
    }
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
                *existing = sail_mesh(shape, *tack, mirror, drawn.cut);
                drawn.shape = *shape;
                drawn.mirror = mirror;
            }
        }
    }

    let Some((_, _, main)) = shapes.iter().find(|(sail, _, _)| *sail == Sail::Main) else {
        return;
    };
    // A rotation of `θ` about render `y` takes the boom tip at `(-E, 0, 0)` to
    // `z = E sin θ`, and render `z` is the engine's `y`, which is starboard. The
    // foot's chord angle is measured to leeward, so `θ` carries the sign of the
    // leeward direction on the body `y` axis — which is what `leeward_sign` is.
    // The boom mesh's origin is the gooseneck, so the pivot is the transform's
    // own and nothing has to be moved back.
    let rotation = Quat::from_rotation_y((leeward * main.chord_angle(0.0)) as f32);
    for mut transform in &mut booms {
        transform.rotation = rotation;
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
