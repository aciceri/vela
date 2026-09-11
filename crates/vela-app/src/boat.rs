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
use bevy::light::TransmittedShadowReceiver;
use bevy::mesh::{Indices, MeshVertexAttribute, MeshVertexBufferLayoutRef, VertexAttributeValues};
use bevy::pbr::{
    ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline,
};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, Face, PrimitiveTopology, RenderPipelineDescriptor, ShaderType,
    SpecializedMeshPipelineError, VertexFormat,
};
use bevy::shader::ShaderRef;
use nalgebra::Vector3;
use vela_core::aero::Sail;
use vela_core::flying::Shape;
use vela_core::frames::file_to_body;
use vela_core::geometry::Point;

use crate::frame;
use crate::ocean::embedded_shader;
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

/// Embeds the model and registers its material extension and one-time upgrade.
pub struct BoatPlugin;

impl Plugin for BoatPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "models/sailboat.glb");
        embedded_asset!(app, "shaders/boat.wgsl");
        embedded_asset!(app, "shaders/boat_prepass.wgsl");
        app.init_resource::<HullContact>()
            .add_plugins(MaterialPlugin::<BoatMaterial>::default())
            .add_systems(Update, (finish_materials, update_surface).chain())
            .add_systems(
                PostUpdate,
                reset_rigid_motion.after(bevy::transform::TransformSystems::Propagate),
            );
    }
}

/// A reset changes the simulation, not the camera. Retain camera velocity while
/// removing the artificial hull/boom travel from the old simulation state.
fn reset_rigid_motion(
    engine: Res<Engine>,
    mut previous_time: Local<Option<f64>>,
    mut meshes: Query<
        (&GlobalTransform, &mut bevy::pbr::PreviousGlobalTransform),
        With<MeshMaterial3d<BoatMaterial>>,
    >,
) {
    let time = engine.sim.time();
    let reset = previous_time.is_none_or(|previous| time < previous || time - previous > 0.25);
    *previous_time = Some(time);
    if reset {
        for (current, mut previous) in &mut meshes {
            previous.0 = current.affine();
        }
    }
}

type BoatMaterial = ExtendedMaterial<StandardMaterial, BoatSurface>;

/// The atlas already classifies metal in its blue ORM channel. The extension
/// preserves that classification, all texture handles and the baked normals.
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
struct BoatSurface {
    /// x: atlas / cloth / hardware; zw: the sail's unrolled foot and luff, m.
    /// Vec4 fields and arrays keep the uniform layout valid on WebGL2.
    #[uniform(100)]
    finish: Vec4,
    #[uniform(101)]
    environment: BoatEnvironment,
}

impl MaterialExtension for BoatSurface {
    fn fragment_shader() -> ShaderRef {
        embedded_shader!("shaders/boat.wgsl")
    }

    fn deferred_fragment_shader() -> ShaderRef {
        embedded_shader!("shaders/boat.wgsl")
    }

    fn prepass_vertex_shader() -> ShaderRef {
        embedded_shader!("shaders/boat_prepass.wgsl")
    }

    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Preserve every StandardMaterial prepass binding and attribute. Only
        // deforming sails carry the extra previous-local-position stream.
        let prepass = descriptor
            .vertex
            .shader_defs
            .iter()
            .any(|def| *def == "PREPASS_PIPELINE".into());
        if prepass && layout.0.contains(PREVIOUS_POSITION) {
            let previous = layout
                .0
                .get_layout(&[PREVIOUS_POSITION.at_shader_location(8)])?;
            descriptor.vertex.buffers[0]
                .attributes
                .extend(previous.attributes);
            descriptor
                .vertex
                .shader_defs
                .push("SAIL_PREVIOUS_POSITION".into());
        }
        Ok(())
    }
}

const CONTACT_COLUMNS: usize = 17;
const CONTACT_ROWS: usize = 5;
const CONTACT_SAMPLES: usize = CONTACT_COLUMNS * CONTACT_ROWS;

/// Contact is sampled in boat-local metres, so its drying history stays on the
/// same hull parcel through translation, yaw and heel. No fragment wave sums.
#[derive(Clone, Copy, Debug, Reflect, ShaderType)]
struct BoatEnvironment {
    boat_from_world: Mat4,
    grid: Vec4,
    sun_time: Vec4,
    /// xy: current sea intersection and draining wet-film height, in local y.
    contact: [Vec4; CONTACT_SAMPLES],
}

impl Default for BoatEnvironment {
    fn default() -> Self {
        Self {
            boat_from_world: Mat4::IDENTITY,
            grid: Vec4::new(0.0, -2.0, 0.75, 1.0),
            sun_time: crate::sky::SUN.normalize().extend(0.0),
            contact: [Vec4::ZERO; CONTACT_SAMPLES],
        }
    }
}

#[derive(Resource, Default)]
struct HullContact {
    environment: BoatEnvironment,
    time: Option<f64>,
}

fn update_surface(
    engine: Res<Engine>,
    mut history: ResMut<HullContact>,
    mut materials: ResMut<Assets<BoatMaterial>>,
) {
    let time = engine.sim.time();
    if history.time == Some(time) {
        return;
    }
    let reset = history.time.is_none_or(|previous| time < previous);
    let dt = history
        .time
        .map_or(0.0, |previous| (time - previous).max(0.0)) as f32;
    history.time = Some(time);
    let state = engine.sim.state();
    let rotation = frame::rotation(&state.attitude);
    let position = frame::to_render(state.position);
    let world_from_boat = Mat4::from_rotation_translation(rotation, position);
    let environment = &mut history.environment;
    environment.boat_from_world = world_from_boat.inverse();
    environment.sun_time = crate::sky::SUN.normalize().extend(time as f32);
    if let Some((min, max)) = engine.hull.bounds() {
        environment.grid = Vec4::new(
            frame::metres(min.x) - 0.15,
            frame::metres(min.y) - 0.25,
            frame::metres(max.x - min.x + 0.30) / (CONTACT_COLUMNS - 1) as f32,
            frame::metres(max.y - min.y + 0.50) / (CONTACT_ROWS - 1) as f32,
        );
    }
    let grid = environment.grid;
    let up = rotation * Vec3::Y;
    // A height field in the boat frame is well-conditioned throughout sailing
    // heel angles. Near a knockdown retain the last finite contact, rather than
    // dividing by a horizontal body-up axis.
    if up.y > 0.15 {
        for row in 0..CONTACT_ROWS {
            for column in 0..CONTACT_COLUMNS {
                let index = row * CONTACT_COLUMNS + column;
                let origin = world_from_boat.transform_point3(Vec3::new(
                    grid.x + column as f32 * grid.z,
                    0.0,
                    grid.y + row as f32 * grid.w,
                ));
                let previous = environment.contact[index];
                let mut height = if reset { -origin.y / up.y } else { previous.x };
                // Fixed-point intersection: heel moves xz as local height
                // changes. Three CPU samples include that displacement rather
                // than sampling the unheeled baseline or a fixed body height.
                for _ in 0..3 {
                    let world = origin + up * height;
                    let sea = engine.sea.as_ref().map_or(0.0, |sea| {
                        frame::metres(sea.elevation(f64::from(world.x), f64::from(world.z), time))
                    });
                    height = ((sea - origin.y) / up.y).clamp(-6.0, 6.0);
                }
                // A draining film recedes at 2.5 cm/s in body coordinates.
                // Its finite 18 cm feather also fades parcels above contact;
                // raising the sea immediately re-wets them.
                let film = if reset {
                    height
                } else {
                    height.max(previous.y - dt * 0.025)
                };
                environment.contact[index] = Vec4::new(height, film, 0.0, 0.0);
            }
        }
    }
    for (_, material) in materials.iter_mut() {
        material.extension.environment = *environment;
    }
}

/// Removed after the loaded material has been upgraded; nothing is cloned or
/// allocated on subsequent frames, including while the boat is being trimmed.
#[derive(Component, Clone, Copy)]
enum BoatFinish {
    Atlas,
    Cloth(Sail),
    Hardware,
}

fn finish_materials(
    mut commands: Commands,
    assets: Res<AssetServer>,
    standard: Res<Assets<StandardMaterial>>,
    mut extended: ResMut<Assets<BoatMaterial>>,
    history: Res<HullContact>,
    pending: Query<(
        Entity,
        &MeshMaterial3d<StandardMaterial>,
        &BoatFinish,
        Option<&DrawnSail>,
    )>,
    mut finished: Local<[Option<Handle<BoatMaterial>>; 5]>,
) {
    for (entity, original, finish, sail) in &pending {
        let index = match finish {
            BoatFinish::Atlas => 0,
            BoatFinish::Cloth(sail) => *sail as usize + 1,
            BoatFinish::Hardware => 4,
        };
        if finished[index].is_none() {
            // The model is asynchronous, including its JPEG dependencies.
            // Procedural cloth is inserted directly into Assets, not loaded.
            if index == 0 && !assets.is_loaded_with_dependencies(original.0.id()) {
                continue;
            }
            let Some(base) = standard.get(&original.0) else {
                continue;
            };
            let mut base = base.clone();
            if index == 0 {
                // Enable the clearcoat shader path; its per-pixel strength is
                // masked to gelcoat/varnish, never to metal, by the extension.
                base.clearcoat = 1.0;
            }
            let finish = if let Some(sail) = sail {
                Vec4::new(
                    1.0,
                    0.0,
                    sail.shape.planform().chord(0.0) as f32,
                    sail.shape.planform().luff() as f32,
                )
            } else {
                Vec4::new(if index == 0 { 0.0 } else { 2.0 }, 0.0, 0.0, 0.0)
            };
            finished[index] = Some(extended.add(BoatMaterial {
                base,
                extension: BoatSurface {
                    finish,
                    environment: history.environment,
                },
            }));
        }
        if let Some(material) = &finished[index] {
            commands
                .entity(entity)
                .remove::<(MeshMaterial3d<StandardMaterial>, BoatFinish)>()
                .insert(MeshMaterial3d(material.clone()));
        }
    }
}

/// Visual tessellation resolves centimetre-amplitude tension folds without
/// changing the engine's flying shape or the hull used for pressure integration.
const CHORDS: usize = 24;

/// Finer spanwise sampling rounds the cloth and resolves the corner fans.
const PANELS: usize = 48;

/// CPU cloth deformation is not a rigid transform or a GPU morph target.
/// An extra vertex attribute works on WebGL2 without storage buffers.
const PREVIOUS_POSITION: MeshVertexAttribute =
    MeshVertexAttribute::new("PreviousSailPosition", 0x5645_4c41, VertexFormat::Float32x3);

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
fn sail_mesh(
    shape: &Shape,
    tack: Point,
    mirror: f64,
    cut: Cut,
    previous: Option<VertexAttributeValues>,
) -> Mesh {
    let tack = file_to_body(tack);
    let station = |along: f64, up: f64| {
        let cloth = Vec2::new(
            (along * shape.planform().chord(up)) as f32,
            (up * shape.planform().luff()) as f32,
        );
        let foot = shape.planform().chord(0.0) as f32;
        let luff = shape.planform().luff() as f32;
        // Radial folds run along corner load paths. More draft and twist mean
        // more slack; flattening/tightening the flying shape reduces them.
        // This is a bounded visual cloth relief, not a new force model.
        let slack = (shape.profile(up).camber() as f32 * 4.0
            + (shape.chord_angle(1.0) - shape.chord_angle(0.0)).abs() as f32 * 0.5)
            .clamp(0.05, 1.0);
        let mut fold = 0.0;
        for corner in [Vec2::ZERO, Vec2::new(foot, 0.0), Vec2::new(0.0, luff)] {
            let delta = cloth - corner;
            let radius = delta.length();
            let angle = delta.y.atan2(delta.x);
            fold += (angle * 8.0).sin() * (-radius / 1.4).exp() * (radius / 0.16).min(1.0);
        }
        // Keep the attachment points and all three cut edges on the engine's
        // surface. The relief mirrors with the cloth, before frame conversion.
        let pinned = ((along * (1.0 - along) * up * (1.0 - up) * 140.0) as f32).clamp(0.0, 1.0);
        let mut file = shape.point(along, up);
        file += shape.camber_direction(up) * f64::from(0.012 * slack * fold * pinned);
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
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(vertices);
    let mut cut_uvs: Vec<[f32; 2]> = Vec::with_capacity(vertices);
    let cloth_coordinate = |along: f64, up: f64| {
        [
            (along * shape.planform().chord(up)) as f32,
            (up * shape.planform().luff()) as f32,
        ]
    };

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
            // Unrolled cloth metres, not normalized triangle coordinates: weave
            // and crosscut seam widths stay physical under trim and on both tacks.
            let cloth_corners = [
                cloth_coordinate(aft, low),
                cloth_coordinate(forward, low),
                cloth_coordinate(forward, high),
                cloth_coordinate(aft, high),
            ];
            let cut_corners = [
                [aft as f32, low as f32],
                [forward as f32, low as f32],
                [forward as f32, high as f32],
                [aft as f32, high as f32],
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
                    uvs.push(cloth_corners[index]);
                    cut_uvs.push(cut_corners[index]);
                }
            }
            for winding in [[0, 2, 1], [0, 3, 2]] {
                for index in winding {
                    positions.push(corners[index].to_array());
                    normals.push((-normal).to_array());
                    uvs.push(cloth_corners[index]);
                    cut_uvs.push(cut_corners[index]);
                }
            }
        }
    }

    let count = positions.len() as u32;
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(
        PREVIOUS_POSITION,
        previous.unwrap_or_else(|| VertexAttributeValues::Float32x3(positions.clone())),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_1, cut_uvs)
    .with_inserted_indices(Indices::U32((0..count).collect()))
}

/// The inspected GLB already contains rails, pulpits, six winches, coachroof
/// handrails and mast spreaders (2,340 hull triangles). Its preparation removed
/// standing rigging. Add that missing load path and its joints, not a second
/// set of deck fittings. All dimensions below are actual metres.
fn standing_rigging(rig: &Rig, engine: &Engine) -> Mesh {
    let mut hardware = HardwareMesh::default();
    let head = Vec3::new(rig.mast_at, rig.masthead - 0.06, 0.0);
    let beam = engine
        .hull
        .bounds()
        .map_or(1.65, |(min, max)| frame::metres(max.y - min.y) * 0.44);
    let bow = Vec3::new(rig.mast_at + rig.foretriangle_base, rig.sheer + 0.06, 0.0);
    let stern = Vec3::new(0.80, rig.sheer + 0.02, 0.0);
    hardware.stay(bow, head);
    // Split backstay clears the cockpit; both lower legs end at real deck
    // height rather than crossing the wheel already modelled in the atlas.
    let split = Vec3::new(1.0, rig.sheer + 2.25, 0.0);
    hardware.tube(split, head, 0.0045, 10);
    for side in [-1.0, 1.0] {
        hardware.stay(stern + Vec3::Z * side * beam * 0.63, split);
        let chainplate = Vec3::new(rig.mast_at + 0.16, rig.sheer, side * beam);
        // Actual GLB spreader-tip bounds: x=-0.16, z=±1.035 and
        // normalized mast y=0.547. Route the cap shroud over that hardware.
        let spreader = Vec3::new(
            rig.mast_at - 0.16,
            rig.sheer + (rig.masthead - rig.sheer) * 0.547,
            side * 1.035,
        );
        hardware.stay(chainplate, spreader);
        hardware.tube(spreader, head, 0.0045, 10);
        hardware.tube(
            spreader - Vec3::Y * 0.027,
            spreader + Vec3::Y * 0.027,
            0.014,
            12,
        );
        hardware.stay(
            chainplate - Vec3::X * 0.65,
            Vec3::new(rig.mast_at, spreader.y, side * 0.10),
        );
    }
    // The gooseneck is a 28 mm pivot with two cheeks and end washers, not a
    // floating boom end. Fixed to the mast; the existing Boom rotates around it.
    let pivot = Vec3::new(rig.mast_at - 0.10, rig.boom, 0.0);
    hardware.tube(pivot - Vec3::Y * 0.22, pivot + Vec3::Y * 0.22, 0.014, 16);
    for end in [-1.0, 1.0] {
        let center = pivot + Vec3::Y * end * 0.20;
        hardware.tube(
            center - Vec3::Y * 0.008,
            center + Vec3::Y * 0.008,
            0.037,
            16,
        );
        hardware.tube(center, center + Vec3::X * 0.22, 0.022, 12);
    }
    hardware.mesh()
}

#[derive(Default)]
struct HardwareMesh {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
}

impl HardwareMesh {
    fn triangle(&mut self, points: [Vec3; 3], normals: [Vec3; 3], uvs: [[f32; 2]; 3]) {
        self.positions.extend(points.map(|point| point.to_array()));
        self.normals.extend(normals.map(|normal| normal.to_array()));
        self.uvs.extend(uvs);
    }

    /// Closed cylinder with explicit cap normals and a round silhouette.
    fn tube(&mut self, start: Vec3, end: Vec3, radius: f32, sides: usize) {
        let axis = (end - start).normalize_or_zero();
        if axis == Vec3::ZERO {
            return;
        }
        let u = axis.any_orthonormal_vector();
        let v = axis.cross(u);
        let length = start.distance(end);
        for side in 0..sides {
            let a = side as f32 * std::f32::consts::TAU / sides as f32;
            let b = (side + 1) as f32 * std::f32::consts::TAU / sides as f32;
            let n0 = u * a.cos() + v * a.sin();
            let n1 = u * b.cos() + v * b.sin();
            let p = [
                start + radius * n0,
                start + radius * n1,
                end + radius * n1,
                end + radius * n0,
            ];
            let uv = [
                [radius * a, 0.0],
                [radius * b, 0.0],
                [radius * b, length],
                [radius * a, length],
            ];
            self.triangle([p[0], p[1], p[2]], [n0, n1, n1], [uv[0], uv[1], uv[2]]);
            self.triangle([p[0], p[2], p[3]], [n0, n1, n0], [uv[0], uv[2], uv[3]]);
            self.triangle([start, p[1], p[0]], [-axis; 3], [[0.0; 2]; 3]);
            self.triangle([end, p[3], p[2]], [axis; 3], [[0.0; 2]; 3]);
        }
    }

    fn stay(&mut self, anchor: Vec3, end: Vec3) {
        let axis = (end - anchor).normalize_or_zero();
        let cross_pin = axis.any_orthonormal_vector() * 0.032;
        // Swaged wire, threaded stud, open turnbuckle cage, clevis and pin.
        self.tube(anchor + axis * 0.37, end, 0.0045, 10);
        self.tube(anchor + axis * 0.28, anchor + axis * 0.43, 0.008, 12);
        for side in [-1.0, 1.0] {
            let offset = cross_pin * side * 0.38;
            self.tube(
                anchor + axis * 0.09 + offset,
                anchor + axis * 0.29 + offset,
                0.0055,
                10,
            );
            self.tube(anchor + offset, anchor + axis * 0.09 + offset, 0.008, 10);
        }
        for distance in [0.09, 0.29] {
            self.tube(
                anchor + axis * (distance - 0.013),
                anchor + axis * (distance + 0.013),
                0.020,
                12,
            );
        }
        self.tube(anchor - cross_pin, anchor + cross_pin, 0.008, 12);
        // Exposed chainplate strap carries the terminal into the deck edge.
        self.tube(anchor - Vec3::Y * 0.18, anchor, 0.016, 8);
    }

    fn mesh(self) -> Mesh {
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::RENDER_WORLD,
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, self.uvs)
    }
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
        base_color: Color::srgb(0.94, 0.935, 0.905),
        perceptual_roughness: 0.84,
        reflectance: 0.38,
        diffuse_transmission: 0.18,
        thickness: 0.0004,
        double_sided: true,
        // sail_mesh already carries both windings with opposed normals.
        cull_mode: Some(Face::Back),
        ..default()
    });

    let rig = Rig::from(&*engine);
    let rigging_mesh = meshes.add(standing_rigging(&rig, &engine));
    let steel = materials.add(StandardMaterial {
        base_color: Color::srgb(0.63, 0.66, 0.69),
        metallic: 0.95,
        perceptual_roughness: 0.27,
        ..default()
    });
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
                BoatFinish::Atlas,
                layers.clone(),
            ));
            boat.spawn((
                Mesh3d(rigging_mesh),
                MeshMaterial3d(steel),
                BoatFinish::Hardware,
                layers.clone(),
            ));

            // Mast: the model's, a unit-high spar at its foot, stood on the
            // sheer and stretched to the masthead. Its width is the model's
            // own, scaled with the hull, and not the file's diameter.
            boat.spawn((
                Mesh3d(assets.load(model_mesh(MODEL_MAST))),
                MeshMaterial3d(skin.clone()),
                BoatFinish::Atlas,
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
                BoatFinish::Atlas,
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
                    Mesh3d(meshes.add(sail_mesh(&shape, tack, mirror, cut, None))),
                    MeshMaterial3d(cloth.clone()),
                    BoatFinish::Cloth(sail),
                    TransmittedShadowReceiver,
                    DrawnSail {
                        sail,
                        shape,
                        mirror,
                        cut,
                        motion_pending: false,
                        time: engine.sim.time(),
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
    /// Settle the previous attribute once after a changed shape, not every frame.
    motion_pending: bool,
    time: f64,
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
            if drawn.sail != *sail {
                continue;
            }
            let time = engine.sim.time();
            let reset = time < drawn.time || time - drawn.time > 0.25;
            drawn.time = time;
            if drawn.shape == *shape && drawn.mirror == mirror {
                if drawn.motion_pending {
                    if let Some(mut existing) = meshes.get_mut(mesh) {
                        let current = existing
                            .attribute(Mesh::ATTRIBUTE_POSITION)
                            .unwrap()
                            .clone();
                        existing.insert_attribute(PREVIOUS_POSITION, current);
                        drawn.motion_pending = false;
                    }
                }
                continue;
            }
            if let Some(mut existing) = meshes.get_mut(mesh) {
                let previous = if reset {
                    None
                } else {
                    existing.remove_attribute(Mesh::ATTRIBUTE_POSITION)
                };
                *existing = sail_mesh(shape, *tack, mirror, drawn.cut, previous);
                drawn.shape = *shape;
                drawn.mirror = mirror;
                drawn.motion_pending = !reset;
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
