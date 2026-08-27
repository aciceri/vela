//! The camera, the light, and the patch of sea that follows the boat.
//!
//! # The sea is a moving window, not an ocean
//!
//! The surface is a finite grid that is kept centred on the boat and snapped to
//! its own cell size. Because the elevation is a closed form of *world* position,
//! moving the grid does not move the water: a wave crest stays where it is while
//! the mesh slides under it. That is the whole reason the sea is shared as a
//! realisation rather than as a height field — a height field would have to be
//! rebuilt every time the window moved, and this does not have to be rebuilt at
//! all.
//!
//! The snapping matters. Without it the grid's vertices would slide continuously
//! and the surface would visibly crawl, because a coarse grid samples a wave
//! differently depending on where its vertices land. Snapping to the cell size
//! keeps every vertex on a fixed world lattice, so the sampling is stationary
//! even while the window moves.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::Indices;
use bevy::prelude::*;
use bevy::render::render_resource::PrimitiveTopology;

use crate::boat::Boat;
use crate::frame;
use crate::ocean::OceanMaterial;
use crate::sim::Engine;

/// Half-width of the drawn sea, m.
///
/// Two hundred metres of visible water around a twelve metre boat: far enough
/// that the edge is not the first thing a viewer sees, near enough that the
/// vertex count stays reasonable.
const REACH: f32 = 200.0;

/// Cells across the sea grid.
///
/// 256 cells over 400 m is a 1.56 m cell. The shortest wave in the default
/// realisation has a period around 1.5 s, so a length near 3.5 m — which this
/// samples with about two cells and therefore does not resolve. That is a
/// deliberate trade and worth stating plainly rather than discovering: the short
/// waves carry little of the variance and the grid is sized for the swell that
/// carries most of it. A viewer sees the sea the boat is riding; the boat feels
/// all of it, because the physics samples the closed form per triangle and never
/// touches this grid.
const CELLS: u32 = 256;

/// Marks the sea surface entity.
#[derive(Component)]
pub struct Sea;

/// Marks the camera that follows the boat.
#[derive(Component)]
pub struct Chase;

/// A flat grid in the render plane, for the ocean shader to displace.
///
/// Positions only: the shader computes the normal from the analytic slope, so an
/// uploaded normal would be overwritten. No UVs either — nothing samples a
/// texture.
fn grid(reach: f32, cells: u32) -> Mesh {
    let step = 2.0 * reach / cells as f32;
    let count = (cells + 1) as usize;
    let mut positions = Vec::with_capacity(count * count);

    for row in 0..=cells {
        for column in 0..=cells {
            positions.push([
                -reach + column as f32 * step,
                0.0,
                -reach + row as f32 * step,
            ]);
        }
    }

    let mut indices = Vec::with_capacity((cells * cells * 6) as usize);
    for row in 0..cells {
        for column in 0..cells {
            let here = row * (cells + 1) + column;
            let next = here + cells + 1;
            // Wound counter-clockwise seen from above, which is what puts the
            // front face towards a camera looking down at the water.
            indices.extend_from_slice(&[here, next, here + 1]);
            indices.extend_from_slice(&[here + 1, next, next + 1]);
        }
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_indices(Indices::U32(indices))
}

/// Spawns the sea, the camera and the light.
pub fn spawn(
    mut commands: Commands,
    engine: Res<Engine>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut oceans: ResMut<Assets<OceanMaterial>>,
) {
    if let Some(sea) = &engine.sea {
        commands.spawn((
            Mesh3d(meshes.add(grid(REACH, CELLS))),
            MeshMaterial3d(oceans.add(OceanMaterial::realising(sea, engine.sim.time()))),
            Transform::default(),
            Sea,
        ));
    }

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-28.0, 12.0, 22.0).looking_at(Vec3::new(0.0, 3.0, 0.0), Vec3::Y),
        // Sky light, a per-view component in this version of Bevy. Without it the
        // side of a sail facing away from the sun renders black, which reads as a
        // rendering fault rather than as shade — and it is wrong besides:
        // sailcloth is thin enough to glow when it is backlit. Not a translucency
        // model, just enough ambient that a shaded sail looks like cloth.
        AmbientLight {
            color: Color::srgb(0.62, 0.72, 0.85),
            brightness: 2_600.0,
            ..default()
        },
        Chase,
    ));

    // One directional light, no shadow map. An ocean plane is both the worst
    // shadow caster and the worst receiver, and the ocean shader lights itself
    // from the analytic normal in any case.
    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            ..default()
        },
        Transform::default().looking_to(Vec3::new(0.4, -0.8, -0.45), Vec3::Y),
    ));
}

/// Pushes the engine's clock into the ocean material.
///
/// The engine's time, not the render clock: see `crate::ocean`. This is the only
/// per-frame traffic between the physics and the water, and it is one float.
pub fn advance_sea(engine: Res<Engine>, mut oceans: ResMut<Assets<OceanMaterial>>) {
    let time = engine.sim.time();
    for (_, material) in oceans.iter_mut() {
        material.set_time(time);
    }
}

/// Keeps the sea grid centred on the boat, snapped to its own cell size.
pub fn follow_sea(engine: Res<Engine>, mut seas: Query<&mut Transform, With<Sea>>) {
    let centre = frame::to_render(engine.sim.state().position);
    let step = 2.0 * REACH / CELLS as f32;
    let snapped = Vec3::new(
        (centre.x / step).round() * step,
        0.0,
        (centre.z / step).round() * step,
    );
    for mut transform in &mut seas {
        transform.translation = snapped;
    }
}

/// Where the chase camera sits, in the boat's neighbourhood.
///
/// Spherical rather than a fixed offset, because the first thing anyone wants
/// from a simulator is to look at the thing from somewhere else — and because a
/// fixed viewpoint makes some questions unanswerable. "Is the sail there?" and
/// "is the boat floating at the right depth?" are both trivial from abeam and
/// both guesswork from astern.
#[derive(Resource)]
pub struct Orbit {
    /// Bearing of the camera from the boat, radians, 0 dead astern.
    pub azimuth: f32,
    /// Elevation above the horizontal, radians.
    pub elevation: f32,
    /// Distance from the boat, m.
    pub distance: f32,
    /// Height above the water the camera looks at, m.
    pub focus: f32,
}

impl Default for Orbit {
    fn default() -> Self {
        // Astern, above and off the quarter. The distance is set by the mast
        // rather than by the hull: the rig stands twenty metres, so a view framed
        // on a twelve metre waterline puts the masthead out of shot.
        Self {
            azimuth: 0.6,
            elevation: 0.28,
            distance: 62.0,
            focus: 8.0,
        }
    }
}

/// Moves the camera on its orbit.
///
/// Deliberately not on the arrow keys: those steer the boat, and a simulator
/// that made looking around and steering the same gesture would be unusable.
pub fn orbit(keys: Res<ButtonInput<KeyCode>>, time: Res<Time>, mut orbit: ResMut<Orbit>) {
    let dt = time.delta_secs();
    if keys.pressed(KeyCode::KeyJ) {
        orbit.azimuth += 1.2 * dt;
    }
    if keys.pressed(KeyCode::KeyL) {
        orbit.azimuth -= 1.2 * dt;
    }
    if keys.pressed(KeyCode::KeyI) {
        orbit.elevation = (orbit.elevation + 0.8 * dt).min(1.4);
    }
    if keys.pressed(KeyCode::KeyK) {
        orbit.elevation = (orbit.elevation - 0.8 * dt).max(-0.2);
    }
    if keys.pressed(KeyCode::KeyU) {
        orbit.distance = (orbit.distance * (1.0 - 0.9 * dt)).max(15.0);
    }
    if keys.pressed(KeyCode::KeyO) {
        orbit.distance = (orbit.distance * (1.0 + 0.9 * dt)).min(400.0);
    }
}

/// Holds the camera on its orbit around the boat, looking at it.
///
/// The orbit is in *world* axes rather than the boat's, so the view does not
/// heel and roll with the hull. A camera rigidly attached to a boat in a seaway
/// is unwatchable, and worse, it hides the motion the simulation exists to show
/// by making the boat the one still thing on screen.
pub fn chase(
    orbit: Res<Orbit>,
    boats: Query<&Transform, (With<Boat>, Without<Chase>)>,
    mut cameras: Query<&mut Transform, With<Chase>>,
) {
    let Ok(boat) = boats.single() else {
        return;
    };
    let (sin_azimuth, cos_azimuth) = orbit.azimuth.sin_cos();
    let (sin_elevation, cos_elevation) = orbit.elevation.sin_cos();
    // Azimuth zero is dead astern: render `x` is north, which is the bow.
    let offset = orbit.distance
        * Vec3::new(
            -cos_azimuth * cos_elevation,
            sin_elevation,
            sin_azimuth * cos_elevation,
        );
    let target = boat.translation + Vec3::Y * orbit.focus;
    for mut camera in &mut cameras {
        camera.translation = target + offset;
        camera.look_at(target, Vec3::Y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The grid must be a closed lattice of quads with the vertex and index
    /// counts that follow from the cell count.
    ///
    /// Cheap, and it catches the classic off-by-one where a grid of `n` cells is
    /// built with `n` vertices per side instead of `n + 1` — which leaves a seam
    /// a viewer sees as a crack in the water.
    #[test]
    fn the_grid_closes() {
        let mesh = grid(10.0, 4);
        assert_eq!(mesh.count_vertices(), 25);
        assert_eq!(
            mesh.indices().map(bevy::mesh::Indices::len),
            Some(4 * 4 * 6)
        );
    }
}
