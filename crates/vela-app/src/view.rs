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
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::ecs::message::MessageReader;
use bevy::input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::light::{CascadeShadowConfigBuilder, NotShadowCaster};
use bevy::mesh::Indices;
use bevy::post_process::bloom::{Bloom, BloomCompositeMode, BloomPrefilter};
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
/// 160 cells over 400 m is a 2.5 m cell. The shortest wave in the default
/// realisation has a period near 1.5 s and so a length near 3.5 m, which this
/// does not resolve — a deliberate trade, and worth stating plainly rather than
/// discovering: the short waves carry little of the variance, the grid is sized
/// for the swell that carries most of it, and the fragment stage puts the missing
/// centimetre scale back as a normal perturbation. A viewer sees the sea the boat
/// is riding; the boat feels all of it, because the physics samples the closed
/// form per hull triangle and never touches this grid.
///
/// Down from 256, which was measured rather than guessed. Every vertex evaluates
/// the whole sixty-component sum, so the grid alone was running eight million
/// transcendentals a frame and the frame budget showed it. Beyond about 130 m the
/// displacement is faded out anyway (see `resolved` in `shaders/ocean.wgsl`), so
/// most of what the extra density bought was detail in water that is drawn flat.
const CELLS: u32 = 160;

/// Outer radius of the far sea, m.
///
/// Chosen so the water runs out where the horizon is rather than before it. The
/// chase camera sits around twenty-five metres above the surface, and a flat
/// plane cut off at eight kilometres puts its edge 0.18 degrees below the eye's
/// level; a curved Earth would put the true horizon 0.16 degrees below it. The
/// seam therefore lands within a fiftieth of a degree — under a pixel at 1080p —
/// of where a viewer already expects the sea to end. That buys the entire
/// impression of distance without modelling curvature, which is a very good
/// trade for one constant.
const HORIZON: f32 = 8_000.0;

/// Angular segments around the far sea.
///
/// The ring is a disc seen almost edge-on, so its silhouette against the sky is
/// the only part of it a viewer really reads, and 128 segments make that
/// silhouette a 2.8 degree polygon — straight enough that no facet shows.
/// Doubling it would double a vertex count for nothing visible.
const HORIZON_SEGMENTS: u32 = 128;

/// Concentric rings across the far sea.
///
/// Forty-eight bands over a fortyfold change in radius, which is where the
/// geometric spacing in [`horizon_ring`] lands: each band about eight per cent
/// wider than the one inside it. The whole ring is then under a tenth of the
/// near grid's vertices, which is the point — uniform spacing fine enough for
/// the inner rim would need thousands of rings, and every band past a kilometre
/// would draw triangles smaller than a pixel.
const HORIZON_RINGS: u32 = 48;

/// Far bound of the chase camera's frustum, m.
///
/// Worth being precise about what this does, because the name misleads in this
/// version of Bevy: the matrix `PerspectiveProjection` builds is reverse-Z and
/// *infinite*, so `far` never clips a fragment and never enters the depth
/// mapping. It is the far plane of the culling frustum, and a mesh whose
/// bounding volume falls wholly outside it is dropped entire. Left at the
/// default kilometre the far sea would not be clipped at the kilometre mark; it
/// would simply never be drawn.
///
/// Twenty kilometres is the eight kilometre ring plus the four hundred metres the
/// orbit can pull the camera back, and then a wide margin, which costs nothing
/// precisely because `far` is not in the matrix.
const FAR_PLANE: f32 = 20_000.0;

/// Near plane of the chase camera, m.
///
/// This is the one that buys depth precision, and it is the reason to state it
/// rather than take Bevy's 0.1: with an infinite reverse-Z projection the depth
/// resolution at any distance scales with `near`, so five times the near plane is
/// five times the resolution eight kilometres out — which is where two
/// tessellations of the same water have to agree about which is in front.
///
/// Half a metre is free here. The orbit will not close inside fifteen metres of
/// the boat and the camera looks at the boat, so nothing ever comes near enough
/// for the foreground to notice.
const NEAR_PLANE: f32 = 0.5;

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

/// A flat annulus from `inner` to `outer`, for the ocean shader to displace.
///
/// The near grid stops at two hundred metres, and beyond it there was
/// background: an edge that reads as a wall rather than as distance. This is the
/// water that carries the eye from there to the horizon, and it can afford to be
/// very coarse, because everything it draws is at least two hundred metres away
/// and the short waves it fails to resolve are waves nobody at that range can
/// see.
///
/// The radial spacing is geometric rather than uniform — each ring's radius a
/// fixed ratio times the one inside it, the ratio derived from the two radii and
/// the ring count rather than tuned — which keeps every band subtending roughly
/// the same angle at the camera. That is the property worth having: it spends
/// vertices in proportion to how much of the screen they cover, whereas uniform
/// spacing is simultaneously too coarse at the inner rim and absurdly fine out
/// near the horizon.
///
/// Positions only, like [`grid`], and for the same reason.
pub fn horizon_ring(inner: f32, outer: f32, segments: u32, rings: u32) -> Mesh {
    // Each ring sits a fixed ratio further out than the last, so that `rings`
    // steps of it cover exactly the span asked for. Radii are computed from the
    // power rather than accumulated, which keeps the outermost ring on `outer`
    // instead of wherever forty-eight roundings drifted to.
    let ratio = (outer / inner).powf(1.0 / rings as f32);
    let arc = std::f32::consts::TAU / segments as f32;

    let mut positions = Vec::with_capacity(((rings + 1) * segments) as usize);
    for ring in 0..=rings {
        let radius = inner * ratio.powi(ring as i32);
        for segment in 0..segments {
            let angle = segment as f32 * arc;
            positions.push([radius * angle.cos(), 0.0, radius * angle.sin()]);
        }
    }

    let mut indices = Vec::with_capacity((rings * segments * 6) as usize);
    for ring in 0..rings {
        for segment in 0..segments {
            // The seam closes by wrapping onto the ring's first column rather
            // than by duplicating it, so there is no pair of coincident vertices
            // to keep in step.
            let beside_segment = (segment + 1) % segments;
            let here = ring * segments + segment;
            let beside = ring * segments + beside_segment;
            let out = (ring + 1) * segments + segment;
            let out_beside = (ring + 1) * segments + beside_segment;
            // Wound counter-clockwise seen from above, as in `grid`: the radial
            // direction stands in for the grid's columns and the angular one for
            // its rows, which makes this the same two triangles per quad.
            indices.extend_from_slice(&[here, beside, out]);
            indices.extend_from_slice(&[out, beside, out_beside]);
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
        // One material, two meshes. They differ only in how finely they sample
        // the same closed form, so sharing the handle is not an optimisation but
        // a statement that it is one sea — and it means the per-frame time push
        // in `advance_sea` cannot leave the two halves on different clocks.
        let ocean = oceans.add(OceanMaterial::realising(sea, engine.sim.time()));

        commands.spawn((
            Mesh3d(meshes.add(grid(REACH, CELLS))),
            MeshMaterial3d(ocean.clone()),
            Transform::default(),
            Sea,
            // The sea casts no shadow. Nothing in this scene is under it, and
            // leaving it a caster put both tessellations -- seventy-eight thousand
            // triangles -- through the shadow pass once per cascade for an effect
            // that cannot exist. Measured: it was most of the frame.
            NotShadowCaster,
        ));

        // Inner radius `REACH`, not `REACH * sqrt(2)`: the circle inscribed in
        // the near grid's square rather than the one drawn around it. The grid's
        // corners then overlap the ring instead of a gap opening between them,
        // and the trade is the right way round. Overlap costs two tessellations
        // of the same surface interpenetrating in four corner wedges, which two
        // hundred metres away is a faint seam; a gap is a hole in the water.
        //
        // Same `Sea` marker, so `follow_sea` carries this with the near grid. Its
        // own vertex spacing is far coarser than the snap step, so the ring's
        // sampling of the wave field is not stationary the way the grid's is —
        // but a sixteen metre triangle drifting by a metre and a half is not
        // something an eye can find at that range.
        commands.spawn((
            Mesh3d(meshes.add(horizon_ring(
                REACH,
                HORIZON,
                HORIZON_SEGMENTS,
                HORIZON_RINGS,
            ))),
            MeshMaterial3d(ocean),
            Transform::default(),
            Sea,
            // The sea casts no shadow. Nothing in this scene is under it, and
            // leaving it a caster put both tessellations -- seventy-eight thousand
            // triangles -- through the shadow pass once per cascade for an effect
            // that cannot exist. Measured: it was most of the frame.
            NotShadowCaster,
        ));
    }

    commands.spawn((
        Camera3d::default(),
        // Overrides the projection `Camera3d` requires into place. See
        // `FAR_PLANE`, which is what gets the far sea drawn at all, and
        // `NEAR_PLANE`, which is what gets it drawn without depth fighting.
        Projection::Perspective(PerspectiveProjection {
            near: NEAR_PLANE,
            far: FAR_PLANE,
            ..default()
        }),
        Transform::from_xyz(-28.0, 12.0, 22.0).looking_at(Vec3::new(0.0, 3.0, 0.0), Vec3::Y),
        // Sky light, a per-view component in this version of Bevy. Without it the
        // side of a sail facing away from the sun renders black, which reads as a
        // rendering fault rather than as shade — and it is wrong besides:
        // sailcloth is thin enough to glow when it is backlit. Not a translucency
        // model, just enough ambient that a shaded sail looks like cloth.
        AmbientLight {
            color: Color::srgb(0.72, 0.79, 0.88),
            brightness: 1_500.0,
            ..default()
        },
        // Sun on water is the one thing in this scene with a specular return far
        // brighter than anything else in frame. Without a display transform it
        // clips flat to white and reads as a hole in the sea rather than as
        // glitter, so the choice of transform is not decoration here.
        //
        // `AcesFitted` rather than Bevy's default `TonyMcMapface`, which needs a
        // lookup table the `tonemapping_luts` feature bakes in: this one is a
        // closed-form curve, so it cannot quietly degrade to the neutral
        // placeholder LUT if that feature is ever trimmed for download size. It
        // also desaturates brights across the spectrum, which is what keeps a
        // glint coloured instead of white.
        Tonemapping::AcesFitted,
        // Bloom brings `Hdr` with it as a required component, and that is the
        // half that matters: it gives the camera a float render target, so a
        // glint can be brighter than white before the transform above pulls it
        // back.
        //
        // Thresholded, which is the whole difference between glitter and haze.
        // Energy-conserving bloom with no prefilter scatters every pixel in
        // proportion to its brightness, and a sunlit sea is uniformly bright, so
        // it would fog the entire surface. Cutting in above 0.7 leaves the water
        // alone and blooms only what the sun is actually mirroring. Bevy couples
        // the two settings and says so: a prefilter this aggressive is not
        // energy-conserving, so the composite has to be additive to match.
        //
        // Intensity well under the 0.15 default. This is a simulator, and the
        // effect is here to make a highlight read, not to glow.
        Bloom {
            intensity: 0.06,
            prefilter: BloomPrefilter {
                threshold: 0.7,
                threshold_softness: 0.3,
            },
            composite_mode: BloomCompositeMode::Additive,
            ..Bloom::NATURAL
        },
        Chase,
    ));

    // The sun, and the only shadow caster. Its direction is `sky::SUN` rather
    // than a vector of its own: the sky draws the disc there, the ocean puts its
    // highlight there, and a light that disagreed with either would produce a
    // boat lit from one side of a sea that glitters on the other.
    //
    // Shadows are on, which the earlier comment here said they should not be. The
    // reasoning was that an ocean plane is a poor shadow receiver, and that is
    // true of a *flat* one — but the surface is displaced, and the material's
    // prepass shader displaces it identically, so the shadow map describes the
    // water that is actually drawn. What it buys is the sails darkening the hull
    // and the rig laying a shadow across the water, which is most of what tells a
    // viewer the boat is in the scene rather than pasted onto it.
    commands.spawn((
        DirectionalLight {
            illuminance: 13_000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        // Fitted to the boat and the water immediately around it. A single
        // cascade stretched over the whole 8 km of visible sea would put the
        // twelve metre boat inside one shadow-map texel, and nothing beyond a
        // couple of hundred metres has anything to cast onto anything.
        CascadeShadowConfigBuilder {
            // WebGL2 supports exactly one cascade and Bevy's own default gates on
            // that; asking for four on the web target fails the pipeline.
            num_cascades: if cfg!(target_arch = "wasm32") { 1 } else { 4 },
            minimum_distance: 0.5,
            first_cascade_far_bound: 40.0,
            maximum_distance: 260.0,
            overlap_proportion: 0.2,
        }
        .build(),
        Transform::default().looking_to(-crate::sky::SUN.normalize(), Vec3::Y),
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

impl Orbit {
    /// Lowest and highest the camera is allowed to go, radians.
    ///
    /// The top stops short of straight down because the orbit is expressed in
    /// spherical angles and `look_at` needs an up vector that is not parallel to
    /// the view. The bottom goes slightly below the horizontal, which is worth
    /// having: a camera just above the water is the only way to see what the sea
    /// state actually looks like from a cockpit.
    const LOWEST: f32 = -0.16;
    const HIGHEST: f32 = 1.45;

    /// Nearest and furthest, m.
    const NEAREST: f32 = 14.0;
    const FURTHEST: f32 = 400.0;

    /// Applies a change in bearing and elevation, keeping both in range.
    fn turn(&mut self, azimuth: f32, elevation: f32) {
        self.azimuth += azimuth;
        self.elevation = (self.elevation + elevation).clamp(Self::LOWEST, Self::HIGHEST);
    }

    /// Scales the distance, keeping it in range.
    ///
    /// Multiplicative rather than additive because that is what "zoom" means to a
    /// hand: a fixed number of metres per notch crawls when you are far out and
    /// jumps through the boat when you are close.
    fn pull(&mut self, factor: f32) {
        self.distance = (self.distance * factor).clamp(Self::NEAREST, Self::FURTHEST);
    }
}

/// Turns the camera with the mouse: drag to orbit, wheel to zoom.
///
/// Held rather than free-look, and on the *right* button. Free-look would need
/// the cursor grabbed, which takes the pointer away from a window that has no
/// other use for it and traps it in a way a physics demo should not. Right rather
/// than left leaves the left button for anything a future UI wants, and matches
/// what every CAD and 3D tool has trained hands to expect.
///
/// The keyboard orbit stays: a drag is better for looking around, and keys are
/// better for a slow deliberate sweep, and neither replaces the other.
pub fn orbit_with_mouse(
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion: MessageReader<MouseMotion>,
    mut wheel: MessageReader<MouseWheel>,
    mut orbit: ResMut<Orbit>,
) {
    // Radians per pixel. Set so that dragging across a 1280-pixel window sweeps
    // most of a turn, which is what makes a drag feel like turning the boat
    // rather than nudging it.
    const PER_PIXEL: f32 = 0.005;

    if buttons.pressed(MouseButton::Right) {
        let mut delta = Vec2::ZERO;
        for moved in motion.read() {
            delta += moved.delta;
        }
        // Dragging right walks the camera to the right around the boat, which
        // means the bearing decreases: the scene moves the other way from the
        // hand, as it does when you push a real object.
        orbit.turn(-delta.x * PER_PIXEL, -delta.y * PER_PIXEL);
    } else {
        // Drain the queue even when not dragging. Left to accumulate, the events
        // would all arrive at once on the frame the button went down and throw the
        // view across the sky.
        motion.clear();
    }

    for scrolled in wheel.read() {
        // A line of scroll is one notch; a pixel of scroll is a fraction of one.
        // Trackpads report pixels and mice report lines, and treating them alike
        // makes a trackpad useless and a mouse violent.
        let notches = match scrolled.unit {
            MouseScrollUnit::Line => scrolled.y,
            MouseScrollUnit::Pixel => scrolled.y / 40.0,
        };
        orbit.pull(0.88_f32.powf(notches));
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
