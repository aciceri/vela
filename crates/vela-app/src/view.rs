//! The camera, the light, and the patch of sea that follows the boat.
//!
//! # The sea is a moving window, not an ocean
//!
//! The surface is a finite disc kept centred on the boat. Because the elevation
//! is a closed form of *world* position, moving the disc does not move the
//! water: a wave crest stays where it is while the mesh slides under it. That is
//! the whole reason the sea is shared as a realisation rather than as a height
//! field — a height field would have to be rebuilt every time the window moved,
//! and this does not have to be rebuilt at all.
//!
//! What moves with the boat is the *sampling*: the vertices slide through the
//! wave field, so the polygonal approximation of a crest shifts as the window
//! goes by. A uniform lattice could have been snapped to its own cell size to
//! hold the sampling still, and an earlier version was; a graded mesh has no
//! lattice to snap to. The trade is taken with open eyes — see [`RINGS`] —
//! because the crawl is a slow low-contrast shimmer in the middle distance,
//! while the faceting a uniform grid produced near the boat was the first thing
//! anyone noticed.

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

/// Radius of the near sea, m.
///
/// Six hundred metres of finely drawn water around a twelve metre boat. It
/// was two hundred, and the edge was the first thing a viewer saw: the
/// displacement fades to nothing at the rim (see `resolved` in
/// `shaders/ocean.wgsl`), and a disc of moving crests inside a plane of still
/// ones reads as a circle drawn on the sea, whatever the shading does. Three
/// times the radius costs a fifth more rings, because the spacing is
/// geometric, and pushes the circle to where a half-metre crest is a pixel.
/// The far ring picks up exactly here.
const REACH: f32 = 600.0;

/// Angular segments around the near disc and the far ring alike.
///
/// One constant for both because the two meshes meet at `REACH`, and they meet
/// cleanly only if their rims are the same polygon: the disc's outermost row
/// and the ring's innermost are then the same vertices, computed the same way,
/// and the seam is watertight without any overlap to fight over depth. A
/// hundred and sixty is 2.25 degrees a segment, which at the rim is a cell just
/// under eight metres across and at the horizon a silhouette straight enough
/// that no facet shows.
const SEGMENTS: u32 = 160;

/// Rings of vertices across the near disc, from [`CORE`] out to [`REACH`].
///
/// Together with [`SEGMENTS`] this is the near sea's vertex budget: two
/// hundred rings of a hundred and sixty is 32,001 vertices with the centre,
/// a quarter over the 161-by-161 square grid it replaced for three times the
/// radius. That budget is measured rather than guessed — every vertex
/// evaluates the whole sixty-component wave sum, and a 257-square was running
/// eight million transcendentals a frame for detail in water that is faded flat
/// anyway.
///
/// The spacing is geometric, each ring a fixed ratio further out than the last
/// — 3.3 per cent here — so a cell grows in proportion to its radius: a metre
/// across at thirty metres, three and a half at a hundred, twenty at the rim.
/// That is what keeps a cell roughly constant in *screen* space for a camera
/// looking down at a plane, and it is why the near water is smooth where a
/// uniform grid of the same budget was a visible mesh of 2.5 m facets against a
/// shortest wave of 3.5 m.
///
/// A square grid graded the same way was tried first and is worth recording as
/// the wrong shape: grading separably along each axis produces cells that are
/// fine in one direction and coarse in the other everywhere except the diagonal,
/// and along the axes through the boat they degenerate to slivers three
/// centimetres by five metres. A disc has one radial direction and grades along
/// it alone.
const RINGS: u32 = 200;

/// Radius of the innermost ring of the near disc, m.
///
/// Inside it is a fan of triangles from a single vertex at the boat's origin.
/// With the ratio above the first cells out from here are close to square —
/// 3.4 per cent of the radius one way, 3.9 the other — and a metre puts the
/// whole fan under the transom.
const CORE: f32 = 1.0;

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

/// Concentric rings across the far sea.
///
/// Sixty-four bands over a thirteenfold change in radius, each about four per
/// cent wider than the one inside it: twenty-five metres at the inner rim,
/// which is what lets the far ring keep *shading* the swell for a while after
/// it has stopped displacing it — see `shaded` in `shaders/ocean.wgsl`. The
/// whole ring is still a third of the near disc's vertices, which is the
/// point: uniform spacing fine enough for the inner rim would need thousands
/// of rings, and every band past a kilometre would draw triangles smaller than
/// a pixel.
const HORIZON_RINGS: u32 = 64;

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

/// Vertices on `rings + 1` concentric rings from `inner` to `outer`, in the render
/// plane, `segments` around each.
///
/// The radial spacing is geometric: each ring sits a fixed ratio further out
/// than the last, so that `rings` steps of it cover exactly the span asked for.
/// Radii are computed from the power rather than accumulated, which keeps the
/// rings where forty-eight roundings would not; and the outermost is set to
/// `outer` outright rather than computed, because a mesh built to meet another
/// at that radius has to land on it exactly, not to within a unit of the last
/// place.
fn polar_rows(inner: f32, outer: f32, segments: u32, rings: u32) -> Vec<[f32; 3]> {
    let ratio = (outer / inner).powf(1.0 / rings as f32);
    let arc = std::f32::consts::TAU / segments as f32;
    let mut positions = Vec::with_capacity(((rings + 1) * segments) as usize);
    for ring in 0..=rings {
        let radius = if ring == rings {
            outer
        } else {
            inner * ratio.powi(ring as i32)
        };
        for segment in 0..segments {
            let angle = segment as f32 * arc;
            positions.push([radius * angle.cos(), 0.0, radius * angle.sin()]);
        }
    }
    positions
}

/// Two triangles per quad between consecutive rings laid out by [`polar_rows`],
/// whose first vertex is at index `first`.
fn polar_bands(first: u32, segments: u32, rings: u32, indices: &mut Vec<u32>) {
    for ring in 0..rings {
        for segment in 0..segments {
            // The seam closes by wrapping onto the ring's first column rather
            // than by duplicating it, so there is no pair of coincident vertices
            // to keep in step.
            let beside_segment = (segment + 1) % segments;
            let here = first + ring * segments + segment;
            let beside = first + ring * segments + beside_segment;
            let out = first + (ring + 1) * segments + segment;
            let out_beside = first + (ring + 1) * segments + beside_segment;
            // Wound counter-clockwise seen from above, which is what puts the
            // front face towards a camera looking down at the water.
            indices.extend_from_slice(&[here, beside, out]);
            indices.extend_from_slice(&[out, beside, out_beside]);
        }
    }
}

/// Positions only, for the ocean shader to displace.
///
/// The shader computes the normal from the analytic slope, so an uploaded
/// normal would be overwritten. No UVs either — nothing samples a texture.
fn sea_mesh(positions: Vec<[f32; 3]>, indices: Vec<u32>) -> Mesh {
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_indices(Indices::U32(indices))
}

/// A flat disc of radius `outer`, for the ocean shader to displace.
///
/// A fan from the centre to the ring at `core`, then `rings - 1` geometric bands
/// out to the rim — the same construction as [`horizon_ring`] from a radius of
/// zero, which is what a near sea and a far sea being one surface ought to look
/// like in the code as well. See [`RINGS`] for why the near sea is a disc and
/// not a square.
fn near_disc(core: f32, outer: f32, segments: u32, rings: u32) -> Mesh {
    let mut positions = Vec::with_capacity((1 + rings * segments) as usize);
    positions.push([0.0, 0.0, 0.0]);
    positions.extend(polar_rows(core, outer, segments, rings - 1));

    let mut indices = Vec::with_capacity(((2 * rings - 1) * segments * 3) as usize);
    for segment in 0..segments {
        let beside = (segment + 1) % segments;
        // Same orientation as the bands: centre, then the far corner, then the
        // near one, so the fan's front face is the bands' front face.
        indices.extend_from_slice(&[0, 1 + beside, 1 + segment]);
    }
    polar_bands(1, segments, rings - 1, &mut indices);

    sea_mesh(positions, indices)
}

/// A flat annulus from `inner` to `outer`, for the ocean shader to displace.
///
/// The near disc stops at six hundred metres, and beyond it there was
/// background: an edge that reads as a wall rather than as distance. This is the
/// water that carries the eye from there to the horizon, and it can afford to be
/// very coarse, because everything it draws is at least six hundred metres away
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
fn horizon_ring(inner: f32, outer: f32, segments: u32, rings: u32) -> Mesh {
    let positions = polar_rows(inner, outer, segments, rings);
    let mut indices = Vec::with_capacity((rings * segments * 6) as usize);
    polar_bands(0, segments, rings, &mut indices);
    sea_mesh(positions, indices)
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
            Mesh3d(meshes.add(near_disc(CORE, REACH, SEGMENTS, RINGS))),
            MeshMaterial3d(ocean.clone()),
            Transform::default(),
            Sea,
            // The sea casts no shadow. Nothing in this scene is under it, and
            // leaving it a caster put both tessellations -- sixty-six thousand
            // triangles -- through the shadow pass once per cascade for an effect
            // that cannot exist. Measured: it was most of the frame.
            NotShadowCaster,
        ));

        // Inner radius `REACH` and the disc's `SEGMENTS`, so that the ring's
        // innermost row is vertex for vertex the disc's rim: the same radius,
        // the same angles, the same arithmetic. The seam is then closed by
        // construction, with no gap for the sky to show through and no overlap
        // for two tessellations to fight over depth in. The square grid this
        // replaced overlapped the ring in four corner wedges and needed to be
        // held a hand's breadth above it to settle the flicker; there is nothing
        // left to settle, and the two sit at the same height.
        //
        // Same `Sea` marker, so `follow_sea` carries this with the disc, and the
        // ring's sampling of the wave field moves with the boat as the disc's
        // does — but a sixteen metre triangle drifting is not something an eye
        // can find at that range, and the displacement is faded out over it
        // anyway.
        commands.spawn((
            Mesh3d(meshes.add(horizon_ring(REACH, HORIZON, SEGMENTS, HORIZON_RINGS))),
            MeshMaterial3d(ocean),
            Transform::default(),
            Sea,
            // The sea casts no shadow. Nothing in this scene is under it, and
            // leaving it a caster put both tessellations -- sixty-six thousand
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
    // true of a *flat* one — but the surface is displaced, and the ocean's
    // fragment stage looks the shadow map up at the *displaced* world position
    // (`fetch_directional_shadow` in `shaders/ocean.wgsl`), so the shadow lands
    // on the water that is actually drawn. The sea itself is in no depth or
    // shadow pass: the map is the boat's alone. What it buys is the sails
    // darkening the hull and the rig laying a shadow across the water, which is
    // most of what tells a viewer the boat is in the scene rather than pasted
    // onto it.
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

/// Pushes the engine's clock and the boat's motion into the ocean material.
///
/// The engine's time, not the render clock: see `crate::ocean`. This and the
/// stern's position and velocity are the whole per-frame traffic between the
/// physics and the water; the material records the stern's track from them
/// and the water computes everything else, the wake included.
pub fn advance_sea(engine: Res<Engine>, mut oceans: ResMut<Assets<OceanMaterial>>) {
    let time = engine.sim.time();
    let state = engine.sim.state();
    let stern = frame::to_render(state.position);
    let velocity = frame::to_render(state.world_velocity());
    for (_, material) in oceans.iter_mut() {
        material.set_time(time);
        material.record(stern, velocity, time);
    }
}

/// Keeps the sea centred on the boat.
///
/// Continuously, not snapped, and the change of mind is worth recording. A
/// uniform grid could be snapped to its own cell size, which kept every vertex
/// on a fixed world lattice so the sampling stayed still while the window slid
/// — the right argument for a uniform grid, and the module documentation says
/// what it cost. A geometrically spaced disc has no lattice to snap to, so
/// snapping would only quantise the window's position into jumps while the
/// sampling moved anyway: all of the lurch and none of the benefit. Following
/// the boat exactly is both simpler and smoother.
pub fn follow_sea(engine: Res<Engine>, mut seas: Query<&mut Transform, With<Sea>>) {
    let centre = frame::to_render(engine.sim.state().position);
    for mut transform in &mut seas {
        // Horizontal only: the mean level is world `y = 0` and the boat's heave
        // is the boat's. The two tessellations share that height, and their
        // shared rim relies on it.
        transform.translation.x = centre.x;
        transform.translation.z = centre.z;
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
///
/// Through [`Orbit::turn`] and [`Orbit::pull`], the same two entries the mouse
/// uses, so the limits live in one place: this once carried its own copies of
/// them with slightly different numbers, and the camera could be pulled to a
/// distance by wheel that the keys then refused to hold.
pub fn orbit(keys: Res<ButtonInput<KeyCode>>, time: Res<Time>, mut orbit: ResMut<Orbit>) {
    let dt = time.delta_secs();
    let held = |key: KeyCode| if keys.pressed(key) { 1.0 } else { 0.0 };
    // Radians per second of bearing and of elevation, and a zoom rate per
    // second, each as the difference of an opposing pair of keys.
    let azimuth = 1.2 * dt * (held(KeyCode::KeyJ) - held(KeyCode::KeyL));
    let elevation = 0.8 * dt * (held(KeyCode::KeyI) - held(KeyCode::KeyK));
    orbit.turn(azimuth, elevation);
    orbit.pull(1.0 + 0.9 * dt * (held(KeyCode::KeyO) - held(KeyCode::KeyU)));
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

    /// The disc must be a closed fan and lattice of quads with the vertex and
    /// index counts that follow from its ring and segment counts.
    ///
    /// Cheap, and it catches the off-by-one where the bands are counted from the
    /// centre rather than from the first ring — which either leaves a crack
    /// around the rim or indexes past the last vertex.
    #[test]
    fn the_disc_closes() {
        let mesh = near_disc(1.0, 10.0, 8, 4);
        assert_eq!(mesh.count_vertices(), 1 + 4 * 8);
        assert_eq!(
            mesh.indices().map(bevy::mesh::Indices::len),
            Some(8 * 3 + 3 * 8 * 6)
        );
    }

    /// The far ring's innermost row must be the near disc's rim, exactly.
    ///
    /// Bitwise, not approximately: the two meshes sit at the same height with no
    /// offset to hide behind, so a rim vertex a unit in the last place off the
    /// ring's would open a hairline gap the sky shows through. Computing the rim
    /// radius by repeated power instead of setting it is the plausible mistake,
    /// and it fails this.
    #[test]
    fn the_seam_is_shared_vertex_for_vertex() {
        let disc = near_disc(CORE, REACH, SEGMENTS, RINGS);
        let ring = horizon_ring(REACH, HORIZON, SEGMENTS, HORIZON_RINGS);
        let positions = |mesh: &Mesh| {
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
                .and_then(bevy::mesh::VertexAttributeValues::as_float3)
                .expect("positions")
                .to_vec()
        };
        let rim = &positions(&disc)[(1 + (RINGS - 1) * SEGMENTS) as usize..];
        let inner = &positions(&ring)[..SEGMENTS as usize];
        assert_eq!(rim.len(), SEGMENTS as usize);
        assert_eq!(rim, inner);
    }
}
