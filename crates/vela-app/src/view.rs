//! The camera, lighting and camera-projected sampling of the physical sea.
//!
//! The ocean grid covers the visible water, independently of the boat's position.
//! Zoom changes metres per cell uniformly across the image rather than revealing
//! a dense disc surrounded by flattened waves. Both colour and motion-vector
//! passes project the same grid and evaluate the same world-space wave field.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::{NoFrustumCulling, RenderLayers};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::ecs::message::MessageReader;
use bevy::input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::light::{CascadeShadowConfigBuilder, NotShadowCaster, ShadowFilteringMethod};
use bevy::mesh::Indices;
use bevy::post_process::bloom::{Bloom, BloomCompositeMode, BloomPrefilter};
use bevy::post_process::motion_blur::MotionBlur;
use bevy::prelude::*;
use bevy::render::render_resource::PrimitiveTopology;
use bevy::render::view::{ColorGrading, ColorGradingSection};
use bevy::ui::IsDefaultUiCamera;

use crate::boat::Boat;
use crate::frame;
use crate::ocean::OceanMaterial;
use crate::reflection;
use crate::sim::Engine;

/// Screen-grid divisions, also supplied as shader definitions to both passes.
/// 31,073 vertices cover visible water at every zoom, rather than concentrating
/// most of the old polar mesh under the boat.
pub(crate) const SEA_COLUMNS: u32 = 192;
pub(crate) const SEA_ROWS: u32 = 160;

/// Culling range for the camera, including the projected sea's 8 km horizon.
const FAR_PLANE: f32 = 20_000.0;

/// Reverse-Z near plane. The 14 m minimum orbit keeps the boat comfortably clear.
const NEAR_PLANE: f32 = 0.5;

/// Both views use analytic sky lighting in the boat material, not a uniform fill.
pub const AMBIENT: AmbientLight = AmbientLight {
    color: Color::WHITE,
    brightness: 0.0,
    affects_lightmapped_meshes: true,
};

/// Shared by direct view and planar reflection; shaders expose radiance once.
pub const EXPOSURE: bevy::camera::Exposure = bevy::camera::Exposure { ev100: 10.0 };

/// Low, hazy sun; ocean.wgsl uses the same 1650 lux and warm spectral balance.
const SUNLIGHT: f32 = 1_650.0;

/// Marks the sea surface entity.
#[derive(Component)]
pub struct Sea;

/// Marks the main camera and holds its presentation-only spring state.
#[derive(Component)]
pub struct Chase {
    /// Azimuth, elevation and logarithmic distance.
    orbit: Vec3,
    orbit_velocity: Vec3,
    target: Vec3,
    target_velocity: Vec3,
}

/// Static NDC coordinates. The shader projects these onto the mean-water plane
/// and displaces them; no CPU mesh rebuild or boat-centred transform is needed.
fn projected_grid(columns: u32, rows: u32) -> Mesh {
    let mut positions = Vec::with_capacity(((columns + 1) * (rows + 1)) as usize);
    for row in 0..=rows {
        for column in 0..=columns {
            positions.push([
                2.0 * column as f32 / columns as f32 - 1.0,
                0.0,
                2.0 * row as f32 / rows as f32 - 1.0,
            ]);
        }
    }
    let mut indices = Vec::with_capacity((columns * rows * 6) as usize);
    for row in 0..rows {
        for column in 0..columns {
            let here = row * (columns + 1) + column;
            let above = here + columns + 1;
            indices.extend_from_slice(&[here, here + 1, above, above, here + 1, above + 1]);
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
    orbit: Res<Orbit>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut oceans: ResMut<Assets<OceanMaterial>>,
) {
    {
        // Still water is a sea with no waves to draw and the same surface to
        // shade, so the water is spawned whatever the boat was released in;
        // `advance_sea` re-realises it when the sea changes.
        //
        // One projected surface reaches the horizon without a near/far seam.
        let ocean = oceans.add(OceanMaterial::realising(
            engine.sea.as_ref(),
            engine.sim.time(),
        ));

        commands.spawn((
            Mesh3d(meshes.add(projected_grid(SEA_COLUMNS, SEA_ROWS))),
            MeshMaterial3d(ocean),
            Transform::default(),
            Sea,
            // The NDC bounding box is not its world-space shader projection.
            NoFrustumCulling,
            // Only the boat casts shadows; water still receives them.
            NotShadowCaster,
        ));
    }

    let chase = Chase::new(&orbit, frame::to_render(engine.sim.state().position));
    let camera_pose = chase.pose();
    commands.spawn((
        Camera3d::default(),
        // Thin rigging needs edge antialiasing. FXAA runs after tonemapping
        // rather than multiplying the full-screen ocean's HDR sample count.
        Msaa::Off,
        bevy::anti_alias::fxaa::Fxaa::default(),
        // Overrides the projection `Camera3d` requires into place. See
        // `FAR_PLANE`, which is what gets the far sea drawn at all, and
        // `NEAR_PLANE`, which is what gets it drawn without depth fighting.
        Projection::Perspective(PerspectiveProjection {
            near: NEAR_PLANE,
            far: FAR_PLANE,
            ..default()
        }),
        camera_pose,
        AMBIENT,
        EXPOSURE,
        // The Castaño filter: nine taps read as a 5×5 Gaussian, which is the
        // filter that turns a shadow map's stair-steps into an edge. Stated
        // rather than inherited, because the water is the one receiver in the
        // scene that draws the shadow *by hand* (`fetch_directional_shadow` in
        // `shaders/ocean.wgsl`) and so has no other way to say which filter it
        // expects; the `Temporal` alternative is a per-frame random rotation of
        // the taps that reads as a crawling edge without TAA to average it.
        ShadowFilteringMethod::Gaussian,
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
        // Neutral white balance and untouched highlights preserve the sail's
        // whites; a small saturation reduction quiets shadows and midtones.
        ColorGrading {
            shadows: ColorGradingSection {
                saturation: 0.94,
                ..default()
            },
            midtones: ColorGradingSection {
                saturation: 0.97,
                ..default()
            },
            ..default()
        },
        MotionBlur {
            shutter_angle: 0.18,
            samples: 3,
        },
        // Bloom brings `Hdr` with it as a required component, and that is the
        // half that matters: it gives the camera a float render target, so a
        // glint can be brighter than white before the transform above pulls it
        // back.
        //
        // Only the bright cores of highlights bloom. Broad low-threshold bloom
        // used to erase wave detail around the sun's already bright reflection.
        Bloom {
            intensity: 0.025,
            prefilter: BloomPrefilter {
                threshold: 2.0,
                threshold_softness: 0.5,
            },
            composite_mode: BloomCompositeMode::Additive,
            ..Bloom::NATURAL
        },
        // Where the HUD goes. Without the marker Bevy picks the highest-order
        // camera drawing to the primary window, which is this one today; the
        // marker says so, and keeps the readout out of `crate::reflection`'s
        // mirror image whatever cameras are added later. Standard scene
        // postprocessing finishes before UI, keeping the HUD sharp and ungraded.
        IsDefaultUiCamera,
        chase,
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
    // on the water that is actually drawn. Water participates in the velocity
    // prepass, not the shadow map: only the boat casts a shadow.
    // The hull and rig shade one another and the water, anchoring the boat
    // in the scene rather than leaving it looking pasted onto the surface.
    //
    // The biases are Bevy's defaults, and that is a decision rather than an
    // omission: acne is a surface comparing against its own depth in the map,
    // and the water is not in the map, so there is nothing for a bias to cure
    // there. The two centimetres of depth bias and the 1.8 texels of normal
    // bias — sixteen centimetres at the default cascade — are what keep the
    // sails from striping the hull, and are far under anything the water's
    // shadow could lose.
    let towards_sun = Transform::default().looking_to(-crate::sky::SUN.normalize(), Vec3::Y);
    commands.spawn((
        DirectionalLight {
            illuminance: SUNLIGHT,
            color: Color::linear_rgb(1.0, 1.54 / 1.65, 1.35 / 1.65),
            shadow_maps_enabled: true,
            ..default()
        },
        // Fitted to the orbit. A single cascade stretched over the whole 8 km of
        // visible sea would put the twelve metre boat inside one shadow-map
        // texel, and nothing beyond the boat has anything to cast onto anything.
        //
        // The first split is at a hundred metres so that the default view —
        // sixty-two metres from the boat, whose shadow reaches fifty metres
        // towards the camera under a sun twenty degrees up — lies entirely in
        // cascade 0. It was forty, which put the water under the hull in the
        // blend between cascades 1 and 2, and the shadow faded and coarsened
        // exactly where it was wanted; the blend now starts at eighty metres,
        // beyond the shadow. The price is cascade 0 spanning a hundred metres of
        // frustum, which is nine-centimetre texels at 2048: two across the mast,
        // dozens across the hull and the sails, which is the shadow that
        // actually reads. The maximum is the far end of the orbit plus the
        // shadow's length, so pulling back never drops the shadow outright — at
        // four hundred metres the last cascade resolves the hull at thirty
        // texels, which is about what the hull itself gets on screen there.
        CascadeShadowConfigBuilder {
            // WebGL2 supports exactly one cascade and Bevy's own default gates on
            // that; asking for four on the web target fails the pipeline.
            num_cascades: if cfg!(target_arch = "wasm32") { 1 } else { 4 },
            minimum_distance: 0.5,
            first_cascade_far_bound: 100.0,
            maximum_distance: 460.0,
            overlap_proportion: 0.2,
        }
        .build(),
        towards_sun,
    ));

    // The same sun once more, without shadows, for the mirror view alone.
    //
    // A light illuminates only the views whose layers it shares, and shadow
    // cascades are built per light *per view*: had the one above been put on
    // the mirror's layer as well, the mirror would have doubled the cascade
    // count — four more 2048-square depth passes and sixty-seven megabytes of
    // depth on an integrated GPU — to shade a reflection the water blurs
    // anyway. So the mirror gets a twin with no map, and the boat in the water
    // is lit as brightly as the boat above it, minus the sails' shadow on the
    // hull, which at reflection quality is not a difference a viewer can find.
    commands.spawn((
        DirectionalLight {
            illuminance: SUNLIGHT,
            color: Color::linear_rgb(1.0, 1.54 / 1.65, 1.35 / 1.65),
            ..default()
        },
        towards_sun,
        RenderLayers::layer(reflection::LAYER),
    ));
}

/// Pushes the engine's clock and the boat's motion into the ocean material.
///
/// The engine's time, not the render clock: see `crate::ocean`. This and the
/// stern's position and velocity are the whole per-frame traffic between the
/// physics and the water; the material records the stern's track from them
/// and the water computes everything else, the wake included.
pub fn advance_sea(
    engine: Res<Engine>,
    mut drawn: Local<Option<crate::sim::SeaPreset>>,
    mut oceans: ResMut<Assets<OceanMaterial>>,
) {
    let time = engine.sim.time();
    let state = engine.sim.state();
    let stern = frame::to_render(state.position);
    let velocity = frame::to_render(state.world_velocity());
    // The hull's axis in the render plane, and how fast the bow is going down
    // through the water it is over: the bow's world velocity is z-down in the
    // engine, and the surface under it moves too, so the plunge is the
    // difference. The bow is the forward end of the stations.
    let forward = frame::to_render(state.to_world(nalgebra::Vector3::x()));
    let heading = Vec2::new(forward.x, forward.z).normalize_or(Vec2::X);
    let bow_body = nalgebra::Vector3::new(engine.hull_length(), 0.0, 0.0);
    let bow_world = state.position + state.to_world(bow_body);
    let bow_velocity = state.to_world(state.point_velocity(bow_body));
    // Encounter derivative, including horizontal travel across a sloping wave.
    // The z-down bow velocity adds to the positive-up surface rate.
    let (immersion, plunge) = engine
        .sea
        .as_ref()
        .map_or((bow_world.z, bow_velocity.z), |sea| {
            let before = bow_world - bow_velocity * 0.01;
            let after = bow_world + bow_velocity * 0.01;
            let rate = (sea.elevation(after.x, after.y, time + 0.01)
                - sea.elevation(before.x, before.y, time - 0.01))
                / 0.02;
            (
                bow_world.z + sea.elevation(bow_world.x, bow_world.y, time),
                bow_velocity.z + rate,
            )
        });
    let hull = Vec2::new(engine.hull_length() as f32, engine.hull_half_beam() as f32);
    // A change of weather: the water is re-realised from the engine's new
    // sea, keeping its clock and the wake's track. Judged by the preset the
    // water was last drawn from, so the realisation is loaded once and not
    // every frame.
    let changed = *drawn != Some(engine.preset);
    *drawn = Some(engine.preset);
    for (_, material) in oceans.iter_mut() {
        if changed {
            material.realise(engine.sea.as_ref());
        }
        material.set_time(time);
        material.record(stern, velocity, time);
        material.set_heading(heading, plunge as f32, hull);
        material.sea.sun.w = immersion as f32;
    }
}

/// Requested orbit around the boat; [`Chase`] smooths it without changing physics.
///
/// Input changes these targets immediately. The camera's separate spring state
/// keeps keyboard, mouse and wheel equally responsive without input-rate filters.
#[derive(Resource)]
pub struct Orbit {
    /// World-space bearing, radians; zero is astern of a north-facing boat.
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
    orbit.pull((0.9 * dt * (held(KeyCode::KeyO) - held(KeyCode::KeyU))).exp());
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
        self.azimuth = wrap_angle(self.azimuth + azimuth);
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

    fn coordinates(&self) -> Vec3 {
        Vec3::new(
            wrap_angle(self.azimuth),
            self.elevation.clamp(Self::LOWEST, Self::HIGHEST),
            self.distance.clamp(Self::NEAREST, Self::FURTHEST).ln(),
        )
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

/// Holds a level horizon while smoothing framing and the boat's translation.
///
/// Runs after boat presentation and both input systems; the reflection follows
/// this result. Neither hull attitude nor simulated time enters the camera rig.
pub fn chase(
    orbit: Res<Orbit>,
    time: Res<Time>,
    boats: Query<&Transform, (With<Boat>, Without<Chase>)>,
    mut cameras: Query<(&mut Transform, &mut Chase)>,
) {
    let Ok(boat) = boats.single() else {
        return;
    };
    let dt = time.delta_secs();
    let requested = orbit.coordinates();
    let target = boat.translation + Vec3::Y * orbit.focus;
    for (mut camera, mut chase) in &mut cameras {
        // Unwrap around the current pose, not zero: crossing ±π takes the short
        // route. Rewrap after integration to preserve precision over many turns.
        let desired = Vec3::new(
            chase.orbit.x + wrap_angle(requested.x - chase.orbit.x),
            requested.y,
            requested.z,
        );
        chase.orbit = damp(chase.orbit, &mut chase.orbit_velocity, desired, 24.0, dt);
        chase.orbit.x = wrap_angle(chase.orbit.x);
        chase.target = damp(chase.target, &mut chase.target_velocity, target, 10.0, dt);
        *camera = chase.pose();
    }
}

impl Chase {
    fn new(orbit: &Orbit, boat: Vec3) -> Self {
        Self {
            orbit: orbit.coordinates(),
            orbit_velocity: Vec3::ZERO,
            target: boat + Vec3::Y * orbit.focus,
            target_velocity: Vec3::ZERO,
        }
    }

    fn pose(&self) -> Transform {
        let (sin_azimuth, cos_azimuth) = self.orbit.x.sin_cos();
        let (sin_elevation, cos_elevation) = self.orbit.y.sin_cos();
        // Render x is north. Zoom stays multiplicative during smoothing too.
        let distance = self.orbit.z.exp().clamp(Orbit::NEAREST, Orbit::FURTHEST);
        let offset = distance
            * Vec3::new(
                -cos_azimuth * cos_elevation,
                sin_elevation,
                sin_azimuth * cos_elevation,
            );
        Transform::from_translation(self.target + offset).looking_at(self.target, Vec3::Y)
    }
}

fn wrap_angle(angle: f32) -> f32 {
    (angle + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI
}

/// Exact critically damped step for a held target, stable even across long frames.
/// Stop residual momentum at a target or reversal: never overshoot a bound or
/// keep moving away from the requested pose. There is no accumulated prediction.
fn damp(current: Vec3, velocity: &mut Vec3, target: Vec3, rate: f32, dt: f32) -> Vec3 {
    if dt <= 0.0 {
        return current;
    }
    let decay = (-rate * dt).exp();
    let error = current - target;
    let tangent = *velocity + rate * error;
    let mut next = target + (error + tangent * dt) * decay;
    *velocity = (*velocity - rate * tangent * dt) * decay;
    for axis in 0..3 {
        let bounded = next[axis].clamp(
            current[axis].min(target[axis]),
            current[axis].max(target[axis]),
        );
        if bounded != next[axis] || bounded == target[axis] {
            velocity[axis] = 0.0;
        }
        next[axis] = bounded;
    }
    next
}
