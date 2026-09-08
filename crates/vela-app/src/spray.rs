//! Bow spray: the one part of the water a surface shader cannot draw.
//!
//! # Why particles, and why here
//!
//! The sea is a height field evaluated in a shader, and everything the water
//! does *as a surface* — the swell, the wake, the foam at the stem — is the
//! shader's to draw. Spray is not a surface: it is water that has left the
//! surface, flying through the air on its own trajectory before falling back.
//! No displacement of a mesh can represent it. So it is drawn as what it is,
//! a few hundred droplets simulated on the CPU as ordinary entities, and this
//! module is the whole of that.
//!
//! The emission is driven by the physics, not by a timer. The bow's plunge
//! rate through the water it is over — the same quantity `view::advance_sea`
//! hands the shader for the stem foam — and the boat's speed set how many
//! droplets are born each second, so a boat sitting in a calm throws none, a
//! boat reaching in flat water throws a steady thin sheet, and a boat burying
//! its bow into a crest throws a burst. The arithmetic for the bow's motion is
//! repeated here rather than shared, deliberately: it is four lines, and a
//! `view` that exported it would be a `view` that owned the spray.
//!
//! # What is drawn is a pool, not a stream
//!
//! [`MAX`] entities are spawned once and recycled. A dead droplet is a hidden
//! entity waiting to be reborn, never a despawn: at a hundred and fifty births
//! a second, spawning and despawning would churn the archetype tables and the
//! render world's entity maps every frame for no visible gain. When the pool is
//! exhausted the extra births are dropped, which at the rates chosen happens
//! only in a burst, and a burst that is a little thinner than it should be is
//! not a thing a viewer can see.
//!
//! # The lies
//!
//! All droplets share one material, so none can fade by alpha on its own: a
//! droplet fades by *shrinking* to nothing over the last third of its life
//! instead. It reads as a droplet dispersing rather than as a droplet
//! vanishing, which is the effect wanted, but it is not what water does.
//!
//! The droplets are screen-aligned quads carrying a soft disc, drawn unlit and
//! blended. They are lit by nothing and shadow nothing — a droplet in the
//! sun's shadow is as bright as one in the sun — and they do not appear in
//! the water's reflection, being on the main view's layer only: a reflected
//! sheet of spray is not worth a second pass over six hundred transforms.
//!
//! WebGL2 is the floor: no compute, no storage buffers, no instancing the
//! renderer does not already do on its own. Six hundred quads through the
//! ordinary mesh path is well inside that, and the renderer batches them
//! because they share a mesh and a material.

use bevy::light::NotShadowCaster;
use bevy::prelude::*;
use nalgebra::Vector3;

use crate::frame;
use crate::sim::Engine;
use crate::view::Chase;

/// Droplets in the pool, and so the most that can be in the air at once.
///
/// At the peak birth rate the formula below reaches in a seaway — some five
/// hundred a second — and the longest life a droplet is given, the steady
/// state is under seven hundred alive. The rest is headroom for a burst.
const MAX: usize = 1200;

/// Droplets born per second per metre-per-second of speed above the
/// threshold, per unit of the plunge factor.
///
/// Chosen so that at 2.8 m/s — the reference boat's working speed — and a
/// plunge of a metre a second about three hundred and fifty droplets a second
/// are born, and in flat water at that speed about a hundred:
/// `140 × 1.8 × 1.4 = 353` and `140 × 1.8 × 0.4 = 101`. A sheet of spray is
/// many small droplets, not a few large ones; the first figure, sixty, read
/// as a handful of confetti.
const BIRTHS_PER_METRE: f32 = 140.0;

/// Boat speed below which there is no spray at all, m/s.
///
/// A boat ghosting along at a knot does not throw water; the stem parts it.
const SPEED_THRESHOLD: f32 = 1.0;

/// The share of the birth rate that speed alone accounts for, with the bow
/// neither rising nor falling through the surface.
///
/// The plunge adds to this rather than multiplying it, so a boat in flat water
/// still throws the thin sheet its speed alone earns.
const PLUNGE_FLOOR: f32 = 0.4;

/// The plunge rate beyond which more plunge throws no more spray, m/s.
///
/// The formula is linear in plunge and a bow falling off a crest can exceed
/// three metres a second; unclamped, that is a burst that empties the pool.
const PLUNGE_CEILING: f32 = 3.0;

/// The fraction of a droplet's life over which it shrinks to nothing.
const FADE: f32 = 0.3;

/// A droplet's side at birth and at death, m.
///
/// A droplet grows as it flies: not because water does, but because a sheet
/// of spray disperses into a mist, and one growing quad is the cheapest
/// picture of that.
const SIZE_BORN: f32 = 0.05;
const SIZE_DEAD: f32 = 0.16;

/// Downward acceleration of a droplet in flight, m/s².
///
/// The same `g` the physics uses; the frontend is not the place for a
/// different gravity.
const GRAVITY: f32 = 9.81;

/// Horizontal air drag on a droplet, per second.
///
/// A fraction of the horizontal velocity lost each second, applied as a
/// first-order decay. Real drag is quadratic in speed and depends on the
/// droplet's size; this one makes the sheet curl back as it flies, which is
/// what the eye checks.
const DRAG: f32 = 1.5;

/// Below this height above the mean level, a droplet is checked against the
/// sea's actual elevation each frame, m.
///
/// The elevation is a sum over sixty wave components, and a droplet still
/// well above the highest crest does not need it evaluated to know it is in
/// the air.
const SEA_CHECK_HEIGHT: f32 = 0.5;

/// A droplet, alive while its age is under its life.
///
/// Position lives in the `Transform`, as it must for the renderer to draw it;
/// what this adds is the state the renderer has no use for.
#[derive(Component)]
pub struct Droplet {
    /// Velocity in the render frame, m/s.
    velocity: Vec3,
    /// Seconds since birth.
    age: f32,
    /// Seconds it will live. Age past this is dead.
    life: f32,
}

/// The emitter's own state, carried between frames.
#[derive(Resource)]
pub struct Emitter {
    /// The generator every random choice draws from.
    rng: Random,
    /// Births owed from previous frames: the fractional remainder of
    /// `rate × dt`, so a rate of forty a second at a hundred frames a second
    /// gives forty a second and not zero.
    owed: f32,
}

/// A small deterministic random generator.
///
/// SplitMix64, the same four lines `vela_core::seaway` uses for its phases,
/// and for the same reason: it has no state to get wrong and is identical on
/// every platform. Fixed seed, so two runs of the same sailing throw the same
/// spray — not a property anything relies on, but a property that costs
/// nothing and makes a screenshot reproducible.
struct Random(u64);

impl Random {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // The top 24 bits: every value exactly representable in an `f32`.
        (z >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in `[low, high)`.
    fn between(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.unit()
    }

    /// `-1` or `+1`, evenly.
    fn side(&mut self) -> f32 {
        if self.unit() < 0.5 {
            -1.0
        } else {
            1.0
        }
    }
}

/// The droplet sprite: a soft disc, white in the middle and clear at the rim.
///
/// Generated rather than shipped, for the same reason the ocean's shader is
/// embedded: no asset path to configure and nothing to fetch on the web. A
/// droplet is not a disc either, but at two to seven pixels a soft disc is
/// what a droplet looks like, and a hard square is what confetti looks like.
fn sprite() -> Image {
    use bevy::asset::RenderAssetUsages;
    use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

    const SIZE: u32 = 32;
    let mut data = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    for row in 0..SIZE {
        for column in 0..SIZE {
            let x = (column as f32 + 0.5) / SIZE as f32 * 2.0 - 1.0;
            let y = (row as f32 + 0.5) / SIZE as f32 * 2.0 - 1.0;
            let radius = (x * x + y * y).sqrt();
            // Full in the middle third, gone at the rim, smooth between.
            let t = ((1.0 - radius) / 0.65).clamp(0.0, 1.0);
            let alpha = t * t * (3.0 - 2.0 * t);
            data.extend_from_slice(&[255, 255, 255, (alpha * 255.0) as u8]);
        }
    }
    Image::new(
        Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    )
}

/// Spawns the pool, hidden, and the emitter that will fill it.
///
/// One quad and one material, shared by every droplet: what lets the renderer
/// draw the pool as one batch, and what forces the fade to be a shrink.
pub fn spawn(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Unit-sized, scaled per droplet: the size is the transform's business.
    let quad = meshes.add(Rectangle::new(1.0, 1.0));
    let water = materials.add(StandardMaterial {
        // Near white with a little of the sky in it. Unlit, so this is the
        // colour on screen and not an albedo the sun works on. The alpha is
        // in the sprite: a square droplet is confetti.
        base_color: Color::srgb(0.90, 0.95, 1.0),
        base_color_texture: Some(images.add(sprite())),
        unlit: true,
        alpha_mode: AlphaMode::Blend,
        // A screen-aligned quad is only ever seen from the front, but the
        // billboard puts the camera behind it for a frame now and then as a
        // droplet crosses the camera's plane, and a flicker there is worse
        // than a culling decision the material does not need.
        double_sided: true,
        cull_mode: None,
        ..default()
    });

    for _ in 0..MAX {
        commands.spawn((
            Mesh3d(quad.clone()),
            MeshMaterial3d(water.clone()),
            Transform::default(),
            Visibility::Hidden,
            Droplet {
                velocity: Vec3::ZERO,
                age: 0.0,
                life: 0.0,
            },
            // Layer 0 only, by default. A droplet in the mirror would be a
            // second pass over the whole pool for a reflection the water blurs.
            NotShadowCaster,
        ));
    }

    commands.insert_resource(Emitter {
        rng: Random::new(7),
        owed: 0.0,
    });
}

/// Births new droplets at the bow and flies the ones in the air.
///
/// One system rather than two, because the two share a pass over the pool: a
/// dead droplet is revived in the same iteration that would otherwise have
/// skipped it, and there is no second query for the spawner to find one with.
///
/// Runs in `Update`, after the engine has stepped — see `main` — with the
/// render frame's `dt`, which is the right one: a droplet is a picture, not a
/// physical body, and a picture advances at the picture's rate.
pub fn emit_and_advance(
    engine: Res<Engine>,
    time: Res<Time>,
    mut emitter: ResMut<Emitter>,
    cameras: Query<&GlobalTransform, With<Chase>>,
    mut droplets: Query<(&mut Droplet, &mut Transform, &mut Visibility)>,
) {
    let dt = time.delta_secs();
    if dt <= 0.0 {
        return;
    }
    let sim_time = engine.sim.time();
    let state = engine.sim.state();
    let sea = engine.sea.as_ref();

    // The bow's motion, as `view::advance_sea` computes it for the stem foam:
    // the forward end of the stations, its world velocity, and how fast it is
    // going down through a surface that is itself moving. Engine `z` is down,
    // so a positive plunge is the bow driving into the water.
    let forward = frame::to_render(state.to_world(Vector3::x()));
    let heading = Vec2::new(forward.x, forward.z).normalize_or(Vec2::X);
    // Perpendicular to the heading in the render plane, to starboard when the
    // boat heads north; which side is which does not matter, since every
    // droplet chooses a side at random.
    let across = Vec3::new(-heading.y, 0.0, heading.x);
    let along = Vec3::new(heading.x, 0.0, heading.y);
    let bow_body = Vector3::new(engine.hull_length(), 0.0, 0.0);
    let bow_world = state.position + state.to_world(bow_body);
    let bow_velocity = state.to_world(state.point_velocity(bow_body));
    let surface_rate = sea.map_or(0.0, |sea| {
        sea.vertical_rate(bow_world.x, bow_world.y, sim_time)
    });
    let plunge = ((bow_velocity.z + surface_rate) as f32).clamp(0.0, PLUNGE_CEILING);
    let world_velocity = state.world_velocity();
    let speed = world_velocity.xy().norm() as f32;
    let bow = frame::to_render(bow_world);
    let boat_velocity = frame::to_render(world_velocity);

    // Births this frame: the rate, integrated, with the remainder carried.
    let rate = BIRTHS_PER_METRE * (speed - SPEED_THRESHOLD).max(0.0) * (PLUNGE_FLOOR + plunge);
    let owed = emitter.owed + rate * dt;
    let mut births = owed.floor();
    emitter.owed = owed - births;

    // Every droplet faces the chase camera: the camera's own rotation, which
    // aligns the quad's plane with the screen. The camera exists from the
    // first frame; identity is only what happens before it does.
    let facing = cameras
        .single()
        .map_or(Quat::IDENTITY, GlobalTransform::rotation);

    let horizontal_decay = (1.0 - DRAG * dt).max(0.0);
    let elevation = |x: f32, z: f32| -> f32 {
        sea.map_or(0.0, |sea| {
            frame::metres(sea.elevation(f64::from(x), f64::from(z), sim_time))
        })
    };

    for (mut droplet, mut transform, mut visibility) in &mut droplets {
        if droplet.age < droplet.life {
            droplet.age += dt;
            droplet.velocity.y -= GRAVITY * dt;
            droplet.velocity.x *= horizontal_decay;
            droplet.velocity.z *= horizontal_decay;
            transform.translation += droplet.velocity * dt;

            // Dead of age, or fallen back into the sea. A droplet below the
            // mean level and above the surface is still in the trough and
            // still in the air, so the surface is what it is checked against.
            let position = transform.translation;
            let drowned = position.y < SEA_CHECK_HEIGHT
                && position.y < elevation(position.x, position.z) - 0.1;
            if droplet.age >= droplet.life || drowned {
                droplet.age = droplet.life;
                *visibility = Visibility::Hidden;
                continue;
            }

            let fraction = droplet.age / droplet.life;
            let size = SIZE_BORN + (SIZE_DEAD - SIZE_BORN) * fraction;
            let fade = ((1.0 - fraction) / FADE).min(1.0);
            transform.rotation = facing;
            transform.scale = Vec3::splat(size * fade);
        } else if births > 0.0 {
            births -= 1.0;
            let rng = &mut emitter.rng;

            // Born on the stem, a little aft of the forward end and to one
            // side of the centreline, at the surface: the water thrown is the
            // water the stem is parting.
            let side = rng.side();
            let aft = rng.between(0.0, 1.5);
            let out = rng.between(0.3, 1.2);
            let x = bow.x - along.x * aft + across.x * side * out;
            let z = bow.z - along.z * aft + across.z * side * out;
            let y = elevation(x, z) + rng.between(0.0, 0.2);
            transform.translation = Vec3::new(x, y, z);
            transform.rotation = facing;
            transform.scale = Vec3::splat(SIZE_BORN);

            // Carried forward with the boat, thrown outward off the flare,
            // and up by an amount the plunge sets: a bow burying itself
            // throws high, a bow skimming throws low.
            let outward = across * (side * rng.between(1.5, 3.5));
            let upward = Vec3::Y * (rng.between(1.5, 4.0) * (0.5 + plunge));
            let jitter = Vec3::new(
                rng.between(-0.5, 0.5),
                rng.between(-0.5, 0.5),
                rng.between(-0.5, 0.5),
            );
            droplet.velocity = boat_velocity * 0.6 + outward + upward + jitter;
            droplet.age = 0.0;
            droplet.life = rng.between(0.6, 1.4);
            *visibility = Visibility::Visible;
        }
    }
}
