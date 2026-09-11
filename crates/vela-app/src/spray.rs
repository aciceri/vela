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
//! Emission requires the bow waterline to meet the actual moving sea while
//! entering it. The encounter rate includes heave, pitch and forward motion
//! into a wave face; world speed alone never creates an impact. Spray and
//! collision use the engine clock, so pausing freezes water and droplets
//! together. These are visual tracers only: no force returns to the physics.
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
//! sheet of spray does not justify a second pass over the particle pool.
//!
//! WebGL2 is the floor: no compute, no storage buffers, no instancing the
//! renderer does not already do on its own. The ordinary mesh path batches
//! these quads because they share a mesh and a material.

use bevy::light::NotShadowCaster;
use bevy::prelude::*;
use nalgebra::Vector3;

use crate::frame;
use crate::sim::Engine;
use crate::view::Chase;

/// Droplets in the pool, and so the most that can be in the air at once.
///
/// Enough for a transient impact sheet; sustained peak entry may exhaust it,
/// in which case excess births are dropped rather than queued or allocated.
const MAX: usize = 1200;

/// Births per second per squared metre-per-second of relative entry.
/// A sharp impact fragments a sheet much faster than a gentle immersion.
const BIRTHS_PER_ENTRY: f32 = 190.0;

/// Ignore slow waterline movement that parts water without atomising it.
const ENTRY_THRESHOLD: f32 = 0.12;
const ENTRY_CEILING: f32 = 3.0;

/// The stem waterline must be wet; deeply buried contact makes submerged
/// turbulence rather than an airborne sheet. Fade over this depth range.
const CONTACT_DEPTH: f32 = 0.55;

/// The fraction of a droplet's life over which it shrinks to nothing.
const FADE: f32 = 0.3;

/// A droplet's side at birth and at death, m.
///
/// A droplet grows as it flies: not because water does, but because a sheet
/// of spray disperses into a mist, and one growing quad is the cheapest
/// picture of that.
const SIZE_BORN: f32 = 0.035;
const SIZE_DEAD: f32 = 0.10;

/// Downward acceleration of a droplet in flight, m/s².
///
/// The same `g` the physics uses; the frontend is not the place for a
/// different gravity.
const GRAVITY: f32 = 9.81;

/// Horizontal air drag on a droplet, per second.
///
/// Exponential decay makes the sheet curl back without depending on render
/// frame rate. Ballistic motion is substepped on the engine clock.
const DRAG: f32 = 1.5;

/// Maximum ballistic step: collision follows crests between render frames.
const FLIGHT_STEP: f32 = 1.0 / 60.0;

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
    /// Last physical instant drawn. The render clock may keep running paused.
    time: Option<f64>,
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
            // A soft centre avoids opaque white confetti where sheets overlap.
            let t = (1.0 - radius).clamp(0.0, 1.0);
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
        time: None,
    });
}

/// Births new droplets at the bow and flies the ones in the air.
///
/// One system rather than two, because the two share a pass over the pool: a
/// dead droplet is revived in the same iteration that would otherwise have
/// skipped it, and there is no second query for the spawner to find one with.
///
/// Runs in `Update`, after the engine has stepped. The physical time difference
/// advances flight, emission and moving-surface collision together. Camera
/// facing still updates while paused, but no water is born or aged.
pub fn emit_and_advance(
    engine: Res<Engine>,
    mut emitter: ResMut<Emitter>,
    cameras: Query<&GlobalTransform, With<Chase>>,
    mut droplets: Query<(&mut Droplet, &mut Transform, &mut Visibility)>,
) {
    let sim_time = engine.sim.time();
    let previous_time = emitter.time.replace(sim_time).unwrap_or(sim_time);
    let dt = (sim_time - previous_time) as f32;
    let facing = cameras
        .single()
        .map_or(Quat::IDENTITY, GlobalTransform::rotation);
    if dt <= 0.0 {
        if dt < 0.0 {
            emitter.owed = 0.0;
        }
        for (mut droplet, mut transform, mut visibility) in &mut droplets {
            if dt < 0.0 {
                droplet.age = droplet.life;
                *visibility = Visibility::Hidden;
            } else if droplet.age < droplet.life {
                // Keep the velocity-aligned billboard facing a moving camera.
                transform.rotation = droplet_facing(facing, droplet.velocity);
            }
        }
        return;
    }
    let state = engine.sim.state();
    let sea = engine.sea.as_ref();
    let elevation = |x: f32, z: f32, time: f64| -> f32 {
        sea.map_or(0.0, |sea| {
            frame::metres(sea.elevation(f64::from(x), f64::from(z), time))
        })
    };
    // A rigorous bound, unlike a fixed mean-level cutoff: choppy displacement
    // changes the parcel under a point but never its sum of wave amplitudes.
    // One cheap sum per frame saves all wave evaluations above the crests.
    let highest_crest = sea.map_or(0.0, |sea| {
        sea.waves()
            .iter()
            .map(|wave| wave.amplitude.abs())
            .sum::<f64>() as f32
            + 1e-4
    });

    let forward = frame::to_render(state.to_world(Vector3::x()));
    let heading = Vec2::new(forward.x, forward.z).normalize_or(Vec2::X);
    let across = Vec3::new(-heading.y, 0.0, heading.x);
    let along = Vec3::new(heading.x, 0.0, heading.y);
    let bow_body = Vector3::new(engine.hull_length(), 0.0, 0.0);
    let bow_world = state.position + state.to_world(bow_body);
    let bow_velocity = state.to_world(state.point_velocity(bow_body));
    let bow = frame::to_render(bow_world);
    let water_height = elevation(bow.x, bow.z, sim_time);
    let immersion = water_height - bow.y;

    // Differentiate the actual sea sampled along the bow's world trajectory,
    // including horizontal encounter and choppy parcel inversion. Adding the
    // bow's z-down velocity gives positive entry, negative emergence.
    let surface_rate = sea.map_or(0.0, |sea| {
        const SAMPLE: f64 = 0.01;
        let before = bow_world - bow_velocity * SAMPLE;
        let after = bow_world + bow_velocity * SAMPLE;
        (sea.elevation(after.x, after.y, sim_time + SAMPLE)
            - sea.elevation(before.x, before.y, sim_time - SAMPLE))
            / (2.0 * SAMPLE)
    });
    let entry =
        ((bow_velocity.z + surface_rate) as f32 - ENTRY_THRESHOLD).clamp(0.0, ENTRY_CEILING);
    let wet = if immersion >= 0.0 && engine.hull_length() > 0.0 {
        let depth = (immersion / CONTACT_DEPTH).clamp(0.0, 1.0);
        1.0 - depth * depth * (3.0 - 2.0 * depth)
    } else {
        0.0
    };
    let rate = BIRTHS_PER_ENTRY * entry * entry * wet;
    // Do not carry fractions across separate impacts: a dry or rising stem
    // cannot bank spray to release during an unrelated later contact.
    let owed = if rate > 0.0 {
        emitter.owed + rate * dt
    } else {
        0.0
    };
    let mut births = owed.floor();
    emitter.owed = owed - births;
    let contact_velocity = frame::to_render(bow_velocity);

    for (mut droplet, mut transform, mut visibility) in &mut droplets {
        if droplet.age < droplet.life {
            droplet.age += dt;
            let mut elapsed = 0.0;
            let mut drowned = false;
            while elapsed < dt && droplet.age < droplet.life {
                let step = (dt - elapsed).min(FLIGHT_STEP);
                let decay = (-DRAG * step).exp();
                // Exact displacement for linear horizontal drag and gravity.
                let travel = (1.0 - decay) / DRAG;
                transform.translation.x += droplet.velocity.x * travel;
                transform.translation.z += droplet.velocity.z * travel;
                transform.translation.y += droplet.velocity.y * step - 0.5 * GRAVITY * step * step;
                droplet.velocity.x *= decay;
                droplet.velocity.z *= decay;
                droplet.velocity.y -= GRAVITY * step;
                elapsed += step;
                let position = transform.translation;
                if position.y <= highest_crest
                    && position.y
                        <= elevation(position.x, position.z, previous_time + f64::from(elapsed))
                {
                    drowned = true;
                    break;
                }
            }
            if droplet.age >= droplet.life || drowned {
                droplet.age = droplet.life;
                *visibility = Visibility::Hidden;
                continue;
            }

            let fraction = droplet.age / droplet.life;
            let size = SIZE_BORN + (SIZE_DEAD - SIZE_BORN) * fraction;
            let fade = ((1.0 - fraction) / FADE).min(1.0);
            let streak = 1.0 + (1.0 - fraction) * (droplet.velocity.length() * 0.18).min(1.5);
            transform.rotation = droplet_facing(facing, droplet.velocity);
            transform.scale = Vec3::new(size / streak.sqrt(), size * streak, size) * fade;
        } else if births > 0.0 {
            births -= 1.0;
            let rng = &mut emitter.rng;
            // A narrow, coherent fan originates at the intersecting stem,
            // not a broad rectangle floating alongside a possibly dry bow.
            let side = rng.side();
            let aft = rng.between(0.0, 0.55);
            let out = rng.between(0.06, 0.26);
            let x = bow.x - along.x * aft + across.x * side * out;
            let z = bow.z - along.z * aft + across.z * side * out;
            let y = elevation(x, z, sim_time) + 0.025;
            transform.translation = Vec3::new(x, y, z);

            let energy = entry.sqrt();
            let outward = across * (side * rng.between(1.0, 2.2) * energy);
            let upward = Vec3::Y * (rng.between(1.2, 2.8) * energy);
            let jitter = Vec3::new(
                rng.between(-0.18, 0.18),
                rng.between(-0.12, 0.12),
                rng.between(-0.18, 0.18),
            );
            // Carry horizontal stem motion; downward entry has already been
            // redirected into the upward/outward sheet, not added a second time.
            droplet.velocity = Vec3::new(contact_velocity.x, 0.0, contact_velocity.z) * 0.6
                + outward
                + upward
                + jitter;
            droplet.age = 0.0;
            droplet.life = rng.between(0.6, 1.25);
            transform.rotation = droplet_facing(facing, droplet.velocity);
            transform.scale = Vec3::new(SIZE_BORN * 0.75, SIZE_BORN * 1.8, SIZE_BORN);
            *visibility = Visibility::Visible;
        }
    }
}

/// Keep the sheet's elongated droplets aligned to their projected velocity.
fn droplet_facing(camera: Quat, velocity: Vec3) -> Quat {
    let local = camera.inverse() * velocity;
    let angle = if local.x * local.x + local.y * local.y > 1e-6 {
        (-local.x).atan2(local.y)
    } else {
        0.0
    };
    camera * Quat::from_rotation_z(angle)
}
