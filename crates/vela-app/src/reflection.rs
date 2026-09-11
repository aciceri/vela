//! The boat, seen in the water.
//!
//! # A second camera, not a cubemap
//!
//! The sea reflects the sky by evaluating the sky's closed form in the
//! reflected direction, and that is the right tool for a thing that is the same
//! from everywhere. The boat is not: its reflection depends on where the boat
//! is, where the viewer is, and which way the water under both is tilted. The
//! standard answer for a single large flat mirror is the one Bevy's own `mirror`
//! example gives — a second camera, the first one reflected through the plane,
//! drawing into an image the water then samples at each fragment's own screen
//! position — and this module is that recipe applied to the plane `y = 0`.
//!
//! What it renders is the boat and nothing else. The sky is already in the water
//! by the closed form above, the sea would be reflecting itself, and the HUD is
//! not a thing that floats. So the boat's meshes sit on a second render layer as
//! well as the first, the mirror camera looks at that layer alone, and the image
//! is cleared to transparent black: where there is no boat there is alpha zero,
//! and the ocean composites the boat over the sky it computes for itself.
//!
//! # Which way up
//!
//! The mirror camera's view matrix is the chase camera's with a reflection
//! composed in front of it, so a point's clip position in the mirror view is the
//! chase view's clip position of that point's *mirror image*. That is exactly
//! the undisturbed reflection's screen position. The ocean projects its perturbed
//! reflected ray into this image, filters it by surface roughness, and fades
//! samples beyond the image bounds. Its orientation needs no additional flip;
//! the boat within it is upside down, as a reflection is.
//!
//! # What it costs
//!
//! The image is half the window in each direction. A reflection is seen through
//! a rough, moving surface, so its detail is thrown away by the water before it
//! reaches the eye, and a quarter of the fragments is a quarter of the fill for
//! a difference that cannot be seen. The pass draws the boat's few thousand
//! triangles over a clear and blits the result: on the integrated GPU at 1280
//! by 720, Bevy's render diagnostics put the mirror's opaque pass at 0.07 to
//! 0.11 ms, its blit at 0.05 to 0.07 and its light clustering at 0.12, about a
//! third of a millisecond against the main view's opaque pass alone at about a
//! millisecond. What it deliberately does *not* do is cast shadows — see the
//! second sun in `view::spawn` — because shadow cascades are built per view,
//! and the mirror would otherwise have doubled them.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::RenderLayers;
use bevy::camera::{Hdr, RenderTarget};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::ecs::message::MessageReader;
use bevy::math::reflection_matrix;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::window::{PrimaryWindow, WindowResized};

use crate::ocean::OceanMaterial;
use crate::view::{Chase, AMBIENT};

/// The render layer the mirror camera sees.
///
/// Layer 0 is what every entity and camera belongs to unless told otherwise.
/// This one holds the boat, the sun that lights it for the mirror, and nothing
/// else, so the mirror can be pointed at the boat without a way to say "not
/// the sea" — which render layers do not have.
pub const LAYER: usize = 1;

/// The layers an entity that should appear in the water belongs to: the main
/// view's, and the mirror's.
///
/// Not inherited by children, which is why the boat's hull, spars and sails
/// each carry it rather than the pose entity above them.
#[must_use]
pub fn mirrored() -> RenderLayers {
    RenderLayers::from_layers(&[0, LAYER])
}

/// Marks the camera that draws the reflection.
#[derive(Component)]
pub struct Mirror;

/// The mirror camera and not the chase camera.
///
/// [`follow`] reads the one and writes the other in a single system, and Bevy
/// needs the exclusion stated to see that the two queries cannot overlap.
type MirrorOnly = (With<Mirror>, Without<Chase>);

/// The image the mirror camera draws into, for a window of the given physical
/// size.
///
/// Half the window in each direction; see the module documentation. Sixteen-bit
/// float rather than eight-bit sRGB because the ocean mixes the sample into its
/// own *linear, pre-tonemap* colour: the boat's sunlit cloth is well over 1.0
/// there, and an eight-bit target would clip it flat before the display
/// transform had a chance to roll it off. It also spares a colour-space
/// conversion each way.
fn target(window: UVec2) -> Image {
    let mut image = Image::new_uninit(
        Extent3d {
            width: (window.x / 2).max(1),
            height: (window.y / 2).max(1),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::Rgba16Float,
        // Both worlds: `camera_system` reads the image's size from the main
        // world's copy to compute the mirror camera's aspect.
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage |=
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::RENDER_ATTACHMENT;
    image
}

/// Hands the water the image to reflect the boat from.
fn attach(image: &Handle<Image>, oceans: &mut Assets<OceanMaterial>) {
    for (_, ocean) in oceans.iter_mut() {
        ocean.set_reflection(Some(image.clone()));
    }
}

/// Spawns the mirror camera and its image, and gives the image to the water.
///
/// After `view::spawn`, which creates the ocean material this attaches to and
/// the chase camera whose projection the mirror copies.
pub fn spawn(
    mut commands: Commands,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut images: ResMut<Assets<Image>>,
    mut oceans: ResMut<Assets<OceanMaterial>>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let image = images.add(target(window.physical_size()));
    attach(&image, &mut oceans);

    commands.spawn((
        Camera3d::default(),
        Camera {
            // Before the chase camera, so the water samples this frame's mirror
            // and not last frame's.
            order: -1,
            // A reflection reverses winding. Without this, back-face culling
            // would keep the inside of the hull and drop the outside.
            invert_culling: true,
            // Transparent where there is no boat: the water composites the boat
            // over its own sky, and an opaque clear colour would paint over it.
            clear_color: ClearColorConfig::Custom(Color::NONE),
            ..default()
        },
        RenderTarget::Image(image.clone().into()),
        // Placed by `follow` before the first frame is drawn; anything here
        // would be overwritten.
        Transform::default(),
        // Same reasons as the chase camera: one full-screen shader's worth of
        // edges is not worth four samples on an integrated GPU.
        Msaa::Off,
        // A float target so the boat's highlights survive to the water. See
        // `target`.
        Hdr,
        // No display transform. The image is *input* to the ocean shader, whose
        // output the chase camera then tonemaps; a boat that had been through
        // ACES once here would go through it twice on the way to the screen and
        // come out darker in the water than above it. Linear in, linear out, and
        // the one transform is the chase camera's.
        Tonemapping::None,
        // The same sky light as the chase camera, or the boat would be shaded
        // differently in the water than above it.
        AMBIENT,
        RenderLayers::layer(LAYER),
        Mirror,
    ));
}

/// Keeps the mirror camera the chase camera's reflection in the water.
///
/// After `view::chase`, which is what moves the chase camera; the reflection is
/// recomputed from the chase camera's `Transform` rather than its
/// `GlobalTransform`, which is not propagated until `PostUpdate` and would put
/// the mirror a frame behind.
///
/// The reflection is composed as *matrices* and only then decomposed into a
/// `Transform`, for the reason the Bevy example gives: a reflection is a
/// non-uniform scale of −1 along the normal, and `Transform` composition cannot
/// represent that in general. The decomposition leaves the negative scale on one
/// axis of the result, which is what `invert_culling` on the camera is for.
///
/// The projection is the chase camera's, with an oblique near plane laid along
/// the water. The mirror camera sits below the surface and looks up through it;
/// anything on its side of the plane — the hull below the waterline — is what
/// the real camera sees *through* the water rather than reflected in it, and
/// the plane clips it out.
///
/// In the mirror camera's view space the plane's normal is the water's normal
/// through the mirrored view, `(−Y)` through the chase camera's rotation, and
/// its distance is the chase camera's height, negated: Bevy wants the negative
/// signed distance from the eye.
///
/// `aspect_ratio` is copied and then overwritten by Bevy's `camera_system`,
/// which computes it from the mirror's own target — the window halved, which
/// rounds to the same ratio within a pixel — so it is the field of view that
/// this actually carries across.
pub fn follow(
    chase: Query<(&Transform, &Projection), With<Chase>>,
    mut mirrors: Query<(&mut Transform, &mut Projection), MirrorOnly>,
) {
    let Ok((camera, Projection::Perspective(projection))) = chase.single() else {
        return;
    };

    let transform =
        Transform::from_matrix(Mat4::from_mat3a(reflection_matrix(Vec3::Y)) * camera.to_matrix());
    let normal = camera.rotation.inverse() * Vec3::NEG_Y;
    let near_clip_plane = normal.extend(-camera.translation.y);

    for (mut mirror, mut mirror_projection) in &mut mirrors {
        *mirror = transform;
        *mirror_projection = Projection::Perspective(PerspectiveProjection {
            near_clip_plane,
            ..projection.clone()
        });
    }
}

/// Re-sizes the image with the window.
///
/// The water samples the image at screen position, so the image has to keep
/// the window's aspect, and half its size is a size that changes with it. The
/// camera's target and the ocean material both take the new handle, and the
/// old image goes when its last handle does.
pub fn resize(
    mut resized: MessageReader<WindowResized>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut mirrors: Query<&mut RenderTarget, With<Mirror>>,
    mut images: ResMut<Assets<Image>>,
    mut oceans: ResMut<Assets<OceanMaterial>>,
) {
    // Once, however many resize messages arrived this frame.
    let Some(resized) = resized.read().last() else {
        return;
    };
    let Ok(window) = windows.get(resized.window) else {
        return;
    };

    let image = images.add(target(window.physical_size()));
    attach(&image, &mut oceans);
    for mut target in &mut mirrors {
        *target = RenderTarget::Image(image.clone().into());
    }
}
