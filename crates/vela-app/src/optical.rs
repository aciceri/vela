//! Camera-resolved optical waves, rendered once at half the screen resolution.
//!
//! A dedicated camera schedule writes raw signed slopes straight into its float
//! image. It has no 3D phase, lights, depth, exposure, tonemapping or output blit.
//! The ocean only receives a target after its draw has been submitted; resize,
//! weather replacement and clock rewind invalidate that acknowledgement.

use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};

use bevy::asset::{embedded_asset, embedded_path, AssetPath, RenderAssetUsages};
use bevy::camera::{
    visibility::RenderLayers, CameraOutputMode, CameraUpdateSystems, Hdr, RenderTarget,
};
use bevy::ecs::schedule::ScheduleLabel;
use bevy::image::ImageSampler;
use bevy::prelude::*;
use bevy::render::{
    camera::CameraRenderGraph,
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_asset::RenderAssets,
    render_resource::{binding_types::uniform_buffer, *},
    renderer::{RenderContext, RenderGraph, RenderGraphSystems, RenderQueue},
    texture::GpuImage,
    view::{ViewUniform, ViewUniformOffset, ViewUniforms},
    RenderApp, RenderStartup,
};
use bevy::shader::{load_shader_library, ShaderDefVal};
use bevy::transform::TransformSystems;
use bevy::window::PrimaryWindow;

use crate::ocean::{OceanMaterial, SeaUniform, ShaderWave, MAX_TRAIL, MAX_WAVES};
use crate::view::Chase;

const LAYER: usize = 2;

pub(crate) struct OpticalPlugin;

impl Plugin for OpticalPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "shaders/ocean_optics.wgsl");
        embedded_asset!(app, "shaders/optical_field.wgsl");
        app.init_resource::<Exchange>()
            .add_plugins((
                ExtractResourcePlugin::<Exchange>::default(),
                ExtractComponentPlugin::<OpticalSource>::default(),
            ))
            .add_systems(Startup, spawn.after(crate::view::spawn))
            // All live sea and camera updates finish in Update. Copy before
            // camera projection calculation and global transform propagation.
            .add_systems(
                PostUpdate,
                synchronize
                    .before(CameraUpdateSystems)
                    .before(TransformSystems::Propagate),
            );
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<FieldGpu>()
            .add_systems(RenderStartup, initialize_pipeline)
            .init_schedule(OpticalRender)
            .add_systems(OpticalRender, render_field)
            .add_systems(
                RenderGraph,
                (
                    begin_frame.in_set(RenderGraphSystems::Begin),
                    acknowledge.in_set(RenderGraphSystems::Finish),
                ),
            );
    }
}

#[derive(ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash)]
struct OpticalRender;

#[derive(Component)]
struct OpticalCamera;

/// Use the actual main-view uniform, including its exact projection at odd
/// viewport sizes. The auxiliary camera only drives pass ordering and size.
#[derive(Component, Clone, ExtractComponent)]
struct OpticalSource;

#[derive(Resource)]
struct Field {
    ocean: Handle<OceanMaterial>,
    image: Handle<Image>,
    neutral: Handle<Image>,
    size: UVec2,
    generation: u32,
    epoch: u64,
    previous_time: f32,
}

#[derive(Resource, Clone, ExtractResource, Default)]
struct Exchange {
    frame: Option<FieldFrame>,
    completed: Arc<AtomicU32>,
}

#[derive(Clone)]
struct FieldFrame {
    image: Handle<Image>,
    generation: u32,
    sea: SeaUniform,
    // A realisation is immutable between epochs. Only the Arc is extracted
    // each frame, and its GPU buffer is rewritten only on replacement.
    waves: Arc<[ShaderWave; MAX_WAVES]>,
}

fn half_size(size: UVec2) -> UVec2 {
    (size / 2).max(UVec2::ONE)
}

fn target(size: UVec2) -> Image {
    let mut image = Image::new_uninit(
        Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::Rgba16Float,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage |=
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::RENDER_ATTACHMENT;
    image.sampler = ImageSampler::linear();
    image
}

fn neutral() -> Image {
    // Half floats: flat slopes, conservative variance 0.125, no breaking.
    // This is warmup data only, never the live optical field.
    let mut image = Image::new_fill(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0, 0, 0x30, 0, 0],
        TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::linear();
    image
}

fn spawn(
    mut commands: Commands,
    windows: Query<&Window, With<PrimaryWindow>>,
    chase: Query<(Entity, &Transform, &Projection), With<Chase>>,
    ocean_meshes: Query<&MeshMaterial3d<OceanMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut oceans: ResMut<Assets<OceanMaterial>>,
    mut exchange: ResMut<Exchange>,
) {
    let (Ok(window), Ok((source, transform, projection)), Ok(material)) =
        (windows.single(), chase.single(), ocean_meshes.single())
    else {
        return;
    };
    let Some(mut ocean) = oceans.get_mut(&material.0) else {
        return;
    };
    let size = half_size(window.physical_size());
    let image = images.add(target(size));
    let neutral = images.add(neutral());
    ocean.optical_field = Some(neutral.clone());
    ocean.sea.temporal.y = 0.0;
    exchange.frame = Some(FieldFrame {
        image: image.clone(),
        generation: 1,
        sea: ocean.sea.clone(),
        waves: Arc::new(ocean.waves),
    });
    commands.insert_resource(Field {
        ocean: material.0.clone(),
        image: image.clone(),
        neutral,
        size,
        generation: 1,
        epoch: ocean.foam_epoch,
        previous_time: ocean.sea.time,
    });
    commands.entity(source).insert(OpticalSource);
    commands.spawn((
        Camera {
            order: -2,
            output_mode: CameraOutputMode::Skip,
            clear_color: ClearColorConfig::None,
            ..default()
        },
        CameraRenderGraph::new(OpticalRender),
        RenderTarget::Image(image.into()),
        *transform,
        projection.clone(),
        Msaa::Off,
        Hdr,
        RenderLayers::layer(LAYER),
        OpticalCamera,
    ));
}

type OpticalOnly = (With<OpticalCamera>, Without<Chase>);

fn synchronize(
    field: Option<ResMut<Field>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    chase: Query<(&Transform, &Projection), With<Chase>>,
    mut cameras: Query<(&mut Transform, &mut Projection, &mut RenderTarget), OpticalOnly>,
    mut images: ResMut<Assets<Image>>,
    mut oceans: ResMut<Assets<OceanMaterial>>,
    mut exchange: ResMut<Exchange>,
) {
    let (Some(mut field), Ok(window), Ok((transform, projection))) =
        (field, windows.single(), chase.single())
    else {
        return;
    };
    let Some(mut ocean) = oceans.get_mut(&field.ocean) else {
        return;
    };
    let size = half_size(window.physical_size());
    let resized = size != field.size;
    let replaced = ocean.foam_epoch != field.epoch;
    if resized {
        field.size = size;
        field.image = images.add(target(size));
    }
    if resized || replaced || ocean.sea.time < field.previous_time {
        field.generation = field.generation.wrapping_add(1).max(1);
    }
    field.epoch = ocean.foam_epoch;
    field.previous_time = ocean.sea.time;
    let ready = exchange.completed.load(Ordering::Acquire) == field.generation;
    let image = if ready { &field.image } else { &field.neutral };
    if ocean.optical_field.as_ref() != Some(image) {
        ocean.optical_field = Some(image.clone());
    }
    ocean.sea.temporal.y = if ready { 1.0 } else { 0.0 };
    if let Some(frame) = &mut exchange.frame {
        frame.image = field.image.clone();
        frame.generation = field.generation;
        frame.sea.clone_from(&ocean.sea);
        if replaced {
            frame.waves = Arc::new(ocean.waves);
        }
    }
    for (mut camera_transform, mut camera_projection, mut camera_target) in &mut cameras {
        *camera_transform = *transform;
        *camera_projection = projection.clone();
        if resized {
            *camera_target = RenderTarget::Image(field.image.clone().into());
        }
    }
}

#[derive(Resource)]
struct FieldPipeline {
    view_layout: BindGroupLayoutDescriptor,
    sea_layout: BindGroupLayoutDescriptor,
    id: CachedRenderPipelineId,
}

fn initialize_pipeline(
    mut commands: Commands,
    assets: Res<AssetServer>,
    cache: Res<PipelineCache>,
) {
    let view_layout = BindGroupLayoutDescriptor::new(
        "optical_view",
        &BindGroupLayoutEntries::single(
            ShaderStages::FRAGMENT,
            uniform_buffer::<ViewUniform>(true),
        ),
    );
    let sea_layout = BindGroupLayoutDescriptor::new(
        "optical_sea",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                uniform_buffer::<SeaUniform>(false),
                uniform_buffer::<[ShaderWave; MAX_WAVES]>(false),
            ),
        ),
    );
    let shader = assets.load(
        AssetPath::from_path_buf(embedded_path!("shaders/optical_field.wgsl"))
            .with_source("embedded"),
    );
    let defs = vec![
        ShaderDefVal::UInt("MATERIAL_BIND_GROUP".into(), 1),
        ShaderDefVal::UInt("MAX_WAVES".into(), MAX_WAVES as u32),
        ShaderDefVal::UInt("MAX_TRAIL".into(), MAX_TRAIL as u32),
        ShaderDefVal::UInt("OCEAN_COLUMNS".into(), crate::view::SEA_COLUMNS),
        ShaderDefVal::UInt("OCEAN_ROWS".into(), crate::view::SEA_ROWS),
    ];
    let id = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("optical_field".into()),
        layout: vec![view_layout.clone(), sea_layout.clone()],
        vertex: VertexState {
            shader: shader.clone(),
            shader_defs: defs.clone(),
            entry_point: Some("vertex".into()),
            ..default()
        },
        fragment: Some(FragmentState {
            shader,
            shader_defs: defs,
            entry_point: Some("fragment".into()),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba16Float,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        ..default()
    });
    commands.insert_resource(FieldPipeline {
        view_layout,
        sea_layout,
        id,
    });
}

#[derive(Resource, Default)]
struct FieldGpu {
    buffers: Option<FieldBuffers>,
    view_binding: Option<(BufferId, BindGroup)>,
    encoded: u32,
}

struct FieldBuffers {
    sea: UniformBuffer<SeaUniform>,
    waves: UniformBuffer<[ShaderWave; MAX_WAVES]>,
    source: Arc<[ShaderWave; MAX_WAVES]>,
    binding: BindGroup,
}

fn begin_frame(mut gpu: ResMut<FieldGpu>) {
    gpu.encoded = 0;
}

fn render_field(
    exchange: Res<Exchange>,
    (pipeline, cache): (Res<FieldPipeline>, Res<PipelineCache>),
    images: Res<RenderAssets<GpuImage>>,
    (views, uniforms): (
        Query<&ViewUniformOffset, With<OpticalSource>>,
        Res<ViewUniforms>,
    ),
    queue: Res<RenderQueue>,
    mut gpu: ResMut<FieldGpu>,
    mut context: RenderContext,
) {
    let (Some(frame), Ok(view)) = (&exchange.frame, views.single()) else {
        return;
    };
    let (Some(pipeline_object), Some(destination), Some(view_buffer)) = (
        cache.get_render_pipeline(pipeline.id),
        images.get(&frame.image),
        uniforms.uniforms.buffer(),
    ) else {
        return;
    };
    let Some(view_binding) = uniforms.uniforms.binding() else {
        return;
    };
    let gpu = &mut *gpu;
    let device = context.render_device();
    if gpu
        .view_binding
        .as_ref()
        .is_none_or(|(id, _)| *id != view_buffer.id())
    {
        gpu.view_binding = Some((
            view_buffer.id(),
            device.create_bind_group(
                "optical_view",
                &cache.get_bind_group_layout(&pipeline.view_layout),
                &BindGroupEntries::single(view_binding),
            ),
        ));
    }
    let buffers = gpu.buffers.get_or_insert_with(|| {
        let mut sea = UniformBuffer::from(frame.sea.clone());
        let mut waves = UniformBuffer::from(*frame.waves);
        sea.write_buffer(device, &queue);
        waves.write_buffer(device, &queue);
        let binding = device.create_bind_group(
            "optical_sea",
            &cache.get_bind_group_layout(&pipeline.sea_layout),
            &BindGroupEntries::sequential((&sea, &waves)),
        );
        FieldBuffers {
            sea,
            waves,
            source: frame.waves.clone(),
            binding,
        }
    });
    buffers.sea.set(frame.sea.clone());
    buffers.sea.write_buffer(device, &queue);
    if !Arc::ptr_eq(&buffers.source, &frame.waves) {
        buffers.waves.set(*frame.waves);
        buffers.waves.write_buffer(device, &queue);
        buffers.source = frame.waves.clone();
    }
    let mut pass = context
        .command_encoder()
        .begin_render_pass(&RenderPassDescriptor {
            label: Some("optical_field"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &destination.texture_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(default()),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    pass.set_pipeline(pipeline_object);
    pass.set_bind_group(1, &buffers.binding, &[]);
    pass.set_bind_group(
        0,
        &gpu.view_binding
            .as_ref()
            .expect("view binding was prepared")
            .1,
        &[view.offset],
    );
    pass.draw(0..6, 0..1);
    drop(pass);
    gpu.encoded = frame.generation;
}

fn acknowledge(exchange: Res<Exchange>, gpu: Res<FieldGpu>) {
    if gpu.encoded != 0 {
        exchange.completed.store(gpu.encoded, Ordering::Release);
    }
}
