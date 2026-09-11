//! Persistent, engine-clock boat foam on an advected parcel-space lattice.
//!
//! A fullscreen raster pass reads one 256² UNORM image and writes the other,
//! before any camera renders. No compute, storage buffers, camera, depth,
//! tonemapping or multisampling is involved. The visible ocean keeps the last
//! submitted image until the render world acknowledges a successful pass; an
//! unavailable pipeline or image therefore cannot flip valid history away.
//! Ambient whitecaps belong to the global optical field, not this bounded wake.

use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};

use bevy::asset::{embedded_asset, embedded_path, AssetId, AssetPath, RenderAssetUsages};
use bevy::core_pipeline::FullscreenShader;
use bevy::image::ImageSampler;
use bevy::prelude::*;
use bevy::render::{
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_asset::RenderAssets,
    render_resource::{
        binding_types::{sampler, texture_2d, uniform_buffer},
        *,
    },
    renderer::{RenderContext, RenderGraph, RenderGraphSystems, RenderQueue},
    texture::GpuImage,
    RenderApp, RenderStartup,
};
use bevy::shader::{load_shader_library, ShaderDefVal};

use crate::ocean::{OceanMaterial, SeaUniform, ShaderWave, TrailPoint, MAX_TRAIL, MAX_WAVES};
use crate::sim::Engine;

const RESOLUTION: u32 = 256;
const SPAN: f32 = 160.0;
const TEXEL: f32 = SPAN / RESOLUTION as f32;
const DRIFT: f32 = 0.055;
const DECAY: f32 = 0.32;

pub(crate) struct FoamPlugin;

impl Plugin for FoamPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "shaders/ocean_surface.wgsl");
        embedded_asset!(app, "shaders/foam_history.wgsl");
        app.init_resource::<History>()
            .init_resource::<Exchange>()
            .add_plugins(ExtractResourcePlugin::<Exchange>::default())
            // The ocean's engine/heading synchronization runs in Update.
            .add_systems(PostUpdate, advance);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<FoamGpu>()
            .add_systems(RenderStartup, initialize_pipeline)
            .add_systems(
                RenderGraph,
                (
                    render_history
                        .after(RenderGraphSystems::Begin)
                        .before(RenderGraphSystems::Render),
                    acknowledge.in_set(RenderGraphSystems::Finish),
                ),
            );
    }
}

#[derive(Resource, Default)]
struct History {
    session: Option<Session>,
    sequence: u32,
}

struct Session {
    source: AssetId<OceanMaterial>,
    epoch: u64,
    images: [Handle<Image>; 2],
    front: usize,
    origin: Vec2,
    time: f64,
    observed_time: f64,
    live: bool,
}

/// At most one request is outstanding. The acknowledgement is published only
/// after RenderGraph's Submit set, and corresponds to this exact snapshot.
#[derive(Resource, ExtractResource, Clone, Default)]
struct Exchange {
    request: Option<Arc<Request>>,
    completed: Arc<AtomicU32>,
    advancing: bool,
}

struct Request {
    sequence: u32,
    source: Handle<Image>,
    destination: Handle<Image>,
    destination_index: usize,
    time: f64,
    sea: SeaUniform,
    waves: [ShaderWave; MAX_WAVES],
    pass: FoamPass,
}

#[derive(ShaderType, Clone, Copy)]
struct FoamPass {
    previous_region: Vec4,
    region: Vec4,
    step: Vec4,
}

fn target() -> Image {
    // Explicit data initializes BOTH ping-pong images to zero, rather than
    // relying on their first render pass ever being ready to clear them.
    let mut image = Image::new_fill(
        Extent3d {
            width: RESOLUTION,
            height: RESOLUTION,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage |= TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC;
    image.sampler = ImageSampler::linear();
    image
}

fn region_origin(boat: Vec2, wind: Vec2, time: f64) -> Vec2 {
    // Translating the sampling lattice with the drift makes every backtrace
    // an integer-texel scroll, not a tiny bilinear blur repeated every frame.
    // It is still a bounded square around the boat, never around the camera.
    let drift = wind * (time * f64::from(DRIFT)) as f32;
    ((boat - drift) / TEXEL).floor() * TEXEL + drift - Vec2::splat(SPAN * 0.5)
}

fn advance(
    engine: Res<Engine>,
    mut history: ResMut<History>,
    mut exchange: ResMut<Exchange>,
    mut images: ResMut<Assets<Image>>,
    mut oceans: ResMut<Assets<OceanMaterial>>,
) {
    let Some((source, ocean)) = oceans.iter().next() else {
        exchange.request = None;
        exchange.advancing = false;
        return;
    };
    let time = engine.sim.time();
    let reset = history.session.as_ref().is_none_or(|session| {
        session.source != source
            || session.epoch != ocean.foam_epoch
            || time < session.observed_time
    });
    if reset {
        history.session = Some(Session {
            source,
            epoch: ocean.foam_epoch,
            images: [images.add(target()), images.add(target())],
            front: 0,
            origin: region_origin(ocean.sea.motion.xy(), ocean.sea.wind.xy(), time),
            time,
            observed_time: time,
            live: false,
        });
        // An acknowledgement from a discarded generation cannot publish its
        // image: the request holding its metadata is discarded with it.
        exchange.request = None;
    }

    let History { session, sequence } = &mut *history;
    let session = session.as_mut().expect("foam session was initialized");
    let advancing = time > session.observed_time;
    session.observed_time = time;
    exchange.advancing = advancing;

    if advancing {
        if let Some(request) = exchange
            .request
            .as_ref()
            .filter(|request| exchange.completed.load(Ordering::Acquire) == request.sequence)
        {
            session.front = request.destination_index;
            session.origin = request.pass.region.xy();
            session.time = request.time;
            session.live = true;
            exchange.request = None;
        }
        if exchange.request.is_none() && time > session.time {
            *sequence = sequence.wrapping_add(1).max(1);
            let destination_index = 1 - session.front;
            let origin = region_origin(ocean.sea.motion.xy(), ocean.sea.wind.xy(), time);
            exchange.request = Some(Arc::new(Request {
                sequence: *sequence,
                source: session.images[session.front].clone(),
                destination: session.images[destination_index].clone(),
                destination_index,
                time,
                sea: ocean.sea.clone(),
                waves: ocean.waves,
                pass: FoamPass {
                    previous_region: Vec4::new(session.origin.x, session.origin.y, SPAN, TEXEL),
                    region: Vec4::new(origin.x, origin.y, SPAN, TEXEL),
                    step: Vec4::new((time - session.time) as f32, DRIFT, DECAY, 0.0),
                },
            }));
        }
    }
    // Paused frames neither swap the published image nor execute a delayed
    // request. A sea/time reset above is the sole exception: both images clear.
    let region = Vec4::new(
        session.origin.x,
        session.origin.y,
        SPAN,
        if session.live { 1.0 } else { 0.0 },
    );
    for (_, ocean) in oceans.iter_mut() {
        ocean.foam_history = Some(session.images[session.front].clone());
        ocean.sea.foam_region = region;
    }
}

#[derive(Resource)]
struct FoamPipeline {
    layout: BindGroupLayoutDescriptor,
    id: CachedRenderPipelineId,
}

fn initialize_pipeline(
    mut commands: Commands,
    assets: Res<AssetServer>,
    fullscreen: Res<FullscreenShader>,
    cache: Res<PipelineCache>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "foam_history",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                uniform_buffer::<SeaUniform>(false),
                uniform_buffer::<[ShaderWave; MAX_WAVES]>(false),
                uniform_buffer::<[TrailPoint; MAX_TRAIL]>(false),
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                uniform_buffer::<FoamPass>(false),
            ),
        ),
    );
    let shader = assets.load(
        AssetPath::from_path_buf(embedded_path!("shaders/foam_history.wgsl"))
            .with_source("embedded"),
    );
    let id = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("foam_history".into()),
        layout: vec![layout.clone()],
        vertex: fullscreen.to_vertex_state(),
        fragment: Some(FragmentState {
            shader,
            shader_defs: vec![
                ShaderDefVal::UInt("MATERIAL_BIND_GROUP".into(), 0),
                ShaderDefVal::UInt("MAX_WAVES".into(), MAX_WAVES as u32),
                ShaderDefVal::UInt("MAX_TRAIL".into(), MAX_TRAIL as u32),
            ],
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        ..default()
    });
    commands.insert_resource(FoamPipeline { layout, id });
}

#[derive(Resource, Default)]
struct FoamGpu {
    buffers: Option<FoamBuffers>,
    encoded: u32,
}

struct FoamBuffers {
    sea: UniformBuffer<SeaUniform>,
    waves: UniformBuffer<[ShaderWave; MAX_WAVES]>,
    trail: UniformBuffer<[TrailPoint; MAX_TRAIL]>,
    pass: UniformBuffer<FoamPass>,
    // Each direction keeps its bind group. Only a reset or image recreation
    // changes the views; fixed-size uniform buffers are reused across frames.
    bindings: [Option<(TextureViewId, BindGroup)>; 2],
}

fn render_history(
    exchange: Res<Exchange>,
    pipeline: Res<FoamPipeline>,
    cache: Res<PipelineCache>,
    images: Res<RenderAssets<GpuImage>>,
    queue: Res<RenderQueue>,
    mut gpu: ResMut<FoamGpu>,
    mut context: RenderContext,
) {
    gpu.encoded = 0;
    let Some(request) = &exchange.request else {
        return;
    };
    if !exchange.advancing || exchange.completed.load(Ordering::Acquire) == request.sequence {
        return;
    }
    let Some(pipeline_object) = cache.get_render_pipeline(pipeline.id) else {
        return;
    };
    let (Some(source), Some(destination)) = (
        images.get(&request.source),
        images.get(&request.destination),
    ) else {
        return;
    };
    // A fixed direction for this acknowledged request guarantees these are
    // different, even when preparation took several frames.
    debug_assert_ne!(request.source.id(), request.destination.id());
    let device = context.render_device();
    let buffers = gpu.buffers.get_or_insert_with(|| {
        let mut trail = UniformBuffer::from([TrailPoint::default(); MAX_TRAIL]);
        trail.write_buffer(device, &queue);
        FoamBuffers {
            sea: UniformBuffer::from(request.sea.clone()),
            waves: UniformBuffer::from(request.waves),
            trail,
            pass: UniformBuffer::from(request.pass),
            bindings: [None, None],
        }
    });
    buffers.sea.set(request.sea.clone());
    buffers.waves.set(request.waves);
    buffers.pass.set(request.pass);
    buffers.sea.write_buffer(device, &queue);
    buffers.waves.write_buffer(device, &queue);
    buffers.pass.write_buffer(device, &queue);

    let binding = &mut buffers.bindings[request.destination_index];
    if binding
        .as_ref()
        .is_none_or(|(id, _)| *id != source.texture_view.id())
    {
        *binding = Some((
            source.texture_view.id(),
            device.create_bind_group(
                "foam_history",
                &cache.get_bind_group_layout(&pipeline.layout),
                &BindGroupEntries::sequential((
                    &buffers.sea,
                    &buffers.waves,
                    &buffers.trail,
                    &source.texture_view,
                    &source.sampler,
                    &buffers.pass,
                )),
            ),
        ));
    }
    let mut pass = context
        .command_encoder()
        .begin_render_pass(&RenderPassDescriptor {
            label: Some("foam_history"),
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
    pass.set_bind_group(
        0,
        &binding.as_ref().expect("foam bind group was prepared").1,
        &[],
    );
    pass.draw(0..3, 0..1);
    drop(pass);
    gpu.encoded = request.sequence;
}

fn acknowledge(exchange: Res<Exchange>, gpu: Res<FoamGpu>) {
    if gpu.encoded != 0 {
        exchange.completed.store(gpu.encoded, Ordering::Release);
    }
}
