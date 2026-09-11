//! The cockpit, as a view onto the engine.
//!
//! Sliders send crew input through the same controls as the keyboard. They do
//! not integrate motion or own a second set of settings: every thumb and number
//! is projected from the live engine. Positive displayed helm means starboard,
//! the opposite sign to the physical rudder angle. Manual helm stays set until
//! another crew input; the explicitly labelled course hold is optional.
//!
//! Stock Bevy widgets handle pointer capture, touch and keyboard navigation.
//! Tab focuses a control, left/right adjusts a slider, Home/End reaches its
//! limits, and Escape or a scene click returns the keys to the boat. The debug
//! readout is opt-in, and still only displays engine telemetry and unit
//! conversions, never an independently calculated physics answer.

use bevy::{
    ecs::{query::QueryData, system::SystemParam},
    input::{keyboard::KeyboardInput, ButtonState},
    input_focus::{
        tab_navigation::{TabGroup, TabIndex},
        FocusedInput, InputFocus,
    },
    picking::hover::Hovered,
    prelude::*,
    ui::{Pressed, RelativeCursorPosition},
    ui_widgets::{
        Activate, Button, ScrollArea, Slider, SliderDragState, SliderOrientation, SliderPrecision,
        SliderRange, SliderStep, SliderThumb, SliderValue, TrackClick, ValueChange,
    },
};
use vela_core::equilibrium::MAX_HELM;

use crate::{helm, sim::Engine};

/// Knots per metre per second. Display only; nothing in the engine knows what a
/// knot is.
const KNOTS: f64 = 1.943_844_492_440_605;
const INK: Color = Color::srgb(0.92, 0.95, 0.98);
const MUTED: Color = Color::srgb(0.64, 0.74, 0.83);
const ACCENT: Color = Color::srgb(0.40, 0.82, 0.87);
const MANUAL: Color = Color::srgb(0.98, 0.78, 0.47);
const BORDER: Color = Color::srgba(0.52, 0.68, 0.81, 0.40);
const THUMB_SIZE: f32 = 18.0;

/// The bottom cockpit. Its cursor position also gates scene-camera gestures.
#[derive(Component)]
pub struct ControlPanel;

/// Marks the optional telemetry viewport, bounded above the cockpit.
#[derive(Component)]
pub struct Readout;

/// The text inside the independently clipped telemetry viewport.
#[derive(Component)]
pub struct ReadoutText;

#[derive(Component, Clone, Copy, PartialEq, Eq)]
enum Control {
    Helm,
    Sheet,
    Traveller,
    Power,
}

impl Control {
    fn name(self) -> &'static str {
        match self {
            Self::Helm => "Helm",
            Self::Sheet => "Main sheet",
            Self::Traveller => "Traveller",
            Self::Power => "Sail power",
        }
    }

    fn range(self) -> SliderRange {
        match self {
            Self::Helm => {
                let limit = MAX_HELM.to_degrees() as f32;
                SliderRange::new(-limit, limit)
            }
            Self::Sheet | Self::Traveller => SliderRange::new(0.0, 1.0),
            Self::Power => SliderRange::new(0.4, 1.0),
        }
    }

    fn value(self, engine: &Engine) -> f32 {
        let controls = engine.sim.controls();
        match self {
            Self::Helm => -controls.rudder_angle.to_degrees() as f32,
            Self::Sheet => controls.shape.sheet as f32,
            Self::Traveller => controls.shape.traveller as f32,
            Self::Power => controls.trim.flat as f32,
        }
    }

    fn displayed(self, value: f32) -> i32 {
        (value * if self == Self::Helm { 10.0 } else { 100.0 }).round() as i32
    }

    fn format(self, displayed: i32) -> String {
        if self == Self::Helm {
            format!("{:+.1} deg", f64::from(displayed) / 10.0)
        } else {
            format!("{displayed}%")
        }
    }
}

/// Only caches the rendered precision, never a setting to write to the engine.
#[derive(Component)]
struct ControlValue {
    control: Control,
    displayed: i32,
}

#[derive(Component)]
struct HelmStatus;

#[derive(Component, Clone, Copy)]
enum CockpitButton {
    CourseHold,
    CenterHelm,
    Sea,
    Telemetry,
}

#[derive(Component)]
struct CourseLabel;

#[derive(Component)]
struct TelemetryLabel;

/// Marks the sea button, and the label inside it.
#[derive(Component)]
pub struct SeaButton;

#[derive(Component)]
pub struct SeaLabel;

fn text(value: impl Into<String>, size: f32, color: Color) -> impl Bundle {
    (
        Text::new(value),
        TextFont {
            font_size: size.into(),
            ..default()
        },
        TextColor(color),
        Pickable::IGNORE,
    )
}

fn button(action: CockpitButton) -> impl Bundle {
    (
        Button,
        action,
        Hovered::default(),
        TabIndex(0),
        Node {
            min_height: px(32),
            min_width: px(if matches!(action, CockpitButton::CourseHold) {
                132.0
            } else {
                0.0
            }),
            padding: UiRect::axes(px(9), px(5)),
            border: UiRect::all(px(1)),
            border_radius: BorderRadius::all(px(5)),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            ..default()
        },
        BorderColor::all(BORDER),
        BackgroundColor(Color::srgba(0.10, 0.20, 0.30, 0.78)),
    )
}

fn slider(control: Control, engine: &Engine) -> impl Bundle {
    let value = control.value(engine);
    let range = control.range();
    let position = range.thumb_position(value) * 100.0;
    let displayed = control.displayed(value);
    let (left, right) = match control {
        Control::Helm => ("PORT", "STBD"),
        Control::Sheet => ("EASED", "HARD"),
        Control::Traveller => ("OUT", "CENTRE"),
        Control::Power => ("40% / FLAT", "100% / FULL"),
    };
    (
        Node {
            flex_basis: px(140),
            flex_grow: 1.0,
            min_width: px(130),
            flex_direction: FlexDirection::Column,
            ..default()
        },
        children![
            (
                Node {
                    justify_content: JustifyContent::SpaceBetween,
                    align_items: AlignItems::Center,
                    column_gap: px(6),
                    ..default()
                },
                children![
                    text(control.name(), 14.0, INK),
                    (
                        text(control.format(displayed), 14.0, ACCENT),
                        ControlValue { control, displayed },
                    ),
                ],
            ),
            (
                control,
                Slider {
                    track_click: TrackClick::Snap,
                    orientation: SliderOrientation::Horizontal,
                },
                SliderValue(value),
                range,
                SliderStep(if control == Control::Helm { 1.0 } else { 0.01 }),
                SliderPrecision(if control == Control::Helm { 1 } else { 2 }),
                Hovered::default(),
                TabIndex(0),
                AccessibleLabel(control.name().into()),
                Node {
                    height: px(32),
                    width: percent(100),
                    justify_content: JustifyContent::Center,
                    flex_direction: FlexDirection::Column,
                    border_radius: BorderRadius::all(px(4)),
                    ..default()
                },
                Outline {
                    width: px(1),
                    offset: px(2),
                    color: Color::NONE,
                },
                children![
                    (
                        Node {
                            height: px(4),
                            border_radius: BorderRadius::MAX,
                            ..default()
                        },
                        BackgroundColor(Color::srgba(0.39, 0.53, 0.66, 0.65)),
                        Pickable::IGNORE,
                    ),
                    (
                        // As in standard_widgets: the thumb's parent is shorter
                        // by its width, so percentages match the stock hit math.
                        Node {
                            position_type: PositionType::Absolute,
                            left: px(0),
                            right: px(THUMB_SIZE),
                            top: px(7),
                            bottom: px(7),
                            ..default()
                        },
                        Pickable::IGNORE,
                        children![(
                            SliderThumb,
                            Node {
                                position_type: PositionType::Absolute,
                                width: px(THUMB_SIZE),
                                height: px(THUMB_SIZE),
                                left: percent(position),
                                border_radius: BorderRadius::MAX,
                                ..default()
                            },
                            BackgroundColor(ACCENT),
                        )],
                    ),
                ],
            ),
            (
                Node {
                    justify_content: JustifyContent::SpaceBetween,
                    ..default()
                },
                children![text(left, 10.0, MUTED), text(right, 10.0, MUTED)],
            ),
        ],
    )
}

/// Spawns the live cockpit and collapsed telemetry, and registers its observers.
///
/// Requires Bevy's UI widgets and picking features plus `TabNavigationPlugin`.
/// Register [`sync_controls`] in `Update` after keyboard input and [`sea_button`].
pub fn spawn(mut commands: Commands, engine: Res<Engine>) {
    commands.add_observer(change_control);
    commands.add_observer(activate_button);
    commands.add_observer(escape_focus);

    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            top: px(12),
            left: px(14),
            // Keep the optional readout to the left of the wind compass.
            max_width: percent(55),
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::FlexStart,
            row_gap: px(6),
            ..default()
        },
        TabGroup::default(),
        children![
            (
                button(CockpitButton::Telemetry),
                children![(text("Telemetry +", 12.0, INK), TelemetryLabel)],
            ),
            (
                Node {
                    display: Display::None,
                    padding: UiRect::all(px(10)),
                    max_height: px(0),
                    overflow: Overflow::scroll_y(),
                    flex_direction: FlexDirection::Column,
                    ..default()
                },
                BackgroundColor(Color::srgba(0.02, 0.07, 0.13, 0.88)),
                Readout,
                ScrollArea,
                RelativeCursorPosition::default(),
                children![(
                    text("", 12.0, INK),
                    Node {
                        flex_shrink: 0.0,
                        ..default()
                    },
                    ReadoutText,
                )],
            ),
        ],
    ));

    commands.spawn((
        ControlPanel,
        RelativeCursorPosition::default(),
        TabGroup::default(),
        Node {
            position_type: PositionType::Absolute,
            left: px(14),
            right: px(14),
            bottom: px(24),
            padding: UiRect::axes(px(12), px(9)),
            border: UiRect::all(px(1)),
            border_radius: BorderRadius::all(px(9)),
            flex_direction: FlexDirection::Column,
            row_gap: px(8),
            ..default()
        },
        BorderColor::all(BORDER),
        BackgroundColor(Color::srgba(0.025, 0.075, 0.14, 0.88)),
        children![
            (
                Node {
                    flex_wrap: FlexWrap::Wrap,
                    column_gap: px(20),
                    row_gap: px(9),
                    ..default()
                },
                children![
                    slider(Control::Helm, &engine),
                    slider(Control::Sheet, &engine),
                    slider(Control::Traveller, &engine),
                    slider(Control::Power, &engine),
                ],
            ),
            (
                Node {
                    flex_wrap: FlexWrap::Wrap,
                    align_items: AlignItems::Center,
                    column_gap: px(7),
                    row_gap: px(6),
                    ..default()
                },
                children![
                    (
                        button(CockpitButton::CourseHold),
                        children![(
                            text(course_label(engine.hold_course), 12.0, INK),
                            CourseLabel,
                        )],
                    ),
                    (
                        button(CockpitButton::CenterHelm),
                        children![text("Center helm", 12.0, INK)],
                    ),
                    (
                        text(helm_status(engine.hold_course), 11.0, ACCENT),
                        HelmStatus,
                        Node {
                            // Switching AUTO/MANUAL must not move a dragged slider.
                            min_width: px(170),
                            ..default()
                        },
                    ),
                    (
                        button(CockpitButton::Sea),
                        SeaButton,
                        children![(text(sea_label(&engine), 12.0, INK), SeaLabel)],
                    ),
                ],
            ),
            text(
                "Keys: Left/Right helm | Up/Down sheet | Q/E traveller | F/R power | M sea\n\
                 Tab + Left/Right sliders | Esc releases focus | Camera: right-drag / wheel | J/L I/K U/O",
                11.0,
                MUTED,
            ),
        ],
    ));
}

fn change_control(
    change: On<ValueChange<f32>>,
    sliders: Query<&Control, With<Slider>>,
    mut engine: ResMut<Engine>,
    mut commands: Commands,
) {
    let Ok(&control) = sliders.get(change.source) else {
        return;
    };
    let value = f64::from(control.range().clamp(change.value));
    if control == Control::Helm {
        helm::set_rudder(&mut engine, -value.to_radians());
    } else {
        let mut controls = *engine.sim.controls();
        match control {
            Control::Sheet => controls.shape.sheet = value,
            Control::Traveller => controls.shape.traveller = value,
            Control::Power => controls.trim = controls.trim.with_flat(value),
            Control::Helm => unreachable!(),
        }
        engine.sim.set_controls(controls);
    }
    // Read back the engine even here. This keeps repeated key events and the
    // stock drag-start offset current within a frame. Inserting SliderValue
    // does not emit ValueChange, so neither this nor sync_controls feeds back.
    commands
        .entity(change.source)
        .insert(SliderValue(control.value(&engine)));
}

fn activate_button(
    activate: On<Activate>,
    buttons: Query<&CockpitButton>,
    mut engine: ResMut<Engine>,
    mut readouts: Query<&mut Node, With<Readout>>,
    mut labels: Query<&mut Text, With<TelemetryLabel>>,
) {
    let Ok(action) = buttons.get(activate.entity) else {
        return;
    };
    match action {
        CockpitButton::CourseHold => {
            let enabled = !engine.hold_course;
            helm::set_course_hold(&mut engine, enabled);
        }
        CockpitButton::CenterHelm => helm::set_rudder(&mut engine, 0.0),
        CockpitButton::Sea => {
            let next = engine.preset.next();
            engine.set_sea(next);
        }
        CockpitButton::Telemetry => {
            for mut node in &mut readouts {
                let expanded = node.display == Display::None;
                node.display = if expanded {
                    Display::Flex
                } else {
                    Display::None
                };
                for mut label in &mut labels {
                    label.0.clear();
                    label.0.push_str(if expanded {
                        "Telemetry -"
                    } else {
                        "Telemetry +"
                    });
                }
            }
        }
    }
}

fn escape_focus(mut input: On<FocusedInput<KeyboardInput>>, mut focus: ResMut<InputFocus>) {
    if input.input.key_code == KeyCode::Escape && input.input.state == ButtonState::Pressed {
        input.propagate(false);
        if focus.get().is_some() {
            focus.clear();
        }
    }
}

fn course_label(enabled: bool) -> &'static str {
    if enabled {
        "Course hold ON"
    } else {
        "Course hold OFF"
    }
}

fn helm_status(enabled: bool) -> &'static str {
    if enabled {
        "AUTO / holding course"
    } else {
        "MANUAL / helm stays set"
    }
}

#[derive(QueryData)]
#[query_data(mutable)]
struct SliderView {
    entity: Entity,
    control: &'static Control,
    value: &'static SliderValue,
    range: &'static SliderRange,
    hovered: &'static Hovered,
    drag: &'static SliderDragState,
    outline: &'static mut Outline,
}

#[derive(QueryData)]
#[query_data(mutable)]
struct ButtonView {
    entity: Entity,
    action: &'static CockpitButton,
    hovered: &'static Hovered,
    pressed: Has<Pressed>,
    background: &'static mut BackgroundColor,
    border: &'static mut BorderColor,
}

type ControlValuesOnly = (Without<CourseLabel>, Without<HelmStatus>);
type ReadoutPanelsOnly = (With<Readout>, Without<SliderThumb>);

/// Disjoint widget queries for the cockpit's single synchronization system.
#[derive(SystemParam)]
pub struct ControlWidgets<'w, 's> {
    sliders: Query<'w, 's, SliderView>,
    children: Query<'w, 's, &'static Children>,
    thumbs: Query<'w, 's, (&'static mut Node, &'static mut BackgroundColor), With<SliderThumb>>,
    values: Query<'w, 's, (&'static mut Text, &'static mut ControlValue), ControlValuesOnly>,
    course: Query<'w, 's, &'static mut Text, (With<CourseLabel>, Without<HelmStatus>)>,
    status: Query<'w, 's, (&'static mut Text, &'static mut TextColor), With<HelmStatus>>,
    buttons: Query<'w, 's, ButtonView, Without<SliderThumb>>,
    panels: Query<'w, 's, (&'static ComputedNode, &'static UiGlobalTransform), With<ControlPanel>>,
    readouts: Query<
        'w,
        's,
        (
            &'static mut Node,
            &'static ComputedNode,
            &'static UiGlobalTransform,
        ),
        ReadoutPanelsOnly,
    >,
}

/// Projects live engine values and focus/hover states into the cockpit.
///
/// This is the only new per-frame HUD system. Value text is rewritten only when
/// its displayed precision changes; unchanged controls never allocate strings.
pub fn sync_controls(
    mut commands: Commands,
    engine: Res<Engine>,
    focus: Res<InputFocus>,
    mut ui: ControlWidgets,
) {
    for mut slider in &mut ui.sliders {
        let value = slider.control.value(&engine);
        if slider.value.0 != value {
            commands.entity(slider.entity).insert(SliderValue(value));
        }
        let focused = focus.get() == Some(slider.entity);
        let outline_color = if focused { ACCENT } else { Color::NONE };
        if slider.outline.color != outline_color {
            slider.outline.color = outline_color;
        }
        let position = percent(slider.range.thumb_position(value) * 100.0);
        let color = if slider.hovered.0 || slider.drag.dragging || focused {
            INK
        } else {
            ACCENT
        };
        for child in ui.children.iter_descendants(slider.entity) {
            if let Ok((mut node, mut background)) = ui.thumbs.get_mut(child) {
                if node.left != position {
                    node.left = position;
                }
                background.set_if_neq(BackgroundColor(color));
            }
        }
    }
    for (mut text, mut label) in &mut ui.values {
        let displayed = label.control.displayed(label.control.value(&engine));
        if label.displayed != displayed {
            label.displayed = displayed;
            text.0 = label.control.format(displayed);
        }
    }
    for mut text in &mut ui.course {
        let wanted = course_label(engine.hold_course);
        if text.0 != wanted {
            text.0.clear();
            text.0.push_str(wanted);
        }
    }
    for (mut text, mut color) in &mut ui.status {
        let wanted = helm_status(engine.hold_course);
        if text.0 != wanted {
            text.0.clear();
            text.0.push_str(wanted);
        }
        color.set_if_neq(TextColor(if engine.hold_course { ACCENT } else { MANUAL }));
    }
    for mut button in &mut ui.buttons {
        let active = matches!(button.action, CockpitButton::CourseHold) && engine.hold_course;
        let color = if button.pressed {
            Color::srgba(0.22, 0.43, 0.52, 0.95)
        } else if button.hovered.0 || active {
            Color::srgba(0.13, 0.31, 0.39, 0.92)
        } else {
            Color::srgba(0.10, 0.20, 0.30, 0.78)
        };
        button.background.set_if_neq(BackgroundColor(color));
        button
            .border
            .set_if_neq(BorderColor::all(if focus.get() == Some(button.entity) {
                ACCENT
            } else {
                BORDER
            }));
    }

    // Use laid-out bounds so wrapped controls and UI scaling are respected.
    // The built-in scroll area keeps every telemetry row reachable without
    // drawing it behind the persistent cockpit.
    if let Ok((panel, transform)) = ui.panels.single() {
        let affine: bevy::math::Affine2 = transform.into();
        let panel_top = affine.translation.y - panel.size().y * 0.5;
        for (mut node, computed, transform) in &mut ui.readouts {
            let affine: bevy::math::Affine2 = transform.into();
            let top = affine.translation.y - computed.size().y * 0.5;
            let height = px(((panel_top - top) * computed.inverse_scale_factor - 10.0).max(0.0));
            if node.max_height != height {
                node.max_height = height;
            }
        }
    }
}

fn sea_label(engine: &Engine) -> String {
    let preset = engine.preset;
    match preset.state() {
        Some((height, period)) => format!(
            "sea: {}  {height:.1} m / {period:.0} s{}",
            preset.name(),
            if preset.sails() {
                ""
            } else {
                "   (beyond the model: the boat will not sail)"
            }
        ),
        None => format!("sea: {}", preset.name()),
    }
}

/// Retains the `m` shortcut and sea/model-envelope label; widget activation is
/// handled by the same stock-button observer as the other cockpit actions.
pub fn sea_button(
    mut engine: ResMut<Engine>,
    keys: Res<ButtonInput<KeyCode>>,
    focus: Res<InputFocus>,
    sliders: Query<(), With<Slider>>,
    mut labels: Query<&mut Text, With<SeaLabel>>,
    mut shown: Local<Option<crate::sim::SeaPreset>>,
) {
    let slider_focused = focus.get().is_some_and(|entity| sliders.contains(entity));
    if !slider_focused && keys.just_pressed(KeyCode::KeyM) {
        let next = engine.preset.next();
        engine.set_sea(next);
    }
    if *shown != Some(engine.preset) {
        *shown = Some(engine.preset);
        let wanted = sea_label(&engine);
        for mut label in &mut labels {
            label.0.clone_from(&wanted);
        }
    }
}

/// A smoothed frame rate, kept here because nothing else needs it.
///
/// Displayed rather than trusted to feel: "it seems slow" is not a number, and
/// the physics step and the render frame are two different rates that can each
/// be the one at fault. The engine's own step rate is fixed and known, so what
/// this adds is the other half.
#[derive(Resource, Default)]
pub struct FrameRate {
    /// Exponentially smoothed frame time, seconds.
    smoothed: f64,
    /// Seconds since the readout was last rewritten.
    since_rewrite: f64,
}

/// How often the readout is rewritten, s.
///
/// Every frame was the first answer, and every frame the text was laid out
/// again - twenty lines of glyphs shaped and re-uploaded - for numbers that
/// change in the third decimal. Ten times a second is as fast as a reader
/// takes a number in, and the smoothed frame time reads the same.
const REWRITE: f64 = 0.1;

/// Rewrites the readout from the engine's telemetry.
pub fn update(
    engine: Res<Engine>,
    time: Res<Time>,
    mut rate: ResMut<FrameRate>,
    mut readouts: Query<&mut Text, With<ReadoutText>>,
    panels: Query<&Node, With<Readout>>,
) {
    let sim = &engine.sim;
    // Smoothed over about half a second: an unsmoothed frame time flickers too
    // fast to read, and the question being asked is about the run rather than
    // about this frame.
    let frame = f64::from(time.delta_secs());
    rate.smoothed = if rate.smoothed <= 0.0 {
        frame
    } else {
        rate.smoothed + 0.06 * (frame - rate.smoothed)
    };
    rate.since_rewrite += frame;
    if rate.since_rewrite < REWRITE {
        return;
    }
    rate.since_rewrite = 0.0;
    if !panels.iter().any(|node| node.display != Display::None) {
        return;
    }

    let state = sim.state();
    let telemetry = sim.telemetry();
    let (heel, trim, _) = state.attitude.euler_angles();
    let speed = state.world_velocity().xy().norm();

    let read = |key: &str| telemetry.get(key);
    let wind_speed = read("aero.apparent_wind.speed");
    let wind_angle = read("aero.apparent_wind.angle");

    // A key that is absent is shown as absent. A HUD that printed zero for a
    // quantity the engine had not computed would be inventing a measurement,
    // which is the one thing this project refuses everywhere else.
    let number = |value: Option<f64>, scale: f64| match value {
        Some(it) => format!("{:>8.2}", it * scale),
        None => format!("{:>8}", "--"),
    };

    let text = format!(
        "speed        {:>8.2} kn\n\
         heel         {:>8.1} deg\n\
         trim         {:>8.2} deg\n\
         sinkage      {:>8.3} m\n\
         \n\
         sail wind    {} kn  at {} deg\n\
         drive        {} N\n\
         heeling      {} N\n\
         hull drag    {} N\n\
         keel C_L     {}\n\
         \n\
         helm         {:>8.1} deg\n\
         sheet        {:>8.2}   traveller {:>5.2}\n\
         flat         {:>8.2}\n\
         \n\
         t            {:>8.1} s\n\
         frame        {:>8.1} fps  {:>5.1} ms\n\
         steps/frame  {:>8.1}",
        speed * KNOTS,
        heel.to_degrees(),
        trim.to_degrees(),
        state.position.z,
        number(wind_speed, KNOTS),
        number(wind_angle, 180.0 / std::f64::consts::PI),
        number(read("aero.driving_force"), 1.0),
        number(read("aero.heeling_force"), 1.0),
        number(read("hull.total"), 1.0),
        number(read("lateral.keel.lift_coefficient"), 1.0),
        sim.controls().rudder_angle.to_degrees(),
        sim.controls().shape.sheet,
        sim.controls().shape.traveller,
        sim.controls().trim.flat,
        sim.time(),
        1.0 / rate.smoothed.max(1e-6),
        rate.smoothed * 1e3,
        rate.smoothed / crate::sim::STEP,
    );

    for mut readout in &mut readouts {
        readout.0.clone_from(&text);
    }
}
