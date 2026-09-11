//! North-up navigation display. Bearings indicate where wind comes from.
//!
//! Winds are horizontal velocities at the boat reference point. Apparent wind
//! is air velocity relative to that moving point, not the heel-corrected wind
//! at the sails' centre of effort reported by the aerodynamic force model.
//! All samples come from the active simulation; this display adds no forces.

use std::fmt::Write;

use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use nalgebra::Vector3;

use crate::sim::Engine;

const KNOTS: f64 = 1.943_844_492_440_605;
const DIAL: f32 = 128.0;
const CARD_WIDTH: f32 = 188.0;
const CARD_HEIGHT: f32 = 270.0;
const INK: Color = Color::srgb(0.92, 0.95, 0.98);
const TRUE_WIND: Color = Color::srgb(0.30, 0.80, 0.89);
const APPARENT_WIND: Color = Color::srgb(1.0, 0.68, 0.36);

pub(crate) struct CompassPlugin;

impl Plugin for CompassPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn).add_systems(Update, update);
    }
}

#[derive(Component)]
struct CompassCard;

#[derive(Component, Clone, Copy)]
enum Needle {
    Heading,
    TrueWind,
    ApparentWind,
}

#[derive(Clone, Copy)]
enum Metric {
    Heading,
    BoatSpeed,
    TrueSpeed,
    ApparentSpeed,
}

#[derive(Component)]
struct Reading {
    metric: Metric,
    shown: Option<i64>,
}

fn label(value: &str, size: f32, color: Color) -> impl Bundle {
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

fn needle(kind: Needle, color: Color, reach: f32) -> impl Bundle {
    (
        Node {
            position_type: PositionType::Absolute,
            width: px(DIAL),
            height: px(DIAL),
            ..default()
        },
        UiTransform::default(),
        kind,
        Pickable::IGNORE,
        children![
            (
                Node {
                    position_type: PositionType::Absolute,
                    left: px(DIAL * 0.5 - 1.0),
                    top: px(DIAL * 0.5 - reach),
                    width: px(2),
                    height: px(reach),
                    ..default()
                },
                BackgroundColor(color),
                Pickable::IGNORE,
            ),
            (
                Node {
                    position_type: PositionType::Absolute,
                    left: px(DIAL * 0.5 - 3.0),
                    top: px(DIAL * 0.5 - reach - 3.0),
                    width: px(6),
                    height: px(6),
                    border_radius: BorderRadius::MAX,
                    ..default()
                },
                BackgroundColor(color),
                Pickable::IGNORE,
            )
        ],
    )
}

fn metric_row(name: &str, metric: Metric, color: Color) -> impl Bundle {
    (
        Node {
            width: percent(100),
            justify_content: JustifyContent::SpaceBetween,
            ..default()
        },
        Pickable::IGNORE,
        children![
            label(name, 13.0, color),
            (
                label("-- kn", 14.0, color),
                Reading {
                    metric,
                    shown: None
                },
            )
        ],
    )
}

fn spawn(mut commands: Commands) {
    commands
        .spawn((
            Name::new("Wind compass"),
            CompassCard,
            Node {
                position_type: PositionType::Absolute,
                top: px(12),
                right: px(14),
                width: px(CARD_WIDTH),
                height: px(CARD_HEIGHT),
                padding: UiRect::all(px(10)),
                border: UiRect::all(px(1)),
                border_radius: BorderRadius::all(px(10)),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                row_gap: px(6),
                ..default()
            },
            UiTransform::default(),
            BackgroundColor(Color::srgba(0.02, 0.05, 0.09, 0.84)),
            BorderColor::all(Color::srgba(0.36, 0.52, 0.61, 0.55)),
            Pickable::IGNORE,
        ))
        .with_children(|card| {
            card.spawn(label("WIND / NORTH UP", 12.0, INK));
            card.spawn((
                Node {
                    width: px(DIAL),
                    height: px(DIAL),
                    flex_shrink: 0.0,
                    ..default()
                },
                Pickable::IGNORE,
            ))
            .with_children(|dial| {
                for index in 0..24 {
                    dial.spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            width: px(DIAL),
                            height: px(DIAL),
                            ..default()
                        },
                        UiTransform {
                            rotation: Rot2::radians(index as f32 * std::f32::consts::TAU / 24.0),
                            ..default()
                        },
                        Pickable::IGNORE,
                        children![(
                            Node {
                                position_type: PositionType::Absolute,
                                left: px(DIAL * 0.5 - 0.5),
                                top: px(14),
                                width: px(1),
                                height: px(if index % 6 == 0 { 7.0 } else { 3.0 }),
                                ..default()
                            },
                            BackgroundColor(Color::srgba(0.70, 0.78, 0.83, 0.6)),
                            Pickable::IGNORE,
                        )],
                    ));
                }
                for (name, x, y) in [
                    ("N", 58.0, 0.0),
                    ("E", 116.0, 56.0),
                    ("S", 58.0, 112.0),
                    ("W", 0.0, 56.0),
                ] {
                    dial.spawn((
                        label(name, 13.0, INK),
                        Node {
                            position_type: PositionType::Absolute,
                            left: px(x),
                            top: px(y),
                            ..default()
                        },
                    ));
                }
                dial.spawn(needle(Needle::TrueWind, TRUE_WIND, 45.0));
                dial.spawn(needle(Needle::ApparentWind, APPARENT_WIND, 35.0));
                dial.spawn(needle(Needle::Heading, INK, 23.0));
            });
            card.spawn((
                label("HDG --", 14.0, INK),
                Reading {
                    metric: Metric::Heading,
                    shown: None,
                },
            ));
            card.spawn((
                Node {
                    width: percent(100),
                    flex_direction: FlexDirection::Column,
                    row_gap: px(2),
                    ..default()
                },
                Pickable::IGNORE,
                children![
                    metric_row("BOAT / SOG", Metric::BoatSpeed, INK),
                    metric_row("TRUE", Metric::TrueSpeed, TRUE_WIND),
                    metric_row("APPARENT", Metric::ApparentSpeed, APPARENT_WIND),
                ],
            ));
            card.spawn(label("Winds FROM / at boat", 10.0, INK));
        });
}

/// Horizontal bearing of a vector pointing toward a direction, clockwise from north.
fn bearing(vector: Vector3<f64>) -> Option<f32> {
    let norm = vector.xy().norm_squared();
    (norm.is_finite() && norm > 1e-12)
        .then(|| vector.y.atan2(vector.x).rem_euclid(std::f64::consts::TAU) as f32)
}

fn update(
    engine: Res<Engine>,
    windows: Query<&Window, With<PrimaryWindow>>,
    panels: Query<&ComputedNode, With<crate::hud::ControlPanel>>,
    mut cards: Query<&mut UiTransform, (With<CompassCard>, Without<Needle>)>,
    mut needles: Query<(&Needle, &mut UiTransform, &mut Visibility)>,
    mut readings: Query<(&mut Reading, &mut Text)>,
) {
    let sim = &engine.sim;
    let state = sim.state();
    let velocity = state.world_velocity();
    let wind = sim.environment().wind(state.position, sim.time());
    let apparent = wind - velocity;
    let heading = bearing(state.to_world(Vector3::x()));
    let true_bearing = bearing(-wind);
    let apparent_bearing = bearing(-apparent);
    for (kind, mut transform, mut visibility) in &mut needles {
        let angle = match kind {
            Needle::Heading => heading,
            Needle::TrueWind => true_bearing,
            Needle::ApparentWind => apparent_bearing,
        };
        *visibility = if angle.is_some() {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        if let Some(angle) = angle {
            transform.rotation = Rot2::radians(angle);
        }
    }
    for (mut reading, mut text) in &mut readings {
        let value = match reading.metric {
            Metric::Heading => {
                heading.map(|angle| (angle.to_degrees().round() as i64).rem_euclid(360))
            }
            metric => {
                let speed = match metric {
                    Metric::BoatSpeed => velocity.xy().norm(),
                    Metric::TrueSpeed => wind.xy().norm(),
                    Metric::ApparentSpeed => apparent.xy().norm(),
                    Metric::Heading => unreachable!(),
                };
                speed
                    .is_finite()
                    .then(|| (speed * KNOTS * 10.0).round() as i64)
            }
        };
        if value == reading.shown {
            continue;
        }
        reading.shown = value;
        let output = &mut **text;
        output.clear();
        match (reading.metric, value) {
            (Metric::Heading, Some(value)) => write!(output, "HDG {value:03} deg"),
            (Metric::Heading, None) => write!(output, "HDG --"),
            (_, Some(value)) => write!(output, "{}.{:01} kn", value / 10, value % 10),
            (_, None) => write!(output, "-- kn"),
        }
        .expect("formatting into a String cannot fail");
    }
    if let Ok(window) = windows.single() {
        let panel_height = panels
            .iter()
            .map(|node| node.size().y * node.inverse_scale_factor())
            .fold(0.0, f32::max);
        let scale = ((window.height() - panel_height - 50.0) / CARD_HEIGHT).clamp(0.6, 1.0);
        for mut transform in &mut cards {
            transform.scale = Vec2::splat(scale);
            transform.translation = Val2::px(
                (1.0 - scale) * CARD_WIDTH * 0.5,
                -(1.0 - scale) * CARD_HEIGHT * 0.5,
            );
        }
    }
}
