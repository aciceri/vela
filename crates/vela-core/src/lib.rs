//! `vela-core` — the Vela sailing yacht physics engine.
//!
//! # Invariants
//!
//! **Renderer-free by construction.** This crate must never depend on a
//! windowing, rendering, audio or asset-pipeline library. Frontends consume it;
//! it knows nothing about them. This is what makes the engine embeddable in a
//! CLI, a test harness, or a wasm frontend interchangeably.
//!
//! **No ambient time or I/O in the simulation path.** Time advances only
//! through explicit fixed-step calls; nothing here reads a wall clock or the
//! filesystem. Determinism follows from fixed step size, fixed call order and
//! seeded randomness.
//!
//! # Conventions
//!
//! Units are SI throughout the API, angles in radians. Two frames exist, and
//! the distinction is load-bearing:
//!
//! - the **file frame** used by boat data files (x forward, y to port, z up),
//! - the **body frame** used by all dynamics (x forward, y to starboard,
//!   z down), following the marine-craft convention of Fossen, *Handbook of
//!   Marine Craft Hydrodynamics and Motion Control* (2011).
//!
//! Boat loading converts once, at the boundary; nothing downstream of the
//! loader sees file-frame quantities. See [`frames`].
//!
//! The world frame is NED (north, east, down), so gravity is `+z`.

pub mod frames;
pub mod mass;
pub mod rigid_body;
pub mod state;
pub mod wrench;

pub use mass::{MassError, MassProperties};
pub use rigid_body::{Acceleration, RigidBody};
pub use state::BodyState;
pub use wrench::Wrench;

/// Standard gravitational acceleration, m/s² (CGPM 1901 conventional value).
pub const STANDARD_GRAVITY: f64 = 9.806_65;

/// Density of standard sea water at 15 °C, kg/m³ (ITTC 7.5-02-01-03).
pub const SEA_WATER_DENSITY: f64 = 1025.9;

/// Density of air at 15 °C and 1013.25 hPa, kg/m³.
pub const AIR_DENSITY: f64 = 1.225;
