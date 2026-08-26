//! The force modules, superposed each step by [`crate::sim::Sim`].
//!
//! # Partitioned by coupling, not by taxonomy
//!
//! A module boundary sits where forces are independent; strongly coupled
//! phenomena live inside one module. That is why keel and rudder are a single
//! [`lateral`] module rather than two: the keel's circulation sets the
//! downwash that changes the rudder's angle of attack, so solving them apart
//! produces a yaw balance that looks reasonable and is wrong.
//!
//! # Who owns which resistance
//!
//! Resistance components are split by the geometry they belong to, and the
//! split is stated here because double counting one is invisible in the total:
//!
//! - [`hull`] owns the canoe body: friction, viscous pressure, residuary, and
//!   the change of residuary with heel.
//! - [`lateral`] owns everything attached to the appendages: their induced
//!   resistance, the keel's own residuary resistance, and its change with heel.
//! - [`sails`] owns everything the air acts on, including rig windage.
//! - [`buoyancy`] owns pressure over the wetted surface, so it produces no
//!   resistance at all in still water — only the vertical force and the
//!   restoring moments.

pub mod buoyancy;
pub mod hull;
pub mod lateral;
pub mod sails;

pub use buoyancy::Buoyancy;
pub use hull::CanoeBody;
pub use lateral::LateralSystem;
pub use sails::Sails;
