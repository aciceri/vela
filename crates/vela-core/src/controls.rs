//! What the crew can change while sailing.
//!
//! Only controls the force models actually consume appear here. A traveller, a
//! backstay or a jib lead would be perfectly reasonable things for a crew to
//! pull, but the transcribed coefficient model has no term for any of them, and
//! a control that silently does nothing is worse than an absent one.

use crate::aero::{EffectiveSpan, SailSet, Trim};

/// The control inputs at one instant.
///
/// # Rudder sign
///
/// `rudder_angle` follows the sign convention of [`crate::appendages`], where
/// the rudder's angle of attack is `β − ε + δ`: a **positive** rudder angle
/// adds to the angle of attack in the same sense as positive leeway, and so
/// increases the rudder's side force in the same direction the leeway already
/// pushes it. This is not a new convention invented here — it is the one the
/// appendage model is written in, and restating it differently would guarantee
/// a sign error at the boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Controls {
    /// Rudder deflection, radians. See the type documentation for the sign.
    pub rudder_angle: f64,
    /// Which sails are set.
    pub sails: SailSet,
    /// Flattening, reefing, and the aspect-ratio regime.
    pub trim: Trim,
}

impl Controls {
    /// Helm amidships, full sail, close-hauled aspect ratio.
    #[must_use]
    pub fn close_hauled(sails: SailSet) -> Self {
        Self {
            rudder_angle: 0.0,
            sails,
            trim: Trim::full(EffectiveSpan::CloseHauled),
        }
    }

    /// Helm amidships, full sail, eased aspect ratio.
    #[must_use]
    pub fn eased(sails: SailSet) -> Self {
        Self {
            rudder_angle: 0.0,
            sails,
            trim: Trim::full(EffectiveSpan::Eased),
        }
    }

    #[must_use]
    pub fn with_rudder(self, rudder_angle: f64) -> Self {
        Self {
            rudder_angle,
            ..self
        }
    }

    #[must_use]
    pub fn with_trim(self, trim: Trim) -> Self {
        Self { trim, ..self }
    }
}
