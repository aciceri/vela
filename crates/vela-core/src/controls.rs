//! What the crew can change while sailing.
//!
//! Two aerodynamic trims live here rather than one, because two force models do.
//! [`crate::aero`]'s tabular model reads `flat` and `reef`; the geometric model of
//! [`crate::sail`] reads line positions. They are not translations of each other —
//! there is no flattening factor that *means* "outhaul three quarters on" — so
//! carrying both and letting each model read its own is the only honest
//! arrangement. A boat runs whichever model its file has data for.

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
    /// Flattening, reefing, and the aspect-ratio regime — the tabular model's trim.
    pub trim: Trim,
    /// Line positions — the geometric model's trim.
    pub shape: crate::flying::Controls,
}

impl Controls {
    /// Helm amidships, full sail, close-hauled aspect ratio.
    #[must_use]
    pub fn close_hauled(sails: SailSet) -> Self {
        Self {
            rudder_angle: 0.0,
            sails,
            trim: Trim::full(EffectiveSpan::CloseHauled),
            // Everything on: a closed leech, a flat sail and the draft forward,
            // which is what close-hauled *means* on the line positions as much as
            // it means an unflattened coefficient on the tabular one.
            shape: crate::flying::Controls::HARD,
        }
    }

    /// Helm amidships, full sail, eased aspect ratio.
    #[must_use]
    pub fn eased(sails: SailSet) -> Self {
        Self {
            rudder_angle: 0.0,
            sails,
            trim: Trim::full(EffectiveSpan::Eased),
            shape: crate::flying::Controls::EASED,
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

    /// Sets the line positions the geometric model reads.
    #[must_use]
    pub fn with_shape(self, shape: crate::flying::Controls) -> Self {
        Self { shape, ..self }
    }
}
