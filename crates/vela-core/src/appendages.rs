//! Keel and rudder forces: side force, induced resistance, and the keel's own
//! residuary resistance.
//!
//! # Provenance
//!
//! Transcribed from the **primary** source: Larsson, Eliasson & Orych,
//! *Principles of Yacht Design*, 5th ed. (Bloomsbury, 2022) — Fig 6.13
//! (hydrodynamic side force, after Keuning & Verwerft 2009, whose lift
//! coefficient is Whicker & Fehlner 1958), Fig 6.14 (induced resistance),
//! Fig 5.19 (residuary resistance of the keel) and Fig 5.23 (its change with
//! heel).
//!
//! As with [`crate::dsyhs`], two secondary reproductions of these tables were
//! checked and found defective, so nothing here was cross-read against a
//! third-party source and nothing here should be "corrected" against one. The
//! tests reproduce the book's own worked examples for the YD-41 instead.
//!
//! # Accuracy, in the book's own words
//!
//! The book warns explicitly that the accuracy of the Fig 6.13 side force
//! model is **too low for optimizing appendages**: it estimates the total side
//! force of the yacht and no more. Treat a keel-versus-rudder split from this
//! module as a plausible distribution, not a measurement, and do not use it to
//! choose between two candidate foils.
//!
//! # Conventions
//!
//! * Every angle in every signature is in **radians**, including sweep. Fig
//!   6.13 prints sweep in degrees, but `cos Λ` does not care which unit the
//!   figure was typeset in.
//! * A [`FoilPlanform`] is the **extended** blade: its root chord sits at the
//!   bottom of the canoe body and its span is measured from there. That is the
//!   geometry both figures want — Fig 6.13's `AR_e = 2 AR` mirroring is about
//!   the hull bottom, and Fig 6.14's `A_E`/`AR_Ee` are explicitly the extended
//!   blade's area and aspect ratio, so the two are the same planform and there
//!   is no second set of fields for it. A rudder that stops short of the hull
//!   bottom must be extended to it by the caller; the figures give no separate
//!   treatment of the gap.
//! * SI units throughout: metres, m², m³, newtons, radians.
//!
//! # Known gap
//!
//! `Z_CBk` in Fig 5.19 — the height of the keel's centre of buoyancy above the
//! bottom of the canoe body — is not tabulated in the book's Appendix 1, so it
//! is a required input on [`Keel`] rather than something this module derives.
//! [`FoilPlanform::planform_centroid_below_root`] offers the planform-centroid
//! estimate, but see the note on that method: it does **not** reproduce the
//! book's worked example, and the tests record the discrepancy rather than
//! papering over it.

use std::fmt;

/// Fraction of boat speed the rudder actually sees. Fig 6.13.
///
/// The rudder works in the wake of hull and keel, so its dynamic pressure is
/// built on `0.9 V`, not `V`. Since the dynamic pressure goes as the square,
/// this is a 19 % reduction of everything the rudder does.
///
/// Forgetting it is a *silent* error. The rudder carries roughly a tenth of the
/// total side force, so the total moves by only two or three per cent — nothing
/// looks broken — but the whole of that error lands on one end of the boat. Yaw
/// balance and therefore helm feel come out wrong while the resistance and
/// speed predictions still look entirely respectable.
pub const RUDDER_WAKE_FRACTION: f64 = 0.9;

/// Heel angles, radians, at which Fig 6.13 tabulates the downwash constant.
///
/// The figure gives 0° and 15°; 15° is exactly `π/12`, so the grid is written
/// that way rather than as a rounded decimal.
const DOWNWASH_HEEL_GRID: [f64; 2] = [0.0, std::f64::consts::PI / 12.0];

/// The downwash constant `a0` of Fig 6.13, one per row of
/// [`DOWNWASH_HEEL_GRID`].
///
/// The heel dependence is almost nothing — 0.7 % across the tabulated range —
/// but the figure gives two rows, so two rows are what is transcribed. Reading
/// it as a single constant would be a judgement call the book did not make.
const DOWNWASH_FACTORS: [f64; 2] = [0.136, 0.137];

/// Froude numbers at which Fig 5.19 tabulates the keel residuary coefficients.
const KEEL_RESIDUARY_FROUDE: [f64; 9] = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60];

/// Lowest Froude number Fig 5.19 was fitted over.
pub const KEEL_RESIDUARY_FROUDE_MIN: f64 = KEEL_RESIDUARY_FROUDE[0];

/// Highest Froude number Fig 5.19 was fitted over.
pub const KEEL_RESIDUARY_FROUDE_MAX: f64 = KEEL_RESIDUARY_FROUDE[KEEL_RESIDUARY_FROUDE.len() - 1];

/// Keel residuary resistance coefficients `A0..A3`, one row per Froude number
/// of [`KEEL_RESIDUARY_FROUDE`]. Fig 5.19.
///
/// Unlike the heel table of Fig 5.22 in [`crate::dsyhs`], these are **not**
/// scaled by 1000: they are used as printed.
#[rustfmt::skip]
const KEEL_RESIDUARY_COEFFICIENTS: [[f64; 4]; 9] = [
    [-0.00104,  0.00172,  0.00117, -0.00008],
    [-0.00550,  0.00597,  0.00390, -0.00009],
    [-0.01110,  0.01421,  0.00069,  0.00021],
    [-0.00713,  0.02632, -0.00232,  0.00039],
    [-0.03581,  0.08649,  0.00999,  0.00017],
    [-0.00470,  0.11592, -0.00064,  0.00035],
    [ 0.00553,  0.07371,  0.05991, -0.00114],
    [ 0.04822,  0.00660,  0.07048, -0.00035],
    [ 0.01021,  0.14173,  0.06409, -0.00192],
];

/// The trapezoidal planform of one appendage, extended to the bottom of the
/// canoe body.
///
/// Area and aspect ratio are derived, not declared: a planform that carries
/// both its chords and its own area invites the two to disagree, and the
/// figures define the aspect ratio from the geometry anyway.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FoilPlanform {
    /// Chord at the root, m — where the blade meets the bottom of the canoe
    /// body.
    pub root_chord: f64,
    /// Chord at the tip, m.
    pub tip_chord: f64,
    /// Span from root to tip, m. For a keel this is the draft below the canoe
    /// body, so the yacht's total draft is the canoe body draft plus this.
    pub span: f64,
    /// Quarter-chord sweep, **radians**, positive aft. Fig 6.13 prints this in
    /// degrees; the API takes radians like every other angle here.
    pub sweep: f64,
}

impl FoilPlanform {
    /// Planform area, m². `A` in Fig 6.13 and `A_E` in Fig 6.14 — the same
    /// area, since this planform is already the extended blade.
    #[must_use]
    pub fn area(&self) -> f64 {
        0.5 * (self.root_chord + self.tip_chord) * self.span
    }

    /// Mean geometric chord, m.
    #[must_use]
    pub fn mean_chord(&self) -> f64 {
        0.5 * (self.root_chord + self.tip_chord)
    }

    /// Geometric aspect ratio, `span² / area`.
    #[must_use]
    pub fn aspect_ratio(&self) -> f64 {
        self.span * self.span / self.area()
    }

    /// Effective aspect ratio `AR_e = 2 AR` of Fig 6.13, and `AR_Ee` of
    /// Fig 6.14.
    ///
    /// The factor 2 is the hull bottom mirroring the foil — the blade behaves
    /// as half of a wing twice as long — not a fudge factor. It is the reason
    /// the planform must be the blade extended to the hull bottom: mirroring a
    /// blade that stops short of the plane of symmetry is meaningless.
    #[must_use]
    pub fn effective_aspect_ratio(&self) -> f64 {
        2.0 * self.aspect_ratio()
    }

    /// Height of the planform's area centroid below the root chord, m.
    ///
    /// The trapezoid centroid, `span (c_root + 2 c_tip) / (3 (c_root + c_tip))`.
    ///
    /// **Do not reach for this as `Z_CBk`.** Fig 5.19 asks for the centroid of
    /// the keel's displaced *volume*, and a planform area centroid is a
    /// different quantity that happens to have the same units. For the YD-41
    /// this one gives 0.911 m and a keel residuary resistance of 43.7 N against
    /// the published 39 N — 12 % high — while inverting the worked example asks
    /// for 1.401 m, 74 % of the way down the keel. No trapezoid's area centroid
    /// reaches that (the range is 47 % to 53 %), but a bulb keel's volume
    /// centroid does: the YD-41 is a high-aspect fin with a torpedo bulb
    /// holding 2300 kg of lead at the tip, so nearly all of its 0.275 m³ sits
    /// at the bottom.
    ///
    /// The method is kept because the two coincide for an unbulbed fin of
    /// roughly uniform thickness, where it is a reasonable estimate. For
    /// anything with a bulb it is simply the wrong number.
    #[must_use]
    pub fn planform_centroid_below_root(&self) -> f64 {
        self.span * (self.root_chord + 2.0 * self.tip_chord)
            / (3.0 * (self.root_chord + self.tip_chord))
    }
}

/// A keel: its planform plus the two volumetric quantities Fig 5.19 needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Keel {
    /// Lateral planform, extended to the bottom of the canoe body.
    pub planform: FoilPlanform,
    /// Displaced volume of the keel, m³ (`∇_k` in Fig 5.19).
    pub volume: f64,
    /// Depth of the keel's centre of buoyancy below the bottom of the canoe
    /// body, m (`Z_CBk` in Fig 5.19).
    ///
    /// Required rather than derived: the book's Appendix 1 does not tabulate
    /// it and the planform centroid does not reproduce the worked example. See
    /// [`FoilPlanform::planform_centroid_below_root`] for the estimate and the
    /// evidence.
    pub centre_of_buoyancy_below_hull_bottom: f64,
}

/// The hull scalars the appendage formulas reach for.
///
/// Deliberately not [`crate::dsyhs::HullParameters`]: the appendage
/// regressions want the *total* draft and nothing about the waterplane or the
/// form coefficients, and a struct that carries fields its formulas ignore
/// invites a caller to believe they matter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HullScalars {
    /// Waterline length, m.
    pub waterline_length: f64,
    /// Waterline beam, m.
    pub waterline_beam: f64,
    /// Canoe body draft, m (`T_c`).
    pub canoe_draft: f64,
    /// Total draft of hull plus keel, m (`T`).
    pub total_draft: f64,
    /// Canoe body displaced volume, m³ (`∇_c`).
    pub canoe_volume: f64,
}

impl HullScalars {
    /// Froude number at a given speed through the water.
    #[must_use]
    pub fn froude(&self, speed: f64, gravity: f64) -> f64 {
        speed / (gravity * self.waterline_length).sqrt()
    }

    /// Checks the speed against the range Fig 5.19 was fitted over.
    ///
    /// This is the *only* envelope the source publishes for the appendage
    /// regressions. Unlike Fig 5.18 for the canoe body, none of Figs 5.19,
    /// 5.23, 6.13 or 6.14 states a range of validity for the geometry, so
    /// there is nothing honest to check there and this function does not
    /// pretend otherwise.
    ///
    /// # Errors
    ///
    /// [`FroudeEnvelopeError`] when the Froude number falls outside
    /// `0.20..=0.60`. Outside that range the resistance functions still return
    /// a number — tapered below, held above, so that a time-domain solver sees
    /// no step — but it is extrapolation, and a caller that cares is told.
    pub fn check_keel_residuary_envelope(
        &self,
        speed: f64,
        gravity: f64,
    ) -> Result<(), FroudeEnvelopeError> {
        let froude = self.froude(speed, gravity);
        if !(KEEL_RESIDUARY_FROUDE_MIN..=KEEL_RESIDUARY_FROUDE_MAX).contains(&froude) {
            Err(FroudeEnvelopeError {
                froude,
                low: KEEL_RESIDUARY_FROUDE_MIN,
                high: KEEL_RESIDUARY_FROUDE_MAX,
            })
        } else {
            Ok(())
        }
    }
}

/// A speed outside the Froude range Fig 5.19 was fitted over.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FroudeEnvelopeError {
    pub froude: f64,
    pub low: f64,
    pub high: f64,
}

impl fmt::Display for FroudeEnvelopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Fn = {:.3} is outside the keel residuary range {:.2}..{:.2}",
            self.froude, self.low, self.high
        )
    }
}

impl std::error::Error for FroudeEnvelopeError {}

/// The flow the appendages are working in.
///
/// Grouped into a struct only for [`appendage_forces`], which would otherwise
/// take nine positional arguments. The component functions below take their
/// scalars directly so each formula can be exercised on its own.
///
/// # Two inflows, not one
///
/// The keel and the rudder are given their own local speed and angle. In steady
/// sailing the two are the same flow — [`FlowState::uniform`] says so in one
/// call — but under a yaw rate they are not, and the difference is the whole of
/// the rudder's yaw damping. With the keel half a metre abaft the centre of
/// gravity and the rudder five metres abaft it, a yaw rate `r` gives the keel a
/// sway of about `0.5 r` and the rudder about `5 r`, opposite in sign and ten
/// times larger. A rudder fed the keel's angle contributes nothing to course
/// stability, which is exactly the thing a fin-keel yacht relies on its rudder
/// for. See `modules::lateral` for where the two inflows come from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlowState {
    /// Speed of the flow at the keel, m/s. This is boat speed in steady
    /// sailing, and reaches the keel's residuary resistance as such.
    pub speed: f64,
    /// Angle of the flow at the keel, radians (`β`): the keel's angle of
    /// attack before downwash, which the keel has none of.
    pub leeway: f64,
    /// Speed of the undisturbed flow at the rudder, m/s. The rudder sees
    /// [`RUDDER_WAKE_FRACTION`] of it.
    pub rudder_speed: f64,
    /// Angle of the flow at the rudder, radians, before the keel's downwash
    /// and the helm are applied.
    pub rudder_leeway: f64,
    /// Heel angle, radians (`φ`).
    pub heel: f64,
    /// Rudder angle, radians (`δ_r`), positive in the same sense as leeway so
    /// that it adds to the rudder's angle of attack.
    pub rudder_angle: f64,
}

impl FlowState {
    /// A flow that is the same at both foils: steady sailing, no rotation.
    ///
    /// This is the case every figure in the book is drawn for, and the one the
    /// tests exercise the component formulas in.
    #[must_use]
    pub fn uniform(speed: f64, leeway: f64, heel: f64, rudder_angle: f64) -> Self {
        Self {
            speed,
            leeway,
            rudder_speed: speed,
            rudder_leeway: leeway,
            heel,
            rudder_angle,
        }
    }
}

/// Appendage forces broken into the components the figures produce, N.
///
/// A bare total would throw away exactly the information a yaw balance needs:
/// which end of the boat each force acts at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AppendageForces {
    /// Horizontal side force of the keel, N (`F_h,k`, Fig 6.13).
    pub keel_side_force: f64,
    /// Horizontal side force of the rudder, N (`F_h,r`, Fig 6.13).
    pub rudder_side_force: f64,
    /// Induced resistance of the keel, N (`R_i,k`, Fig 6.14).
    pub keel_induced_resistance: f64,
    /// Induced resistance of the rudder, N (`R_i,r`, Fig 6.14).
    pub rudder_induced_resistance: f64,
    /// Upright residuary resistance of the keel, N (Fig 5.19).
    pub keel_residuary_resistance: f64,
    /// Change in keel residuary resistance with heel, N (Fig 5.23). Signed by
    /// the regression, so it can be negative for some hulls.
    pub keel_heel_residuary_delta: f64,
    /// Keel lift coefficient at the leeway angle.
    pub keel_lift_coefficient: f64,
    /// Rudder lift coefficient at its own angle of attack.
    pub rudder_lift_coefficient: f64,
    /// Downwash angle the keel imposes on the rudder, radians (`ε`).
    pub downwash_angle: f64,
    /// The rudder's angle of attack, radians (`α_r`), after downwash and helm.
    pub rudder_angle_of_attack: f64,
}

impl AppendageForces {
    /// Total horizontal side force, N.
    #[must_use]
    pub fn side_force(&self) -> f64 {
        self.keel_side_force + self.rudder_side_force
    }

    /// Total appendage resistance, N: induced plus residuary plus the heel
    /// correction.
    ///
    /// Appendage *friction* is not here. It is a wetted-surface calculation on
    /// the real blade, not the extended one, and it belongs with the rest of
    /// the friction bookkeeping rather than in a model of lift-dependent drag.
    #[must_use]
    pub fn resistance(&self) -> f64 {
        self.keel_induced_resistance
            + self.rudder_induced_resistance
            + self.keel_residuary_resistance
            + self.keel_heel_residuary_delta
    }
}

/// Lift curve slope of a swept foil, per radian. Fig 6.13, after Whicker &
/// Fehlner (1958).
///
/// `5.7 AR_e / (1.8 + cos Λ (AR_e²/cos⁴Λ + 4)^0.5)`.
///
/// Prandtl's lifting line would give a slope too, and more cheaply, but it has
/// no sweep term at all; a 10° rudder sweep is not negligible once the whole
/// point of the module is yaw balance.
#[must_use]
pub fn lift_slope(foil: &FoilPlanform) -> f64 {
    let effective_aspect_ratio = foil.effective_aspect_ratio();
    let cos_sweep = foil.sweep.cos();
    let cos_fourth = cos_sweep * cos_sweep * cos_sweep * cos_sweep;
    let denominator = 1.8
        + cos_sweep * (effective_aspect_ratio * effective_aspect_ratio / cos_fourth + 4.0).sqrt();
    5.7 * effective_aspect_ratio / denominator
}

/// Lift coefficient of a foil at an angle of attack in radians. Fig 6.13.
#[must_use]
pub fn lift_coefficient(foil: &FoilPlanform, angle_of_attack: f64) -> f64 {
    lift_slope(foil) * angle_of_attack
}

/// The downwash constant `a0` at a given heel angle in radians. Fig 6.13.
///
/// Linearly interpolated between the figure's two rows and clamped outside
/// them, the same treatment the coefficient tables get. Heel is taken as a
/// magnitude: the figure is written for positive heel, and downwash does not
/// care which way the boat is leaning.
#[must_use]
pub fn downwash_factor(heel: f64) -> f64 {
    let heel = heel.abs();
    if heel <= DOWNWASH_HEEL_GRID[0] {
        return DOWNWASH_FACTORS[0];
    }
    if heel >= DOWNWASH_HEEL_GRID[1] {
        return DOWNWASH_FACTORS[1];
    }
    let t = (heel - DOWNWASH_HEEL_GRID[0]) / (DOWNWASH_HEEL_GRID[1] - DOWNWASH_HEEL_GRID[0]);
    DOWNWASH_FACTORS[0] + t * (DOWNWASH_FACTORS[1] - DOWNWASH_FACTORS[0])
}

/// Downwash angle the keel imposes on the rudder, radians. Fig 6.13.
///
/// `ε = a0 (C_L,k / AR_e,k)^0.5`.
///
/// This is load bearing, not a refinement. For the YD-41 at 4° of leeway it is
/// 1.94° — roughly half the leeway angle — so a rudder modelled without it
/// works at nearly twice its real angle of attack. The side force total barely
/// moves, because the rudder is only about a tenth of it, but every yaw moment
/// and therefore the whole of the helm feel is wrong.
///
/// The figure is written for positive leeway. The square root is taken on the
/// magnitude and the sign of the leeway is restored afterwards, so that
/// downwash always *reduces* the rudder's angle of attack whichever tack the
/// boat is on. Without that the model would be silently one-tack-only.
#[must_use]
pub fn keel_downwash_angle(keel: &FoilPlanform, leeway: f64, heel: f64) -> f64 {
    let coefficient = lift_coefficient(keel, leeway);
    let magnitude =
        downwash_factor(heel) * (coefficient.abs() / keel.effective_aspect_ratio()).sqrt();
    magnitude * leeway.signum()
}

/// The rudder's angle of attack, radians. Fig 6.13.
///
/// `α_r = β − ε + δ_r`: the leeway it shares with the keel, less the keel's
/// downwash, plus whatever the helmsman is asking for.
#[must_use]
pub fn rudder_angle_of_attack(
    keel: &FoilPlanform,
    leeway: f64,
    heel: f64,
    rudder_angle: f64,
) -> f64 {
    leeway - keel_downwash_angle(keel, leeway, heel) + rudder_angle
}

/// The hull factor `c_hull = 1.8 (T_c / T_k) + 1` of Fig 6.13.
///
/// `T_k` is the keel's own span, measured down from the bottom of the canoe
/// body, so the yacht's total draft is `T_c + T_k`. The factor is the hull's
/// share of the side force, carried implicitly: a deep canoe body on a short
/// keel lifts more than the keel alone accounts for.
#[must_use]
pub fn hull_side_force_factor(canoe_draft: f64, keel_span: f64) -> f64 {
    1.8 * (canoe_draft / keel_span) + 1.0
}

/// The heel factor `c_heel = 1 − 0.382 φ` of Fig 6.13, with `φ` in radians.
///
/// Heel is taken as a magnitude, since heeling to port must cost the same as
/// to starboard and the figure is written for positive `φ`. The result is
/// floored at zero: the linear fit crosses zero near 150° of heel and would
/// then *reverse* the side force, which is not a physical statement the figure
/// is making. Anything near that floor is a capsize, not a sailing condition.
#[must_use]
pub fn heel_side_force_factor(heel: f64) -> f64 {
    (1.0 - 0.382 * heel.abs()).max(0.0)
}

/// Horizontal side force of one foil, N. Fig 6.13.
///
/// `F_h = C_L * 0.5 ρ V_local² * A * c_hull * c_heel`, where `V_local` is the
/// speed that foil actually sees — boat speed for the keel,
/// [`RUDDER_WAKE_FRACTION`] of it for the rudder.
#[must_use]
pub fn side_force(
    foil: &FoilPlanform,
    angle_of_attack: f64,
    local_speed: f64,
    canoe_draft: f64,
    keel_span: f64,
    heel: f64,
    density: f64,
) -> f64 {
    let lift = lift_coefficient(foil, angle_of_attack)
        * dynamic_pressure(local_speed, density)
        * foil.area();
    lift * hull_side_force_factor(canoe_draft, keel_span) * heel_side_force_factor(heel)
}

/// Induced resistance of one foil, N. Fig 6.14.
///
/// `R_i = C_Dφ * 0.5 ρ V_local² * A_E` with `C_Dφ = C_Lφ² / (π AR_Ee)`, where
/// `C_Lφ` is built from the side force **in the heeled plane** — the caller's
/// horizontal `F_h` divided by `cos φ`. The lift a foil actually generates is
/// perpendicular to its own span, and it is that force, not its horizontal
/// component, that trails the vortices.
///
/// Squaring the force means an error in side force costs twice as much here,
/// which is the main reason the side force model's own accuracy caveat
/// propagates into the resistance.
#[must_use]
pub fn induced_resistance(
    foil: &FoilPlanform,
    horizontal_side_force: f64,
    local_speed: f64,
    heel: f64,
    density: f64,
) -> f64 {
    let reference = dynamic_pressure(local_speed, density) * foil.area();
    if reference <= 0.0 {
        return 0.0;
    }
    let heeled_force = horizontal_side_force / heel.cos();
    let coefficient = heeled_force / reference;
    coefficient * coefficient / (std::f64::consts::PI * foil.effective_aspect_ratio()) * reference
}

/// The keel residuary resistance coefficient `R_Rk / (∇_k ρ g)` of Fig 5.19.
///
/// The raw regression, with the coefficient rows interpolated linearly in
/// Froude number and clamped at both ends of the table. No taper is applied
/// here — this is the coefficient the book prints, so it is what a test can
/// compare against; [`keel_residuary_resistance`] is where the low-speed taper
/// lives.
#[must_use]
pub fn keel_residuary_coefficient(hull: &HullScalars, keel: &Keel, froude: f64) -> f64 {
    let a = interpolate(&KEEL_RESIDUARY_FROUDE, &KEEL_RESIDUARY_COEFFICIENTS, froude);
    let keel_cube_root = keel.volume.cbrt();

    a[0] + a[1] * hull.total_draft / hull.waterline_beam
        + a[2] * (hull.canoe_draft + keel.centre_of_buoyancy_below_hull_bottom) / keel_cube_root
        + a[3] * hull.canoe_volume / keel.volume
}

/// Upright residuary resistance of the keel, N. Fig 5.19, valid `Fn = 0.20`
/// to `0.60`.
///
/// Both ends of the tabulated range are handled the way [`crate::dsyhs`]
/// handles Fig 5.18's, and for the same reason: the published table simply
/// stops, but a force that appears out of nothing at a threshold speed is a
/// step discontinuity, and a step in a force jolts a time-domain solver as the
/// boat accelerates through it. Below `Fn = 0.20` the result tapers linearly
/// to zero at rest, which is also the physically right answer — a keel at rest
/// makes no waves. Above `Fn = 0.60` the last row is held, since extrapolating
/// the regression is worse than admitting the table has ended.
#[must_use]
pub fn keel_residuary_resistance(
    hull: &HullScalars,
    keel: &Keel,
    speed: f64,
    density: f64,
    gravity: f64,
) -> f64 {
    let froude = hull.froude(speed, gravity);
    let taper = low_speed_taper(froude, KEEL_RESIDUARY_FROUDE_MIN);
    if taper <= 0.0 {
        return 0.0;
    }
    taper * keel_residuary_coefficient(hull, keel, froude) * keel.volume * density * gravity
}

/// The heel coefficient `C_H` of Fig 5.23.
///
/// Purely geometric — no Froude number enters — which is why
/// [`keel_heel_residuary_delta`] needs no coefficient table.
#[must_use]
pub fn keel_heel_residuary_coefficient(hull: &HullScalars) -> f64 {
    let draft_ratio = hull.canoe_draft / hull.total_draft;
    let beam_draft = hull.waterline_beam / hull.canoe_draft;

    -3.5837 * draft_ratio - 0.0518 * beam_draft
        + 0.5958 * draft_ratio * beam_draft
        + 0.2055 * hull.waterline_length / hull.canoe_volume.cbrt()
}

/// Change in keel residuary resistance with heel, N. Fig 5.23.
///
/// `ΔR_RKφ = ∇_k ρ g * C_H * Fn² * φ`, with `φ` in radians and taken as a
/// magnitude — heeling to port must cost what heeling to starboard costs.
///
/// Unlike [`keel_residuary_resistance`] this needs no edge treatment at either
/// end of the Froude range, and adding one would be a mistake rather than a
/// courtesy. `C_H` is speed-independent, so there is no coefficient row to
/// interpolate or to hold above `Fn = 0.60`; and the `Fn²` factor already goes
/// smoothly to zero at rest, so there is no step at `Fn = 0.20` to taper away.
/// Clamping `Fn` at the top of Fig 5.19's range would flatten a force that the
/// formula says keeps growing, which is a fabricated discontinuity in the
/// derivative in exchange for nothing.
#[must_use]
pub fn keel_heel_residuary_delta(
    hull: &HullScalars,
    keel: &Keel,
    speed: f64,
    heel: f64,
    density: f64,
    gravity: f64,
) -> f64 {
    let froude = hull.froude(speed, gravity);
    keel.volume
        * density
        * gravity
        * keel_heel_residuary_coefficient(hull)
        * froude
        * froude
        * heel.abs()
}

/// Every appendage force at one operating point.
///
/// The order matters: the keel's lift sets the downwash, the downwash sets the
/// rudder's angle of attack, and both side forces then set the induced
/// resistances. Computing the rudder before the keel is the classic way to get
/// a plausible-looking yaw balance that is wrong.
#[must_use]
pub fn appendage_forces(
    hull: &HullScalars,
    keel: &Keel,
    rudder: &FoilPlanform,
    flow: &FlowState,
    density: f64,
    gravity: f64,
) -> AppendageForces {
    // The rudder sits in the wake of hull and keel; every dynamic pressure it
    // sees is built on this speed, not on boat speed. The speed is the flow at
    // the rudder's own position, which differs from the keel's under a rate.
    let rudder_speed = RUDDER_WAKE_FRACTION * flow.rudder_speed;

    // The downwash is the keel's, set by the keel's own angle of attack, and
    // it is applied to the flow the rudder is actually in.
    let downwash_angle = keel_downwash_angle(&keel.planform, flow.leeway, flow.heel);
    let rudder_alpha = flow.rudder_leeway - downwash_angle + flow.rudder_angle;

    let keel_side_force = side_force(
        &keel.planform,
        flow.leeway,
        flow.speed,
        hull.canoe_draft,
        keel.planform.span,
        flow.heel,
        density,
    );
    let rudder_side_force = side_force(
        rudder,
        rudder_alpha,
        rudder_speed,
        hull.canoe_draft,
        keel.planform.span,
        flow.heel,
        density,
    );

    AppendageForces {
        keel_side_force,
        rudder_side_force,
        keel_induced_resistance: induced_resistance(
            &keel.planform,
            keel_side_force,
            flow.speed,
            flow.heel,
            density,
        ),
        rudder_induced_resistance: induced_resistance(
            rudder,
            rudder_side_force,
            rudder_speed,
            flow.heel,
            density,
        ),
        keel_residuary_resistance: keel_residuary_resistance(
            hull, keel, flow.speed, density, gravity,
        ),
        keel_heel_residuary_delta: keel_heel_residuary_delta(
            hull, keel, flow.speed, flow.heel, density, gravity,
        ),
        keel_lift_coefficient: lift_coefficient(&keel.planform, flow.leeway),
        rudder_lift_coefficient: lift_coefficient(rudder, rudder_alpha),
        downwash_angle,
        rudder_angle_of_attack: rudder_alpha,
    }
}

/// Dynamic pressure `0.5 ρ V²`, Pa.
#[must_use]
pub fn dynamic_pressure(speed: f64, density: f64) -> f64 {
    0.5 * density * speed * speed
}

/// Fraction of a tabulated regression to apply below its first Froude number.
///
/// Deliberately the same rule as the private helper of the same name in
/// [`crate::dsyhs`], because Fig 5.19's table stops for the same reason
/// Fig 5.18's does. Duplicated rather than shared so neither module owns the
/// other's edge behaviour; if a third table ever needs it, it should move to a
/// shared helper.
fn low_speed_taper(froude: f64, first_tabulated: f64) -> f64 {
    if froude <= 0.0 {
        0.0
    } else if froude >= first_tabulated {
        1.0
    } else {
        froude / first_tabulated
    }
}

/// Linear interpolation between tabulated coefficient rows, clamped at both
/// ends.
fn interpolate<const N: usize, const M: usize>(
    grid: &[f64; M],
    rows: &[[f64; N]; M],
    at: f64,
) -> [f64; N] {
    if at <= grid[0] {
        return rows[0];
    }
    if at >= grid[M - 1] {
        return rows[M - 1];
    }
    let upper = grid.iter().position(|g| *g >= at).unwrap_or(M - 1);
    let lower = upper - 1;
    let t = (at - grid[lower]) / (grid[upper] - grid[lower]);

    let mut out = [0.0; N];
    for i in 0..N {
        out[i] = rows[lower][i] + t * (rows[upper][i] - rows[lower][i]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn keel_planform() -> FoilPlanform {
        FoilPlanform {
            root_chord: 1.00,
            tip_chord: 0.78,
            span: 1.90,
            sweep: 5.5_f64.to_radians(),
        }
    }

    #[test]
    fn planform_geometry_is_derived_from_the_chords() {
        let keel = keel_planform();
        // Trapezoid: 0.5 (1.00 + 0.78) 1.90 = 1.691 m².
        assert_relative_eq!(keel.area(), 1.691, epsilon = 1e-12);
        // span² / area = 3.61 / 1.691.
        assert_relative_eq!(keel.aspect_ratio(), 3.61 / 1.691, epsilon = 1e-12);
        assert_relative_eq!(
            keel.effective_aspect_ratio(),
            2.0 * keel.aspect_ratio(),
            epsilon = 1e-12
        );
        // span / mean chord is the same aspect ratio by another route.
        assert_relative_eq!(
            keel.aspect_ratio(),
            keel.span / keel.mean_chord(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn the_trapezoid_centroid_lies_between_mid_span_and_the_root() {
        let keel = keel_planform();
        // Tapered towards the tip, so the centroid sits above mid-span.
        let centroid = keel.planform_centroid_below_root();
        assert!(
            centroid < 0.5 * keel.span,
            "centroid {centroid} below mid-span"
        );
        assert!(centroid > 0.0);
        // A rectangular planform must give exactly mid-span.
        let rectangle = FoilPlanform {
            tip_chord: 1.00,
            ..keel
        };
        assert_relative_eq!(
            rectangle.planform_centroid_below_root(),
            0.5 * rectangle.span,
            epsilon = 1e-12
        );
    }

    #[test]
    fn the_downwash_constant_interpolates_and_clamps() {
        assert_relative_eq!(downwash_factor(0.0), 0.136, epsilon = 1e-12);
        assert_relative_eq!(
            downwash_factor(15.0_f64.to_radians()),
            0.137,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            downwash_factor(7.5_f64.to_radians()),
            0.1365,
            epsilon = 1e-12
        );
        // Held outside the two tabulated rows, in both directions.
        assert_relative_eq!(
            downwash_factor(40.0_f64.to_radians()),
            0.137,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            downwash_factor(-20.0_f64.to_radians()),
            0.137,
            epsilon = 1e-12
        );
    }

    #[test]
    fn coefficient_interpolation_hits_the_grid_and_clamps() {
        let a = interpolate(&KEEL_RESIDUARY_FROUDE, &KEEL_RESIDUARY_COEFFICIENTS, 0.35);
        assert_relative_eq!(a[0], -0.00713);
        assert_relative_eq!(a[3], 0.00039);

        let midpoint = interpolate(&KEEL_RESIDUARY_FROUDE, &KEEL_RESIDUARY_COEFFICIENTS, 0.325);
        assert_relative_eq!(midpoint[0], 0.5 * (-0.01110 + -0.00713), epsilon = 1e-12);

        let low = interpolate(&KEEL_RESIDUARY_FROUDE, &KEEL_RESIDUARY_COEFFICIENTS, 0.01);
        assert_relative_eq!(low[0], -0.00104);
        let high = interpolate(&KEEL_RESIDUARY_FROUDE, &KEEL_RESIDUARY_COEFFICIENTS, 9.0);
        assert_relative_eq!(high[0], 0.01021);
    }

    #[test]
    fn the_heel_factor_never_reverses_the_side_force() {
        assert_relative_eq!(heel_side_force_factor(0.0), 1.0, epsilon = 1e-12);
        // Symmetric in the sign of the heel angle.
        let twenty = 20.0_f64.to_radians();
        assert_relative_eq!(
            heel_side_force_factor(-twenty),
            heel_side_force_factor(twenty),
            epsilon = 1e-12
        );
        // 1 - 0.382 φ crosses zero at φ = 2.618 rad; floored, never negative.
        assert_relative_eq!(heel_side_force_factor(170.0_f64.to_radians()), 0.0);
    }
}
