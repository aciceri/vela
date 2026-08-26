//! Sail and rig aerodynamics after Hazen (1980).
//!
//! # Provenance
//!
//! Everything here — every area formula, every coefficient, the induced-drag
//! form and the two trim factors — is transcribed from Larsson, Eliasson &
//! Orych, *Principles of Yacht Design*, 5th ed. (Bloomsbury, 2022), **Fig 8.19**
//! (areas and centres of effort) and **Table 8.1** (lift and viscous drag
//! coefficients). Those figures present the model of Hazen, G. (1980), "A model
//! of sail aerodynamics for diverse rig types", which is the basis of the
//! IMS/ORC aerodynamic model.
//!
//! No term has been added to the model and no constant has been adjusted to
//! make a test pass. Where the transcription is silent — the combination rule
//! for the centres of effort, the apparent wind angle at which the close-hauled
//! aspect ratio gives way to the eased one, the geometry of a mizzen staysail —
//! the gap is stated in the doc comment of the item that would need it, and,
//! where a caller could otherwise be misled, in the API itself.
//!
//! # Scope
//!
//! Sails, mast and topsides windage: everything the air acts on. The hull below
//! the waterline is [`crate::dsyhs`]'s business.
//!
//! This is a coefficient model, not a flow solver: the sails do not interact,
//! there is no wind gradient or twist correction (the IMS model has one; the
//! transcription does not carry it), and heel enters only through the caller's
//! apparent wind — the coefficients themselves are not heel-corrected.
//!
//! # Conventions
//!
//! SI units and radians throughout. The one exception is Table 8.1, which is
//! published against apparent wind angle in **degrees**; the conversion happens
//! inside [`lift_coefficient`] and [`viscous_drag_coefficient`], and degrees
//! appear nowhere in a signature.
//!
//! Sign and frame conventions, all of which are stated because a mistake in any
//! of them produces plausible-looking nonsense:
//!
//! - The apparent wind angle is measured from the bow, so 0 is dead ahead and
//!   π (180°) is dead astern. Its **sign is ignored**: the model has no
//!   port/starboard asymmetry, so an angle and its negation give the same
//!   forces. Angles outside `[-π, π]` are folded, so `190°` is read as `170°`.
//! - **Lift** is normal to the apparent wind, **drag** parallel to it, both in
//!   the horizontal plane. Resolving them onto the yacht's centreline gives the
//!   two components a caller wants, with `β` the apparent wind angle:
//!   `driving = L·sin β − D·cos β` and `heeling = L·cos β + D·sin β`.
//! - [`SailForces::driving_force`] is positive **forward**, along the
//!   centreline. It goes negative when the yacht is pinched up so far that
//!   `D·cos β` beats `L·sin β`, which is correct and worth not clamping.
//! - [`SailForces::heeling_force`] is positive **to leeward**, athwartships.
//!   Which body-frame axis that is depends on the tack, and the caller knows
//!   the tack from the sign of the apparent wind angle it passed in: leeward is
//!   `−sign(angle)` in a body frame whose `y` is to starboard. Deliberately not
//!   decided here — this module never sees a tack.
//! - [`SailForces::heeling_moment_about_waterline`] is the heeling force times
//!   the height of the centre of effort **above the water**, positive heeling
//!   to leeward. It is a moment about the waterline, not the full heeling
//!   moment of the yacht: the lateral force on keel and rudder acts below the
//!   waterline and adds its own arm, which belongs to the lateral-plane model,
//!   not here.

use std::f64::consts::{PI, TAU};
use std::fmt;

/// Drag coefficient assumed for mast and topsides frontal area. Fig 8.19.
const WINDAGE_DRAG_COEFFICIENT: f64 = 1.13;

/// Hazen's addition to the induced drag factor.
///
/// Not a fudge factor, and not part of the classical `C_L²/(π·AR)`: some of the
/// viscous drag from separation on the leeward side of a sail also grows with
/// the square of the lift, so Hazen carries it along with the induced term
/// rather than in `C_DP`, which is a function of apparent wind angle alone.
const HAZEN_SEPARATION_ADDITION: f64 = 0.005;

/// Mirror factor on the effective span of the sail plan.
///
/// The water surface acts as a reflecting plane, so the sail plan behaves like
/// a wing of 110 % of its height above the water. Close-hauled the jib closes
/// the gap between foot and deck and the whole height above the water counts;
/// eased, the gap opens and only the mast above the sheer does. See
/// [`EffectiveSpan`].
const SURFACE_MIRROR_FACTOR: f64 = 1.1;

/// Apparent wind angles, in **degrees**, at which Table 8.1 is tabulated.
///
/// Degrees here and only here: the published table is indexed by degrees, and
/// converting the table would put five hand-typed radian constants between the
/// code and the page it was read from.
const TABULATED_ANGLES: [f64; ANGLE_COUNT] = [27.0, 50.0, 80.0, 100.0, 180.0];

const ANGLE_COUNT: usize = 5;
const SAIL_COUNT: usize = 5;

/// One row per angle of [`TABULATED_ANGLES`], one column per sail in the order
/// of [`Sail::ALL`], which is the column order of the published table.
type CoefficientTable = [[f64; SAIL_COUNT]; ANGLE_COUNT];

/// Table 8.1(a) — lift coefficients.
#[rustfmt::skip]
const LIFT: CoefficientTable = [
    //  main   jib    spinnaker  mizzen  mizzen staysail
    [   1.5,   1.5,   0.0,       1.3,    0.0  ], //  27°
    [   1.5,   0.5,   1.5,       1.4,    0.75 ], //  50°
    [   0.95,  0.3,   1.0,       1.0,    1.0  ], //  80°
    [   0.85,  0.0,   0.85,      0.8,    0.8  ], // 100°
    [   0.0,   0.0,   0.0,       0.0,    0.0  ], // 180°
];

/// Table 8.1(b) — viscous drag coefficients.
#[rustfmt::skip]
const VISCOUS_DRAG: CoefficientTable = [
    //  main   jib    spinnaker  mizzen  mizzen staysail
    [   0.02,  0.02,  0.0,       0.02,   0.0  ], //  27°
    [   0.15,  0.25,  0.25,      0.15,   0.1  ], //  50°
    [   0.8,   0.15,  0.9,       0.75,   0.75 ], //  80°
    [   1.0,   0.0,   1.2,       1.0,    1.0  ], // 100°
    [   0.9,   0.0,   0.66,      0.8,    0.0  ], // 180°
];

/// A sail the model has coefficients for.
///
/// The discriminants are the column order of Table 8.1 and are used to index
/// it, so they are not free to reorder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sail {
    Main = 0,
    Jib = 1,
    Spinnaker = 2,
    Mizzen = 3,
    MizzenStaysail = 4,
}

impl Sail {
    /// Every sail in Table 8.1, in the table's own column order.
    pub const ALL: [Sail; SAIL_COUNT] = [
        Sail::Main,
        Sail::Jib,
        Sail::Spinnaker,
        Sail::Mizzen,
        Sail::MizzenStaysail,
    ];

    const fn column(self) -> usize {
        self as usize
    }

    const fn bit(self) -> u8 {
        1 << (self as u8)
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Sail::Main => "main",
            Sail::Jib => "jib",
            Sail::Spinnaker => "spinnaker",
            Sail::Mizzen => "mizzen",
            Sail::MizzenStaysail => "mizzen staysail",
        }
    }
}

impl fmt::Display for Sail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Which sails are set.
///
/// Modelled explicitly rather than inferred from the apparent wind angle,
/// because it is a *choice*: a crew that carries a jib to 140° or a spinnaker
/// to 60° is making a bad choice, not an impossible one, and a velocity
/// prediction is precisely the tool for showing which choice is bad where. The
/// area-weighted coefficient sums run over the set sails only, so a sail that
/// is not set contributes nothing — while still counting towards the nominal
/// area if it is part of the rig's reference triangles (see
/// [`SailPlan::nominal_area`]).
///
/// The empty set is legal and means bare poles: no lift, no viscous drag, and
/// the mast-and-topsides windage of [`SailPlan::parasitic_drag_coefficient`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SailSet(u8);

impl SailSet {
    /// Bare poles.
    #[must_use]
    pub const fn none() -> Self {
        SailSet(0)
    }

    /// Main and jib: the upwind set.
    #[must_use]
    pub const fn upwind() -> Self {
        SailSet::none().with(Sail::Main).with(Sail::Jib)
    }

    /// Main and spinnaker: the downwind set.
    #[must_use]
    pub const fn downwind() -> Self {
        SailSet::none().with(Sail::Main).with(Sail::Spinnaker)
    }

    #[must_use]
    pub const fn with(self, sail: Sail) -> Self {
        SailSet(self.0 | sail.bit())
    }

    #[must_use]
    pub const fn without(self, sail: Sail) -> Self {
        SailSet(self.0 & !sail.bit())
    }

    #[must_use]
    pub const fn contains(self, sail: Sail) -> bool {
        self.0 & sail.bit() != 0
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The set sails, in Table 8.1 column order.
    pub fn iter(self) -> impl Iterator<Item = Sail> {
        Sail::ALL
            .into_iter()
            .filter(move |sail| self.contains(*sail))
    }
}

impl fmt::Display for SailSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return f.write_str("bare poles");
        }
        for (index, sail) in self.iter().enumerate() {
            if index > 0 {
                f.write_str(" + ")?;
            }
            f.write_str(sail.name())?;
        }
        Ok(())
    }
}

/// Which effective span the induced drag is computed on.
///
/// Fig 8.19 gives two aspect ratios and the physical reason for the pair: the
/// span that matters is the height over which the sail plan carries load, and
/// close-hauled the jib reaches the deck so the load runs all the way down to
/// the water, while eased there is a gap under the foot and only the mast above
/// the sheer is loaded.
///
/// The transcription does **not** give the apparent wind angle at which one
/// definition gives way to the other, so this module will not invent a
/// threshold: the caller states which regime it is in. That is honest rather
/// than convenient — a hidden threshold would put a step in the drag curve at
/// an angle nobody chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectiveSpan {
    /// Sheeted hard: span is `1.1 × (EHM + FA)`, masthead height above the water.
    CloseHauled,
    /// Eased: span is `1.1 × EHM`, masthead height above the sheer.
    Eased,
}

/// The extra dimensions of a mizzen, for a ketch or a yawl.
///
/// A sloop has none, and `None` is the common case. These mirror the mainsail's
/// `P`, `E` and `BAD` exactly, which is why supporting them costs nothing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MizzenDimensions {
    /// `PY`, mizzen hoist, m.
    pub mizzen_hoist: f64,
    /// `EY`, foot of the mizzen, m.
    pub mizzen_foot: f64,
    /// `BADY`, height of the mizzen boom above the sheer, m.
    pub mizzen_boom_above_sheer: f64,
}

/// Rig dimensions in the IOR notation of Fig 8.19, all in metres.
///
/// This is the module's own input struct: the sail plan is described here by
/// eleven lengths and nothing else, independently of how a boat file happens to
/// spell them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigDimensions {
    /// `P`, mainsail hoist.
    pub main_hoist: f64,
    /// `E`, foot of the mainsail.
    pub main_foot: f64,
    /// `I`, height of the foretriangle.
    pub foretriangle_height: f64,
    /// `J`, base of the foretriangle.
    pub foretriangle_base: f64,
    /// `LPG`, perpendicular of the longest jib.
    pub jib_perpendicular: f64,
    /// `SL`, spinnaker leech length.
    pub spinnaker_leech: f64,
    /// `BAD`, height of the main boom above the sheer.
    pub boom_above_sheer: f64,
    /// `BMAX`, maximum beam of the yacht.
    pub max_beam: f64,
    /// `FA`, average freeboard.
    pub average_freeboard: f64,
    /// `EHM`, mast height above the sheer.
    pub mast_height_above_sheer: f64,
    /// `EMDC`, average mast diameter.
    pub mast_diameter: f64,
    /// Mizzen dimensions, `None` for a sloop.
    pub mizzen: Option<MizzenDimensions>,
}

impl RigDimensions {
    /// Area of the foretriangle, `A_F = 0.5·I·J`, m².
    ///
    /// A reference triangle, not a sail: it is part of the nominal area whether
    /// or not a headsail is set.
    #[must_use]
    pub fn foretriangle_area(&self) -> f64 {
        0.5 * self.foretriangle_height * self.foretriangle_base
    }

    /// Area of one sail, m², or `None` where the rig does not define it.
    ///
    /// `None` has two meanings, and they are different gaps:
    ///
    /// - [`Sail::Mizzen`] on a rig with no [`MizzenDimensions`] — the yacht
    ///   simply has no mizzen.
    /// - [`Sail::MizzenStaysail`], always. Fig 8.19 gives its area as
    ///   `0.5·YSD·(YSMG + YSF)`, but the transcription this module was built
    ///   from carries those three symbols without their definitions, and
    ///   guessing what a "mid girth" is measured between would be inventing
    ///   geometry. Its Table 8.1 coefficient columns *are* transcribed and are
    ///   returned by [`lift_coefficient`] and [`viscous_drag_coefficient`], so
    ///   the only thing missing is the area, and [`SailPlan::new`] refuses to
    ///   set the sail rather than quietly giving it zero area.
    #[must_use]
    pub fn area(&self, sail: Sail) -> Option<f64> {
        match sail {
            Sail::Main => Some(0.5 * self.main_hoist * self.main_foot),
            Sail::Jib => {
                let luff = (self.foretriangle_height * self.foretriangle_height
                    + self.foretriangle_base * self.foretriangle_base)
                    .sqrt();
                Some(0.5 * luff * self.jib_perpendicular)
            }
            // 1.15 is the book's allowance for the spinnaker's curved leeches
            // and mid-girth: a spinnaker is fuller than the flat triangle
            // SL × J its two dimensions describe.
            Sail::Spinnaker => Some(1.15 * self.spinnaker_leech * self.foretriangle_base),
            Sail::Mizzen => self
                .mizzen
                .map(|mizzen| 0.5 * mizzen.mizzen_hoist * mizzen.mizzen_foot),
            Sail::MizzenStaysail => None,
        }
    }

    /// Height of a sail's centre of effort above the sheer, m, or `None` where
    /// the rig does not define it.
    ///
    /// The fractions are Fig 8.19's: `0.39·P + BAD` for the main, `0.39·I` for
    /// the jib, `0.59·I` for the spinnaker — a spinnaker's centre is much
    /// higher because its area is up in the head, not down at the foot.
    ///
    /// The mizzen staysail has a centre of effort here (the figure gives it the
    /// same expression as the mizzen) even though it has no [`Self::area`].
    #[must_use]
    pub fn centre_of_effort_height(&self, sail: Sail) -> Option<f64> {
        let mizzen_centre = || {
            self.mizzen
                .map(|mizzen| 0.39 * mizzen.mizzen_hoist + mizzen.mizzen_boom_above_sheer)
        };
        match sail {
            Sail::Main => Some(0.39 * self.main_hoist + self.boom_above_sheer),
            Sail::Jib => Some(0.39 * self.foretriangle_height),
            Sail::Spinnaker => Some(0.59 * self.foretriangle_height),
            Sail::Mizzen | Sail::MizzenStaysail => mizzen_centre(),
        }
    }

    /// Nominal area `A_N = A_F + A_M + A_Y`, m².
    ///
    /// The reference area of **every** coefficient in this model. It is a
    /// property of the rig, not of what is currently hoisted: the foretriangle
    /// counts even under spinnaker, and the mainsail counts when reefed away.
    /// So `A_S/A_N` can exceed one, which is not a bug — a coefficient
    /// referenced to a fixed area is exactly how the model keeps forces
    /// comparable across sail sets.
    ///
    /// Prefer [`SailPlan`], which computes this once and then divides by
    /// nothing else.
    #[must_use]
    pub fn nominal_area(&self) -> f64 {
        let mizzen = self.area(Sail::Mizzen).unwrap_or(0.0);
        self.foretriangle_area() + self.area(Sail::Main).unwrap_or(0.0) + mizzen
    }
}

/// A rig and sail set that cannot be turned into a sail plan.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SailPlanError {
    /// The set names a sail the rig has no dimensions for.
    SailNotCarried(Sail),
    /// The set names a sail whose area this model cannot compute. See
    /// [`RigDimensions::area`].
    AreaUnavailable(Sail),
    /// A dimension the sail plan needs is out of range.
    Dimension {
        name: &'static str,
        value: f64,
        minimum: f64,
    },
}

impl fmt::Display for SailPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SailPlanError::SailNotCarried(sail) => {
                write!(f, "the rig carries no {sail}, so it cannot be set")
            }
            SailPlanError::AreaUnavailable(sail) => write!(
                f,
                "the area of a {sail} is not part of this model, so it cannot be set"
            ),
            SailPlanError::Dimension {
                name,
                value,
                minimum,
            } => write!(f, "rig dimension {name} = {value}, expected > {minimum}"),
        }
    }
}

impl std::error::Error for SailPlanError {}

/// The apparent wind at the sail plan.
///
/// A named struct because `(speed, angle)` as two bare `f64` arguments is the
/// classic silently-swapped pair, and because there is exactly one apparent
/// wind for the whole plan: this model has no wind gradient and no twist
/// correction, so a caller that wants one applies it before calling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ApparentWind {
    /// Apparent wind speed, m/s.
    pub speed: f64,
    /// Apparent wind angle from the bow, radians. The sign is ignored; see the
    /// module documentation.
    pub angle: f64,
}

/// The two trim controls of the model, plus the aspect-ratio regime.
///
/// `flat` and `reef` are independent and physically asymmetric, which is the
/// interesting part of the model — see [`SailPlan::forces`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Trim {
    /// Flattening factor `F`; 1.0 is a normally trimmed sail, less is flatter.
    pub flat: f64,
    /// Reef factor `R`; 1.0 is full sail, less is reefed.
    pub reef: f64,
    /// Which effective span the induced drag uses.
    pub effective_span: EffectiveSpan,
}

impl Trim {
    /// Full sail, unflattened.
    #[must_use]
    pub const fn full(effective_span: EffectiveSpan) -> Self {
        Trim {
            flat: 1.0,
            reef: 1.0,
            effective_span,
        }
    }

    #[must_use]
    pub const fn with_flat(self, flat: f64) -> Self {
        Trim { flat, ..self }
    }

    #[must_use]
    pub const fn with_reef(self, reef: f64) -> Self {
        Trim { reef, ..self }
    }
}

/// The aerodynamic force on a sail plan, broken down.
///
/// The breakdown is the point: a caller debugging a velocity prediction needs
/// to see which coefficient moved, and a total tells it nothing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SailForces {
    /// Lift, N, normal to the apparent wind.
    pub lift: f64,
    /// Total drag, N, along the apparent wind.
    pub drag: f64,
    /// Force along the centreline, N, positive forward. May be negative.
    pub driving_force: f64,
    /// Force athwartships, N, positive to leeward.
    pub heeling_force: f64,
    /// Heeling moment about the waterline, N·m, positive heeling to leeward.
    pub heeling_moment_about_waterline: f64,
    /// Height of the centre of effort above the water, m, after reefing.
    pub centre_of_effort_height: f64,
    /// Lift coefficient, after flattening and reefing.
    pub lift_coefficient: f64,
    /// Viscous drag coefficient, after reefing.
    pub viscous_drag_coefficient: f64,
    /// Induced drag coefficient, including Hazen's separation addition.
    pub induced_drag_coefficient: f64,
    /// Parasitic drag coefficient of mast and topsides. Not reefed.
    pub parasitic_drag_coefficient: f64,
    /// Aspect ratio the induced drag was computed on.
    pub aspect_ratio: f64,
    /// Reference area of every coefficient above, m².
    pub nominal_area: f64,
}

impl SailForces {
    /// Total drag coefficient `C_D = C_DP + C_DI + C_D0`.
    #[must_use]
    pub fn drag_coefficient(&self) -> f64 {
        self.viscous_drag_coefficient
            + self.induced_drag_coefficient
            + self.parasitic_drag_coefficient
    }
}

/// A rig with a chosen sail set: areas, centres of effort, and the coefficients
/// that follow from them.
///
/// The nominal area is computed once, here, and kept private. Every coefficient
/// this type produces is divided by that one stored value and every force is
/// multiplied back by it, and no public function in this module takes an area
/// as an argument. That is deliberate: `A_N` is the reference for all of
/// Table 8.1, and a coefficient divided by the wrong area is a mistake that
/// produces entirely believable numbers.
///
/// The sail set lives here rather than in [`Trim`] because it is what makes the
/// areas valid: a plan can only ever weight sails the rig actually carries.
/// Changing sails means building a new plan, which costs a handful of
/// multiplications.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SailPlan {
    rig: RigDimensions,
    set: SailSet,
    nominal_area: f64,
    /// Area of each sail, m²; zero for a sail that is not set, so the weighted
    /// sums need no branch.
    areas: [f64; SAIL_COUNT],
    /// Centre of effort above the sheer of each sail, m; zero when not set.
    centres: [f64; SAIL_COUNT],
}

impl SailPlan {
    /// Builds a sail plan from rig dimensions and a sail set.
    ///
    /// # Errors
    ///
    /// [`SailPlanError`] when the set names a sail the rig cannot describe, or
    /// when a dimension the plan needs is not positive. The checks are not
    /// pedantry: a zero nominal area divides every coefficient by zero, and a
    /// zero mast height does the same to the aspect ratio, both of which
    /// produce infinities several call frames away from the cause.
    ///
    /// `P`, `E`, `I` and `J` are required whatever is set, because they define
    /// the nominal area and hence every coefficient. `LPG` is required only
    /// with a jib set, `SL` only with a spinnaker.
    pub fn new(rig: &RigDimensions, set: SailSet) -> Result<Self, SailPlanError> {
        require_positive("P, main hoist", rig.main_hoist)?;
        require_positive("E, main foot", rig.main_foot)?;
        require_positive("I, foretriangle height", rig.foretriangle_height)?;
        require_positive("J, foretriangle base", rig.foretriangle_base)?;
        require_positive("EHM, mast height above sheer", rig.mast_height_above_sheer)?;
        require_non_negative("BAD, boom above sheer", rig.boom_above_sheer)?;
        require_non_negative("FA, average freeboard", rig.average_freeboard)?;
        require_non_negative("BMAX, maximum beam", rig.max_beam)?;
        require_non_negative("EMDC, average mast diameter", rig.mast_diameter)?;
        if set.contains(Sail::Jib) {
            require_positive("LPG, jib perpendicular", rig.jib_perpendicular)?;
        }
        if set.contains(Sail::Spinnaker) {
            require_positive("SL, spinnaker leech", rig.spinnaker_leech)?;
        }
        if let Some(mizzen) = rig.mizzen {
            require_positive("PY, mizzen hoist", mizzen.mizzen_hoist)?;
            require_positive("EY, mizzen foot", mizzen.mizzen_foot)?;
            require_non_negative(
                "BADY, mizzen boom above sheer",
                mizzen.mizzen_boom_above_sheer,
            )?;
        }

        let mut areas = [0.0; SAIL_COUNT];
        let mut centres = [0.0; SAIL_COUNT];
        for sail in set.iter() {
            let area = match sail {
                Sail::MizzenStaysail => return Err(SailPlanError::AreaUnavailable(sail)),
                _ => rig.area(sail).ok_or(SailPlanError::SailNotCarried(sail))?,
            };
            let centre = rig
                .centre_of_effort_height(sail)
                .ok_or(SailPlanError::SailNotCarried(sail))?;
            areas[sail.column()] = area;
            centres[sail.column()] = centre;
        }

        let nominal_area = rig.nominal_area();
        require_positive("A_N, nominal area", nominal_area)?;

        Ok(SailPlan {
            rig: *rig,
            set,
            nominal_area,
            areas,
            centres,
        })
    }

    #[must_use]
    pub fn rig(&self) -> &RigDimensions {
        &self.rig
    }

    #[must_use]
    pub fn set(&self) -> SailSet {
        self.set
    }

    /// The reference area of every coefficient this plan produces, m².
    #[must_use]
    pub fn nominal_area(&self) -> f64 {
        self.nominal_area
    }

    /// Area of a set sail, m². Zero for a sail that is not set.
    #[must_use]
    pub fn area(&self, sail: Sail) -> f64 {
        self.areas[sail.column()]
    }

    /// Total area actually hoisted, m². Not the reference area — see
    /// [`Self::nominal_area`].
    #[must_use]
    pub fn set_area(&self) -> f64 {
        self.areas.iter().sum()
    }

    /// Area-weighted mean lift coefficient of the set sails, referenced to the
    /// nominal area: `C_L = Σ(C_Li·A_i) / A_N`.
    #[must_use]
    pub fn lift_coefficient(&self, apparent_wind_angle: f64) -> f64 {
        self.area_weighted(&LIFT, apparent_wind_angle)
    }

    /// Area-weighted mean viscous drag coefficient of the set sails,
    /// referenced to the nominal area: `C_DP = Σ(C_DPi·A_i) / A_N`.
    #[must_use]
    pub fn viscous_drag_coefficient(&self, apparent_wind_angle: f64) -> f64 {
        self.area_weighted(&VISCOUS_DRAG, apparent_wind_angle)
    }

    fn area_weighted(&self, table: &CoefficientTable, apparent_wind_angle: f64) -> f64 {
        let degrees = folded_angle(apparent_wind_angle).to_degrees();
        let weighted: f64 = self
            .set
            .iter()
            .map(|sail| self.areas[sail.column()] * interpolate(table, sail, degrees))
            .sum();
        weighted / self.nominal_area
    }

    /// Aspect ratio of the whole sail plan, `span²/A_N`.
    ///
    /// One aspect ratio for the plan, not one per sail: the sails share a
    /// single trailing vortex system, so the span that matters is the plan's.
    #[must_use]
    pub fn aspect_ratio(&self, effective_span: EffectiveSpan) -> f64 {
        let height = match effective_span {
            EffectiveSpan::CloseHauled => {
                self.rig.mast_height_above_sheer + self.rig.average_freeboard
            }
            EffectiveSpan::Eased => self.rig.mast_height_above_sheer,
        };
        let span = SURFACE_MIRROR_FACTOR * height;
        span * span / self.nominal_area
    }

    /// Induced drag coefficient `C_DI = C_L²·(1/(π·AR) + 0.005)`.
    ///
    /// Computed **once for the whole sail plan** from the plan's own lift
    /// coefficient, never per sail and summed: induced drag is a property of
    /// the circulation the whole plan sheds, and `Σ C_Li²` is not `(Σ C_Li)²`
    /// — summing per-sail induced drags would understate it badly.
    ///
    /// The lift coefficient is passed in rather than recomputed so that the
    /// caller can hand over the *trimmed* value. That is what makes flattening
    /// pay: `C_DI` follows `C_L²`, so a flattening factor `F` on the lift
    /// arrives here squared.
    #[must_use]
    pub fn induced_drag_coefficient(
        &self,
        lift_coefficient: f64,
        effective_span: EffectiveSpan,
    ) -> f64 {
        let aspect_ratio = self.aspect_ratio(effective_span);
        let induced_factor = 1.0 / (PI * aspect_ratio) + HAZEN_SEPARATION_ADDITION;
        lift_coefficient * lift_coefficient * induced_factor
    }

    /// Parasitic drag coefficient of mast and topsides,
    /// `C_D0 = 1.13·(BMAX·FA + EHM·EMDC)/A_N`.
    ///
    /// Topsides frontal area is average freeboard × maximum beam and the mast's
    /// is mean diameter × height above the sheer, both at an assumed drag
    /// coefficient of 1.13. Independent of apparent wind angle and of trim —
    /// notably, reefing does not touch it, because the mast is still standing
    /// there.
    #[must_use]
    pub fn parasitic_drag_coefficient(&self) -> f64 {
        let topsides = self.rig.max_beam * self.rig.average_freeboard;
        let mast = self.rig.mast_height_above_sheer * self.rig.mast_diameter;
        WINDAGE_DRAG_COEFFICIENT * (topsides + mast) / self.nominal_area
    }

    /// Height of the sail plan's centre of effort above the **sheer**, m,
    /// before reefing.
    ///
    /// Area-weighted mean of the set sails' individual centres. Fig 8.19 gives
    /// the per-sail heights but the transcription does not state the
    /// combination rule, so the plain area weighting is used: it is the
    /// centroid of the sail area, a geometric property of the plan.
    ///
    /// The alternative was to weight by each sail's force contribution
    /// (`C_L·A`), which is arguably better physics — the centre of *effort*
    /// should follow the effort. It loses because it would make the heeling arm
    /// a function of the apparent wind angle, a coupling the source does not
    /// sanction, and because it would then differ from every published centre
    /// of effort a user could compare against.
    ///
    /// Zero for bare poles: with no sails set there is no sail area to take a
    /// centroid of. The windage that remains does have a centre of its own,
    /// which this model does not locate — see [`Self::forces`].
    #[must_use]
    pub fn centre_of_effort_height(&self) -> f64 {
        let total_area = self.set_area();
        if total_area <= 0.0 {
            return 0.0;
        }
        let weighted: f64 = self
            .set
            .iter()
            .map(|sail| self.areas[sail.column()] * self.centres[sail.column()])
            .sum();
        weighted / total_area
    }

    /// The aerodynamic force on the plan, resolved onto the yacht's centreline.
    ///
    /// # Trim: why `F` and `R` are not symmetric
    ///
    /// Flattening factor `F` multiplies `C_L` only. Flattening a sail takes
    /// camber out of it, not cloth: the area in the flow is unchanged, so
    /// `C_DP` is unchanged, and the height of the centre of effort is unchanged
    /// too. But `C_DI` follows `C_L²`, so lift falls as `F` while induced drag
    /// falls as `F²`. The resultant therefore **rotates forward** — less side
    /// force, disproportionately less drag — which is the whole reason a crew
    /// flattens before it reefs, and why this code applies `F` first even
    /// though the multiplications commute. The order in the source below is a
    /// statement about seamanship, not about arithmetic.
    ///
    /// Reef factor `R` is a linear scale on the sail plan's dimensions, so it
    /// multiplies `C_L` and `C_DP` by `R²` — an area — and the centre of effort
    /// by `R` — a length. It does **not** touch `C_D0`: reefing the main does
    /// nothing about the mast and the topsides, whose windage is exactly the
    /// drag you are left with in a gale. So the force falls roughly as `R²`
    /// while the moment falls as `R³`, and reefing buys more stability than
    /// force at a rate flattening cannot match. Note also that `C_DI`, computed
    /// from the already-reefed `C_L`, falls as `R⁴`.
    ///
    /// Both factors are clamped to `[0, 1]`. Values outside that range are a
    /// caller bug, but a trim solver searching for an equilibrium will overshoot
    /// its bracket, and it must not be answered with a sail plan that makes more
    /// lift than the sail can.
    ///
    /// # Known gap
    ///
    /// Under bare poles the only force left is the mast-and-topsides windage,
    /// whose centre of pressure this model does not locate. The moment then
    /// rests on the freeboard alone and understates the mast's contribution.
    /// Locating that centre needs a term Fig 8.19 does not provide.
    #[must_use]
    pub fn forces(&self, wind: ApparentWind, trim: Trim, air_density: f64) -> SailForces {
        let flat = trim.flat.clamp(0.0, 1.0);
        let reef = trim.reef.clamp(0.0, 1.0);
        let reefed_area = reef * reef;
        let angle = folded_angle(wind.angle);

        // Flatten first, then reef: see above.
        let flattened_lift = self.lift_coefficient(angle) * flat;
        let lift_coefficient = flattened_lift * reefed_area;
        let viscous_drag_coefficient = self.viscous_drag_coefficient(angle) * reefed_area;
        let induced_drag_coefficient =
            self.induced_drag_coefficient(lift_coefficient, trim.effective_span);
        let parasitic_drag_coefficient = self.parasitic_drag_coefficient();
        let drag_coefficient =
            viscous_drag_coefficient + induced_drag_coefficient + parasitic_drag_coefficient;

        let reference = 0.5 * air_density * wind.speed * wind.speed * self.nominal_area;
        let lift = reference * lift_coefficient;
        let drag = reference * drag_coefficient;

        let (sin, cos) = angle.sin_cos();
        let driving_force = lift * sin - drag * cos;
        let heeling_force = lift * cos + drag * sin;

        // Reef lowers the sail plan's own centre; it does not lower the deck,
        // so the freeboard is added after the reef factor, not scaled by it.
        let centre_of_effort_height =
            reef * self.centre_of_effort_height() + self.rig.average_freeboard;

        SailForces {
            lift,
            drag,
            driving_force,
            heeling_force,
            heeling_moment_about_waterline: heeling_force * centre_of_effort_height,
            centre_of_effort_height,
            lift_coefficient,
            viscous_drag_coefficient,
            induced_drag_coefficient,
            parasitic_drag_coefficient,
            aspect_ratio: self.aspect_ratio(trim.effective_span),
            nominal_area: self.nominal_area,
        }
    }
}

/// Lift coefficient of a single sail at an apparent wind angle. Table 8.1(a).
///
/// `apparent_wind_angle` is in radians, like every other angle in this crate;
/// the table it indexes is published in degrees.
///
/// # Interpolation
///
/// The book draws **splines** through the five tabulated angles; this is
/// **linear** interpolation between them. The difference is small and
/// one-sided: a spline rounding over a maximum lies above the chords on either
/// side of it, so linear interpolation slightly *undercuts* every peak. The
/// two curves with an interior maximum are the ones to watch — the mizzen,
/// whose lift peaks at the tabulated 50°, and the spinnaker, whose lift climbs
/// from nothing at 27° to its maximum at 50°. At the tabulated angles the two
/// interpolations agree exactly.
///
/// Below 27° the 27° value is **held**, not extrapolated. That is the book's
/// own modelling choice — it draws the curves horizontal below 27° — and not an
/// artefact of a clamped table lookup: a close-hauled sail plan is at its
/// maximum lift, and the model declines to say anything about what happens
/// inside the angle at which a yacht stops sailing. Above 180° nothing needs
/// held, because angles are folded into `[0, π]` first.
#[must_use]
pub fn lift_coefficient(sail: Sail, apparent_wind_angle: f64) -> f64 {
    interpolate(&LIFT, sail, folded_angle(apparent_wind_angle).to_degrees())
}

/// Viscous drag coefficient of a single sail at an apparent wind angle.
/// Table 8.1(b). Same interpolation and same held value below 27° as
/// [`lift_coefficient`].
///
/// This is the drag of the sail's own boundary layer and separated flow at a
/// given angle; the drag that grows with lift is in
/// [`SailPlan::induced_drag_coefficient`], and the rig's own windage in
/// [`SailPlan::parasitic_drag_coefficient`].
#[must_use]
pub fn viscous_drag_coefficient(sail: Sail, apparent_wind_angle: f64) -> f64 {
    interpolate(
        &VISCOUS_DRAG,
        sail,
        folded_angle(apparent_wind_angle).to_degrees(),
    )
}

/// Linear interpolation of one column of Table 8.1, in degrees.
///
/// Held at both ends: below 27° because the book holds it, above 180° because
/// the caller cannot get there.
fn interpolate(table: &CoefficientTable, sail: Sail, degrees: f64) -> f64 {
    let column = sail.column();
    if degrees <= TABULATED_ANGLES[0] {
        return table[0][column];
    }
    if degrees >= TABULATED_ANGLES[ANGLE_COUNT - 1] {
        return table[ANGLE_COUNT - 1][column];
    }
    let upper = TABULATED_ANGLES
        .iter()
        .position(|angle| *angle >= degrees)
        .unwrap_or(ANGLE_COUNT - 1);
    let lower = upper - 1;
    let fraction =
        (degrees - TABULATED_ANGLES[lower]) / (TABULATED_ANGLES[upper] - TABULATED_ANGLES[lower]);
    table[lower][column] + fraction * (table[upper][column] - table[lower][column])
}

/// Folds an apparent wind angle onto `[0, π]`.
///
/// The model is symmetric about the centreline, so an angle to port and its
/// mirror to starboard give the same forces, and 190° is 170° on the other
/// tack. Folding here rather than asking the caller to normalise means a
/// heading integrated over many tacks cannot walk off the end of the table.
///
/// The absolute value is taken *before* the modulo so that the two tacks are
/// bit-identical rather than merely equal to within an ulp: folding `-x`
/// through `τ − (τ − x)` loses a bit, and a solver that compares the two tacks
/// should not see a force differing in the last place.
fn folded_angle(apparent_wind_angle: f64) -> f64 {
    let wrapped = apparent_wind_angle.abs().rem_euclid(TAU);
    if wrapped > PI {
        TAU - wrapped
    } else {
        wrapped
    }
}

fn require_positive(name: &'static str, value: f64) -> Result<(), SailPlanError> {
    if value > 0.0 {
        Ok(())
    } else {
        Err(SailPlanError::Dimension {
            name,
            value,
            minimum: 0.0,
        })
    }
}

fn require_non_negative(name: &'static str, value: f64) -> Result<(), SailPlanError> {
    if value >= 0.0 {
        Ok(())
    } else {
        Err(SailPlanError::Dimension {
            name,
            value,
            minimum: 0.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn table_columns_are_in_the_published_order() {
        // Guards the discriminant-as-column-index trick that every lookup
        // depends on. The 100° row is the discriminating one: four distinct
        // lift values and a zero exactly where the jib is.
        assert_relative_eq!(LIFT[3][Sail::Main.column()], 0.85);
        assert_relative_eq!(LIFT[3][Sail::Jib.column()], 0.0);
        assert_relative_eq!(LIFT[3][Sail::Spinnaker.column()], 0.85);
        assert_relative_eq!(LIFT[3][Sail::Mizzen.column()], 0.8);
        assert_relative_eq!(LIFT[3][Sail::MizzenStaysail.column()], 0.8);
        // ... and the 180° viscous row separates main from spinnaker.
        assert_relative_eq!(VISCOUS_DRAG[4][Sail::Main.column()], 0.9);
        assert_relative_eq!(VISCOUS_DRAG[4][Sail::Spinnaker.column()], 0.66);
    }

    #[test]
    fn interpolation_is_linear_between_tabulated_angles() {
        // Main lift halfway between 50° and 80°: (1.5 + 0.95)/2.
        let midpoint = interpolate(&LIFT, Sail::Main, 65.0);
        assert_relative_eq!(midpoint, 1.225, epsilon = 1e-12);
    }

    #[test]
    fn angles_fold_onto_the_half_circle() {
        assert_relative_eq!(folded_angle(-0.5), 0.5, epsilon = 1e-12);
        assert_relative_eq!(folded_angle(190.0_f64.to_radians()), 170.0_f64.to_radians());
        assert_relative_eq!(folded_angle(TAU + 0.25), 0.25, epsilon = 1e-12);
        assert_relative_eq!(folded_angle(PI), PI, epsilon = 1e-12);
    }

    #[test]
    fn a_sail_set_lists_its_sails_in_table_order() {
        let set = SailSet::downwind().with(Sail::Mizzen);
        let sails: Vec<Sail> = set.iter().collect();
        assert_eq!(sails, vec![Sail::Main, Sail::Spinnaker, Sail::Mizzen]);
        assert!(!set.contains(Sail::Jib));
        assert!(SailSet::upwind().without(Sail::Jib).contains(Sail::Main));
        assert!(SailSet::none().is_empty());
    }
}
