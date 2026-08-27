//! Flying shape: the surface the controls actually leave behind.
//!
//! # What this module is
//!
//! A sail is not a rigid mesh. Halyard, cunningham, outhaul, vang, sheet and
//! traveller do not move a shape around — they *change* it, and the shape they
//! leave is the flying shape. This module is the parametric description of that
//! shape: a planform, a section family, and how the section parameters vary up the
//! sail. It produces a [`Lattice`] for [`crate::vlm`] and nothing else; it has no
//! opinion about wind, forces or trim inputs.
//!
//! # Why parametric rather than a membrane solve
//!
//! The honest alternative is fluid-structure interaction: a membrane FEM in the
//! loop, re-solved against the pressure field until shape and load agree. That is
//! how the reference measurements this module is checked against were *reproduced*
//! numerically, and it is out of scope at sixty frames a second.
//!
//! What is in scope is the observation that a real sail's shape is nearly
//! low-dimensional. Sailmakers describe a section with three numbers — camber,
//! draft position, entry angle — and a sail with how those vary up the leech. So
//! the parameters are carried directly and the controls map onto them, which keeps
//! every control physically meaningful (vang tension reduces twist, which changes
//! the vertical force distribution, which changes the heeling moment) at the cost
//! of a few closed forms.
//!
//! # Three decisions
//!
//! **The section is the NACA four-digit mean line, and entry angle is not a free
//! parameter.** A parabolic arc is the obvious family and it is wrong for a sail:
//! its draft is pinned at mid chord, and a sail's sits between a third and a half.
//! The four-digit mean line is two parabolas joined at the draft position with
//! matching value *and* matching slope, so it moves the draft, stays smooth, and
//! degenerates exactly to the parabolic arc at half chord.
//!
//! Entry and exit angles then follow from camber and draft rather than being
//! carried alongside them: `tan(entry) = 2ε/p`, `tan(exit) = -2ε/(1-p)`. That is a
//! restriction and it is the physically right one. Draft forward means a sharper
//! entry and a flatter exit, which is what moving the draft forward *does*, and a
//! model with an independent entry angle can express sections no sail can fly.
//!
//! **Sections are horizontal, and twist is a rotation about the vertical.** Not
//! about the luff tangent, which would be the choice for a swept wing. This is the
//! convention the flying-shape measurements use — twist is reported as the angle
//! between the horizontal projections of the root and head chords — so it is the
//! convention that makes a comparison against them mean something.
//!
//! **Placement is not shape.** The luff runs up the `z` axis from the origin. Mast
//! rake, mast bend, position on deck and the tack's height above the water belong
//! to the rig that carries the sail, not to the sail's shape, and putting them here
//! would mean every consumer of a shape had to know which of them had been applied
//! already.
//!
//! # Frame and tack
//!
//! The file frame of [`crate::frames`]: `x` forward, `y` to port, `z` up. The
//! shape is built with its chord swinging to port and its camber bulging to port,
//! which is a sail on **starboard tack**. A boat on port tack mirrors in `y`.
//! Carrying a signed tack in here would have put a factor of `±1` in every formula
//! below to save one mirror at the call site.
//!
//! # Verified against
//!
//! Closed forms for everything geometric, and for the aerodynamics the full-scale
//! wind-tunnel campaign on an Olympic windsurf sail of [[Zhang et al.
//! 2025]](https://arxiv.org/abs/2501.13254), which measured flying shape and forces
//! at the same time. Three of its findings are reproduced here: twist reduces lift
//! at a given root incidence, twist lowers the centre of effort, and twist leaves
//! the head at positive incidence rather than shadowing it.
//!
//! Its fourth finding is reproduced only in part, and the part that is missing is
//! informative. They measured *more* camber giving *less* lift, which is backwards
//! for a section and is explained by their own shape data: the high-camber setting
//! also twisted more. This module separates the two, so camber alone increases lift
//! here — and their result is recovered only once a control mapping couples camber
//! to twist. The coupling belongs in that mapping, not in the geometry.

use nalgebra::Vector3;

use crate::geometry::Point;
use crate::vlm::Lattice;

/// A section's camber line: the NACA four-digit mean line.
///
/// Two parabolas joined at the draft position, carrying camber and draft as
/// fractions of the chord. Offsets are also fractions of the chord, so one profile
/// serves every station of a tapered sail.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Profile {
    camber: f64,
    draft: f64,
}

impl Profile {
    /// A profile from camber ratio and draft position, both fractions of chord.
    ///
    /// Returns `None` unless the camber is finite and non-negative and the draft is
    /// strictly inside the chord. Draft *at* an end is not a degenerate sail, it is
    /// a division by zero: the mean line's two halves have widths `p` and `1 - p`.
    #[must_use]
    pub fn new(camber: f64, draft: f64) -> Option<Self> {
        if !camber.is_finite() || camber < 0.0 || !draft.is_finite() || draft <= 0.0 || draft >= 1.0
        {
            return None;
        }
        Some(Self { camber, draft })
    }

    /// Maximum camber, as a fraction of chord.
    #[must_use]
    pub fn camber(&self) -> f64 {
        self.camber
    }

    /// Chordwise position of maximum camber, as a fraction of chord.
    #[must_use]
    pub fn draft(&self) -> f64 {
        self.draft
    }

    /// Camber offset at a chordwise fraction, as a fraction of chord.
    ///
    /// `(m/p²)(2px - x²)` ahead of the draft and `(m/(1-p)²)((1-2p) + 2px - x²)`
    /// behind it. Both give `m` at `x = p` and both give zero slope there, which is
    /// what makes the join invisible to a lattice that samples across it.
    #[must_use]
    pub fn offset(&self, along: f64) -> f64 {
        let (m, p) = (self.camber, self.draft);
        if along <= p {
            m / (p * p) * (2.0 * p * along - along * along)
        } else {
            m / ((1.0 - p) * (1.0 - p)) * ((1.0 - 2.0 * p) + 2.0 * p * along - along * along)
        }
    }

    /// Slope of the camber line at a chordwise fraction.
    ///
    /// `2m(p - x)` over the width of whichever half `x` falls in — so it is
    /// positive ahead of the draft, negative behind it, and zero at it.
    #[must_use]
    pub fn slope(&self, along: f64) -> f64 {
        let (m, p) = (self.camber, self.draft);
        let width = if along <= p { p } else { 1.0 - p };
        2.0 * m * (p - along) / (width * width)
    }

    /// Entry angle at the luff, radians, positive into the wind.
    #[must_use]
    pub fn entry_angle(&self) -> f64 {
        self.slope(0.0).atan()
    }

    /// Exit angle at the leech, radians. Negative: the flow leaves closing.
    #[must_use]
    pub fn exit_angle(&self) -> f64 {
        self.slope(1.0).atan()
    }

    /// Arc length of the camber line, as a fraction of chord.
    ///
    /// A cambered section's surface is longer than its chord, so a lofted sail has
    /// more cloth in it than its planform area. Integrated rather than closed
    /// because the closed form is an `asinh` in two pieces and this is called once
    /// per shape, not per frame.
    #[must_use]
    pub fn arc_length(&self) -> f64 {
        let steps = 2048;
        let mut total = 0.0;
        let mut previous = (1.0 + self.slope(0.0).powi(2)).sqrt();
        for step in 1..=steps {
            let along = step as f64 / steps as f64;
            let current = (1.0 + self.slope(along).powi(2)).sqrt();
            total += 0.5 * (previous + current) / steps as f64;
            previous = current;
        }
        total
    }
}

/// How a section parameter varies up the sail.
///
/// Linear in height fraction, between a value at the foot and a value at the head.
/// A real camber distribution is not linear — it peaks low and flattens toward the
/// head — but the shape of that peak is not something the published flying-shape
/// data pins down, whereas the two endpoint values are what the controls move. Two
/// numbers that are honest beat four that are invented.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Distribution {
    /// Value at the foot.
    pub foot: f64,
    /// Value at the head.
    pub head: f64,
}

impl Distribution {
    /// The same value everywhere.
    #[must_use]
    pub const fn uniform(value: f64) -> Self {
        Self {
            foot: value,
            head: value,
        }
    }

    /// A distribution from foot and head values.
    #[must_use]
    pub const fn new(foot: f64, head: f64) -> Self {
        Self { foot, head }
    }

    /// The value at a height fraction.
    #[must_use]
    pub fn at(&self, up: f64) -> f64 {
        self.foot + (self.head - self.foot) * up
    }
}

/// The sail's outline.
///
/// Chord against height, as a linear taper plus a leech round. Enough for a
/// mainsail, a headsail and a flat-cut downwind sail; a full sailmaker's outline
/// with luff round and a shaped head would need the luff and leech as curves, and
/// would not change any force this model computes by as much as the coefficient
/// uncertainty it is compared against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Planform {
    luff: f64,
    foot: f64,
    head: f64,
    roach: f64,
}

impl Planform {
    /// A planform from luff height, foot chord, head chord and leech round.
    ///
    /// `roach` is the leech round as a fraction of the foot chord, applied as a
    /// half sine so that it vanishes at both ends and peaks at mid height.
    ///
    /// A head chord of zero is allowed and is the ordinary triangular sail: the top
    /// panels become triangles, which have a perfectly good normal and area. A foot
    /// chord of zero is not, because it is a sail with no area.
    #[must_use]
    pub fn new(luff: f64, foot: f64, head: f64, roach: f64) -> Option<Self> {
        let finite = [luff, foot, head, roach].iter().all(|v| v.is_finite());
        if !finite || luff <= 0.0 || foot <= 0.0 || head < 0.0 || roach < 0.0 {
            return None;
        }
        Some(Self {
            luff,
            foot,
            head,
            roach,
        })
    }

    /// Luff height, m.
    #[must_use]
    pub fn luff(&self) -> f64 {
        self.luff
    }

    /// Chord at a height fraction, m.
    #[must_use]
    pub fn chord(&self, up: f64) -> f64 {
        self.foot * (1.0 - up)
            + self.head * up
            + self.roach * self.foot * (std::f64::consts::PI * up).sin()
    }

    /// Planform area, m².
    ///
    /// Closed form: the taper contributes the mean of its ends and the leech round
    /// contributes `2/π` of its peak, since that is `∫sin` over a half period.
    #[must_use]
    pub fn area(&self) -> f64 {
        self.luff
            * (0.5 * (self.foot + self.head) + self.roach * self.foot * 2.0 / std::f64::consts::PI)
    }

    /// Aspect ratio, luff squared over area.
    ///
    /// The rig convention — span over *mean* chord — rather than the sailmaker's
    /// luff over foot. It is the one that enters an induced-drag estimate, and the
    /// one the wind-tunnel literature quotes.
    #[must_use]
    pub fn aspect_ratio(&self) -> f64 {
        self.luff * self.luff / self.area()
    }

    /// Area-weighted mean height fraction: the height the planform's area balances
    /// at.
    ///
    /// Closed form, from the first moment of the chord distribution. The taper
    /// contributes `foot/6 + head/3` and the leech round contributes `1/π` of its
    /// peak, since that is `∫u sin πu` over a half period.
    ///
    /// It exists because it is the weight any sail-wide average of a quantity that
    /// varies with height has to use. A triangular sail carries two thirds of its
    /// cloth in its bottom half, so averaging twist over *height* would put the
    /// mean where very little sail is.
    #[must_use]
    pub fn mean_height(&self) -> f64 {
        let pi = std::f64::consts::PI;
        let moment = self.foot / 6.0 + self.head / 3.0 + self.roach * self.foot / pi;
        let area = 0.5 * (self.foot + self.head) + self.roach * self.foot * 2.0 / pi;
        moment / area
    }
}

/// A sail's flying shape.
///
/// The planform, the section parameters and how they vary up the sail, plus the
/// angle the whole sail is trimmed to. Everything a lattice needs and nothing
/// about the wind.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    planform: Planform,
    camber: Distribution,
    draft: Distribution,
    twist: Distribution,
    trim: f64,
}

impl Shape {
    /// A shape from a planform and its parameter distributions.
    ///
    /// `trim` is the chord angle at the foot, radians, positive easing to leeward.
    /// `twist` adds to it going up, so the head's chord angle is `trim +
    /// twist.head` — which makes twist the quantity the flying-shape measurements
    /// report, and a `twist` of zero a sail with no twist rather than a sail with
    /// no trim.
    ///
    /// Returns `None` if any distribution reaches a camber or draft the section
    /// family cannot represent. Checked at the ends, which is sufficient: both
    /// distributions are linear, so an endpoint pair inside the range keeps every
    /// station inside it.
    #[must_use]
    pub fn new(
        planform: Planform,
        camber: Distribution,
        draft: Distribution,
        twist: Distribution,
        trim: f64,
    ) -> Option<Self> {
        if !trim.is_finite() || !twist.foot.is_finite() || !twist.head.is_finite() {
            return None;
        }
        Profile::new(camber.foot, draft.foot)?;
        Profile::new(camber.head, draft.head)?;
        Some(Self {
            planform,
            camber,
            draft,
            twist,
            trim,
        })
    }

    /// The planform.
    #[must_use]
    pub fn planform(&self) -> &Planform {
        &self.planform
    }

    /// The section at a height fraction.
    #[must_use]
    pub fn profile(&self, up: f64) -> Profile {
        Profile {
            camber: self.camber.at(up),
            draft: self.draft.at(up),
        }
    }

    /// Chord angle at a height fraction, radians: trim plus twist.
    #[must_use]
    pub fn chord_angle(&self, up: f64) -> f64 {
        self.trim + self.twist.at(up)
    }

    /// Unit vector from luff to leech at a height fraction.
    ///
    /// Aft and to leeward: `(-cos α, sin α, 0)`. Horizontal by construction — see
    /// the module's second decision.
    #[must_use]
    pub fn chord_direction(&self, up: f64) -> Vector3<f64> {
        let angle = self.chord_angle(up);
        Vector3::new(-angle.cos(), angle.sin(), 0.0)
    }

    /// Unit vector the camber bulges along at a height fraction.
    ///
    /// The chord direction turned a quarter turn to leeward, so that a positive
    /// camber offset is a sail bellied the way a sail bellies.
    #[must_use]
    pub fn camber_direction(&self, up: f64) -> Vector3<f64> {
        let angle = self.chord_angle(up);
        Vector3::new(angle.sin(), angle.cos(), 0.0)
    }

    /// A point on the surface, at chordwise fraction `along` and height fraction
    /// `up`.
    #[must_use]
    pub fn point(&self, along: f64, up: f64) -> Point {
        let chord = self.planform.chord(up);
        let luff = Point::new(0.0, 0.0, up * self.planform.luff);
        luff + self.chord_direction(up) * (along * chord)
            + self.camber_direction(up) * (self.profile(up).offset(along) * chord)
    }

    /// The lattice, sampled uniformly in both directions.
    ///
    /// Uniform and not cosine-spaced: the chordwise convergence of a cambered
    /// lattice is limited by how well flat panels carry a curved surface's normal,
    /// which no spacing fixes, and cosine spacing was measured to make it very
    /// slightly worse. Spanwise, a sail's loading has its structure at the head and
    /// foot rather than at one tip, so there is no single end to crowd.
    #[must_use]
    pub fn lattice(&self, chordwise: usize, spanwise: usize) -> Option<Lattice> {
        Lattice::from_shape(chordwise, spanwise, |along, up| self.point(along, up))
    }

    /// Geometric incidence of a section against a horizontal flow direction,
    /// radians.
    ///
    /// The angle between the flow and the chord, positive when the flow meets the
    /// windward side — so a section at negative incidence is being back-winded.
    /// Geometric, not effective: a cambered section still lifts at zero incidence,
    /// and the zero-lift angle is the lattice's business rather than the geometry's.
    #[must_use]
    pub fn incidence(&self, up: f64, flow: Vector3<f64>) -> f64 {
        let chord = self.chord_direction(up);
        let horizontal = Vector3::new(flow.x, flow.y, 0.0);
        if horizontal.norm() < 1e-12 {
            return 0.0;
        }
        let along = horizontal.normalize();
        // Both run downstream - the chord from luff to leech, the flow onto the
        // luff and away - so they are nearly parallel and compare directly.
        (chord.y * along.x - chord.x * along.y).atan2(chord.dot(&along))
    }

    /// Area-weighted mean geometric incidence against a flow, radians.
    ///
    /// The single number a sail-wide stall criterion can be read against, and where
    /// twist earns its keep: twist lowers the mean, so a twisted sail reaches any
    /// given onset at a higher wind angle.
    ///
    /// Exact rather than quadrature. Sections are horizontal and the flow is
    /// projected onto the horizontal, so incidence is linear in height fraction —
    /// it is `α - twist(up)` and nothing else — and the area weighting therefore
    /// collapses to the planform's own mean height. Interpolating between the two
    /// ends keeps the sign conventions in [`Shape::incidence`] alone, rather than
    /// repeating them here where they could drift.
    #[must_use]
    pub fn mean_incidence(&self, flow: Vector3<f64>) -> f64 {
        let (foot, head) = (self.incidence(0.0, flow), self.incidence(1.0, flow));
        foot + (head - foot) * self.planform.mean_height()
    }
}

/// The trim controls of one sail, each normalised: 0 fully eased, 1 fully hard.
///
/// Normalised rather than physical because the physical alternative does not
/// exist yet. A control's real input is a line tension in newtons or a car
/// position in millimetres, and the transfer from either of those to a shape has
/// been measured only in a handful of wind-tunnel campaigns whose numbers live in
/// figures behind paywalls — and it depends on dynamic pressure as well as on the
/// control, because cloth stretches. Declaring what a control *reaches* (see
/// [`Response`]) asks the boat's designer for two numbers they know, instead of
/// asking this module to invent a coefficient.
///
/// Values outside `[0, 1]` are clamped rather than refused. These come from
/// sliders and solvers, which overshoot by a bit-width and would otherwise make a
/// trim fail; and a control at 1.2 is not a physical statement whose error needs
/// preserving.
///
/// Distinct from [`crate::controls::Controls`], which is the boat-level state —
/// sail selection, rudder, and which sails are set. This is the shape of one sail.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Controls {
    /// Mainsheet, or the headsail sheet: closes the angle *and* tensions the leech.
    pub sheet: f64,
    /// Traveller car, or the sheet lead: closes the angle without touching the
    /// leech.
    pub traveller: f64,
    /// Vang, or the leech line: tensions the leech without closing the angle.
    pub vang: f64,
    /// Outhaul, or the foot tension: flattens the sail.
    pub outhaul: f64,
    /// Luff tension — cunningham and halyard together, since they do the same
    /// thing to the shape: moves the draft forward.
    pub cunningham: f64,
}

impl Controls {
    /// Everything eased.
    pub const EASED: Self = Self {
        sheet: 0.0,
        traveller: 0.0,
        vang: 0.0,
        outhaul: 0.0,
        cunningham: 0.0,
    };

    /// Everything hard on.
    pub const HARD: Self = Self {
        sheet: 1.0,
        traveller: 1.0,
        vang: 1.0,
        outhaul: 1.0,
        cunningham: 1.0,
    };

    /// Every control clamped into its range, and any non-finite one eased.
    #[must_use]
    pub fn clamped(self) -> Self {
        let fix = |value: f64| {
            if value.is_finite() {
                value.clamp(0.0, 1.0)
            } else {
                0.0
            }
        };
        Self {
            sheet: fix(self.sheet),
            traveller: fix(self.traveller),
            vang: fix(self.vang),
            outhaul: fix(self.outhaul),
            cunningham: fix(self.cunningham),
        }
    }

    /// Leech tension, which is what actually sets twist.
    ///
    /// `max(sheet, vang)`, and the choice of `max` over a sum is the physics. A
    /// leech has one tension; sheet and vang are two ways of applying it, and
    /// whichever is tighter is the one holding it. That is *why* a boat carries
    /// both — the vang takes over the leech tension the sheet gives up, which is
    /// what lets a trimmer open the angle without letting the head twist off.
    #[must_use]
    pub fn leech_tension(self) -> f64 {
        self.sheet.max(self.vang)
    }

    /// How far the boom is brought in, as a fraction of its range.
    ///
    /// The product of sheet and traveller, because the angle needs *both*. Easing
    /// the sheet lets the boom out however the car is set, and dropping the car to
    /// leeward lets it out however hard the sheet is in. So either at zero leaves
    /// the angle open — which is exactly the classic depowering trim, traveller
    /// down and sheet hard: an open angle with a closed leech, and this model
    /// reproduces it because the two quantities are computed from different rules.
    #[must_use]
    pub fn boom_in(self) -> f64 {
        self.sheet * self.traveller
    }
}

/// What one sail's controls reach: the shape at each end of each control's travel.
///
/// This is the calibration, and it is the boat file's to declare rather than this
/// module's to know. A sailmaker knows that a given main goes from sixteen per cent
/// camber with the outhaul off to eight with it hard on; nothing in the published
/// literature knows it for a sail in general.
///
/// Validated once, on construction. After that [`Response::shape`] is infallible:
/// every value it produces is an interpolation between two declared endpoints that
/// were already checked, so there is no trim setting a simulator can reach that
/// fails. A control that could break the model would be worse than one that cannot
/// express something.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Response {
    angle: (f64, f64),
    twist: (f64, f64),
    camber: (f64, f64),
    head_camber: f64,
    draft: (f64, f64),
    camber_twist: f64,
}

impl Response {
    /// Declares a sail's control travel.
    ///
    /// Each pair is `(eased, hard)`:
    ///
    /// - `angle`: chord angle at the foot, radians, with the boom fully out and
    ///   fully in.
    /// - `twist`: head twist, radians, with the leech slack and fully tensioned.
    ///   Slack must be the larger: tensioning a leech removes twist.
    /// - `camber`: camber ratio at the foot, with the outhaul off and hard on. Off
    ///   must be the larger: an outhaul flattens.
    /// - `draft`: draft position, with the luff slack and hard on. Slack must be
    ///   the larger: luff tension moves the draft forward.
    ///
    /// `head_camber` is the head's camber as a fraction of the foot's — a sail's
    /// own taper, which the controls do not change. One number rather than a second
    /// pair, because no control acts on the head's camber independently.
    ///
    /// Returns `None` if any value is unusable or any pair runs the wrong way. The
    /// direction checks are the point: they are the qualitative derivatives the
    /// literature *does* give, so a boat file with a sign error is caught at load
    /// rather than sailed.
    #[must_use]
    pub fn new(
        angle: (f64, f64),
        twist: (f64, f64),
        camber: (f64, f64),
        head_camber: f64,
        draft: (f64, f64),
    ) -> Option<Self> {
        let finite = [
            angle.0,
            angle.1,
            twist.0,
            twist.1,
            camber.0,
            camber.1,
            head_camber,
            draft.0,
            draft.1,
        ]
        .iter()
        .all(|value| value.is_finite());
        if !finite
            || twist.0 < twist.1
            || camber.0 < camber.1
            || draft.0 < draft.1
            || !(0.0..=1.0).contains(&head_camber)
        {
            return None;
        }
        // Both ends of both distributions have to be sections a sail can fly, and
        // the head's camber is scaled from the foot's so it is checked separately.
        Profile::new(camber.0, draft.0)?;
        Profile::new(camber.1, draft.1)?;
        Profile::new(camber.0 * head_camber, draft.0)?;
        Some(Self {
            angle,
            twist,
            camber,
            head_camber,
            draft,
            camber_twist: 0.0,
        })
    }

    /// Declares that a fuller sail twists more, by `radians` of extra head twist at
    /// full camber.
    ///
    /// The coupling [[Zhang et al. 2025]](https://arxiv.org/abs/2501.13254)
    /// measured and this engine refuses to guess. Easing their outhaul increased
    /// camber and twist together, and the twist won — their high-camber setting made
    /// *less* lift, which is backwards for a section and is the whole reason the
    /// geometry of [`Shape`] holds camber and twist apart.
    ///
    /// The mechanism is a slacker leech: a fuller sail has more cloth in it, so the
    /// same sheet load leaves the head freer. The *strength* is not published for
    /// any sail, so it defaults to zero — which is a complete model of a sail whose
    /// designer declares no coupling, not a placeholder. Sign only is known, and
    /// negative values are refused for that reason.
    #[must_use]
    pub fn with_camber_twist_coupling(self, radians: f64) -> Option<Self> {
        if !radians.is_finite() || radians < 0.0 {
            return None;
        }
        Some(Self {
            camber_twist: radians,
            ..self
        })
    }

    /// The shape a planform takes at a control setting.
    ///
    /// Infallible by construction — see the type's own documentation.
    #[must_use]
    pub fn shape(&self, planform: Planform, controls: Controls) -> Shape {
        let controls = controls.clamped();
        let between = |range: (f64, f64), at: f64| range.0 + (range.1 - range.0) * at;

        let camber = between(self.camber, controls.outhaul);
        // Fullness above the flattest the sail goes, as a fraction of its range,
        // which is what the coupling is declared against.
        let fullness = if self.camber.0 > self.camber.1 {
            (camber - self.camber.1) / (self.camber.0 - self.camber.1)
        } else {
            0.0
        };
        let twist = between(self.twist, controls.leech_tension()) + self.camber_twist * fullness;

        Shape {
            planform,
            camber: Distribution::new(camber, camber * self.head_camber),
            draft: Distribution::uniform(between(self.draft, controls.cunningham)),
            twist: Distribution::new(0.0, twist),
            trim: between(self.angle, controls.boom_in()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    use crate::vlm::Solver;

    const AIR: f64 = 1.225;
    const SPEED: f64 = 8.0;

    /// The Olympic windsurf sail of Zhang et al. 2025, as a planform.
    ///
    /// Eight square metres and aspect ratio 3.36, both published, with a two-metre
    /// maximum chord and a 4.9 m mast. Those four numbers over-determine a
    /// four-parameter planform by one, and the roach is what absorbs it: a lot of
    /// roach, which is what a fully-battened windsurf sail has.
    fn windsurf() -> Planform {
        Planform::new(5.185, 2.0, 0.10, 0.387).expect("a real sail's dimensions")
    }

    /// A shape over the windsurf planform, trimmed to the centreline so that the
    /// wind angle *is* the root section's angle of attack.
    fn sail(camber: f64, draft: f64, twist_degrees: f64) -> Shape {
        Shape::new(
            windsurf(),
            Distribution::uniform(camber),
            Distribution::uniform(draft),
            Distribution::new(0.0, twist_degrees.to_radians()),
            0.0,
        )
        .expect("a trimmable sail")
    }

    /// Lift and drag coefficients and centre-of-effort height, in the conventions
    /// the measurements are reported in.
    ///
    /// Coefficients on *planform* area, not the lattice's surface area, because
    /// that is what the wind-tunnel literature divides by. Centre of effort as
    /// rolling moment over horizontal force, `Zr = Mr/Fa`, which is the published
    /// definition rather than a load centroid — the two differ, and only one of
    /// them can be compared.
    fn measure(shape: &Shape, wind_degrees: f64, panels: (usize, usize)) -> (f64, f64, f64) {
        let beta = wind_degrees.to_radians();
        let flow = Vector3::new(-SPEED * beta.cos(), SPEED * beta.sin(), 0.0);
        let across = Vector3::new(beta.sin(), beta.cos(), 0.0);
        let solution = Solver::new(shape.lattice(panels.0, panels.1).expect("a grid"), flow)
            .expect("a well-formed sail")
            .solve(|_| flow, AIR)
            .expect("a sail solves");
        let force = solution.force();
        let moment = solution.moment_about(Point::zeros());
        let dynamic = 0.5 * AIR * SPEED * SPEED * shape.planform().area();
        let horizontal = force.x.hypot(force.y);
        (
            force.dot(&across) / dynamic,
            force.dot(&(flow / SPEED)) / dynamic,
            moment.x.hypot(moment.y) / horizontal,
        )
    }

    /// The mean line peaks at its draft position, with the camber it was given.
    ///
    /// The definition, and the thing a two-piece formula gets wrong by algebra
    /// rather than by physics.
    #[test]
    fn the_mean_line_peaks_at_its_draft_position() {
        for draft in [0.30_f64, 0.40, 0.50, 0.65] {
            let profile = Profile::new(0.12, draft).expect("a real section");
            assert_relative_eq!(profile.offset(draft), 0.12, max_relative = 1e-14);
            for step in 0..=400 {
                let along = step as f64 / 400.0;
                assert!(
                    profile.offset(along) <= 0.12 + 1e-12,
                    "offset {} exceeded the camber at {along}",
                    profile.offset(along)
                );
            }
            // And it closes at both ends, which is what makes it a chord.
            assert_relative_eq!(profile.offset(0.0), 0.0, epsilon = 1e-15);
            assert_relative_eq!(profile.offset(1.0), 0.0, epsilon = 1e-15);
        }
    }

    /// The join between the two parabolas is invisible in value and in slope.
    ///
    /// The reason for this family over two arcs that merely meet. A slope
    /// discontinuity at the draft would put a kink in every section, and the
    /// lattice would answer it with a load spike wherever a panel edge happened to
    /// land near it — a mesh-dependent force, which is the worst kind.
    #[test]
    fn the_mean_line_is_smooth_at_its_join() {
        let profile = Profile::new(0.15, 0.35).expect("a real section");
        let step = 1e-7;
        let (before, after) = (0.35 - step, 0.35 + step);
        assert_relative_eq!(
            profile.offset(before),
            profile.offset(after),
            epsilon = 1e-13
        );
        assert!(profile.slope(before) > 0.0 && profile.slope(after) < 0.0);
        assert!(
            profile.slope(before).abs() < 1e-6 && profile.slope(after).abs() < 1e-6,
            "the slope does not vanish at the draft"
        );
    }

    /// Analytic slope, entry and exit angles agree with differencing the offset.
    #[test]
    fn the_slope_agrees_with_the_offset_it_differentiates() {
        let profile = Profile::new(0.10, 0.42).expect("a real section");
        let step = 1e-6;
        for along in [0.05_f64, 0.2, 0.41, 0.6, 0.95] {
            let numeric =
                (profile.offset(along + step) - profile.offset(along - step)) / (2.0 * step);
            assert_relative_eq!(profile.slope(along), numeric, max_relative = 1e-6);
        }
        assert_relative_eq!(
            profile.entry_angle().tan(),
            2.0 * 0.10 / 0.42,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            profile.exit_angle().tan(),
            -2.0 * 0.10 / (1.0 - 0.42),
            max_relative = 1e-14
        );
    }

    /// At half chord the family is exactly the parabolic arc.
    ///
    /// `4εx(1-x)`, the section every closed-form sail result is quoted for — so
    /// this is the case where the geometry and the aerodynamic literature are
    /// talking about the same shape, and it must be exact rather than close.
    #[test]
    fn half_draft_is_exactly_the_parabolic_arc() {
        let profile = Profile::new(0.08, 0.5).expect("a real section");
        for step in 0..=200 {
            let along = step as f64 / 200.0;
            let arc = 4.0 * 0.08 * along * (1.0 - along);
            assert_relative_eq!(profile.offset(along), arc, epsilon = 1e-15);
        }
        // And its entry angle is the arc's `4ε`.
        assert_relative_eq!(
            profile.entry_angle().tan(),
            4.0 * 0.08,
            max_relative = 1e-14
        );
    }

    /// Moving the draft forward sharpens the entry and flattens the exit.
    ///
    /// The restriction the module argues for, as a test: entry and exit are not
    /// independent knobs, and the direction they move when the draft moves is what
    /// a sailmaker means by draft position. A model that let them move freely could
    /// express a section with a sharp entry *and* a sharp exit at the same camber,
    /// which is a shape no cloth takes up.
    #[test]
    fn moving_the_draft_forward_sharpens_the_entry() {
        let mut previous_entry = f64::INFINITY;
        let mut previous_exit = 0.0;
        for draft in [0.30_f64, 0.35, 0.40, 0.45, 0.50] {
            let profile = Profile::new(0.12, draft).expect("a real section");
            let entry = profile.entry_angle();
            let exit = profile.exit_angle().abs();
            assert!(
                entry < previous_entry,
                "entry did not soften at draft {draft}"
            );
            assert!(
                exit > previous_exit,
                "exit did not sharpen at draft {draft}"
            );
            previous_entry = entry;
            previous_exit = exit;
        }
    }

    /// Planform area matches its closed form, and the roach contributes `2/π`.
    #[test]
    fn planform_area_matches_its_closed_form() {
        let planform = Planform::new(5.0, 2.0, 0.2, 0.15).expect("a real planform");
        let mut integral = 0.0;
        let steps = 20_000;
        for step in 0..steps {
            let up = (step as f64 + 0.5) / steps as f64;
            integral += planform.chord(up) * planform.luff() / steps as f64;
        }
        assert_relative_eq!(planform.area(), integral, max_relative = 1e-8);

        // The roach term alone, isolated against a straight-leech planform.
        let straight = Planform::new(5.0, 2.0, 0.2, 0.0).expect("a real planform");
        assert_relative_eq!(
            planform.area() - straight.area(),
            5.0 * 0.15 * 2.0 * 2.0 / PI,
            max_relative = 1e-14
        );
    }

    /// The area-weighted mean height is the planform's centroid, and it is a
    /// triangle's third and a rectangle's half exactly.
    ///
    /// Two closed forms that need no source: the centroid of a triangle is a third
    /// of the way up, and a rectangle's is half. If a mean chord had been used as
    /// the weight by mistake, or the height integrated unweighted, the rectangle
    /// would still come out at a half and only the triangle would give it away.
    #[test]
    fn the_mean_height_is_the_planforms_centroid() {
        let triangle = Planform::new(5.0, 2.0, 0.0, 0.0).expect("a triangle");
        assert_relative_eq!(triangle.mean_height(), 1.0 / 3.0, max_relative = 1e-14);

        let rectangle = Planform::new(5.0, 2.0, 2.0, 0.0).expect("a rectangle");
        assert_relative_eq!(rectangle.mean_height(), 0.5, max_relative = 1e-14);

        // And against the integral it stands for, on a planform with every term.
        let planform = windsurf();
        let steps = 200_000;
        let (mut moment, mut area) = (0.0, 0.0);
        for step in 0..steps {
            let up = (step as f64 + 0.5) / steps as f64;
            let chord = planform.chord(up);
            moment += chord * up;
            area += chord;
        }
        assert_relative_eq!(planform.mean_height(), moment / area, max_relative = 1e-9);

        // A sail carries its cloth low, so the balance point is below mid height.
        assert!(planform.mean_height() < 0.45);
    }

    /// Mean incidence is the area-weighted average of the sections' own, and twist
    /// is what moves it.
    ///
    /// Checked against the quadrature it replaces, because the closed form takes a
    /// shortcut that is only valid while incidence is linear in height — which it is
    /// here, and which a later shape family with a curved luff would break silently.
    #[test]
    fn mean_incidence_is_the_area_weighted_average() {
        let flow = Vector3::new(-1.0, 0.0, 0.0);
        for twist in [0.0_f64, 8.0, 20.0] {
            let shape = sail(0.12, 0.40, twist);
            let steps = 100_000;
            let (mut moment, mut area) = (0.0, 0.0);
            for step in 0..steps {
                let up = (step as f64 + 0.5) / steps as f64;
                let chord = shape.planform().chord(up);
                moment += chord * shape.incidence(up, flow);
                area += chord;
            }
            assert_relative_eq!(
                shape.mean_incidence(flow),
                moment / area,
                max_relative = 1e-8,
                epsilon = 1e-12
            );
        }

        // Untwisted, every section is at the wind angle and so is the mean.
        let beta = 14.0_f64.to_radians();
        let angled = Vector3::new(-beta.cos(), beta.sin(), 0.0);
        assert_relative_eq!(
            sail(0.12, 0.40, 0.0).mean_incidence(angled),
            beta,
            max_relative = 1e-12
        );
        // Twisted, it falls - by less than the head does, because the head is where
        // the sail has least area.
        let twisted = sail(0.12, 0.40, 12.0);
        let mean = twisted.mean_incidence(angled);
        assert!(mean < beta);
        assert!(mean > twisted.incidence(1.0, angled));
        assert!(
            beta - mean < 0.5 * 12.0_f64.to_radians(),
            "the mean lost more than half the head's twist"
        );
    }

    /// A cambered sail carries more cloth than its planform area, by the arc length
    /// of its sections.
    ///
    /// The lattice's reference area is surface area and the coefficients the
    /// literature quotes are on planform area. The two differ by a factor this
    /// module states in closed form, and confusing them is a silent few per cent on
    /// every coefficient — five per cent at a sail's camber, which is half the
    /// uncertainty band the whole downwind model inherits.
    ///
    /// Exact on an untapered planform, where the lofted surface is a cylinder and
    /// every panel is planar.
    ///
    /// Tapered, it becomes a *bound* rather than an identity, and a bound is what
    /// gets asserted: the surface between two differently-scaled sections is warped
    /// rather than developable, so it can only hold more area than the cylinder
    /// identity predicts. How much more is the second half of the claim — under one
    /// per cent for a four-to-one taper at twenty per cent camber, which is beyond
    /// any real sail on both counts. So the closed form is usable for converting
    /// between the two coefficient conventions, and this says by how much it can be
    /// trusted instead of hiding the gap inside a tolerance.
    #[test]
    fn the_lofted_surface_exceeds_the_planform_by_its_arc_length() {
        let flat = Profile::new(0.0, 0.4).expect("a flat section");
        assert_relative_eq!(flat.arc_length(), 1.0, epsilon = 1e-15);

        let with_camber = |head: f64, camber: f64| {
            Shape::new(
                Planform::new(5.0, 2.0, head, 0.0).expect("a real planform"),
                Distribution::uniform(camber),
                Distribution::uniform(0.45),
                Distribution::uniform(0.0),
                0.0,
            )
            .expect("a real sail")
        };

        let mut excesses = Vec::new();
        for camber in [0.0_f64, 0.05, 0.12, 0.20] {
            let untapered = with_camber(2.0, camber);
            let surface = untapered.lattice(400, 8).expect("a grid").reference_area();
            let identity = untapered.planform().area() * untapered.profile(0.5).arc_length();
            assert_relative_eq!(surface, identity, max_relative = 1e-5);

            let tapered = with_camber(0.5, camber);
            let surface = tapered.lattice(200, 60).expect("a grid").reference_area();
            let identity = tapered.planform().area() * tapered.profile(0.5).arc_length();
            let excess = surface / identity - 1.0;
            assert!(
                (-1e-12..0.01).contains(&excess),
                "a tapered sail at camber {camber} held {:.3} % more area than the \
                 cylinder identity",
                100.0 * excess
            );
            excesses.push(excess);
        }
        // The warp is a camber effect, so it vanishes on a flat sail and grows.
        assert!(excesses[0].abs() < 1e-12);
        assert!(excesses[3] > excesses[1] && excesses[1] > 0.0);
    }

    /// Twist is recoverable from the geometry, by the definition the measurements
    /// use.
    ///
    /// The angle between the horizontal projections of the foot and head chords.
    /// Pinned because it is the one number this module and the wind-tunnel papers
    /// have to agree on before any force comparison means anything.
    #[test]
    fn twist_is_recoverable_from_the_geometry_the_way_it_is_measured() {
        for degrees in [0.0_f64, 5.0, 12.0, 20.0] {
            let shape = sail(0.12, 0.40, degrees);
            let chord_at = |up: f64| {
                let along = shape.point(1.0, up) - shape.point(0.0, up);
                (-along.y).atan2(-along.x)
            };
            let measured = chord_at(0.0) - chord_at(1.0);
            assert_relative_eq!(
                measured,
                degrees.to_radians(),
                max_relative = 1e-9,
                epsilon = 1e-12
            );
        }
    }

    /// Camber and draft are recoverable from the sampled surface.
    ///
    /// The parameters go in as fractions of chord and come back out of the
    /// three-dimensional points, through a taper, a trim rotation and a twist. This
    /// is the test that the placement arithmetic does not quietly rescale the
    /// section it places.
    #[test]
    fn camber_and_draft_survive_the_trip_through_three_dimensions() {
        let shape = Shape::new(
            windsurf(),
            Distribution::new(0.14, 0.07),
            Distribution::new(0.38, 0.48),
            Distribution::new(0.0, 18.0_f64.to_radians()),
            12.0_f64.to_radians(),
        )
        .expect("a real sail");

        for up in [0.0_f64, 0.25, 0.6, 0.9] {
            let luff = shape.point(0.0, up);
            let chord = shape.point(1.0, up) - luff;
            let length = chord.norm();
            let along = chord / length;

            let mut best = (0.0_f64, 0.0_f64);
            let steps = 4000;
            for step in 0..=steps {
                let fraction = step as f64 / steps as f64;
                let from_luff = shape.point(fraction, up) - luff;
                let offset = from_luff - along * from_luff.dot(&along);
                if offset.norm() > best.0 {
                    best = (offset.norm(), from_luff.dot(&along) / length);
                }
            }
            assert_relative_eq!(
                best.0 / length,
                shape.profile(up).camber(),
                max_relative = 2e-3
            );
            assert_relative_eq!(best.1, shape.profile(up).draft(), max_relative = 5e-3);
        }
    }

    /// The lift slope sits below the reference the measurements are drawn against,
    /// by what a triangular planform costs.
    ///
    /// `dC_L/dα = 2πλ/(λ+2)` with `λ = 3.36` is the band Zhang et al. plot their
    /// data against, and it assumes elliptical loading. This sail is a tapered
    /// triangle with a large roach and a head chord of a tenth of a metre, so its
    /// loading is nothing like elliptical and it must come in below — refined to
    /// twenty by forty-eight panels, at 0.81 of the reference.
    ///
    /// Below is the only defensible direction and it is the same direction
    /// [`crate::vlm`] already establishes for an ellipse at this aspect ratio,
    /// where the gap is ten per cent. Nineteen for a triangle is the planform's
    /// share of it. A lattice that *matched* the reference here would mean the
    /// planform had stopped mattering, which is the one thing this module exists to
    /// make it do.
    #[test]
    fn the_lift_slope_sits_below_the_elliptical_loading_reference() {
        let aspect = windsurf().aspect_ratio();
        assert_relative_eq!(aspect, 3.36, max_relative = 0.01);
        assert_relative_eq!(windsurf().area(), 8.0, max_relative = 0.01);

        let reference = 2.0 * PI * aspect / (aspect + 2.0);
        let flat = sail(0.0, 0.40, 0.0);
        let slope = |panels: (usize, usize)| {
            let low = measure(&flat, 4.0, panels).0;
            let high = measure(&flat, 12.0, panels).0;
            (high - low) / 8.0_f64.to_radians()
        };

        let (coarse, fine, finest) = (slope((6, 16)), slope((14, 34)), slope((20, 48)));
        let ratio = fine / reference;
        assert!(
            (0.75..0.90).contains(&ratio),
            "the lattice gave {ratio:.4} of the reference slope"
        );
        // And it is converging from above rather than wandering.
        assert!(coarse > fine && fine > finest);
        assert!(
            (fine - finest).abs() / reference < 0.01,
            "the slope is still moving by more than a per cent per refinement"
        );
    }

    /// The zero-lift angle is the arc's, and its deficit is second order in camber.
    ///
    /// A parabolic-arc section lifts at zero incidence and stops lifting at
    /// `α₀ = -2ε`. That value is *aspect-ratio independent* in lifting-line theory
    /// for an untwisted wing of uniform section, so unlike the lift slope above
    /// there is no planform excuse here: the number has to come out.
    ///
    /// It does, to within a deficit that grows quadratically — 0.7 % at four per
    /// cent camber, 4.2 % at twelve. That is the same second-order term
    /// [`crate::vlm`] measured directly on camber lift, arriving by a completely
    /// different route: there by extrapolating a lattice in panel count against
    /// `4πε`, here by hunting a three-dimensional sail's zero crossing. Two
    /// independent measurements of one discarded term, and the agreement is what
    /// says the section family is wired into the geometry the way the section
    /// family thinks it is.
    #[test]
    fn the_zero_lift_angle_is_the_arcs_and_its_deficit_is_second_order() {
        // Bracket the theoretical value and interpolate: the lift curve's curvature
        // over four degrees is far below the deficit being measured.
        let zero_lift = |camber: f64| {
            let shape = sail(camber, 0.5, 0.0);
            let theory = -(2.0 * camber).to_degrees();
            let (low, high) = (theory - 2.0, theory + 2.0);
            let (at_low, at_high) = (
                measure(&shape, low, (16, 40)).0,
                measure(&shape, high, (16, 40)).0,
            );
            assert!(
                at_low < 0.0 && at_high > 0.0,
                "the bracket does not bracket"
            );
            low - at_low * (high - low) / (at_high - at_low)
        };

        let mut deficits = Vec::new();
        for camber in [0.04_f64, 0.08, 0.12] {
            let theory = -(2.0 * camber).to_degrees();
            let ratio = zero_lift(camber) / theory;
            assert!(
                (0.94..1.0).contains(&ratio),
                "at camber {camber} the zero-lift angle was {ratio:.4} of the arc's"
            );
            deficits.push(1.0 - ratio);
        }

        // Tripling the camber grows the deficit by about nine if it is quadratic,
        // by three if it is linear. The range excludes linear.
        let growth = deficits[2] / deficits[0];
        assert!(
            (5.0..14.0).contains(&growth),
            "the deficit grew by {growth:.2} from four to twelve per cent camber, \
             which is not second order"
        );
    }

    /// The drag is induced drag, just above the elliptic minimum for this aspect
    /// ratio, and the excess grows with camber.
    ///
    /// There is no viscous drag in a lattice, so `C_D` here is entirely induced and
    /// `C_L²/πλ` is its theoretical floor. Being *above* the floor is the physical
    /// content: the floor belongs to elliptical loading and nothing else attains
    /// it. Three per cent above for a flat sail, nine at twelve per cent camber —
    /// because camber loads the sail further from elliptical, not because the
    /// arithmetic drifted.
    ///
    /// The measured excess understates the real one. [`crate::vlm`] establishes
    /// that near-field induced drag comes in two to three per cent low, and that
    /// bias is inside these numbers.
    #[test]
    fn the_drag_is_the_induced_minimum_for_this_aspect_ratio() {
        let aspect = windsurf().aspect_ratio();
        let mut excesses = Vec::new();
        for camber in [0.0_f64, 0.06, 0.12] {
            let shape = sail(camber, 0.40, 0.0);
            let mut worst: f64 = 0.0;
            for wind in [4.0_f64, 10.0, 16.0] {
                let (lift, drag, _) = measure(&shape, wind, (12, 30));
                let minimum = lift * lift / (PI * aspect);
                let ratio = drag / minimum;
                assert!(
                    (0.98..1.15).contains(&ratio),
                    "at camber {camber} and {wind} deg the drag was {ratio:.4} of the minimum"
                );
                worst = worst.max(ratio);
            }
            excesses.push(worst);
        }
        assert!(
            excesses[2] > excesses[0],
            "camber did not push the loading further from elliptical: {excesses:?}"
        );
    }

    /// Twist reduces lift and lowers the centre of effort.
    ///
    /// Both measured on a full-scale sail with its shape recorded at the same time
    /// [Zhang et al. 2025]. Twist eases the head, shedding load from the highest
    /// part of the sail — so total lift falls *and* what is left sits lower.
    ///
    /// The second half is why twist is a depowering control rather than merely a
    /// lift-losing one, and it is the thing a coefficient model cannot produce at
    /// all: with lift and centre of effort as separate table lookups there is
    /// nothing to make them move together. Twenty degrees of twist here costs 37 %
    /// of the lift and 13 % of the arm it acts on.
    #[test]
    fn twist_reduces_lift_and_lowers_the_centre_of_effort() {
        let mut previous = (f64::INFINITY, f64::INFINITY);
        for degrees in [0.0_f64, 6.0, 12.0, 20.0] {
            let (lift, _, height) = measure(&sail(0.12, 0.40, degrees), 14.0, (8, 20));
            assert!(
                lift < previous.0,
                "twist of {degrees} deg did not reduce lift: {lift:.4} after {:.4}",
                previous.0
            );
            assert!(
                height < previous.1,
                "twist of {degrees} deg did not lower the effort: {height:.3} m after {:.3} m",
                previous.1
            );
            previous = (lift, height);
        }

        // And the effect is worth having on both counts.
        let (upright, _, high) = measure(&sail(0.12, 0.40, 0.0), 14.0, (8, 20));
        assert!(
            previous.0 / upright < 0.75,
            "twenty degrees of twist cost only {:.1} % of the lift",
            100.0 * (1.0 - previous.0 / upright)
        );
        assert!(
            previous.1 / high < 0.95,
            "twenty degrees of twist lowered the effort by only {:.1} %",
            100.0 * (1.0 - previous.1 / high)
        );
    }

    /// A twisted head loses incidence exactly, and aligns with the wind when the
    /// twist reaches the angle of attack.
    ///
    /// Zhang et al.'s figure 6 plots twist against angle of attack with the
    /// diagonal marked: on it the head is aligned with the wind, and every measured
    /// point fell below it — the rig twisted off without ever back-winding its own
    /// head. This pins the geometry that statement is made in, so that the
    /// statement can be checked rather than assumed: incidence is `α - twist`, to
    /// the last bit, and the diagonal is where it reaches zero.
    #[test]
    fn a_twisted_head_loses_incidence_exactly_as_measured() {
        let attack = 16.0_f64;
        let beta = attack.to_radians();
        let flow = Vector3::new(-beta.cos(), beta.sin(), 0.0);

        // On the diagonal, the head is aligned with the flow.
        assert_relative_eq!(
            sail(0.12, 0.4, attack).incidence(1.0, flow),
            0.0,
            epsilon = 1e-12
        );

        // Below it, positive everywhere and falling linearly with height.
        let shape = sail(0.12, 0.4, 10.0);
        let mut previous = f64::INFINITY;
        for step in 0..=20 {
            let up = step as f64 / 20.0;
            let incidence = shape.incidence(up, flow);
            assert!(incidence > 0.0, "the section at {up} was back-winded");
            assert!(incidence < previous, "incidence did not fall with height");
            assert_relative_eq!(
                incidence,
                (attack - 10.0 * up).to_radians(),
                epsilon = 1e-12
            );
            previous = incidence;
        }

        // Above it the geometry can shadow the head, which is a shape a rig can be
        // trimmed into even though the measured rigs never reached it.
        assert!(sail(0.12, 0.4, 24.0).incidence(1.0, flow) < 0.0);
    }

    /// Camber alone increases lift, which is the *opposite* of what the full-scale
    /// measurements report — and the discrepancy is the point.
    ///
    /// Zhang et al. measured the high-camber setting giving *less* lift than the
    /// low-camber one, and explained it with their own shape data: easing the
    /// outhaul increased camber and twist together, and the twist won. This module
    /// holds the two apart, so here camber must act the way a section acts.
    ///
    /// Reproducing their result therefore needs a control mapping that couples the
    /// two, and this test is the evidence that the coupling has to live *there*: if
    /// the geometry already lost lift with camber, a coupling on top would
    /// double-count it and nothing would reveal the error.
    ///
    /// Nothing here asserts *how much* twist an eased outhaul brings. A pair of
    /// camber-and-twist settings chosen to reproduce their ordering would be
    /// asserting a coupling strength no published measurement pins down — which is
    /// the calibration this module declines to invent, so it does not get smuggled
    /// in as a test either.
    #[test]
    fn camber_alone_increases_lift() {
        let mut previous = 0.0;
        for camber in [0.0_f64, 0.02, 0.06, 0.10, 0.14] {
            let (lift, _, _) = measure(&sail(camber, 0.40, 0.0), 8.0, (8, 20));
            assert!(
                lift > previous,
                "camber {camber} lost lift: {lift:.4} after {previous:.4}"
            );
            previous = lift;
        }
    }

    /// Sections that no sail can fly are refused.
    #[test]
    fn a_degenerate_shape_is_refused() {
        assert!(Profile::new(-0.01, 0.4).is_none());
        assert!(Profile::new(0.1, 0.0).is_none());
        assert!(Profile::new(0.1, 1.0).is_none());
        assert!(Profile::new(f64::NAN, 0.4).is_none());
        assert!(Planform::new(0.0, 2.0, 0.2, 0.1).is_none());
        assert!(Planform::new(5.0, 0.0, 0.2, 0.1).is_none());
        assert!(Planform::new(5.0, 2.0, -0.1, 0.1).is_none());
        assert!(Planform::new(f64::INFINITY, 2.0, 0.2, 0.1).is_none());

        let planform = windsurf();
        let ok = Distribution::uniform(0.4);
        assert!(Shape::new(planform, Distribution::uniform(0.1), ok, ok, f64::NAN).is_none());
        // A draft distribution that reaches the leading edge at the head.
        assert!(Shape::new(
            planform,
            Distribution::uniform(0.1),
            Distribution::new(0.4, 0.0),
            Distribution::uniform(0.0),
            0.0
        )
        .is_none());

        // A triangular sail with no headboard is fine, and has area: the top panels
        // become triangles, and a triangle's normal comes off its diagonals like any
        // other panel's.
        let pointed = Planform::new(5.0, 2.0, 0.0, 0.0).expect("a triangle");
        let shape = Shape::new(
            pointed,
            Distribution::uniform(0.1),
            Distribution::uniform(0.4),
            Distribution::uniform(0.0),
            0.1,
        )
        .expect("a triangular sail");
        let lattice = shape.lattice(4, 8).expect("a grid");
        assert!(lattice.reference_area() > 0.0);
        assert!(Solver::new(lattice, Vector3::new(-1.0, 0.0, 0.0)).is_some());
    }

    /// A plausible mainsail's control travel, for the mapping tests.
    fn response() -> Response {
        Response::new(
            (25.0_f64.to_radians(), 4.0_f64.to_radians()),
            (22.0_f64.to_radians(), 3.0_f64.to_radians()),
            (0.16, 0.08),
            0.55,
            (0.48, 0.34),
        )
        .expect("a real sail's travel")
    }

    /// The declared endpoints come back exactly at the ends of the travel.
    ///
    /// The contract with whoever fills in a boat file: the two numbers they wrote
    /// are the two shapes the sail actually reaches, not the ends of some internal
    /// range the model rescales.
    #[test]
    fn the_declared_endpoints_are_reached_exactly() {
        let planform = windsurf();
        let response = response();

        let eased = response.shape(planform, Controls::EASED);
        assert_relative_eq!(
            eased.chord_angle(0.0),
            25.0_f64.to_radians(),
            max_relative = 1e-14
        );
        assert_relative_eq!(eased.profile(0.0).camber(), 0.16, max_relative = 1e-14);
        assert_relative_eq!(eased.profile(0.0).draft(), 0.48, max_relative = 1e-14);
        assert_relative_eq!(
            eased.chord_angle(1.0) - eased.chord_angle(0.0),
            22.0_f64.to_radians(),
            max_relative = 1e-14
        );

        let hard = response.shape(planform, Controls::HARD);
        assert_relative_eq!(
            hard.chord_angle(0.0),
            4.0_f64.to_radians(),
            max_relative = 1e-14
        );
        assert_relative_eq!(hard.profile(0.0).camber(), 0.08, max_relative = 1e-14);
        assert_relative_eq!(hard.profile(0.0).draft(), 0.34, max_relative = 1e-14);
        assert_relative_eq!(
            hard.chord_angle(1.0) - hard.chord_angle(0.0),
            3.0_f64.to_radians(),
            max_relative = 1e-14
        );

        // And the head's camber is the declared fraction of the foot's, at both.
        assert_relative_eq!(
            eased.profile(1.0).camber() / eased.profile(0.0).camber(),
            0.55,
            max_relative = 1e-14
        );
    }

    /// The leech has one tension, and the vang takes over what the sheet gives up.
    ///
    /// `max(sheet, vang)` rather than a sum, and this is the test that says why the
    /// distinction is physical rather than notational: easing the sheet under a hard
    /// vang must not let the head twist off, because the vang is still holding the
    /// leech. A model that summed would make the pair of controls twice as powerful
    /// as either, and easing the sheet would open a leech the vang has hold of.
    #[test]
    fn the_leech_has_one_tension_and_the_vang_holds_it() {
        let hard_sheet = Controls {
            sheet: 1.0,
            vang: 0.0,
            ..Controls::EASED
        };
        let hard_vang = Controls {
            sheet: 0.0,
            vang: 1.0,
            ..Controls::EASED
        };
        let both = Controls {
            sheet: 1.0,
            vang: 1.0,
            ..Controls::EASED
        };
        assert_eq!(hard_sheet.leech_tension(), 1.0);
        assert_eq!(hard_vang.leech_tension(), 1.0);
        // Both hard is not tighter than either: there is no more leech to tension.
        assert_eq!(both.leech_tension(), 1.0);
        assert_eq!(Controls::EASED.leech_tension(), 0.0);

        // Easing the sheet under a hard vang leaves the twist alone.
        let planform = windsurf();
        let response = response();
        let twist_of = |controls: Controls| {
            let shape = response.shape(planform, controls);
            shape.chord_angle(1.0) - shape.chord_angle(0.0)
        };
        assert_relative_eq!(
            twist_of(hard_sheet),
            twist_of(hard_vang),
            max_relative = 1e-14
        );
        assert_relative_eq!(twist_of(both), twist_of(hard_vang), max_relative = 1e-14);
        assert!(twist_of(Controls::EASED) > twist_of(hard_vang));
    }

    /// Traveller down with the sheet hard opens the angle and closes the leech.
    ///
    /// The classic depowering trim, and it comes out because the two quantities are
    /// computed by different rules — a product for the angle, a maximum for the
    /// leech. A single "sheet in / sheet out" scalar could not express it, which is
    /// the reason the two controls are separate here.
    #[test]
    fn traveller_down_with_a_hard_sheet_opens_the_angle_and_shuts_the_leech() {
        let planform = windsurf();
        let response = response();

        let depowered = Controls {
            sheet: 1.0,
            traveller: 0.0,
            ..Controls::EASED
        };
        let powered = Controls {
            sheet: 1.0,
            traveller: 1.0,
            ..Controls::EASED
        };
        let (open, closed) = (
            response.shape(planform, depowered),
            response.shape(planform, powered),
        );

        // The angle is wide open - as wide as with no sheet at all.
        assert_relative_eq!(
            open.chord_angle(0.0),
            25.0_f64.to_radians(),
            max_relative = 1e-14
        );
        assert!(closed.chord_angle(0.0) < open.chord_angle(0.0));
        // And yet the leech is fully tensioned in both, so the twist is the same.
        let twist = |shape: &Shape| shape.chord_angle(1.0) - shape.chord_angle(0.0);
        assert_relative_eq!(twist(&open), twist(&closed), max_relative = 1e-14);
        assert_relative_eq!(twist(&open), 3.0_f64.to_radians(), max_relative = 1e-14);
    }

    /// Each control moves its own parameter, monotonically and the right way.
    ///
    /// The qualitative derivatives are what the sail-trim literature actually
    /// supplies, so they are what gets pinned: an outhaul flattens, luff tension
    /// moves the draft forward, leech tension removes twist, and sheeting in closes
    /// the angle. A boat file with a sign error is refused at load, and this is the
    /// test that the mapping itself does not reintroduce one.
    #[test]
    fn every_control_moves_its_own_parameter_the_right_way() {
        let planform = windsurf();
        let response = response();
        let at = |controls: Controls| response.shape(planform, controls);

        let mut previous = (f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY);
        for step in 0..=10 {
            let value = step as f64 / 10.0;
            let camber = at(Controls {
                outhaul: value,
                ..Controls::EASED
            })
            .profile(0.0)
            .camber();
            let draft = at(Controls {
                cunningham: value,
                ..Controls::EASED
            })
            .profile(0.0)
            .draft();
            let shape = at(Controls {
                vang: value,
                ..Controls::EASED
            });
            let twist = shape.chord_angle(1.0) - shape.chord_angle(0.0);
            let angle = at(Controls {
                sheet: value,
                traveller: 1.0,
                ..Controls::EASED
            })
            .chord_angle(0.0);

            if step > 0 {
                assert!(camber < previous.0, "the outhaul did not flatten");
                assert!(
                    draft < previous.1,
                    "luff tension did not move the draft forward"
                );
                assert!(twist < previous.2, "leech tension did not remove twist");
                assert!(angle < previous.3, "sheeting in did not close the angle");
            }
            previous = (camber, draft, twist, angle);
        }
    }

    /// Controls outside their range are clamped, and a trim never fails.
    ///
    /// A slider or a solver will hand over 1.0000001, and a `Response` validated at
    /// construction has no interpolation that can leave the flyable set — so
    /// [`Response::shape`] returns a shape rather than an option. A trim setting
    /// that could break the model would be worse than one that cannot express
    /// something.
    #[test]
    fn controls_out_of_range_are_clamped_and_a_trim_never_fails() {
        let planform = windsurf();
        let response = response();
        let wild = Controls {
            sheet: 4.0,
            traveller: -2.0,
            vang: f64::NAN,
            outhaul: 1e9,
            cunningham: f64::NEG_INFINITY,
        };
        let clamped = wild.clamped();
        assert_eq!(clamped.sheet, 1.0);
        assert_eq!(clamped.traveller, 0.0);
        assert_eq!(clamped.vang, 0.0);
        assert_eq!(clamped.outhaul, 1.0);
        assert_eq!(clamped.cunningham, 0.0);

        // And the shape it produces is a real one that solves.
        let shape = response.shape(planform, wild);
        assert!(shape.profile(0.5).camber() >= 0.0);
        let lattice = shape.lattice(6, 14).expect("a grid");
        assert!(Solver::new(lattice, Vector3::new(-1.0, 0.0, 0.0)).is_some());

        // Sweeping the whole control cube produces nothing unflyable.
        for i in 0..3_usize.pow(5) {
            let digit = |place: u32| ((i / 3_usize.pow(place)) % 3) as f64 / 2.0;
            let shape = response.shape(
                planform,
                Controls {
                    sheet: digit(0),
                    traveller: digit(1),
                    vang: digit(2),
                    outhaul: digit(3),
                    cunningham: digit(4),
                },
            );
            let profile = shape.profile(0.7);
            assert!(profile.camber() >= 0.0 && (0.0..1.0).contains(&profile.draft()));
        }
    }

    /// A boat file whose control travel runs backwards is refused.
    #[test]
    fn a_response_with_a_sign_error_is_refused() {
        let good = (0.16, 0.08);
        let angle = (0.4, 0.1);
        let twist = (0.4, 0.05);
        let draft = (0.48, 0.34);
        assert!(Response::new(angle, twist, good, 0.55, draft).is_some());
        // An outhaul that adds camber.
        assert!(Response::new(angle, twist, (0.08, 0.16), 0.55, draft).is_none());
        // Leech tension that adds twist.
        assert!(Response::new(angle, (0.05, 0.4), good, 0.55, draft).is_none());
        // Luff tension that moves the draft aft.
        assert!(Response::new(angle, twist, good, 0.55, (0.34, 0.48)).is_none());
        // A head fuller than its foot, or a section no sail can fly.
        assert!(Response::new(angle, twist, good, 1.4, draft).is_none());
        assert!(Response::new(angle, twist, good, 0.55, (0.48, 0.0)).is_none());
        assert!(Response::new(angle, twist, (0.16, -0.01), 0.55, draft).is_none());
        assert!(Response::new(angle, twist, good, 0.55, (f64::NAN, 0.34)).is_none());

        // The coupling takes a sign, not a guess.
        let response = Response::new(angle, twist, good, 0.55, draft).expect("a real travel");
        assert!(response.with_camber_twist_coupling(-0.1).is_none());
        assert!(response.with_camber_twist_coupling(f64::NAN).is_none());
        assert!(response.with_camber_twist_coupling(0.0).is_some());
    }

    /// The coupling adds twist only where the sail is full, and defaults to none.
    #[test]
    fn the_coupling_acts_only_on_a_full_sail() {
        let planform = windsurf();
        let coupled = response()
            .with_camber_twist_coupling(12.0_f64.to_radians())
            .expect("a declared coupling");
        let twist_of = |response: &Response, outhaul: f64| {
            let shape = response.shape(
                planform,
                Controls {
                    outhaul,
                    ..Controls::EASED
                },
            );
            shape.chord_angle(1.0) - shape.chord_angle(0.0)
        };

        // Outhaul hard: the sail is at its flattest, so there is nothing to couple.
        assert_relative_eq!(
            twist_of(&coupled, 1.0),
            twist_of(&response(), 1.0),
            max_relative = 1e-14
        );
        // Outhaul off: the whole declared coupling.
        assert_relative_eq!(
            twist_of(&coupled, 0.0) - twist_of(&response(), 0.0),
            12.0_f64.to_radians(),
            max_relative = 1e-12
        );
        // Monotone in between, and zero by default everywhere.
        for step in 0..=8 {
            let outhaul = step as f64 / 8.0;
            assert!(twist_of(&coupled, outhaul) >= twist_of(&response(), outhaul));
        }
    }

    /// With a coupling declared, the measured paradox appears: a fuller sail makes
    /// less lift.
    ///
    /// This closes the loop that [`Shape`]'s own tests deliberately left open.
    /// Zhang et al. measured the high-camber setting producing *less* lift, which is
    /// backwards for a section, and attributed it to the twist that came with the
    /// camber. The geometry holds the two apart so that camber alone increases lift
    /// there; here, with the coupling declared, the ordering reverses.
    ///
    /// What this test asserts is that the model can *express* the observation, not
    /// that any particular sail has that coupling. The coupling used is an input.
    ///
    /// Worth recording, as a property of the model rather than of any sail: the
    /// coupling needed to flip the sign on this planform is about sixteen degrees of
    /// head twist across the camber range. That is large, and it is the number a
    /// campaign would have to confirm or refute before the coupling could be
    /// declared for a real boat.
    #[test]
    fn a_declared_coupling_reproduces_the_measured_paradox() {
        let planform = windsurf();
        let full = Controls {
            outhaul: 0.0,
            ..Controls::EASED
        };
        let flat = Controls {
            outhaul: 1.0,
            ..Controls::EASED
        };
        let lift = |response: &Response, controls: Controls| {
            measure(&response.shape(planform, controls), 14.0, (8, 20)).0
        };

        // Uncoupled, a fuller sail makes more lift - the section's own behaviour.
        let plain = response();
        assert!(lift(&plain, full) > lift(&plain, flat));

        // Coupled hard enough, it makes less, which is what was measured.
        let coupled = plain
            .with_camber_twist_coupling(18.0_f64.to_radians())
            .expect("a declared coupling");
        assert!(
            lift(&coupled, full) < lift(&coupled, flat),
            "the declared coupling did not reverse the ordering: {:.4} against {:.4}",
            lift(&coupled, full),
            lift(&coupled, flat)
        );

        // And a small coupling does not: the reversal needs a real amount of twist,
        // which is what makes the magnitude a question for a campaign.
        let slight = plain
            .with_camber_twist_coupling(4.0_f64.to_radians())
            .expect("a declared coupling");
        assert!(lift(&slight, full) > lift(&slight, flat));
    }
}
