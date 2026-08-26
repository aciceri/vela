//! Balance: where the rig has to sit relative to the keel.
//!
//! # Provenance
//!
//! Larsson, Eliasson & Orych, *Principles of Yacht Design*, 5th ed., Chapter 9 —
//! Fig 9.2 for the centre of lateral resistance, the geometric method of Fig 9.3
//! for the centre of effort of the sails, and the recommended leads of the
//! *Lead* section.
//!
//! # Why this module exists
//!
//! Every force this engine computes acts at a point, and for the sails and the
//! appendages the **longitudinal** part of that point is not in the data. The
//! book's published particulars for the YD-41 give the sail plan's dimensions
//! and the keel's planform but not where either sits along the hull, so yaw has
//! been restrained from the beginning: not because the physics is missing but
//! because the arm is.
//!
//! This module is how that gap gets closed without inventing a number. Chapter 9
//! is a design chapter: it says where the rig goes relative to the keel, and it
//! says it in a form that can be computed. Given the one number a boat file must
//! supply — where the keel is — everything else follows from the source, and the
//! result can be checked against the source's own published answer for its own
//! boat.
//!
//! # What is derived and what is required
//!
//! **Required**: the longitudinal position of the keel. It is a design decision,
//! not a consequence of anything else, and no rule in the book produces it.
//!
//! **Derived**: the centre of lateral resistance, from the keel's planform and
//! that position. The centre of effort of the sails, from the rig dimensions,
//! relative to the mast. And therefore the mast position, from the lead.
//!
//! **Checked**: the lead that results, against the band the source recommends
//! for the rig type. The YD-41 is the test case, and the source publishes its
//! answer — 2.2 % of the waterline length — which makes this one of the few
//! parts of the engine with an exact external number to hit.

use std::ops::RangeInclusive;

/// A keel extended up to the waterline, as Fig 9.2 draws it.
///
/// The planform this engine carries elsewhere has its root at the *bottom of the
/// canoe body*; Fig 9.2's construction runs to the *waterline*, so the depth
/// here is the total draft and the quarter-chord point is the one at the
/// waterline, where the extended leading and trailing edges reach it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtendedKeel {
    /// Longitudinal position of the quarter-chord point at the waterline, m,
    /// measured forward from the aft perpendicular in the file frame.
    ///
    /// This is the number a boat file has to supply. Everything else in this
    /// module is a consequence.
    pub quarter_chord_at_waterline: f64,
    /// Total draft, m: canoe body plus keel.
    pub total_draft: f64,
    /// Quarter-chord sweep, radians, positive aft.
    pub sweep: f64,
}

impl ExtendedKeel {
    /// The centre of lateral resistance, as `(x, depth below waterline)` in m.
    ///
    /// Fig 9.2 in one sentence: *"CLR is easily found by connecting the points at
    /// 25 % of the local chord at the waterline and at the tip of the keel by a
    /// straight line, and finding the point at 45 % of the draft on this line."*
    ///
    /// The line joining those two points is the quarter-chord line, which is what
    /// [`ExtendedKeel::sweep`] is the angle of — so the construction reduces to
    /// walking 45 % of the draft down that line. Doing it this way rather than
    /// interpolating two separately computed points is not a shortcut; it is the
    /// same line, and it means the sweep convention here cannot drift from the
    /// one the lift model uses.
    ///
    /// The rudder and the forebody are absent by the source's instruction, not by
    /// omission: *"for most fin-keel yachts the effect of the rudder and the
    /// forebody cancel each other reasonably well, so as a first approximation
    /// they may both be neglected."* That cancellation is the reason the method is
    /// restricted to fin keels, and [`Balance::from_lead`] carries the warning.
    #[must_use]
    pub fn centre_of_lateral_resistance(&self) -> (f64, f64) {
        const DEPTH_FRACTION: f64 = 0.45;
        let depth = DEPTH_FRACTION * self.total_draft;
        (
            self.quarter_chord_at_waterline - depth * self.sweep.tan(),
            depth,
        )
    }
}

/// A masthead or fractional sloop, which is what decides the recommended lead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RigType {
    /// The foretriangle reaches essentially the top of the mast.
    MastheadSloop,
    /// The foretriangle stops short of it.
    FractionalSloop,
}

impl RigType {
    /// Classifies a rig by how far up the mast the foretriangle reaches.
    ///
    /// The source names the two types and gives each a lead band, but does not
    /// define the boundary, so the threshold here is stated rather than sourced:
    /// a foretriangle reaching 95 % of the mast is called masthead. The YD-41 is
    /// 87 % and is comfortably fractional either way, so nothing in the test
    /// suite depends on where exactly the line is drawn.
    #[must_use]
    pub fn classify(foretriangle_height: f64, mast_height: f64) -> Self {
        if mast_height > 0.0 && foretriangle_height / mast_height >= 0.95 {
            Self::MastheadSloop
        } else {
            Self::FractionalSloop
        }
    }

    /// The lead the source recommends for this rig on a fin keel, as a fraction
    /// of waterline length.
    ///
    /// From the *Lead* section, for the extended keel method: masthead sloops
    /// 5–9 %, fractional sloops 2–6 %. The long-keel bands (12–16 % and so on)
    /// are deliberately absent — they belong to the geometric CLR, which this
    /// module does not compute, and mixing a lead band with the wrong CLR method
    /// is exactly the mistake the source warns about.
    #[must_use]
    pub fn recommended_lead(self) -> RangeInclusive<f64> {
        match self {
            Self::MastheadSloop => 0.05..=0.09,
            Self::FractionalSloop => 0.02..=0.06,
        }
    }
}

/// The sail plan's own geometry, independent of where the mast is.
///
/// The fore triangle and the main triangle are both fixed relative to the mast
/// by the rig dimensions, so the centre of effort's *offset* from the mast is
/// known before the mast has a position. That is what breaks the circularity:
/// the offset comes from the rig, the lead comes from the source, and together
/// they place the mast against the keel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SailPlan {
    /// `I` — foretriangle height above the sheer, m.
    pub foretriangle_height: f64,
    /// `J` — foretriangle base, m, measured forward from the mast.
    pub foretriangle_base: f64,
    /// `P` — mainsail hoist, m.
    pub main_hoist: f64,
    /// `E` — mainsail foot, m, measured aft from the mast.
    pub main_foot: f64,
    /// `BAD` — boom height above the sheer, m.
    pub boom_above_sheer: f64,
}

impl SailPlan {
    /// Area of the fore triangle, m².
    #[must_use]
    pub fn foretriangle_area(&self) -> f64 {
        0.5 * self.foretriangle_height * self.foretriangle_base
    }

    /// Area of the main triangle, m².
    #[must_use]
    pub fn mainsail_area(&self) -> f64 {
        0.5 * self.main_hoist * self.main_foot
    }

    /// The centre of effort as `(distance forward of the mast, height above the
    /// sheer)`, in m.
    ///
    /// The geometric method of Fig 9.3: each triangle's centre is its centroid,
    /// and the two are combined along the line joining them in proportion to
    /// their areas — which is the area-weighted mean of the centroids.
    ///
    /// The fore triangle's centroid sits `J/3` forward of the mast and `I/3`
    /// above the sheer; the main's sits `E/3` aft and `BAD + P/3` above. The
    /// forward result is often *negative* for a fractional rig, because a big
    /// mainsail pulls the combined centre aft of the mast — 0.19 m aft for the
    /// YD-41 — and a sign convention that hid that would be worse than useless.
    #[must_use]
    pub fn centre_of_effort_from_mast(&self) -> (f64, f64) {
        let fore = self.foretriangle_area();
        let main = self.mainsail_area();
        let total = fore + main;
        if total <= 0.0 {
            return (0.0, 0.0);
        }
        let forward =
            (fore * (self.foretriangle_base / 3.0) - main * (self.main_foot / 3.0)) / total;
        let height = (fore * (self.foretriangle_height / 3.0)
            + main * (self.boom_above_sheer + self.main_hoist / 3.0))
            / total;
        (forward, height)
    }
}

/// A resolved longitudinal layout: where the keel is, where the mast goes, and
/// what lead that gives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Balance {
    /// Longitudinal position of the centre of lateral resistance, m.
    pub lateral_resistance_at: f64,
    /// Depth of that centre below the waterline, m.
    pub lateral_resistance_depth: f64,
    /// Longitudinal position of the mast, m.
    pub mast_at: f64,
    /// Longitudinal position of the sails' centre of effort, m.
    pub centre_of_effort_at: f64,
    /// Height of that centre above the sheer, m.
    pub centre_of_effort_height: f64,
    /// Lead as a fraction of waterline length, positive with the sails forward.
    pub lead: f64,
    /// Whether that lead falls in the band the source recommends.
    pub lead_is_recommended: bool,
}

impl Balance {
    /// Places the mast to achieve a wanted lead.
    ///
    /// `lead` is a fraction of `waterline_length`, positive with the centre of
    /// effort forward of the centre of lateral resistance — which the source says
    /// it always is: *"In all the methods used, CE is in front of CLR."*
    ///
    /// This is the direction a designer works in, and the direction that fills in
    /// a boat file: the keel's position is a decision, the lead is read off the
    /// source's band for the rig type, and the mast follows.
    #[must_use]
    pub fn from_lead(
        keel: &ExtendedKeel,
        plan: &SailPlan,
        rig: RigType,
        waterline_length: f64,
        lead: f64,
    ) -> Self {
        let (lateral_resistance_at, lateral_resistance_depth) = keel.centre_of_lateral_resistance();
        let (effort_forward_of_mast, centre_of_effort_height) = plan.centre_of_effort_from_mast();
        let centre_of_effort_at = lateral_resistance_at + lead * waterline_length;
        Self {
            lateral_resistance_at,
            lateral_resistance_depth,
            mast_at: centre_of_effort_at - effort_forward_of_mast,
            centre_of_effort_at,
            centre_of_effort_height,
            lead,
            lead_is_recommended: rig.recommended_lead().contains(&lead),
        }
    }

    /// Measures the lead of a layout that already has a mast position.
    ///
    /// The inverse of [`Balance::from_lead`], for checking a boat file rather
    /// than filling one in.
    #[must_use]
    pub fn from_positions(
        keel: &ExtendedKeel,
        plan: &SailPlan,
        rig: RigType,
        waterline_length: f64,
        mast_at: f64,
    ) -> Self {
        let (lateral_resistance_at, lateral_resistance_depth) = keel.centre_of_lateral_resistance();
        let (effort_forward_of_mast, centre_of_effort_height) = plan.centre_of_effort_from_mast();
        let centre_of_effort_at = mast_at + effort_forward_of_mast;
        let lead = if waterline_length > 0.0 {
            (centre_of_effort_at - lateral_resistance_at) / waterline_length
        } else {
            0.0
        };
        Self {
            lateral_resistance_at,
            lateral_resistance_depth,
            mast_at,
            centre_of_effort_at,
            centre_of_effort_height,
            lead,
            lead_is_recommended: rig.recommended_lead().contains(&lead),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// The YD-41's rig, from the boat file, which is Appendix 1 of the source.
    fn yd41_plan() -> SailPlan {
        SailPlan {
            foretriangle_height: 16.20,
            foretriangle_base: 5.10,
            main_hoist: 16.70,
            main_foot: 5.60,
            boom_above_sheer: 1.70,
        }
    }

    /// Its keel, with the quarter chord at the waterline put at midships — a
    /// placeholder for the one number the source does not publish. Nothing that
    /// follows depends on it except by a translation, which is the point.
    fn yd41_keel() -> ExtendedKeel {
        ExtendedKeel {
            quarter_chord_at_waterline: 5.95,
            total_draft: 2.30,
            sweep: 5.5_f64.to_radians(),
        }
    }

    const WATERLINE: f64 = 11.90;

    /// The source publishes this boat's rig type implicitly, and the
    /// classification has to agree.
    ///
    /// `I / EHM` is 16.20 / 18.60 = 0.87, so the foretriangle stops well short of
    /// the masthead. That matters because it selects the lead band, and the
    /// source's own published lead for this boat — 2.2 % — falls inside the
    /// fractional band (2–6 %) and *outside* the masthead one (5–9 %). The two
    /// statements corroborate each other, which is a check worth having: if the
    /// classification were wrong, the source would appear to contradict itself.
    #[test]
    fn the_yd41_is_a_fractional_rig_and_its_published_lead_agrees() {
        let rig = RigType::classify(16.20, 18.60);
        assert_eq!(rig, RigType::FractionalSloop);
        assert!(
            rig.recommended_lead().contains(&0.022),
            "the source's own 2.2 % should sit in the band it recommends"
        );
        assert!(
            !RigType::MastheadSloop.recommended_lead().contains(&0.022),
            "and outside the other one, or the classification would not matter"
        );
    }

    /// The centre of effort of a fractional rig sits *aft* of the mast.
    ///
    /// Worth pinning because it is counter-intuitive and because a sign error
    /// here would move the mast by twice the offset. For the YD-41 the mainsail
    /// is the larger triangle (46.8 m² against 41.3 m²) and its centroid is
    /// 1.87 m aft of the mast against the foretriangle's 1.70 m forward, so the
    /// combination lands behind the mast.
    #[test]
    fn a_big_mainsail_pulls_the_centre_of_effort_aft_of_the_mast() {
        let plan = yd41_plan();
        assert!(plan.mainsail_area() > plan.foretriangle_area());

        let (forward, height) = plan.centre_of_effort_from_mast();
        assert!(
            forward < 0.0,
            "expected the centre of effort aft of the mast, got {forward} m forward"
        );
        assert_relative_eq!(forward, -0.1936, epsilon = 1e-3);
        // And it is high up: about a third of the way up a 16 m rig, plus the boom.
        assert!(
            (5.0..8.0).contains(&height),
            "centre of effort at {height} m"
        );
    }

    /// Placing the mast for a lead and then measuring it gives the lead back.
    ///
    /// The two directions are each other's inverse, which is the only thing that
    /// makes the pair safe to use: one fills a boat file in, the other checks it,
    /// and if they disagreed a file could pass its own check and still be wrong.
    #[test]
    fn placing_and_measuring_are_inverses() {
        for &wanted in &[0.022_f64, 0.03, 0.05] {
            let placed = Balance::from_lead(
                &yd41_keel(),
                &yd41_plan(),
                RigType::FractionalSloop,
                WATERLINE,
                wanted,
            );
            let measured = Balance::from_positions(
                &yd41_keel(),
                &yd41_plan(),
                RigType::FractionalSloop,
                WATERLINE,
                placed.mast_at,
            );
            assert_relative_eq!(measured.lead, wanted, max_relative = 1e-12);
            assert_relative_eq!(
                measured.centre_of_effort_at,
                placed.centre_of_effort_at,
                max_relative = 1e-12
            );
        }
    }

    /// The published lead puts the mast a definite distance from the keel.
    ///
    /// This is the number that closes the data gap: 2.2 % of an 11.90 m waterline
    /// is 0.262 m of lead, and the centre of effort is 0.194 m aft of the mast,
    /// so the mast stands 0.456 m forward of the centre of lateral resistance.
    /// Everything about the rig's longitudinal position follows from that and the
    /// keel's own position.
    #[test]
    fn the_published_lead_fixes_the_mast_against_the_keel() {
        let keel = yd41_keel();
        let balance = Balance::from_lead(
            &keel,
            &yd41_plan(),
            RigType::FractionalSloop,
            WATERLINE,
            0.022,
        );
        assert!(balance.lead_is_recommended);

        let (clr_at, _) = keel.centre_of_lateral_resistance();
        assert_relative_eq!(
            balance.centre_of_effort_at - clr_at,
            0.022 * WATERLINE,
            max_relative = 1e-12
        );
        assert_relative_eq!(balance.mast_at - clr_at, 0.4556, epsilon = 1e-3);
    }

    /// Sweep moves the centre of lateral resistance aft, and by how much.
    ///
    /// The construction walks 45 % of the draft down the quarter-chord line, so a
    /// swept keel puts its centre aft of the point where it meets the waterline.
    /// For the YD-41's 5.5° over 2.30 m of draft that is a tenth of a metre —
    /// small, but a third of the whole lead, so getting the sign wrong would
    /// matter more than the number suggests.
    #[test]
    fn sweep_carries_the_lateral_centre_aft() {
        let upright = ExtendedKeel {
            sweep: 0.0,
            ..yd41_keel()
        };
        let swept = yd41_keel();
        let (upright_at, upright_depth) = upright.centre_of_lateral_resistance();
        let (swept_at, swept_depth) = swept.centre_of_lateral_resistance();

        assert_relative_eq!(upright_depth, swept_depth, max_relative = 1e-15);
        assert_relative_eq!(upright_depth, 0.45 * 2.30, max_relative = 1e-12);
        assert!(
            swept_at < upright_at,
            "sweeping the keel aft must move its centre aft"
        );
        assert_relative_eq!(
            upright_at - swept_at,
            0.45 * 2.30 * 5.5_f64.to_radians().tan(),
            max_relative = 1e-12
        );
    }

    /// A rig with no sails has no centre of effort to speak of.
    #[test]
    fn an_empty_sail_plan_has_no_centre() {
        let empty = SailPlan {
            foretriangle_height: 0.0,
            foretriangle_base: 0.0,
            main_hoist: 0.0,
            main_foot: 0.0,
            boom_above_sheer: 1.7,
        };
        assert_eq!(empty.centre_of_effort_from_mast(), (0.0, 0.0));
    }
}
