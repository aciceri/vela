//! Separated flow: what the lattice cannot see, and where to hand over to it.
//!
//! # The problem
//!
//! [`crate::vlm`] is potential flow. Its plate at forty degrees comes out at
//! `2π sin 40°` and keeps climbing, and its drag is induced only. A sail spends its
//! downwind life at incidences where that is not merely inaccurate but the wrong
//! shape of curve: past stall, lift falls and drag rises toward the bluff-body
//! limit, and on a dead run a spinnaker is a parachute whose only relevant number
//! is projected area.
//!
//! So the attached model needs a companion and a smooth handover. This module is
//! both.
//!
//! # The separated branch is fitted, not universal
//!
//! The tempting choice is Hoerner's inclined flat plate — closed form, no fitting,
//! exact limits. It is wrong here, and measurably: scaled to this aspect ratio it
//! puts `C_L` at twenty degrees near 0.5 where the lattice says 1.75. Blending onto
//! it would drop a sail's lift by seventy per cent at stall, where a real sail
//! loses ten or twenty.
//!
//! The reason is that a *cambered* surface keeps far more lift past stall than a
//! flat plate does — which is why [[Viterna & Corrigan
//! 1981]](https://ntrs.nasa.gov/citations/19830010962) fit their post-stall curve
//! to the surface's own pre-stall values instead of assuming a plate, and why
//! [[Spera 2008]](https://ntrs.nasa.gov/citations/20090001311) states as a finding
//! that post-stall behaviour must not be taken as a plate's. So the branch here is
//! Viterna-Corrigan, anchored on whatever the attached model says at the stall
//! angle:
//!
//! ```text
//! C_L = A₁ sin 2α + A₂ cos²α / sin α
//! C_D = B₁ sin²α + B₂ cos α
//! ```
//!
//! with the four constants chosen so the branch passes exactly through
//! `(C_L, C_D)` at stall and exactly through `(0, C_Dmax)` at ninety degrees. Both
//! ends are then right by construction rather than by luck: the handover is
//! continuous, and the dead-run limit is bluff-body drag on projected area, which
//! is the answer the downwind literature says dominates there.
//!
//! # The handover is `C¹`, and that is not decoration
//!
//! The branch matches the attached model's *value* at stall but not its slope, so a
//! hard switch would put a kink in the force. A kink is a discontinuity in the
//! derivative the boat's dynamics integrate against — it shows up as a rig that
//! chatters when a gust walks the trim across the handover.
//!
//! A smoothstep weight fixes it exactly. With `w = t²(3-2t)` the weight and its
//! derivative both vanish at the onset, so the blended curve leaves the attached
//! branch tangentially, and both vanish again at the far end so it joins the
//! separated branch the same way. Nothing is `C⁰`-glued anywhere.
//!
//! # Where the blend is applied, and the sectional model that is *not* here
//!
//! At the sail, on its area-weighted mean incidence.
//!
//! Per *section* would be better physics and it is deliberately absent. The
//! attraction is real — a twisted sail's head is at lower incidence, so it should
//! stall later, and blending strip by strip would produce that for free. The
//! obstacle is that it needs a sectional stall criterion, and the sectional lift
//! coefficient a lattice can supply is `2Γ/cV`, which on a sail's tapered head has
//! a chord going to zero underneath it. Measured on the reference sail it climbs to
//! 2.45 at the head against 1.86 at mid span — an artefact of the normalisation,
//! not a section about to stall. Calibrating a criterion against that would be
//! calibrating against a division by a small number.
//!
//! Blending on the mean incidence keeps the part of the twist story that *is*
//! supportable. Twist lowers the mean, so a twisted sail reaches any given onset at
//! a higher wind angle, and stall is delayed — which is the published observation
//! [[Zhang et al. 2025]](https://arxiv.org/abs/2501.13254), who measured stall
//! moving from about 17° on a twist-free rigid model to 20° on a twisting
//! full-scale sail. Sign and order, not the number: the two are different sails and
//! the twist at stall is not reported.
//!
//! # What is a parameter, and what a parameter is worth
//!
//! `C_Dmax = 1.11 + 0.018·AR` is Viterna's own aspect-ratio fit and is not tunable.
//!
//! The stall angle *is* a parameter, because it is a property of a sail — camber,
//! Reynolds number, leading-edge geometry, cloth — and no closed form covers it.
//! The two anchors above are what the literature offers. It is a required argument
//! rather than a defaulted one so that a boat file has to state it.
//!
//! The transition width is a parameter and is **not** a measurement. It is the
//! smoothing scale, and it is named as such: nothing in the data resolves how fast
//! a sail's lift collapses past stall. Coefficient uncertainty downwind is 10–15 %
//! across wind tunnels before this model's own error is counted, so a width in the
//! five-to-ten-degree range is inside the noise it is smoothing.

use std::f64::consts::FRAC_PI_2;

/// Aspect ratio above which the drag at ninety degrees stops growing.
///
/// Viterna's fit is linear in aspect ratio and would run away; past this it is the
/// two-dimensional plate's value.
const SLENDER_ASPECT_RATIO: f64 = 50.0;

/// Drag coefficient at ninety degrees for a two-dimensional surface.
const SLENDER_MAXIMUM_DRAG: f64 = 2.01;

/// Drag coefficient at ninety degrees, from Viterna's aspect-ratio fit.
///
/// `1.11 + 0.018·AR`, flattening to 2.01 for a surface long enough that its ends
/// stop relieving the pressure behind it. A sail's aspect ratio of three or four
/// puts it near 1.18, which is a flat plate broadside — the number a dead run
/// should produce.
#[must_use]
pub fn maximum_drag(aspect_ratio: f64) -> f64 {
    if aspect_ratio >= SLENDER_ASPECT_RATIO {
        SLENDER_MAXIMUM_DRAG
    } else {
        1.11 + 0.018 * aspect_ratio
    }
}

/// The separated-flow branch: Viterna-Corrigan past a surface's stall.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Separated {
    stall: f64,
    maximum_drag: f64,
    lift_amplitude: f64,
    lift_shape: f64,
    drag_shape: f64,
}

impl Separated {
    /// Fits the branch to an attached model's own values at the stall angle.
    ///
    /// `stall` in radians, `lift` and `drag` the attached branch's coefficients
    /// there. The fit is exact at both ends — through `(lift, drag)` at `stall` and
    /// through `(0, maximum_drag)` at ninety degrees — so this is a change of
    /// functional form, not an approximation of one.
    ///
    /// Returns `None` unless the stall angle is strictly inside the first quadrant.
    /// At zero there is no attached regime to leave and at ninety degrees the
    /// `cos²α/sin α` term's anchor divides by a vanishing cosine.
    #[must_use]
    pub fn fit(aspect_ratio: f64, stall: f64, lift: f64, drag: f64) -> Option<Self> {
        if !(aspect_ratio.is_finite() && aspect_ratio > 0.0)
            || !stall.is_finite()
            || stall <= 0.0
            || stall >= FRAC_PI_2
            || !lift.is_finite()
            || !drag.is_finite()
        {
            return None;
        }
        let maximum_drag = maximum_drag(aspect_ratio);
        let (sine, cosine) = (stall.sin(), stall.cos());
        Some(Self {
            stall,
            maximum_drag,
            // `A₁ = B₁/2`, so that `A₁ sin 2α` carries `C_Dmax sin α cos α`.
            lift_amplitude: 0.5 * maximum_drag,
            lift_shape: (lift - maximum_drag * sine * cosine) * sine / (cosine * cosine),
            drag_shape: (drag - maximum_drag * sine * sine) / cosine,
        })
    }

    /// The stall angle the branch was anchored at, radians.
    #[must_use]
    pub fn stall(&self) -> f64 {
        self.stall
    }

    /// Drag coefficient at ninety degrees.
    #[must_use]
    pub fn maximum_drag(&self) -> f64 {
        self.maximum_drag
    }

    /// Lift coefficient of the separated branch at an incidence, radians.
    ///
    /// Clamped to the first quadrant. Below the stall angle the branch is being
    /// evaluated outside its fit and returns its value there, which is the attached
    /// model's — so a caller that blends carelessly gets a continuous answer rather
    /// than an extrapolated one. Above ninety degrees the sail is backed, and that
    /// is a different problem than this one.
    #[must_use]
    pub fn lift(&self, incidence: f64) -> f64 {
        let angle = incidence.clamp(self.stall, FRAC_PI_2);
        let (sine, cosine) = (angle.sin(), angle.cos());
        self.lift_amplitude * (2.0 * angle).sin() + self.lift_shape * cosine * cosine / sine
    }

    /// Drag coefficient of the separated branch at an incidence, radians.
    #[must_use]
    pub fn drag(&self, incidence: f64) -> f64 {
        let angle = incidence.clamp(self.stall, FRAC_PI_2);
        let sine = angle.sin();
        self.maximum_drag * sine * sine + self.drag_shape * angle.cos()
    }
}

/// The separated branch plus the smoothstep that hands over to it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Blend {
    separated: Separated,
    width: f64,
}

impl Blend {
    /// A blend that leaves the attached branch at the fit's stall angle and reaches
    /// the separated branch `width` radians later.
    ///
    /// Returns `None` for a non-positive width, which would be a hard switch and a
    /// kink, or for a width that runs the handover past ninety degrees, where the
    /// separated branch has nothing left to hand over to.
    #[must_use]
    pub fn new(separated: Separated, width: f64) -> Option<Self> {
        if !width.is_finite() || width <= 0.0 || separated.stall() + width > FRAC_PI_2 {
            return None;
        }
        Some(Self { separated, width })
    }

    /// The separated branch.
    #[must_use]
    pub fn separated(&self) -> &Separated {
        &self.separated
    }

    /// Weight given to the separated branch at an incidence, radians.
    ///
    /// `t²(3 - 2t)`: zero below the stall angle, one above the far end, and with a
    /// vanishing derivative at both. The vanishing derivative is the whole point —
    /// it is what makes the blended curve leave one branch and join the other
    /// tangentially instead of at a corner.
    #[must_use]
    pub fn weight(&self, incidence: f64) -> f64 {
        let t = ((incidence - self.separated.stall()) / self.width).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    /// Blended lift and drag coefficients.
    ///
    /// `attached` is what the lattice produced at this incidence, on the same
    /// reference area as the separated branch's coefficients — planform area, which
    /// is what the wind-tunnel sources the branch is calibrated from divide by.
    #[must_use]
    pub fn coefficients(&self, attached: (f64, f64), incidence: f64) -> (f64, f64) {
        let weight = self.weight(incidence);
        if weight == 0.0 {
            return attached;
        }
        (
            attached.0 * (1.0 - weight) + self.separated.lift(incidence) * weight,
            attached.1 * (1.0 - weight) + self.separated.drag(incidence) * weight,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    use nalgebra::Vector3;

    use crate::flying::{Distribution, Planform, Shape};
    use crate::vlm::Solver;

    const AIR: f64 = 1.225;
    const SPEED: f64 = 8.0;

    /// A sail's aspect ratio, and a stall angle from the reference measurements.
    const ASPECT: f64 = 3.36;
    const STALL: f64 = 17.0;

    fn fitted(lift: f64, drag: f64) -> Separated {
        Separated::fit(ASPECT, STALL.to_radians(), lift, drag).expect("a real stall")
    }

    /// The reference sail of [`crate::flying`], so that the two modules are talking
    /// about one boat.
    fn windsurf() -> Planform {
        Planform::new(5.185, 2.0, 0.10, 0.387).expect("a real sail's dimensions")
    }

    fn sail(twist_degrees: f64) -> Shape {
        Shape::new(
            windsurf(),
            Distribution::uniform(0.12),
            Distribution::uniform(0.40),
            Distribution::new(0.0, twist_degrees.to_radians()),
            0.0,
        )
        .expect("a trimmable sail")
    }

    /// The lattice's lift and drag coefficients on planform area.
    fn attached(shape: &Shape, wind_degrees: f64) -> (f64, f64) {
        let beta = wind_degrees.to_radians();
        let flow = Vector3::new(-SPEED * beta.cos(), SPEED * beta.sin(), 0.0);
        let across = Vector3::new(beta.sin(), beta.cos(), 0.0);
        let force = Solver::new(shape.lattice(10, 24).expect("a grid"), flow)
            .expect("a well-formed sail")
            .solve(|_| flow, AIR)
            .expect("a sail solves")
            .force();
        let dynamic = 0.5 * AIR * SPEED * SPEED * shape.planform().area();
        (
            force.dot(&across) / dynamic,
            force.dot(&(flow / SPEED)) / dynamic,
        )
    }

    /// The branch passes exactly through the attached values it was fitted to.
    ///
    /// The property that makes this a change of functional form rather than an
    /// approximation. Both constants are chosen for it, and getting either wrong
    /// leaves a step at the handover that a blend cannot remove.
    #[test]
    fn the_branch_passes_through_its_anchor_exactly() {
        for &(lift, drag) in &[(1.6_f64, 0.24_f64), (0.9, 0.10), (2.1, 0.35)] {
            let separated = fitted(lift, drag);
            assert_relative_eq!(
                separated.lift(STALL.to_radians()),
                lift,
                max_relative = 1e-13
            );
            assert_relative_eq!(
                separated.drag(STALL.to_radians()),
                drag,
                max_relative = 1e-13
            );
        }
    }

    /// At ninety degrees the branch is a bluff body: no lift, and the drag of a flat
    /// plate broadside.
    ///
    /// The dead-run limit, and the reason this form was chosen over one that merely
    /// decays. The downwind literature's own summary is that projected area is what
    /// dominates on a run; `C_L = 0` with `C_D = C_Dmax` on planform area *is* that
    /// statement, and it comes out of the fit rather than being pasted on.
    #[test]
    fn ninety_degrees_is_a_bluff_body() {
        let separated = fitted(1.6, 0.24);
        assert_relative_eq!(separated.lift(FRAC_PI_2), 0.0, epsilon = 1e-13);
        assert_relative_eq!(
            separated.drag(FRAC_PI_2),
            separated.maximum_drag(),
            max_relative = 1e-13
        );
        // And that drag is a plate's, not a wing's.
        assert!((1.1..1.3).contains(&separated.maximum_drag()));
    }

    /// The drag at ninety degrees is Viterna's aspect-ratio fit, and it saturates.
    #[test]
    fn the_maximum_drag_follows_the_aspect_ratio() {
        assert_relative_eq!(
            maximum_drag(3.36),
            1.11 + 0.018 * 3.36,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            maximum_drag(50.0),
            SLENDER_MAXIMUM_DRAG,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            maximum_drag(1e6),
            SLENDER_MAXIMUM_DRAG,
            max_relative = 1e-14
        );
        // Monotone below the knee, and a sail sits well under a plate's value.
        assert!(maximum_drag(2.0) < maximum_drag(8.0));
        assert!(maximum_drag(8.0) < SLENDER_MAXIMUM_DRAG);
    }

    /// Past stall, lift falls and drag rises to the bluff-body value.
    ///
    /// The shape of the curve, which is the reason for having a second branch at
    /// all: the lattice does the opposite in both quantities and never stops.
    #[test]
    fn past_stall_lift_falls_and_drag_rises() {
        let separated = fitted(1.6, 0.24);
        let mut previous = (f64::INFINITY, 0.0_f64);
        for degrees in [18.0_f64, 25.0, 40.0, 60.0, 80.0, 90.0] {
            let angle = degrees.to_radians();
            let (lift, drag) = (separated.lift(angle), separated.drag(angle));
            assert!(lift < previous.0, "lift rose at {degrees} deg");
            assert!(drag > previous.1, "drag fell at {degrees} deg");
            assert!(lift >= -1e-12, "lift went negative at {degrees} deg");
            previous = (lift, drag);
        }
    }

    /// The weight is a smoothstep: zero, one, monotone, and flat at both ends.
    #[test]
    fn the_weight_is_a_smoothstep() {
        let blend = Blend::new(fitted(1.6, 0.24), 8.0_f64.to_radians()).expect("a real blend");
        let onset = STALL.to_radians();
        let width = 8.0_f64.to_radians();

        assert_eq!(blend.weight(onset - 0.1), 0.0);
        assert_eq!(blend.weight(onset), 0.0);
        assert_relative_eq!(blend.weight(onset + width), 1.0, max_relative = 1e-14);
        assert_eq!(blend.weight(onset + width + 0.5), 1.0);
        assert_relative_eq!(blend.weight(onset + 0.5 * width), 0.5, max_relative = 1e-14);

        // Centred exactly on each end, where the true derivative is zero: a
        // difference centred a step *past* the onset would measure the ramp already
        // leaving, which is a property of the offset and not of the ramp.
        let step = 1e-6;
        let slope = |at: f64| (blend.weight(at + step) - blend.weight(at - step)) / (2.0 * step);
        assert!(slope(onset).abs() < 1e-4, "the ramp starts with a corner");
        assert!(
            slope(onset + width).abs() < 1e-4,
            "the ramp ends with a corner"
        );
        let mut previous = -1.0;
        for i in 0..=40 {
            let weight = blend.weight(onset + width * i as f64 / 40.0);
            assert!(weight >= previous, "the ramp is not monotone");
            previous = weight;
        }
    }

    /// The blended curve leaves the attached branch tangentially.
    ///
    /// `C¹` at the handover, checked against the attached branch it is leaving
    /// rather than against itself. A hard switch would match the value here — the
    /// fit guarantees that — and break the derivative, which is the failure that
    /// does not show up in a force plot and does show up as a rig that chatters when
    /// a gust walks the trim across the onset.
    #[test]
    fn the_handover_is_tangent_to_the_attached_branch() {
        let separated = fitted(1.6, 0.24);
        let blend = Blend::new(separated, 8.0_f64.to_radians()).expect("a real blend");
        let onset = separated.stall();

        // A stand-in attached branch with the fit's values at the onset and a
        // plausible slope: what matters is that the blend does not disturb either.
        let slope = 3.2;
        let attached = |angle: f64| (1.6 + slope * (angle - onset), 0.24 + 0.9 * (angle - onset));
        let blended = |angle: f64| blend.coefficients(attached(angle), angle);

        assert_relative_eq!(blended(onset).0, 1.6, max_relative = 1e-13);
        assert_relative_eq!(blended(onset).1, 0.24, max_relative = 1e-13);

        let step = 1e-5;
        let derivative = (blended(onset + step).0 - blended(onset - step).0) / (2.0 * step);
        assert_relative_eq!(derivative, slope, max_relative = 2e-3);

        // And nowhere in the handover does the blended lift leave the interval its
        // two branches span, which is what a weight outside `[0, 1]` would do.
        for i in 0..=60 {
            let angle = onset + 8.0_f64.to_radians() * i as f64 / 60.0;
            let (low, high) = (
                attached(angle).0.min(separated.lift(angle)),
                attached(angle).0.max(separated.lift(angle)),
            );
            let got = blended(angle).0;
            assert!(got >= low - 1e-12 && got <= high + 1e-12);
        }
    }

    /// Below the onset the attached branch is returned untouched, bit for bit.
    ///
    /// Upwind is where this simulator is expected to be accurate and where the
    /// lattice is the whole argument for computing forces from geometry. An
    /// empirical branch that leaked a per cent into the close-hauled answer would
    /// be spending the accuracy that justifies the method.
    #[test]
    fn below_the_onset_nothing_is_touched() {
        let blend = Blend::new(fitted(1.6, 0.24), 8.0_f64.to_radians()).expect("a real blend");
        for degrees in [0.0_f64, 5.0, 12.0, 16.9, 17.0] {
            let attached = (0.83, 0.041);
            let got = blend.coefficients(attached, degrees.to_radians());
            assert_eq!(got.0.to_bits(), attached.0.to_bits());
            assert_eq!(got.1.to_bits(), attached.1.to_bits());
        }
    }

    /// Twist delays stall, because it lowers the mean incidence the blend reads.
    ///
    /// The published observation: stall moved from about 17° on a twist-free rigid
    /// windsurf model to 20° on a twisting full-scale sail [Zhang et al. 2025]. Here
    /// the mechanism is explicit — the onset is a fixed property of the cloth and
    /// twist changes when the sail *reaches* it — so the delay is a consequence
    /// rather than a fitted parameter.
    ///
    /// Sign and order only. The two measured sails are different sails and their
    /// twist at stall is not reported, so a test asserting three degrees would be
    /// asserting a coincidence.
    #[test]
    fn twist_delays_the_onset_of_stall() {
        let onset = STALL.to_radians();
        // The wind angle at which the sail's mean incidence first reaches the onset.
        let stalls_at = |shape: &Shape| {
            let mut low = 0.0_f64;
            let mut high = 60.0_f64;
            for _ in 0..40 {
                let mid = 0.5 * (low + high);
                let beta = mid.to_radians();
                let flow = Vector3::new(-beta.cos(), beta.sin(), 0.0);
                if shape.mean_incidence(flow) < onset {
                    low = mid;
                } else {
                    high = mid;
                }
            }
            0.5 * (low + high)
        };

        // Untwisted, every section is at the wind angle, so the two coincide.
        assert_relative_eq!(stalls_at(&sail(0.0)), STALL, max_relative = 1e-6);

        let mut previous = STALL;
        for twist in [4.0_f64, 8.0, 12.0, 20.0] {
            let delayed = stalls_at(&sail(twist));
            assert!(
                delayed > previous,
                "twist of {twist} deg did not delay stall: {delayed:.2} after {previous:.2}"
            );
            previous = delayed;
        }
        // Eight degrees of twist buys a few degrees, not a fraction and not twenty.
        let delay = stalls_at(&sail(8.0)) - STALL;
        assert!(
            (1.0..6.0).contains(&delay),
            "eight degrees of twist delayed stall by {delay:.2} deg"
        );
    }

    /// The blend turns the lattice's runaway lift into a stall, on a real sail.
    ///
    /// The whole point, end to end: geometry through the lattice through the
    /// handover. Past the onset the lattice keeps climbing — that is what a
    /// potential-flow method does — and the blended curve peaks and comes down.
    #[test]
    fn a_real_sail_stalls_once_it_is_blended() {
        let shape = sail(0.0);
        let onset = STALL.to_radians();
        let (lift_at_stall, drag_at_stall) = attached(&shape, STALL);
        let blend = Blend::new(
            Separated::fit(
                shape.planform().aspect_ratio(),
                onset,
                lift_at_stall,
                drag_at_stall,
            )
            .expect("a real stall"),
            8.0_f64.to_radians(),
        )
        .expect("a real blend");

        let mut lattice_only = Vec::new();
        let mut blended = Vec::new();
        for degrees in [10.0_f64, 17.0, 21.0, 25.0, 30.0, 45.0] {
            let raw = attached(&shape, degrees);
            let beta = degrees.to_radians();
            let flow = Vector3::new(-beta.cos(), beta.sin(), 0.0);
            lattice_only.push(raw.0);
            blended.push(blend.coefficients(raw, shape.mean_incidence(flow)).0);
        }

        // The lattice never stops. This is the defect, and it is real rather than
        // asserted: potential flow has no mechanism to stop.
        for pair in lattice_only.windows(2) {
            assert!(
                pair[1] > pair[0],
                "the lattice stopped climbing: {lattice_only:?}"
            );
        }

        // The blended sail turns over. Structural rather than a pinned peak
        // position: with an eight-degree handover the curve is a plateau between
        // seventeen and twenty-one degrees, which is the shape a real sail's lift
        // curve has near stall, and which of the two samples is the higher by a
        // fraction of a per cent is not a claim worth making.
        let peak = blended.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            peak > *blended.last().expect("a sample"),
            "the blended sail never turned over: {blended:?}"
        );
        assert!(
            blended[1] > 0.98 * peak,
            "the peak is nowhere near the stall angle: {blended:?}"
        );
        // And past the plateau it is falling, sample after sample.
        for pair in blended[2..].windows(2) {
            assert!(pair[1] < pair[0], "the stalled sail recovered: {blended:?}");
        }

        // Drag has gone the other way. Not four times the stall value - it is two
        // and a half - but past half of the bluff-body limit, which says the
        // `C_Dmax sin²α` term has taken the curve over from the attached branch.
        let far = attached(&shape, 45.0);
        let beta = 45.0_f64.to_radians();
        let flow = Vector3::new(-beta.cos(), beta.sin(), 0.0);
        let (_, drag) = blend.coefficients(far, shape.mean_incidence(flow));
        assert!(
            drag > 2.0 * drag_at_stall && drag > 0.5 * blend.separated().maximum_drag(),
            "drag at forty-five degrees was {drag:.3} against {drag_at_stall:.3} at stall \
             and a limit of {:.3}",
            blend.separated().maximum_drag()
        );
    }

    /// Fits and blends that have no regime to describe are refused.
    #[test]
    fn a_degenerate_model_is_refused() {
        assert!(Separated::fit(ASPECT, 0.0, 1.6, 0.24).is_none());
        assert!(Separated::fit(ASPECT, FRAC_PI_2, 1.6, 0.24).is_none());
        assert!(Separated::fit(ASPECT, -0.2, 1.6, 0.24).is_none());
        assert!(Separated::fit(0.0, 0.3, 1.6, 0.24).is_none());
        assert!(Separated::fit(ASPECT, f64::NAN, 1.6, 0.24).is_none());
        assert!(Separated::fit(ASPECT, 0.3, f64::INFINITY, 0.24).is_none());

        let separated = fitted(1.6, 0.24);
        assert!(Blend::new(separated, 0.0).is_none());
        assert!(Blend::new(separated, -0.1).is_none());
        assert!(Blend::new(separated, f64::NAN).is_none());
        // A width that would run the handover past ninety degrees.
        assert!(Blend::new(separated, FRAC_PI_2).is_none());
        // And one that reaches it exactly is fine.
        assert!(Blend::new(separated, FRAC_PI_2 - separated.stall()).is_some());
    }
}
