//! What the radiation pipeline costs, and where.
//!
//! This exists because "is it real time?" is a fair question with a measurable
//! answer, and because the answer was once *no* for a reason no amount of
//! reading would have found: the Gauss-Legendre rule was rebuilt on every
//! section solve, which is nine-tenths of the work of a solve and identical
//! every time. [`SectionSolver`] exists to make that mistake unavailable.
//!
//! Run with `cargo run --release -p vela-core --example radiation_cost`.
//!
//! Nothing measured here runs per frame. The whole pipeline is paid once, when a
//! boat is loaded; what the simulation loop evaluates afterwards is a state-space
//! model of a handful of states per degree of freedom.

use std::time::Instant;
use vela_core::lewis::{LewisForm, SectionGeometry};
use vela_core::tasai::{SectionSolver, TasaiOptions};

fn main() {
    let form = LewisForm::fit(&SectionGeometry {
        beam: 3.2,
        draft: 0.6,
        area: 0.8 * 3.2 * 0.6,
    });

    println!("one section, one frequency:");
    for (multipoles, quadrature) in [(12, 48), (12, 96), (24, 128), (48, 160)] {
        let options = TasaiOptions {
            multipoles,
            quadrature,
        };
        let start = Instant::now();
        let solver = SectionSolver::new(options);
        let build = start.elapsed().as_secs_f64();

        let reps = 2000;
        let start = Instant::now();
        let mut sink = 0.0;
        for i in 0..reps {
            let omega = 0.5 + 4.0 * (i as f64) / reps as f64;
            sink += solver
                .heave(&form, omega, 1025.0, 9.81)
                .expect("a positive frequency has a solution")
                .added_mass;
        }
        let each = start.elapsed().as_secs_f64() / reps as f64;
        println!(
            "  M={multipoles:3} nq={quadrature:3}: rule {:6.1} us once, then {:6.1} us a solve  (sink {sink:.0})",
            build * 1e6,
            each * 1e6
        );
    }

    // A whole boat: a hull's worth of sections over a frequency grid fine enough
    // for the cosine transform that gives the retardation function, out to where
    // the damping tail has died away.
    let solver = SectionSolver::new(TasaiOptions::default());
    let forms: Vec<LewisForm> = (0..17)
        .map(|i| {
            let taper = 1.0 - 0.8 * ((i as f64 / 8.0) - 1.0).abs().powi(2);
            LewisForm::fit(&SectionGeometry {
                beam: 3.2 * taper,
                draft: 0.6 * taper,
                area: 0.8 * 3.2 * 0.6 * taper * taper,
            })
        })
        .collect();

    println!("\na hull's frequency sweep, which is the whole load-time cost:");
    for points in [120, 240] {
        let start = Instant::now();
        let mut sink = 0.0;
        for k in 1..=points {
            let omega = 30.0 * f64::from(k) / f64::from(points);
            for form in &forms {
                sink += solver
                    .heave(form, omega, 1025.0, 9.81)
                    .map_or(0.0, |solved| solved.damping);
            }
        }
        println!(
            "  {} sections x {points:3} frequencies: {:5.0} ms  (sink {sink:.0})",
            forms.len(),
            start.elapsed().as_secs_f64() * 1e3
        );
    }
}
