//! Headless driver for `vela-core`.
//!
//! This binary exists to keep the engine honest. If a command here needed a
//! window, an event loop or a renderer, the separation the project is built on
//! would already be broken — so the CLI is not a convenience, it is the
//! standing proof that `vela-core` is usable on its own. It is also what
//! generates reference output for validation.
//!
//! Argument parsing is done by hand: a handful of commands does not justify a
//! dependency, and this file should stay boring.

use std::process::ExitCode;

use vela_core::aero::{EffectiveSpan, SailSet, Trim};
use vela_core::assembly::velocity_prediction_sim;
use vela_core::boat::HullSpec;
use vela_core::dsyhs::{hull_resistance, HullParameters};
use vela_core::equilibrium::{self, Equilibrium, EquilibriumOptions};
use vela_core::geometry::Point;
use vela_core::hydrostatics::{solve_flotation, FlotationOptions};
use vela_core::lewis::{area_coefficient_bounds, station_geometry, LewisForm};
use vela_core::sections::{hull_form, FormOptions};
use vela_core::{
    loft_hull, BoatSpec, Controls, LoftOptions, RigidBody, Sim, StillWater, TriMesh, UniformWind,
    Water, SEA_WATER_DENSITY,
};

/// Knots per metre per second, for reporting only. Nothing in the engine knows
/// what a knot is.
const KNOTS_PER_METRE_PER_SECOND: f64 = 1.943_844_492_440_605;

/// Relative speed gain a trim must deliver to be preferred over a less
/// depowered one.
///
/// A hundredth of a per cent: two orders below the precision the polar is
/// printed at, so it never changes a reported speed, and it stops a plateau
/// from being awarded to whichever candidate came last.
const TRIM_MARGIN: f64 = 1e-4;

const USAGE: &str = "\
vela — sailing yacht physics engine

USAGE:
    vela-cli mesh          <boat.ron> [--points N]
    vela-cli hydrostatics  <boat.ron> [--points N]
    vela-cli form          <boat.ron> [--points N]
    vela-cli resistance       <boat.ron> [--speed M/S | --froude F] [--heel DEG]
    vela-cli resistance-curve <boat.ron> [--heel DEG]
    vela-cli sail             <boat.ron> [--tws M/S] [--twa DEG] [--downwind]
    vela-cli polar            <boat.ron> [--tws M/S] [--points N]
    vela-cli lewis            <boat.ron> [--waterline M]

COMMANDS:
    mesh              Report the lofted physics mesh without solving anything
    hydrostatics      Solve the floating equilibrium and report hull properties
    form              Compare the geometry against the declared parameters
    resistance        Canoe body resistance at one speed, by component
    resistance-curve  Canoe body resistance across the Froude range
    sail              Solve one sailing condition and show its force balance
    polar             Solve the whole polar, optimising sails and trim
    lewis             Fit Lewis forms to the stations and check the envelope

OPTIONS:
    --points N      Contour points per station (default 16)
    --speed M/S     Speed through the water
    --froude F      Speed given as a Froude number instead
    --heel DEG      Heel angle in degrees (default 0)
    --tws M/S       True wind speed (default 6)
    --twa DEG       True wind angle off the bow (default 40)
    --downwind      Carry main and spinnaker on an eased aspect ratio
    --flat F        Flattening factor, 1.0 full (depowers lift, keeps the arm)
    --reef R        Reef factor, 1.0 full sail (depowers area and lowers the arm)
    --waterline M   Waterline height above the baseline (default: design draft)

Commands needing geometry require a boat file with hull offsets; commands
needing form parameters require the parameters block. Most boats have one or
the other, some have both.
";

struct Options {
    loft: LoftOptions,
    speed: Option<f64>,
    froude: Option<f64>,
    heel: f64,
    /// True wind speed, m/s.
    wind: f64,
    /// True wind angle, radians, measured from the bow.
    wind_angle: f64,
    downwind: bool,
    /// Flattening factor `F`: 1.0 is a normally trimmed sail, less is flatter.
    flat: f64,
    /// Reef factor `R`: 1.0 is full sail, less is reefed.
    reef: f64,
    /// Waterline height above the baseline, m, overriding the design waterline.
    waterline: Option<f64>,
}

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    match run(&arguments) {
        Ok(output) => {
            print!("{output}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run(arguments: &[String]) -> Result<String, String> {
    let Some(command) = arguments.first() else {
        return Err(format!("no command given\n\n{USAGE}"));
    };
    let Some(path) = arguments.get(1) else {
        return Err(format!("{command} needs a boat file\n\n{USAGE}"));
    };
    let options = parse_options(&arguments[2..])?;

    let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let spec = BoatSpec::parse_ron(&text).map_err(|error| format!("{path}: {error}"))?;

    match command.as_str() {
        "mesh" => Ok(report_mesh(&spec, &lofted(&spec, &options)?)),
        "hydrostatics" => report_hydrostatics(&spec, &lofted(&spec, &options)?),
        "form" => report_form(&spec, &lofted(&spec, &options)?),
        "resistance" => report_resistance(&spec, &options),
        "resistance-curve" => report_resistance_curve(&spec, &options),
        "polar" => report_polar(&spec, &options),
        "sail" => report_sail(&spec, &options),
        "lewis" => report_lewis(&spec, &options),
        other => Err(format!("unknown command {other}\n\n{USAGE}")),
    }
}

fn parse_options(arguments: &[String]) -> Result<Options, String> {
    let mut options = Options {
        loft: LoftOptions::default(),
        speed: None,
        froude: None,
        heel: 0.0,
        wind: 6.0,
        wind_angle: 40.0_f64.to_radians(),
        downwind: false,
        flat: 1.0,
        reef: 1.0,
        waterline: None,
    };
    let mut rest = arguments.iter();
    while let Some(flag) = rest.next() {
        match flag.as_str() {
            "--points" => {
                let raw = rest.next().ok_or("--points needs a value")?;
                options.loft.points_per_station = raw
                    .parse()
                    .map_err(|_| format!("--points expects an integer, got {raw}"))?;
            }
            "--speed" => options.speed = Some(number(rest.next(), "--speed")?),
            "--froude" => options.froude = Some(number(rest.next(), "--froude")?),
            "--heel" => options.heel = number(rest.next(), "--heel")?.to_radians(),
            "--tws" => options.wind = number(rest.next(), "--tws")?,
            "--twa" => options.wind_angle = number(rest.next(), "--twa")?.to_radians(),
            "--downwind" => options.downwind = true,
            "--flat" => options.flat = number(rest.next(), "--flat")?,
            "--reef" => options.reef = number(rest.next(), "--reef")?,
            "--waterline" => options.waterline = Some(number(rest.next(), "--waterline")?),
            other => return Err(format!("unknown option {other}\n\n{USAGE}")),
        }
    }
    if options.speed.is_some() && options.froude.is_some() {
        return Err("give either --speed or --froude, not both".into());
    }
    Ok(options)
}

fn number(raw: Option<&String>, flag: &str) -> Result<f64, String> {
    let raw = raw.ok_or_else(|| format!("{flag} needs a value"))?;
    raw.parse()
        .map_err(|_| format!("{flag} expects a number, got {raw}"))
}

/// Lofts the hull, or explains that this boat has no geometry to loft.
fn lofted(spec: &BoatSpec, options: &Options) -> Result<TriMesh, String> {
    let hull: &HullSpec = spec.hull.as_ref().ok_or_else(|| {
        format!(
            "{} has no hull offsets, so it cannot be lofted; \
             only parameter-based commands work on it",
            spec.name
        )
    })?;
    Ok(loft_hull(hull, &options.loft))
}

fn parameters(spec: &BoatSpec) -> Result<HullParameters, String> {
    spec.hull_parameters().ok_or_else(|| {
        format!(
            "{} declares no hull parameters block, which the resistance model needs",
            spec.name
        )
    })
}

fn report_mesh(spec: &BoatSpec, mesh: &TriMesh) -> String {
    let mut out = String::new();
    out.push_str(&format!("boat                {}\n", spec.name));
    out.push_str(&format!("triangles           {}\n", mesh.triangle_count()));
    out.push_str(&format!("vertices            {}\n", mesh.vertices().len()));
    out.push_str(&format!(
        "hull volume         {:>10.3} m^3\n",
        mesh.signed_volume()
    ));
    out.push_str(&format!(
        "hull surface        {:>10.3} m^2\n",
        mesh.surface_area()
    ));
    // A closed surface has zero net area-weighted normal; anything else means
    // the mesh leaks and every integral over it is suspect.
    let closure: f64 = mesh
        .triangles()
        .map(|t| t.area_normal())
        .sum::<Point>()
        .norm();
    out.push_str(&format!("closure residual    {closure:>10.2e} m^2\n"));
    out
}

fn report_hydrostatics(spec: &BoatSpec, mesh: &TriMesh) -> Result<String, String> {
    let properties = spec
        .mass_properties()
        .map_err(|error| format!("mass properties: {error}"))?;
    let body = RigidBody::new(properties).map_err(|error| format!("rigid body: {error}"))?;
    let water = Water::default();

    let flotation = solve_flotation(mesh, &body, &water, &FlotationOptions::default())
        .map_err(|error| error.to_string())?;
    let hydro = &flotation.hydrostatics;

    let mut out = report_mesh(spec, mesh);
    out.push('\n');
    out.push_str(&format!(
        "mass                {:>10.1} kg\n",
        body.mass_properties().mass()
    ));
    out.push_str(&format!(
        "displacement        {:>10.1} kg\n",
        hydro.displacement(&water)
    ));
    out.push_str(&format!("displaced volume    {:>10.4} m^3\n", hydro.volume));
    out.push_str(&format!(
        "draft               {:>10.4} m\n",
        flotation.draft
    ));
    out.push_str(&format!(
        "heel                {:>10.4} deg\n",
        flotation.heel.to_degrees()
    ));
    out.push_str(&format!(
        "trim                {:>10.4} deg (positive = bow up)\n",
        flotation.trim.to_degrees()
    ));
    out.push_str(&format!(
        "wetted area         {:>10.3} m^2\n",
        hydro.wetted_area
    ));
    out.push_str(&format!(
        "waterplane area     {:>10.3} m^2\n",
        hydro.waterplane_area
    ));
    out.push_str(&format!(
        "centre of buoyancy  x {:>8.4}  y {:>8.4}  z {:>8.4} m (body frame)\n",
        hydro.centre_of_buoyancy.x, hydro.centre_of_buoyancy.y, hydro.centre_of_buoyancy.z
    ));
    let cog = body.mass_properties().cog();
    out.push_str(&format!(
        "centre of gravity   x {:>8.4}  y {:>8.4}  z {:>8.4} m (body frame)\n",
        cog.x, cog.y, cog.z
    ));
    out.push_str(&format!(
        "solver iterations   {:>10}\n",
        flotation.iterations
    ));
    Ok(out)
}

fn report_resistance(spec: &BoatSpec, options: &Options) -> Result<String, String> {
    let hull = parameters(spec)?;
    let gravity = vela_core::STANDARD_GRAVITY;
    let speed = match (options.speed, options.froude) {
        (Some(speed), _) => speed,
        (None, Some(froude)) => froude * (gravity * hull.waterline_length).sqrt(),
        (None, None) => return Err("resistance needs --speed or --froude".into()),
    };

    let mut out = String::new();
    out.push_str(&format!("boat                {}\n", spec.name));
    match hull.check_envelope() {
        Ok(()) => out.push_str("DSYHS envelope      inside\n"),
        // Not an error: a hull can be real and still outside the series. The
        // number is reported with the caveat attached rather than withheld.
        Err(error) => out.push_str(&format!("DSYHS envelope      OUTSIDE\n{error}\n")),
    }
    out.push_str(&format!("speed               {speed:>10.3} m/s\n"));
    out.push_str(&format!(
        "froude number       {:>10.4}\n",
        hull.froude(speed, gravity)
    ));
    out.push_str(&format!(
        "heel                {:>10.2} deg\n\n",
        options.heel.to_degrees()
    ));

    let resistance = hull_resistance(&hull, speed, options.heel, SEA_WATER_DENSITY, gravity);
    out.push_str(&format!(
        "friction            {:>10.1} N\n",
        resistance.friction
    ));
    out.push_str(&format!(
        "viscous pressure    {:>10.1} N\n",
        resistance.viscous_pressure
    ));
    out.push_str(&format!(
        "residuary           {:>10.1} N\n",
        resistance.residuary
    ));
    out.push_str(&format!(
        "residuary, heel     {:>10.1} N\n",
        resistance.heel_residuary
    ));
    out.push_str(&format!("                    {:>10}\n", "----------"));
    out.push_str(&format!(
        "canoe body total    {:>10.1} N\n",
        resistance.total()
    ));
    out.push_str("\nAppendages are not included: keel and rudder resistance belong to the\nappendage model, which the series also treats separately.\n");
    Ok(out)
}

/// Measures the hull's form parameters from its geometry, and — when the file
/// also declares them — checks the two against each other.
///
/// This is the point of supporting both descriptions. A boat file that declares
/// a prismatic coefficient contradicting its own offsets is wrong, and until
/// something compares them nobody finds out.
fn report_form(spec: &BoatSpec, mesh: &TriMesh) -> Result<String, String> {
    let properties = spec
        .mass_properties()
        .map_err(|error| format!("mass properties: {error}"))?;
    let body = RigidBody::new(properties).map_err(|error| format!("rigid body: {error}"))?;
    let water = Water::default();

    // Measure at the floating waterline rather than an arbitrary draft: form
    // parameters are only comparable with published ones at the design
    // displacement.
    let flotation = solve_flotation(mesh, &body, &water, &FlotationOptions::default())
        .map_err(|error| error.to_string())?;
    let options = FormOptions::at_draft(mesh, flotation.draft)
        .ok_or("hull has no immersed volume at the floating waterline")?;
    let derived = hull_form(mesh, &options).map_err(|error| error.to_string())?;

    let mut out = String::new();
    out.push_str(&format!("boat                {}\n", spec.name));
    out.push_str(&format!(
        "draft               {:>10.4} m\n\n",
        flotation.draft
    ));
    out.push_str("                        derived");
    if spec.parameters.is_some() {
        out.push_str("    declared      diff");
    }
    out.push('\n');

    let declared = spec.hull_parameters();
    let rows: [(&str, f64, Option<f64>); 10] = [
        (
            "waterline length  m",
            derived.waterline_length,
            declared.map(|d| d.waterline_length),
        ),
        (
            "waterline beam    m",
            derived.waterline_beam,
            declared.map(|d| d.waterline_beam),
        ),
        (
            "canoe draft       m",
            derived.canoe_draft,
            declared.map(|d| d.canoe_draft),
        ),
        (
            "volume          m^3",
            derived.canoe_volume,
            declared.map(|d| d.canoe_volume),
        ),
        (
            "wetted surface  m^2",
            derived.wetted_surface,
            declared.map(|d| d.wetted_surface),
        ),
        (
            "waterplane area m^2",
            derived.waterplane_area,
            declared.map(|d| d.waterplane_area),
        ),
        ("midship area    m^2", derived.midship_area, None),
        (
            "prismatic Cp       ",
            derived.prismatic,
            declared.map(|d| d.prismatic),
        ),
        (
            "midship Cm         ",
            derived.midship,
            declared.map(|d| d.midship),
        ),
        (
            "LCF               %",
            derived.lcf * 100.0,
            declared.map(|d| d.lcf * 100.0),
        ),
    ];

    for (label, derived_value, declared_value) in rows {
        out.push_str(&format!("{label} {derived_value:>12.4}"));
        if let Some(declared_value) = declared_value {
            let scale = derived_value.abs().max(declared_value.abs()).max(1e-12);
            out.push_str(&format!(
                " {:>11.4} {:>8.1}%",
                declared_value,
                100.0 * (derived_value - declared_value) / scale
            ));
        }
        out.push('\n');
    }
    out.push_str(&format!(
        "LCB               % {:>12.4}\n",
        derived.lcb * 100.0
    ));

    if declared.is_none() {
        out.push_str(
            "\nThis file declares no parameters block, so there is nothing to check the\n\
             geometry against. Add one to have the two compared.\n",
        );
    }
    Ok(out)
}

/// The DSYHS resistance curve at prescribed speeds.
///
/// This was called `polar` and was not one: a polar is what a boat *does*, and
/// a table of resistance at speeds nobody solved for is an input to that
/// question rather than an answer to it. It keeps its own command because it is
/// the only report a parameters-only boat can produce — no geometry, no
/// appendages and no rig needed — and because comparing it against a solved
/// polar is how the resistance side gets checked on its own.
fn report_resistance_curve(spec: &BoatSpec, options: &Options) -> Result<String, String> {
    let hull = parameters(spec)?;
    let gravity = vela_core::STANDARD_GRAVITY;

    let mut out = String::new();
    out.push_str(&format!(
        "{} — canoe body resistance, heel {:.1} deg\n\n",
        spec.name,
        options.heel.to_degrees()
    ));
    out.push_str("   Fn    speed      friction   residuary        heel       total\n");
    out.push_str("        [m/s]           [N]         [N]         [N]         [N]\n");

    let mut froude = 0.15;
    while froude <= 0.75001 {
        let speed = froude * (gravity * hull.waterline_length).sqrt();
        let r = hull_resistance(&hull, speed, options.heel, SEA_WATER_DENSITY, gravity);
        out.push_str(&format!(
            "{:5.2} {:8.3} {:13.1} {:11.1} {:11.1} {:11.1}\n",
            froude,
            speed,
            r.friction + r.viscous_pressure,
            r.residuary,
            r.heel_residuary,
            r.total()
        ));
        froude += 0.05;
    }
    Ok(out)
}

/// The depowering path, from full sail downwards.
///
/// Flattening comes before reefing, and that order is the source's rather than
/// a guess: *Principles of Yacht Design*, 5th ed., on Hazen's trim factors —
/// the lift is proportional to `F` while the induced drag goes as `F²`, so
/// flattening rotates the resultant forward and *"it is thus better to flatten
/// the sail than to reef it to reduce heeling, a fact well-known by most
/// sailors."* Reefing also lowers the heeling arm, which is why it is what
/// remains once flattening has run out.
///
/// The path is coarse on purpose. It is searched exhaustively at every point of
/// the polar, so its length is multiplied by the number of cells; and the
/// speed it is being searched for is flat near its own maximum, so a finer grid
/// buys decimals of a knot for a linear cost in solves.
fn power_path() -> impl Iterator<Item = (f64, f64)> {
    const FLAT: [f64; 5] = [1.0, 0.9, 0.8, 0.7, 0.6];
    const REEF: [f64; 4] = [0.9, 0.8, 0.7, 0.6];
    FLAT.into_iter()
        .map(|flat| (flat, 1.0))
        .chain(REEF.into_iter().map(|reef| (0.6, reef)))
}

/// The sail sets a crew would consider, each with the aspect-ratio regime that
/// goes with it.
///
/// Both are tried at every angle rather than switching at a threshold. The
/// aerodynamic model has coefficients for a spinnaker at 27° and for a jib at
/// 180°, and it is perfectly willing to report how badly each does there — so
/// the crossover is something the model can be asked for instead of something
/// this file has to assert.
fn sail_choices() -> [(SailSet, EffectiveSpan); 2] {
    [
        (SailSet::upwind(), EffectiveSpan::CloseHauled),
        (SailSet::downwind(), EffectiveSpan::Eased),
    ]
}

/// The fastest sailing condition available at one wind angle.
struct Fastest {
    solution: Equilibrium,
    sails: SailSet,
    flat: f64,
    reef: f64,
}

/// Searches sail set and trim for the fastest solved condition.
///
/// Maximising boat speed is the whole criterion, and it needs no heel limit to
/// go with it. An overpowered yacht's only equilibrium is a knockdown, and a
/// knockdown is *slow* — the hull is dragging its topsides through the water —
/// so a search for speed discards it without anyone having to name the angle at
/// which sailing stops. That is worth stating because the alternative,
/// depowering until heel falls under a chosen limit, would put an arbitrary
/// number in the middle of every prediction.
/// Depowering is walked only as far as it keeps paying, and it has to pay by a
/// margin: a trim is preferred over a less depowered one only if it is faster
/// by [`TRIM_MARGIN`]. Without that, a plateau is won by whichever candidate
/// happens to come last, and the report claims a crew flattened the sails for a
/// gain in the fourth decimal — dead downwind, where the lift coefficient is
/// 0.001 and flattening provably does nothing, the search dutifully reported
/// `flat 0.6`.
///
/// Speed along the path is unimodal — power buys speed until heel and induced
/// drag take it back — so the first step that fails to pay is where the search
/// for that sail set stops. Stated as the heuristic it is: a pathological
/// double-peaked response would hide behind it, and the whole path can be
/// walked by removing one `break` if that is ever suspected.
fn fastest_at(sim: &mut Sim, reference_length: f64, start: &EquilibriumOptions) -> Option<Fastest> {
    let mut best: Option<Fastest> = None;

    for (sails, span) in sail_choices() {
        let mut previous: Option<f64> = None;

        for (flat, reef) in power_path() {
            let trim = Trim::full(span).with_flat(flat).with_reef(reef);
            sim.set_controls(Controls {
                rudder_angle: 0.0,
                sails,
                trim,
            });

            let Ok(solution) = equilibrium::solve(sim, reference_length, start) else {
                // A trim with no solution says nothing about the next one: the
                // knockdown branch can be unreachable at full sail and the
                // depowered condition perfectly ordinary.
                continue;
            };

            let speed = solution.speed;
            let worth_it = |reference: f64| speed > reference * (1.0 + TRIM_MARGIN);

            if best
                .as_ref()
                .is_none_or(|current| worth_it(current.solution.speed))
            {
                best = Some(Fastest {
                    solution,
                    sails,
                    flat,
                    reef,
                });
            }
            if previous.is_some_and(|earlier| !worth_it(earlier)) {
                break;
            }
            previous = Some(speed);
        }
    }
    best
}

/// The solved polar: what the boat does at every wind angle, at one wind speed.
fn report_polar(spec: &BoatSpec, options: &Options) -> Result<String, String> {
    let hull = parameters(spec)?;
    let start = EquilibriumOptions {
        initial_sinkage: hull.canoe_draft,
        ..EquilibriumOptions::default()
    };

    let mut out = String::new();
    out.push_str(&format!(
        "{} — solved polar, TWS {:.2} m/s ({:.1} kn)\n\n",
        spec.name,
        options.wind,
        options.wind * KNOTS_PER_METRE_PER_SECOND
    ));
    out.push_str(
        "  TWA     speed    speed       VMG     heel   leeway   sails            trim\n\
         [deg]     [m/s]     [kn]      [kn]    [deg]    [deg]                 flat  reef\n",
    );

    let mut angle_deg: f64 = 30.0;
    while angle_deg <= 180.001 {
        let environment =
            StillWater::new(UniformWind::uniform(options.wind, angle_deg.to_radians()));
        // Rebuilt per angle because the environment is fixed at construction;
        // trim and sails are then swept on this one simulation.
        let mut sim = velocity_prediction_sim(
            spec,
            Box::new(environment),
            Controls::close_hauled(SailSet::upwind()),
            &options.loft,
        )
        .map_err(|error| format!("{}: {error}", spec.name))?;

        match fastest_at(&mut sim, hull.waterline_length, &start) {
            Some(best) => {
                let knots = best.solution.speed * KNOTS_PER_METRE_PER_SECOND;
                out.push_str(&format!(
                    "{:5.0} {:9.3} {:8.2} {:9.2} {:8.1} {:8.2}   {:<14} {:4.1} {:5.1}\n",
                    angle_deg,
                    best.solution.speed,
                    knots,
                    knots * angle_deg.to_radians().cos(),
                    best.solution.heel.to_degrees(),
                    best.solution.leeway.to_degrees(),
                    best.sails.to_string(),
                    best.flat,
                    best.reef
                ));
            }
            // Reported rather than skipped: a hole in a polar is a result, and
            // a blank row says where the model ran out of sailing conditions.
            None => out.push_str(&format!("{angle_deg:5.0}         —\n")),
        }
        angle_deg += 10.0;
    }

    out.push_str(
        "\nVMG is positive to windward and negative to leeward. Trim is the fastest\n\
         depowering found, flattening before reefing; 1.0 / 1.0 is full sail.\n",
    );
    Ok(out)
}

/// Solves the steady sailing condition and reports it with its force balance.
///
/// The boat is left heading north and the wind is placed at the requested angle
/// off the bow, which makes the true wind angle and the wind's compass bearing
/// the same number. That is a choice of where to put the origin of the
/// heading, not a physical assumption: nothing in the model depends on absolute
/// direction.
fn report_sail(spec: &BoatSpec, options: &Options) -> Result<String, String> {
    let sails = if options.downwind {
        SailSet::downwind()
    } else {
        SailSet::upwind()
    };
    // The aspect-ratio regime is the caller's to state, since the source gives
    // no apparent wind angle at which one gives way to the other.
    let base = if options.downwind {
        Controls::eased(sails)
    } else {
        Controls::close_hauled(sails)
    };
    let controls = base.with_trim(base.trim.with_flat(options.flat).with_reef(options.reef));

    let wind = UniformWind::uniform(options.wind, options.wind_angle);
    let environment = StillWater::new(wind).with_water(Water::default());

    let mut sim = velocity_prediction_sim(spec, Box::new(environment), controls, &options.loft)
        .map_err(|error| format!("{}: {error}", spec.name))?;

    let hull = parameters(spec)?;
    let start = EquilibriumOptions {
        initial_sinkage: hull.canoe_draft,
        ..EquilibriumOptions::default()
    };
    let solution = equilibrium::solve(&mut sim, hull.waterline_length, &start)
        .map_err(|error| format!("{}: {error}", spec.name))?;

    let mut out = String::new();
    out.push_str(&format!(
        "{} — steady sailing, TWS {:.2} m/s, TWA {:.1} deg, {}\n\n",
        spec.name,
        options.wind,
        options.wind_angle.to_degrees(),
        sails
    ));
    out.push_str(&format!(
        "boat speed          {:>10.3} m/s   ({:.2} kn)\n",
        solution.speed,
        solution.speed * 1.943_844
    ));
    out.push_str(&format!(
        "heel                {:>10.2} deg\n",
        solution.heel.to_degrees()
    ));
    out.push_str(&format!(
        "leeway              {:>10.2} deg\n",
        solution.leeway.to_degrees()
    ));
    out.push_str(&format!(
        "sinkage             {:>10.3} m\n",
        solution.sinkage
    ));
    out.push_str(&format!(
        "residual            {:>10.2e}   ({} iterations)\n\n",
        solution.residual, solution.iterations
    ));

    out.push_str("force breakdown\n");
    for (key, value) in sim.telemetry().iter() {
        out.push_str(&format!("  {key:<38} {value:>12.3}\n"));
    }
    Ok(out)
}

/// Lewis conformal-mapping fit of every station, with the validity envelope.
///
/// The report exists to answer one question before the expensive part of strip
/// theory is written: **is this hull mappable at all?** A station outside the
/// envelope has to be clamped, and a clamped fit no longer reproduces the
/// section's area, so whatever added mass is computed from it inherits that
/// error. Knowing how many stations that happens to, and where, decides whether
/// two-parameter Lewis forms are enough for this hull or whether it needs a
/// close-fit mapping with more parameters.
///
/// The waterline is taken as the declared canoe body draft — the design
/// waterline — rather than from a flotation solve, because this is a question
/// about the shape of the hull rather than about how it happens to be loaded.
fn report_lewis(spec: &BoatSpec, options: &Options) -> Result<String, String> {
    let hull = spec.hull.as_ref().ok_or_else(|| {
        format!(
            "{} has no hull offsets, so it has no sections to map",
            spec.name
        )
    })?;
    let waterline = match options.waterline {
        Some(height) => height,
        None => parameters(spec)?.canoe_draft,
    };

    let mut out = String::new();
    out.push_str(&format!(
        "{} — Lewis conformal mapping, waterline {:.4} m above baseline\n\n",
        spec.name, waterline
    ));
    out.push_str(
        "     x     beam    draft    sigma       H0       a1       a3   status\n\
         \x20  [m]      [m]      [m]\n",
    );

    let mut mapped = 0;
    let mut clamped = 0;
    let mut dry = 0;

    for station in &hull.stations {
        let Some(section) = station_geometry(station, waterline) else {
            dry += 1;
            out.push_str(&format!("{:6.2}        —  (nothing immersed)\n", station.x));
            continue;
        };
        let form = LewisForm::fit(&section);
        let (lower, _) = area_coefficient_bounds(section.ratio());
        if form.clamped {
            clamped += 1;
        } else {
            mapped += 1;
        }

        out.push_str(&format!(
            "{:6.2} {:8.3} {:8.3} {:8.4} {:8.3} {:8.4} {:8.4}   {}\n",
            station.x,
            section.beam,
            section.draft,
            section.area_coefficient(),
            section.ratio(),
            form.a1,
            form.a3,
            if form.clamped {
                format!("CLAMPED to {lower:.4}")
            } else {
                "ok".to_string()
            }
        ));
    }

    out.push_str(&format!(
        "\n{mapped} station(s) mapped, {clamped} clamped, {dry} dry.\n"
    ));
    if clamped > 0 {
        out.push_str(
            "A clamped station's area is not reproduced by its Lewis form. Two-parameter\n\
             mapping cannot make a section that fine; a close-fit mapping with more\n\
             parameters can. Journee & Massie, Offshore Hydromechanics, section 7.3.\n",
        );
    }
    Ok(out)
}
