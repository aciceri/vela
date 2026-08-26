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

use vela_core::boat::HullSpec;
use vela_core::dsyhs::{hull_resistance, HullParameters};
use vela_core::geometry::Point;
use vela_core::hydrostatics::{solve_flotation, FlotationOptions};
use vela_core::sections::{hull_form, FormOptions};
use vela_core::{loft_hull, BoatSpec, LoftOptions, RigidBody, TriMesh, Water, SEA_WATER_DENSITY};

const USAGE: &str = "\
vela — sailing yacht physics engine

USAGE:
    vela-cli mesh          <boat.ron> [--points N]
    vela-cli hydrostatics  <boat.ron> [--points N]
    vela-cli form          <boat.ron> [--points N]
    vela-cli resistance    <boat.ron> [--speed M/S | --froude F] [--heel DEG]

COMMANDS:
    mesh            Report the lofted physics mesh without solving anything
    hydrostatics    Solve the floating equilibrium and report hull properties
    resistance      Canoe body resistance at one speed, by component
    polar           Resistance across the Froude range the series covers

OPTIONS:
    --points N      Contour points per station (default 16)
    --speed M/S     Speed through the water
    --froude F      Speed given as a Froude number instead
    --heel DEG      Heel angle in degrees (default 0)

Commands needing geometry require a boat file with hull offsets; commands
needing form parameters require the parameters block. Most boats have one or
the other, some have both.
";

struct Options {
    loft: LoftOptions,
    speed: Option<f64>,
    froude: Option<f64>,
    heel: f64,
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
        "polar" => report_polar(&spec, &options),
        other => Err(format!("unknown command {other}\n\n{USAGE}")),
    }
}

fn parse_options(arguments: &[String]) -> Result<Options, String> {
    let mut options = Options {
        loft: LoftOptions::default(),
        speed: None,
        froude: None,
        heel: 0.0,
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

fn report_polar(spec: &BoatSpec, options: &Options) -> Result<String, String> {
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
