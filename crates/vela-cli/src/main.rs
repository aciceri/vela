//! Headless driver for `vela-core`.
//!
//! This binary exists to keep the engine honest. If a command here needs a
//! window, an event loop or a renderer, the separation the project is built on
//! has already been broken — so the CLI is not a convenience, it is the
//! standing proof that `vela-core` is usable on its own. It is also what
//! generates reference output for validation.
//!
//! Argument parsing is done by hand: two commands do not justify a dependency,
//! and this file should stay boring.

use std::process::ExitCode;

use vela_core::hydrostatics::{solve_flotation, FlotationOptions};
use vela_core::{loft_hull, BoatSpec, LoftOptions, RigidBody, Water};

const USAGE: &str = "\
vela — sailing yacht physics engine

USAGE:
    vela-cli hydrostatics <boat.ron> [--points N]
    vela-cli mesh <boat.ron> [--points N]

COMMANDS:
    hydrostatics    Solve the floating equilibrium and report hull properties
    mesh            Report the lofted physics mesh without solving anything

OPTIONS:
    --points N      Contour points per station (default 16)
";

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

    let mut options = LoftOptions::default();
    let mut rest = arguments[2..].iter();
    while let Some(flag) = rest.next() {
        match flag.as_str() {
            "--points" => {
                let value = rest.next().ok_or("--points needs a value")?;
                options.points_per_station = value
                    .parse()
                    .map_err(|_| format!("--points expects an integer, got {value}"))?;
            }
            other => return Err(format!("unknown option {other}\n\n{USAGE}")),
        }
    }

    let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let spec = BoatSpec::parse_ron(&text).map_err(|error| format!("{path}: {error}"))?;
    let mesh = loft_hull(&spec.hull, &options);

    match command.as_str() {
        "mesh" => Ok(report_mesh(&spec, &mesh)),
        "hydrostatics" => report_hydrostatics(&spec, &mesh),
        other => Err(format!("unknown command {other}\n\n{USAGE}")),
    }
}

fn report_mesh(spec: &BoatSpec, mesh: &vela_core::TriMesh) -> String {
    let mut out = String::new();
    out.push_str(&format!("boat                {}\n", spec.name));
    out.push_str(&format!(
        "stations            {}\n",
        spec.hull.stations.len()
    ));
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
        .sum::<vela_core::geometry::Point>()
        .norm();
    out.push_str(&format!("closure residual    {closure:>10.2e} m^2\n"));
    out
}

fn report_hydrostatics(spec: &BoatSpec, mesh: &vela_core::TriMesh) -> Result<String, String> {
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
        "heave stiffness     {:>10.1} N/m\n",
        hydro.heave_stiffness(&water, body.gravity())
    ));
    out.push_str(&format!(
        "solver iterations   {:>10}\n",
        flotation.iterations
    ));
    Ok(out)
}
