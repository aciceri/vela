//! One run of every command on the reference boat.
//!
//! These are the standing proof that the binary still drives the engine
//! end to end: each test runs the real executable on `boats/yd41-form-study.ron`,
//! demands a clean exit, and looks for the one line the command exists to
//! print. Nothing here checks a number — the physics has its own tests in
//! `vela-core` — so a change in a coefficient never fails a smoke test, and a
//! command that stops printing its answer always does.
//!
//! The seaway run is cut to five seconds. The command integrates at 200 Hz
//! and a minute of it takes most of a minute unoptimised, which is longer
//! than a smoke test should be; five seconds exercises the same path.

use std::path::{Path, PathBuf};
use std::process::Command;

fn boat() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../boats/yd41-form-study.ron")
}

/// Runs the binary and returns its standard output, failing the test on a
/// non-zero exit with the standard error attached.
fn run(command: &str, file: &Path, flags: &[&str]) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_vela-cli"))
        .arg(command)
        .arg(file)
        .args(flags)
        .output()
        .expect("the binary runs");
    assert!(
        output.status.success(),
        "`vela-cli {command}` failed with {}:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("the report is text")
}

/// Runs a command on the reference boat and asserts that the report carries
/// `key`, the line that is the command's answer.
fn expect(command: &str, flags: &[&str], key: &str) {
    let report = run(command, &boat(), flags);
    assert!(
        report.contains(key),
        "`vela-cli {command}` printed no `{key}` line:\n{report}"
    );
}

#[test]
fn mesh() {
    expect("mesh", &[], "hull volume");
}

#[test]
fn hydrostatics() {
    expect("hydrostatics", &[], "displaced volume");
}

#[test]
fn form() {
    expect("form", &[], "prismatic Cp");
}

#[test]
fn resistance() {
    expect(
        "resistance",
        &["--froude", "0.35"],
        "DSYHS envelope      inside",
    );
}

#[test]
fn resistance_curve() {
    expect("resistance-curve", &[], "residuary");
}

#[test]
fn sail() {
    // The reference boat carries a layout, so the helm is part of the answer.
    expect("sail", &[], "helm");
}

#[test]
fn polar() {
    expect("polar", &[], "VMG");
}

#[test]
fn lewis() {
    expect("lewis", &[], "station(s) mapped");
}

#[test]
fn radiation() {
    expect("radiation", &[], "A_inf");
}

#[test]
fn balance() {
    // The mast position comes from the file's layout when none is given.
    expect("balance", &[], "With the mast at 6.301 m (layout)");
}

#[test]
fn seaway() {
    expect(
        "seaway",
        &["--seconds", "5"],
        "over the second half of the run:",
    );
}

/// A `.json` file is read as JSON, which the extension alone decides.
#[test]
fn json_intake() {
    let ron = std::fs::read_to_string(boat()).expect("the reference boat exists");
    let spec = vela_core::BoatSpec::parse_ron(&ron).expect("the reference boat parses");
    let json = serde_json::to_string(&spec).expect("the spec serialises");
    let file = std::env::temp_dir().join(format!("vela-smoke-{}.json", std::process::id()));
    std::fs::write(&file, json).expect("the temporary file is writable");

    let report = run("balance", &file, &[]);
    std::fs::remove_file(&file).expect("the temporary file is removable");
    assert!(
        report.contains("With the mast at 6.301 m (layout)"),
        "the JSON boat lost its layout:\n{report}"
    );
}
