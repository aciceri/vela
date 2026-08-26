//! The boat data format: geometry and mass properties, nothing else.
//!
//! Files are RON (canonical) or JSON (accepted for interop); both map onto the
//! same serde types. Everything is SI, and the file works in the **file frame**
//! documented in [`crate::frames`] — x forward, y half-breadth, z up from the
//! baseline. The loader converts once, and no other code sees file-frame
//! quantities.
//!
//! The hull is stored as **station offsets**, not a mesh. Strip theory, the
//! Michell integral and the DSYHS parameter extraction all consume sections
//! natively, so a mesh would be converted back to sections anyway, with loss.
//! The physics mesh is lofted from these at load time by [`crate::loft`]; the
//! visual mesh is a frontend concern and deliberately absent from this format.

use crate::dsyhs::HullParameters;
use crate::frames::file_to_body;
use crate::mass::{MassError, MassProperties};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Format version understood by this build. Additive changes keep the number;
/// anything that invalidates existing files bumps it.
pub const SCHEMA_VERSION: u32 = 1;

/// A complete boat definition.
///
/// A hull may be given as **geometry** (station offsets), as **scalar form
/// parameters**, or both. Neither alone covers every need:
///
/// - Buoyancy, wave loads and stability are surface integrals and demand
///   geometry.
/// - The DSYHS resistance regressions are statistical fits over form
///   parameters and never touch geometry. Published hull data — including the
///   design yacht of the series' own textbook — is routinely available as
///   parameters with no offset table anywhere in print.
///
/// When both are present they describe the same hull twice, which makes them
/// checkable against each other: a file whose declared prismatic coefficient
/// contradicts its own offsets is wrong, and the loader can say so.
///
/// RON files need `#![enable(implicit_some)]` at the top to write optional
/// blocks without spelling out `Some(...)`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BoatSpec {
    pub schema_version: u32,
    pub name: String,
    #[serde(default)]
    pub hull: Option<HullSpec>,
    #[serde(default)]
    pub parameters: Option<ParametersSpec>,
    #[serde(default)]
    pub appendages: Option<AppendagesSpec>,
    #[serde(default)]
    pub rig: Option<RigSpec>,
    #[serde(default)]
    pub layout: Option<LayoutSpec>,
    pub mass: MassSpec,
}

/// Where the rig and the appendages sit along the hull.
///
/// Absent from every published particulars table, and therefore absent from this
/// format until now, which is why yaw was restrained: a force with no
/// longitudinal arm makes no yawing moment, so a boat without this block cannot
/// be steered — and the engine says so rather than fabricating positions.
///
/// All three are measured **forward from the aft perpendicular**, in metres, the
/// same convention as a station's `x`.
///
/// Filling it in is a design exercise, not a measurement, and [`crate::balance`]
/// is the tool for it: the keel's position is a decision, and the mast's follows
/// from it through the lead the source recommends for the rig type.
/// `vela-cli balance` prints both.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct LayoutSpec {
    /// Keel quarter chord where it meets the waterline, m.
    pub keel_at: f64,
    /// Rudder quarter chord where it meets the waterline, m.
    pub rudder_at: f64,
    /// Mast, at the sheer, m.
    pub mast_at: f64,
}

/// Scalar hull form parameters, as published in yacht data tables.
///
/// Lengths in metres, areas in square metres, volumes in cubic metres. `lcb`
/// and `lcf` are **percentages** of the waterline length measured from
/// midship, positive forward — the convention used by the published tables, so
/// that numbers can be copied across without a sign hunt.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct ParametersSpec {
    pub waterline_length: f64,
    pub waterline_beam: f64,
    pub canoe_draft: f64,
    pub canoe_volume: f64,
    pub wetted_surface: f64,
    pub waterplane_area: f64,
    pub prismatic: f64,
    pub midship: f64,
    pub lcb_percent: f64,
    pub lcf_percent: f64,
}

impl ParametersSpec {
    /// Converts to the engine's form, turning the published percentages into
    /// fractions.
    #[must_use]
    pub fn to_hull_parameters(&self) -> HullParameters {
        HullParameters {
            waterline_length: self.waterline_length,
            waterline_beam: self.waterline_beam,
            canoe_draft: self.canoe_draft,
            canoe_volume: self.canoe_volume,
            wetted_surface: self.wetted_surface,
            waterplane_area: self.waterplane_area,
            prismatic: self.prismatic,
            midship: self.midship,
            lcb: self.lcb_percent / 100.0,
            lcf: self.lcf_percent / 100.0,
        }
    }
}

/// Hull geometry as a longitudinal sequence of transverse sections.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HullSpec {
    /// Stations ordered from aft to forward, strictly increasing in `x`.
    pub stations: Vec<Station>,
}

/// One transverse section.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Station {
    /// Longitudinal position, m, measured forward from the aft perpendicular.
    pub x: f64,
    /// Contour points ordered from the keel outward and upward to the deck
    /// edge. The hull is assumed symmetric, so these are half-breadths.
    pub points: Vec<Offset>,
}

/// A single offset: half-breadth and height above the baseline.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct Offset {
    /// Half-breadth, m. Zero on the centerline, never negative — the loader
    /// mirrors to both sides.
    pub y: f64,
    /// Height above the baseline, m.
    pub z: f64,
}

/// Mass, centre of gravity and radii of gyration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MassSpec {
    pub displacement_kg: f64,
    /// Centre of gravity in the file frame, m.
    pub cog: Point3Spec,
    /// Radii of gyration about the body axes through the CoG, m.
    pub gyradii: GyradiiSpec,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct Point3Spec {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// A lifting surface: keel, rudder, centreboard.
///
/// Planform only. Area and aspect ratio are *derived*, never declared, because
/// a declared area that disagrees with the chords and span is a silent
/// inconsistency with no right answer.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct FoilSpec {
    /// Chord at the root, where the foil meets the hull, m.
    pub root_chord: f64,
    /// Chord at the tip, m.
    pub tip_chord: f64,
    /// Span from root to tip, m.
    pub span: f64,
    /// Sweep of the quarter-chord line, degrees. Positive aft.
    pub sweep_deg: f64,
}

/// Keel and rudder.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct AppendagesSpec {
    pub keel: FoilSpec,
    pub rudder: FoilSpec,
    /// Displaced volume of the keel, m³. Enters the keel residuary resistance
    /// directly, which is why it is asked for rather than guessed from the
    /// planform and an assumed thickness.
    pub keel_volume: f64,
    /// Height of the keel's centre of buoyancy above the bottom of the hull, m.
    pub keel_cb_height: f64,
}

/// Rig dimensions, in the IOR notation the aerodynamic model is written in.
///
/// Kept in the published notation on purpose: these numbers are copied off sail
/// plans and rating certificates, and renaming them to something more
/// descriptive would make transcription error-prone for the sake of readability
/// nobody wants here. The doc comments carry the meaning.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct RigSpec {
    /// `I` — height of the foretriangle above the sheer, m.
    pub foretriangle_height: f64,
    /// `J` — base of the foretriangle, m.
    pub foretriangle_base: f64,
    /// `P` — mainsail hoist, m.
    pub main_hoist: f64,
    /// `E` — foot of the mainsail, m.
    pub main_foot: f64,
    /// `LPG` — perpendicular of the longest jib, m.
    pub jib_perpendicular: f64,
    /// `SL` — spinnaker leech length, m.
    pub spinnaker_leech: f64,
    /// `BAD` — height of the main boom above the sheer, m.
    pub boom_above_sheer: f64,
    /// `EHM` — mast height above the sheer, m.
    pub mast_above_sheer: f64,
    /// `EMDC` — average mast diameter, m.
    pub mast_diameter: f64,
    /// `FA` — average freeboard, m.
    ///
    /// A hull dimension rather than a rig one, but the windage model is its
    /// only consumer, so it lives where it is used.
    pub average_freeboard: f64,
    /// `BMAX` — maximum beam of the hull, m. Here for the same reason as
    /// [`Self::average_freeboard`].
    pub max_beam: f64,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct GyradiiSpec {
    pub rx: f64,
    pub ry: f64,
    pub rz: f64,
}

/// Why a boat file was rejected.
///
/// Loading is strict on purpose: a hull that is subtly malformed produces
/// plausible-looking but wrong hydrostatics, which is far more expensive to
/// discover later than a refusal now.
#[derive(Debug, Clone, PartialEq)]
pub enum SpecError {
    Syntax(String),
    UnsupportedSchema {
        found: u32,
        expected: u32,
    },
    /// Neither offsets nor parameters were given.
    NoHullDescription,
    /// A declared hull parameter is not physically possible.
    NonPositiveParameter {
        name: &'static str,
        value: f64,
    },
    TooFewStations(usize),
    StationsNotIncreasing {
        index: usize,
        x: f64,
    },
    TooFewPoints {
        station: usize,
        count: usize,
    },
    KeelOffCentreline {
        station: usize,
        y: f64,
    },
    NegativeHalfBreadth {
        station: usize,
        point: usize,
        y: f64,
    },
    HeightNotAscending {
        station: usize,
        point: usize,
    },
    DegenerateSection {
        station: usize,
    },
    Mass(MassError),
}

impl fmt::Display for SpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax(message) => write!(f, "malformed boat file: {message}"),
            Self::UnsupportedSchema { found, expected } => {
                write!(f, "schema_version {found} is not supported (expected {expected})")
            }
            Self::NoHullDescription => write!(
                f,
                "a boat needs either hull offsets or scalar hull parameters; neither was given"
            ),
            Self::NonPositiveParameter { name, value } => {
                write!(f, "hull parameter {name} must be positive, got {value}")
            }
            Self::TooFewStations(n) => {
                write!(f, "a hull needs at least 2 stations, found {n}")
            }
            Self::StationsNotIncreasing { index, x } => write!(
                f,
                "station {index} at x = {x} does not lie forward of its predecessor"
            ),
            Self::TooFewPoints { station, count } => write!(
                f,
                "station {station} has {count} offsets; at least 2 are needed to form a contour"
            ),
            Self::KeelOffCentreline { station, y } => write!(
                f,
                "station {station} starts at half-breadth {y}; contours must start on the centerline"
            ),
            Self::NegativeHalfBreadth { station, point, y } => write!(
                f,
                "station {station} offset {point} has negative half-breadth {y}"
            ),
            Self::HeightNotAscending { station, point } => write!(
                f,
                "station {station} offset {point} drops below the previous one; \
                 contours run from keel to deck and cannot fold back"
            ),
            Self::DegenerateSection { station } => {
                write!(f, "station {station} has zero extent")
            }
            Self::Mass(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for SpecError {}

impl From<MassError> for SpecError {
    fn from(error: MassError) -> Self {
        Self::Mass(error)
    }
}

impl BoatSpec {
    /// Parses and validates RON.
    ///
    /// # Errors
    ///
    /// [`SpecError`] on malformed syntax or a hull that fails validation.
    pub fn parse_ron(text: &str) -> Result<Self, SpecError> {
        let spec: Self =
            ron::from_str(text).map_err(|error| SpecError::Syntax(error.to_string()))?;
        spec.validate()?;
        Ok(spec)
    }

    /// Parses and validates JSON.
    ///
    /// # Errors
    ///
    /// As [`BoatSpec::parse_ron`].
    pub fn parse_json(text: &str) -> Result<Self, SpecError> {
        let spec: Self =
            serde_json::from_str(text).map_err(|error| SpecError::Syntax(error.to_string()))?;
        spec.validate()?;
        Ok(spec)
    }

    /// # Errors
    ///
    /// [`SpecError`] describing the first problem found.
    pub fn validate(&self) -> Result<(), SpecError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(SpecError::UnsupportedSchema {
                found: self.schema_version,
                expected: SCHEMA_VERSION,
            });
        }
        if self.hull.is_none() && self.parameters.is_none() {
            return Err(SpecError::NoHullDescription);
        }
        if let Some(hull) = &self.hull {
            hull.validate()?;
        }
        if let Some(parameters) = &self.parameters {
            parameters.validate()?;
        }
        self.mass_properties()?;
        Ok(())
    }

    /// The scalar form parameters, if the file declares them.
    ///
    /// Deliberately not derived from geometry here: computing a prismatic
    /// coefficient needs a sectional area curve, which the hydrostatics module
    /// owns. This accessor answers "what did the file say", and the cross-check
    /// against geometry belongs where both are in hand.
    #[must_use]
    pub fn hull_parameters(&self) -> Option<HullParameters> {
        self.parameters
            .as_ref()
            .map(ParametersSpec::to_hull_parameters)
    }

    /// Mass properties converted into the body frame.
    ///
    /// Radii of gyration survive the conversion untouched: the file and body
    /// frames share their axes and differ only in the direction of two of them,
    /// and a radius about an axis does not care which way the axis points.
    ///
    /// # Errors
    ///
    /// [`SpecError::Mass`] if the mass or gyradii are not physical.
    pub fn mass_properties(&self) -> Result<MassProperties, SpecError> {
        let cog = file_to_body(Vector3::new(
            self.mass.cog.x,
            self.mass.cog.y,
            self.mass.cog.z,
        ));
        let gyradii = Vector3::new(
            self.mass.gyradii.rx,
            self.mass.gyradii.ry,
            self.mass.gyradii.rz,
        );
        Ok(MassProperties::from_gyradii(
            self.mass.displacement_kg,
            cog,
            gyradii,
        )?)
    }
}

impl ParametersSpec {
    /// Rejects parameters that describe no physical hull.
    ///
    /// Only positivity is checked here. Whether the hull falls inside the range
    /// of models the DSYHS regressions were fitted to is a separate question
    /// with a separate answer — see `dsyhs::HullParameters::check_envelope` —
    /// because a hull can be perfectly real and still outside the series.
    ///
    /// # Errors
    ///
    /// [`SpecError::NonPositiveParameter`] for the first offending value.
    pub fn validate(&self) -> Result<(), SpecError> {
        let checks = [
            ("waterline_length", self.waterline_length),
            ("waterline_beam", self.waterline_beam),
            ("canoe_draft", self.canoe_draft),
            ("canoe_volume", self.canoe_volume),
            ("wetted_surface", self.wetted_surface),
            ("waterplane_area", self.waterplane_area),
            ("prismatic", self.prismatic),
            ("midship", self.midship),
        ];
        for (name, value) in checks {
            if !value.is_finite() || value <= 0.0 {
                return Err(SpecError::NonPositiveParameter { name, value });
            }
        }
        Ok(())
    }
}

impl HullSpec {
    /// # Errors
    ///
    /// [`SpecError`] describing the first problem found.
    pub fn validate(&self) -> Result<(), SpecError> {
        if self.stations.len() < 2 {
            return Err(SpecError::TooFewStations(self.stations.len()));
        }

        for (index, station) in self.stations.iter().enumerate() {
            if index > 0 && station.x <= self.stations[index - 1].x {
                return Err(SpecError::StationsNotIncreasing {
                    index,
                    x: station.x,
                });
            }
            if station.points.len() < 2 {
                return Err(SpecError::TooFewPoints {
                    station: index,
                    count: station.points.len(),
                });
            }
            if station.points[0].y.abs() > 1e-9 {
                return Err(SpecError::KeelOffCentreline {
                    station: index,
                    y: station.points[0].y,
                });
            }
            for (point, offset) in station.points.iter().enumerate() {
                if offset.y < 0.0 {
                    return Err(SpecError::NegativeHalfBreadth {
                        station: index,
                        point,
                        y: offset.y,
                    });
                }
                if point > 0 && offset.z < station.points[point - 1].z - 1e-12 {
                    return Err(SpecError::HeightNotAscending {
                        station: index,
                        point,
                    });
                }
            }
            if station.extent() <= 0.0 {
                return Err(SpecError::DegenerateSection { station: index });
            }
        }
        Ok(())
    }

    /// Waterline length between the outermost stations, m.
    #[must_use]
    pub fn length(&self) -> f64 {
        match (self.stations.first(), self.stations.last()) {
            (Some(first), Some(last)) => last.x - first.x,
            _ => 0.0,
        }
    }

    /// Greatest half-breadth anywhere on the hull, m.
    #[must_use]
    pub fn max_half_breadth(&self) -> f64 {
        self.stations
            .iter()
            .flat_map(|s| s.points.iter())
            .map(|p| p.y)
            .fold(0.0, f64::max)
    }
}

impl Station {
    /// Arc length of the contour, used to reject sections that are a single
    /// repeated point.
    #[must_use]
    pub fn extent(&self) -> f64 {
        self.points
            .windows(2)
            .map(|pair| {
                let (a, b) = (pair[0], pair[1]);
                ((b.y - a.y).powi(2) + (b.z - a.z).powi(2)).sqrt()
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"#![enable(implicit_some)]
        (
            schema_version: 1,
            name: "test",
            hull: (
                stations: [
                    (x: 0.0, points: [(y: 0.0, z: 0.0), (y: 1.0, z: 1.0)]),
                    (x: 2.0, points: [(y: 0.0, z: 0.0), (y: 1.0, z: 1.0)]),
                ],
            ),
            mass: (
                displacement_kg: 1000.0,
                cog: (x: 1.0, y: 0.0, z: 0.5),
                gyradii: (rx: 0.5, ry: 1.0, rz: 1.0),
            ),
        )
    "#;

    fn spec() -> BoatSpec {
        BoatSpec::parse_ron(MINIMAL).expect("valid spec")
    }

    fn hull(spec: &BoatSpec) -> &HullSpec {
        spec.hull.as_ref().expect("this fixture has geometry")
    }

    #[test]
    fn parses_a_minimal_hull() {
        let spec = spec();
        assert_eq!(spec.name, "test");
        assert_eq!(hull(&spec).stations.len(), 2);
        assert!((hull(&spec).length() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn cog_is_converted_into_the_body_frame() {
        let mass = spec().mass_properties().expect("valid mass");
        // File z is up from the baseline, body z is down: the sign flips.
        assert!((mass.cog().z + 0.5).abs() < 1e-12);
        assert!((mass.cog().x - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_a_future_schema() {
        let text = MINIMAL.replace("schema_version: 1", "schema_version: 99");
        assert_eq!(
            BoatSpec::parse_ron(&text).expect_err("must be rejected"),
            SpecError::UnsupportedSchema {
                found: 99,
                expected: SCHEMA_VERSION
            }
        );
    }

    #[test]
    fn rejects_stations_out_of_order() {
        let text = MINIMAL.replace("x: 2.0", "x: -1.0");
        assert!(matches!(
            BoatSpec::parse_ron(&text),
            Err(SpecError::StationsNotIncreasing { .. })
        ));
    }

    #[test]
    fn rejects_a_contour_that_misses_the_centerline() {
        let text = MINIMAL.replacen("(y: 0.0, z: 0.0)", "(y: 0.3, z: 0.0)", 1);
        assert!(matches!(
            BoatSpec::parse_ron(&text),
            Err(SpecError::KeelOffCentreline { .. })
        ));
    }

    #[test]
    fn rejects_a_contour_that_folds_back_downward() {
        let text = MINIMAL.replacen(
            "(y: 0.0, z: 0.0), (y: 1.0, z: 1.0)",
            "(y: 0.0, z: 0.0), (y: 1.0, z: 1.0), (y: 1.2, z: 0.5)",
            1,
        );
        assert!(matches!(
            BoatSpec::parse_ron(&text),
            Err(SpecError::HeightNotAscending { .. })
        ));
    }

    #[test]
    fn rejects_non_physical_mass() {
        let text = MINIMAL.replace("displacement_kg: 1000.0", "displacement_kg: -5.0");
        assert!(matches!(
            BoatSpec::parse_ron(&text),
            Err(SpecError::Mass(_))
        ));
    }

    #[test]
    fn json_and_ron_agree() {
        let from_ron = spec();
        let json = serde_json::to_string(&from_ron).expect("serializable");
        let from_json = BoatSpec::parse_json(&json).expect("valid json");
        assert_eq!(from_ron.name, from_json.name);
        assert_eq!(
            hull(&from_ron).stations.len(),
            hull(&from_json).stations.len()
        );
    }
}
