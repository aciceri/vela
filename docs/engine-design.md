# Vela — Engine Design: Boundaries, Data Format, Force-Module Contract

Status: draft for discussion. Companion to `functional-analysis.md` (which owns the
physics rationale and bibliography). This document owns the *interfaces*.
Code fragments below are illustrative signatures, not implementation.

---

## 1. Crate boundaries

Cargo workspace, dependency arrows only pointing left:

```
vela-core  <--  vela-cli      (headless: polars, batch runs, validation vs oracles)
           <--  vela-app      (Bevy frontend, wasm + native window)
           <--  [future: python bindings, server-side race replay, ...]
```

### `vela-core` — the engine

- Pure Rust. Allowed deps: `nalgebra`, `faer`, `rustfft`, `serde`, `ron`/`serde_json`.
  **Forbidden deps: Bevy, wgpu, winit, anything windowing/rendering/audio.**
  Enforced structurally (separate crate), not by convention.
- No wall clock, no filesystem access in the simulation path: boats load from
  `&[u8]`/`&str`, time advances only through `step(dt)`. This is what makes the
  engine embeddable (wasm, CLI, tests) and deterministic.
- Deterministic given (boat, environment params, seed, control trace): fixed
  `dt`, seeded RNG for spectra/gusts, no platform intrinsics. Native and wasm
  runs agree to FP noise; bit-exactness is a non-goal.

### Public API surface (abstract)

```rust
// Loading & preprocessing (the "load stage" of the two-stage pipeline)
let spec  = BoatSpec::parse(bytes)?;          // format detect: RON | JSON
let boat  = Boat::prepare(&spec, &PrepOptions)?;  // Michell table, strip theory,
                                                  // state-space fits, AIC panelization.
                                                  // Seconds; reports progress via callback.

// Simulation
let mut sim = Sim::new(boat, EnvSpec { wind, sea, water, seed })?;
sim.set_controls(Controls { rudder, mainsheet, traveler, vang, ... });
sim.step(FIXED_DT);                            // advances physics exactly one substep

// Observation (renderer/HUD/tests all use the same API)
sim.state()      -> &BodyState;                // pose + velocities
sim.telemetry()  -> &Telemetry;                // per-module force breakdown (see §3.6)
sim.wave_probe(x, y) -> WaveSample;            // elevation/kinematics for CPU consumers
sim.sea_realization() -> &SeaRealization;      // spectrum + seed + grid → GPU (see §4)
```

The frontend owns the frame loop and accumulates render time into fixed physics
substeps; the engine never sees a frame rate.

### `vela-cli` — headless frontend, exists from phase 1

Not an afterthought: it is how validation (§10 of the functional analysis) runs
in CI, and how polars are generated. If the engine API can't drive a batch polar
sweep without a window, the separation has failed. The CLI is the proof.

---

## 2. Boat data format

### 2.1 Container and syntax

- **Canonical format: RON.** Human-authored files need comments and trailing
  commas; RON has both, maps 1:1 onto the serde data model, and expresses Rust
  enums (sail kinds, appendage kinds) without stringly-typed tags.
- **JSON accepted** on load (same serde schema, detected by content/extension)
  for interop with external tooling. Cost of dual intake under serde: ~zero.
- One boat = **one self-contained file**. Optional large binary sidecars
  (offline-CFD override tables, §2.6) are referenced by relative path; a bundle
  container (zip) is a future option, not v1.
- `schema_version: u32` at top level, checked on load. Additive evolution
  preferred; breaking changes bump the version and get a migration note.

### 2.2 Conventions (stated once, enforced by the loader)

- **Units: SI throughout.** No unit annotations in the format. Angles in the
  file are **degrees** (human authoring); the API is radians.
- **File geometry frame** (naval-architecture style): origin at the
  intersection of baseline, centerline, and aft perpendicular; **x forward,
  y to port, z up** (right-handed). The hull is symmetric about `y = 0` and
  offsets are stored as non-negative **half-breadths**, which the loader
  mirrors to both sides — the way published offset tables are written, and it
  sidesteps the question of which side the file means.
- **Dynamics frame** (engine-internal): Fossen body convention — x forward,
  y starboard, z down, origin at a fixed body reference point. The loader
  converts once; no other code ever sees the file frame.

### 2.3 Hull: station offsets, not mesh

Canonical hull representation is **sections** (station offset tables):

```ron
hull: (
    stations: [
        ( x: 0.0,  points: [ (y: 0.0, z: 0.26), (y: 0.53, z: 0.34), ... ] ),
        ( x: 0.77, points: [ ... ] ),
        // ... 15–30 stations, transom to stem
    ],
)
```

Each contour runs from the keel — which must sit on the centerline — outward
and upward to the deck edge, with heights non-decreasing. The last point *is*
the deck edge: there is no separate `deck_z`, because a single deck height
cannot express sheer, and the deck line is already implied by where the
contours stop. Bulbous forms, where breadth grows and then height falls back,
are outside this representation and are rejected at load.

Rationale:

- Strip theory, the Michell integral, and DSYHS parameter extraction all
  consume sections *natively*. A mesh would be converted back to sections
  anyway, with quality loss.
- The watertight triangle mesh needed by the buoyancy clipper (§5.5 of the
  analysis) is **derived by lofting** at load time, at a physics-appropriate
  resolution — decoupled from whatever high-poly mesh the renderer uses for
  visuals.
- Sections are compact, diffable, and hand-editable; meshes are none of those.

For users who start from a mesh (STL/OBJ from a hull modeler), a converter
belongs in `vela-cli` — mesh-to-sections is a tooling problem, not an engine
format problem. Not implemented yet.

The *visual* hull model is explicitly **not** part of the boat physics file; a
frontend may associate one by name/reference. Physics files must not grow
rendering payloads.

### 2.4 Appendages

```ron
appendages: [
    (
        kind: Keel,                      // Keel | Rudder | Centerboard | Skeg | Foil(future)
        root: (x: 4.2, z: -0.4),         // root-chord leading edge, file frame
        planform: ( root_chord: 1.1, tip_chord: 0.7, span: 1.8, sweep_deg: 12.0 ),
        section: Naca("63-010"),         // or Polar(path) for measured section data
        movable: None,                   // rudder: Some(( axis, max_deflection_deg ))
    ),
]
```

`section` resolves against a small built-in library of section polars
(lift-slope, profile drag, stall angle per common NACA families); `Polar(path)`
overrides with user data. Ballast bulbs contribute mass (in `mass`) and
parasitic drag, not lift.

### 2.5 Rig and sails: parametric flying shape

```ron
rig: (
    spars: [ ( kind: Mast, base: (x: 3.9, z: 0.9), height: 12.0, diameter: 0.15 ), ... ],
    sails: [
        (
            kind: Main,                  // Main | Jib | Genoa | Spinnaker | Gennaker | Code0
            corners: ( tack: ..., head: ..., clew: ... ),
            shape: (
                camber:    [ (h: 0.0, v: 0.10), (h: 0.5, v: 0.12), (h: 1.0, v: 0.08) ],
                draft_pos: [ ... ],      // chordwise position of max camber vs height
                twist_deg: [ ... ],      // relative to boom, vs height fraction h
            ),
            // OPTIONAL: control-response derivatives (d shape / d control).
            // Omitted → engine defaults per sail kind (documented, tunable).
            response: None,
        ),
    ],
)
```

Design point: the file stores the **design flying shape** plus optionally how it
responds to controls; the engine ships default response models per sail kind so
that a minimal boat file still trims realistically. Defaults-with-override,
same philosophy as §2.6. Sail cloth/membrane properties are deliberately absent
(no runtime FSI — see functional analysis §4.3).

### 2.6 Mass and overrides

```ron
mass: (
    displacement_kg: 3800.0,
    cog: (x: 4.1, y: 0.0, z: 0.15),
    gyradii: (rx: 1.1, ry: 2.6, rz: 2.7),   // or full inertia tensor
),

overrides: (                                 // every field optional; empty in most files
    hull_resistance: Some(Table("cfd/rw_table.bin")),  // replaces Michell/DSYHS stage
    radiation:       None,                             // replaces strip-theory fit
    sail_coeffs:     None,                             // replaces downwind blend tables
),
```

The `overrides` block is the forward-compatibility slot from the functional
analysis (§9): any pipeline stage can be replaced by externally computed data
(offline CFD, tank tests, future ML surrogates) without schema changes.

### 2.7 Load-time validation (loader responsibilities)

Hard errors: non-monotonic stations, open sections, negative displacement,
CoG outside hull bounds, appendages detached from hull, sail corners
inconsistent with spar geometry, hydrostatic solve fails to find a floating
equilibrium.
Warnings + model routing: DSYHS envelope check (phase 1 refuses out-of-range
hulls; phase 3 routes them to the generic pipeline), slenderness check for
Michell, Fn range vs Savitsky applicability.

---

## 3. Force-module contract

### 3.1 Granularity: modules follow *coupling*, not taxonomy

A module boundary is drawn where forces are (quasi-)independent; strongly
coupled phenomena live inside one module. This is the key decision — the naive
"one module per named force" splits systems that must be solved jointly:

| Module | Contains | Why one unit |
|---|---|---|
| `Aero` | all sails (one VLM system) + spar/rig windage | jib–main slot interaction: the VLM must solve all lifting surfaces in a single AIC system; splitting per sail is physically wrong |
| `HullResistance` | friction + wave-making + planing blend (DSYHS in phase 1) | one scalar pipeline over shared hull state (Fn, heel, wetted area) |
| `LateralSystem` | keel + rudder + hull side-force share, incl. keel→rudder downwash, induced drag, yaw moment | downwash couples keel circulation to rudder inflow; CLR emerges only from the joint solution |
| `BuoyancyFK` | mesh clip vs wave surface, hydrostatic + Froude-Krylov pressure integration | one traversal of the submerged triangle set |
| `Radiation` | fluid-memory state-space + roll damping correction | owns internal ODE states |

Gravity is not a module; it belongs to the integrator (constant, exact).

### 3.2 Module inputs: the context, read-only

Every module receives the same immutable context each substep:

```rust
struct StepCtx<'a> {
    state:    &'a BodyState,      // pose, body-frame linear/angular velocity
    controls: &'a Controls,       // rudder angle, sheet/traveler/vang settings, ...
    env:      &'a dyn Environment, // wind + wave + water sampling (see §4)
    t:        f64,                // simulation time
    dt:       f64,                // fixed substep
}
```

Modules do not see each other. Cross-module physical coupling either lives
inside one module (§3.1) or flows through `BodyState` (e.g. appendage inflow
depends on leeway, which is state, not another module's output).

### 3.3 Module outputs: wrench + telemetry

```rust
struct Wrench { force: Vector3, moment: Vector3 }   // body frame, about body origin

trait ForceModule {
    fn step(&mut self, ctx: &StepCtx) -> Wrench;
    fn telemetry(&self, out: &mut TelemetryWriter);  // named sub-components, §3.6
}
```

Conventions, fixed once:

- **All wrenches in body frame, moments about the body reference origin** (not
  CoG). The integrator owns the origin→CoG transfer; CoG can move (crew,
  ballast) without touching modules.
- `step` takes `&mut self`: modules are **stateful by design** (VLM cached
  factorization, radiation ODE states, wetted-surface memory for smoothing).
  Determinism comes from fixed call order and fixed `dt`, not purity.
- Module `step` order is fixed and documented; since modules only read shared
  state, order affects nothing physical — fixing it merely pins determinism.

### 3.4 Added mass: not a wrench

Added mass multiplies acceleration; feeding it back as a force would require
the acceleration being solved for. The equation of motion is

$$(M_{RB} + M_A^{\infty})\,\dot\nu = \sum_i \tau_i(\nu, \eta, t)$$

so the contract splits contributions by *where they enter*:

- **Setup-time registration**: `Radiation` (and `Aero`, for the small air added
  mass — negligible, likely omitted) contribute a constant matrix
  $M_A^{\infty}$ once, at `Sim::new`. The integrator assembles and factorizes
  $(M_{RB} + M_A^{\infty})$ once (refactorized only if mass properties change).
- **Step-time wrenches**: everything velocity/position/time-dependent,
  including the radiation memory term (the state-space output *is* a force).

This keeps the integrator a plain linear solve per substep and forbids the
classic instability of treating added mass explicitly.

### 3.5 Stability note (recorded, not solved here)

Velocity-proportional damping terms integrated explicitly bound the usable
`dt`. Mitigation order: (1) fixed substepping at 120+ Hz — cheap, likely
sufficient; (2) if a term proves stiff (radiation states, roll damping at
large amplitude), integrate *that module's internal ODE* implicitly — the
module owns its states, so this is a module-local decision invisible to the
contract. The contract must not change to accommodate stiffness.

**Precedent already set, in the integrator itself.** The rigid-body inertial
terms `C(ν) ν` turned out to be exactly this case, and the resolution followed
this rule. Evaluating the gyroscopic term `ω × (I ω)` explicitly makes a freely
tumbling body *gain* energy (measured +1.6 % over 60 s at 120 Hz); evaluating it
at the end of the step merely flips the error's sign; evaluating it at the
**midpoint**, reached by two fixed-point iterations, conserves quadratic
invariants and cuts the drift to −2e-8 with `dt³` scaling. Cost: two
back-substitutions against the already-factorized mass matrix, and *zero* extra
force evaluations — the one-evaluation-per-step budget is intact, and the
`ForceModule` contract never learned about any of it.

### 3.6 Telemetry: first-class, not debug leftovers

Every module publishes a named breakdown
(`aero.main.lift`, `aero.jib.induced_drag`, `lateral.rudder.stall_fraction`,
`hull.rw`, `radiation.roll_damping`, ...) each substep into a flat
`Telemetry` map with stable keys.

Consumers: the HUD (trim feedback is the *product* — a physics simulator where
you can't see why the boat heels is a failed simulator), CI validation
(component-wise comparison against oracles, per functional analysis §10), and
polar generation. Stable keys are an API contract; renaming is a breaking
change.

### 3.7 Testability

Each module is constructible from (boat-derived data + mock `Environment`) with
no `Sim`. Golden tests per module against the §10 oracles; `Sim` integration
tests only for cross-module invariants (equilibrium sailing, tack transient,
energy sanity).

---

## 4. Environment interface and the shared sea surface

```rust
trait Environment {
    fn wind(&self, pos: Point3, t: f64) -> Vector3;          // incl. shear/twist/gusts
    fn wave(&self, x: f64, y: f64, t: f64) -> WaveSample;    // elevation, orbital vel,
                                                              // dyn pressure at depth, normal
    fn water(&self) -> &WaterProps;                           // rho, nu, gravity
}
```

The engine owns the sea state. The renderer must draw *the same* surface the
physics feels, without a per-frame CPU→GPU upload of height fields. Contract:

- `sim.sea_realization()` exposes the **spectral realization**: spectrum
  parameters, RNG seed, grid size, domain size, and the exact synthesis
  convention (spectrum discretization, phase generation, choppy-displacement
  factor).
- The renderer reruns the same synthesis on GPU; the engine samples it on CPU
  (small physics patch around the boat, inverse FFT or direct evaluation).
  Same realization + documented convention ⇒ same surface, two independent
  evaluators, zero per-frame traffic.
- The synthesis convention is therefore **part of the public API** and version-
  locked: a change in phase convention is a breaking change, tested by a golden
  test (fixed seed → fixed elevation samples).

This is the one place where engine and renderer share an algorithm rather than
an interface; it is deliberate and documented, and the golden test is the
fence.


---

## 4a. Built: what the implementation added to this design

The contract above is implemented. Three things had to be decided that this
document did not anticipate, and all three came from running the thing.

### Kinematics live on `StepCtx`, not in the modules

Every force model needs heel, trim, speed through the water, leeway and
apparent wind, and every one of those carries a sign convention. They are
derived **once**, on `StepCtx`, and no module is permitted its own definition.
A module that recomputed leeway would be free to pick the other sign and
nothing would catch it: the boat would still sail, with the keel lifting the
wrong way.

This paid for itself twice over. Two independent module authors flagged the
same sign error in `trim_angle` — it returned the negated pitch on the strength
of a plausible argument about the right-hand rule in a z-down frame, and the
argument was backwards. Because the derivation had one home, the fix was one
line, and it is now pinned by a test that asserts where the bow points rather
than what the convention is called.

### Captive degrees of freedom

`Captive` restrains chosen modes, exactly as a towing-tank dynamometer
restrains a model, and the assembled simulation runs with **trim and yaw held**.

This is a data gap made structural, not a shortcut. The boat format carries no
longitudinal position for keel, rudder or mast — published particulars do not
include them — so every lateral and aerodynamic force acts at `x = 0` and
produces no yaw moment. Integrating those modes would integrate an
identically-zero moment where the real one is not zero, and the boat would
settle to a heading and a trim that look like results and are artefacts. What
remains free — surge, sway, heave, roll — is exactly the balance a classical
velocity prediction solves, and needs no longitudinal information at all.

The restraint absorbs force without hiding it: the residual moment stays
visible in `last_wrench` and in the telemetry, which is what a captive
measurement is for. Lifting the restriction is a schema change plus sourced
positions, not a change to any force model.

### Steady sailing is solved, not waited for

`equilibrium::solve` drives the four free degrees of freedom to zero with a
Newton iteration over `Sim::applied_wrench`. It exists because **there is no
hydrodynamic damping in heave or roll yet**: buoyancy restores, nothing
dissipates, and an undamped oscillator never settles. A time-domain run
therefore cannot converge until radiation damping lands, and adding a damping
coefficient nobody measured would put a fabricated number underneath every
predicted speed.

When damping arrives, a long run must settle to what this solver returns, which
makes each a check on the other.

Three properties of the force models had to be respected to make it converge,
and each was found by measurement rather than by reading the code:

- **Finite differences are sized by the mesh, not by floating point.** Buoyancy
  is a pressure integral over a clipped triangle mesh, so a perturbation that
  moves the waterline by far less than a triangle's height differentiates the
  discretization. 1e-5 rad of heel — fifteen microns of waterline — produced a
  Jacobian of quantisation noise; 1e-3 rad produces a derivative.
- **The iteration must not start on a non-differentiable point.** The keel's
  downwash on the rudder goes as `sqrt(|C_L|)`, so its derivative with respect
  to leeway is *infinite* at zero leeway; the appendage heel factors are
  written in `|φ|`, so upright is a corner. An iteration started from rest sits
  on both. Seeding a degree of leeway and a few of heel, both taking the sign of
  the initial athwartships force, puts the first Jacobian on a smooth branch.
  Before that seeding, one tack converged and its mirror image stalled — the
  most confusing possible symptom of a perfectly symmetric model.
- **Newton steps need a physically sized trust region.** A full step from an
  upright guess lands with the deck under water, where the models are so
  nonlinear that halving never recovers. Capping the excursion per iteration —
  a quarter of a metre per second, a centimetre of immersion, three degrees of
  heel — keeps the direction and bounds the leap.

### One model gap closed from the source

Sail forces were not corrected for heel, and the consequence was not subtle: a
41-footer solved to 40° of heel in a working breeze because nothing ever
spilled wind out of the rig. The treatment is in the source after all —
Larsson, Eliasson & Orych, Fig 8.22, on Hazen's model: the apparent wind is
computed *in a plane that heels with the yacht*, the along-hull component
unchanged and the athwartships one scaled by `cos φ`. It now lives in
`StepCtx::apparent_wind_at`, which is where heel belongs: in the wind the rig
sees, not in the coefficients.
---

## 5. Decisions taken here (summary for review)

1. Workspace: `vela-core` (physics, renderer-free, enforced by crate boundary),
   `vela-app` (Bevy), `vela-cli` (headless; exists from phase 1 as separation
   proof and CI driver).
2. Boat format: RON canonical + JSON accepted; one self-contained file;
   SI units, degrees in file / radians in API; file frame x-fwd/y-port/z-up,
   converted once to Fossen body frame.
3. Hull as station offsets; physics mesh lofted at load; visual mesh out of
   scope for the physics file; STL import is a CLI tool.
4. Sails as parametric flying shape with engine-default control response,
   overridable per file.
5. `overrides` block reserved from v1 for external coefficient tables.
6. Force modules partitioned by coupling: `Aero`, `HullResistance`,
   `LateralSystem`, `BuoyancyFK`, `Radiation`. Wrenches in body frame about the
   body origin; stateful modules; fixed call order.
7. Added mass enters the mass matrix at setup, never as a wrench.
8. Telemetry with stable keys is a public API, driving HUD, CI, and polars.
9. Sea surface shared with the renderer by *realization* (seed + spectrum +
   convention), not by data transfer; convention is version-locked and
   golden-tested.
10. Convention-sensitive kinematics (heel, trim, leeway, apparent wind) are
    derived once on `StepCtx`; modules are forbidden their own definitions.
11. `Captive` restrains trim and yaw, because the boat format has no
    longitudinal positions. The restraint absorbs force without hiding it.
12. Steady sailing is solved by Newton over the free modes, not reached by
    integration, until radiation damping exists.
13. Heel enters the sail forces through the apparent wind in the heeled plane
    (Fig 8.22), not through the coefficients.

Known defect, tracked rather than absorbed: the lofted hull is not exactly
symmetric — its upright centre of buoyancy sits 0.8 mm off the centreline — so
the two tacks agree to 0.03 % rather than to machine precision. The offsets are
symmetric half-breadths, so this is the lofting, not the data.
