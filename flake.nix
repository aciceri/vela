{
  description = "Vela — sailing yacht physics engine (renderer-free core)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    flake-parts.url = "github:hercules-ci/flake-parts";
    flake-parts.inputs.nixpkgs-lib.follows = "nixpkgs";

    # Binary-distributed Rust toolchains: needed for the wasm32-unknown-unknown
    # target, which nixpkgs' rustc does not ship by default.
    rust-overlay.url = "github:oxalica/rust-overlay";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";

    nix-github-actions.url = "github:nix-community/nix-github-actions";
    nix-github-actions.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    inputs@{
      self,
      nixpkgs,
      flake-parts,
      rust-overlay,
      nix-github-actions,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      # CI matrix: one GitHub Actions runner per check attribute.
      # Restricted to systems GitHub actually provides free runners for.
      flake.githubActions = nix-github-actions.lib.mkGithubMatrix {
        checks = nixpkgs.lib.getAttrs [ "x86_64-linux" ] self.checks;
      };

      perSystem =
        { pkgs, system, ... }:
        let
          lib = pkgs.lib;

          # What the derivations build with: `minimal` plus exactly the
          # components the checks invoke. The `default` profile drags in
          # rust-docs (hundreds of megabytes) that no build step ever reads,
          # which is pure CI download time.
          buildToolchain = pkgs.rust-bin.stable.latest.minimal.override {
            extensions = [
              "clippy"
              "rustfmt"
            ];
            # Reserved for the future Bevy/wasm frontend; the core must keep
            # compiling for it even while no frontend exists.
            targets = [ "wasm32-unknown-unknown" ];
          };

          # What a human gets from `nix develop`: the full profile plus the
          # editor-facing components.
          devToolchain = pkgs.rust-bin.stable.latest.default.override {
            extensions = [
              "rust-src"
              "rust-analyzer"
            ];
            targets = [ "wasm32-unknown-unknown" ];
          };

          rustPlatform = pkgs.makeRustPlatform {
            cargo = buildToolchain;
            rustc = buildToolchain;
          };

          # Keep the store path free of build artifacts, editor state and docs
          # churn so that a docs-only edit does not invalidate the build.
          src = lib.fileset.toSource {
            root = ./.;
            fileset = lib.fileset.unions [
              ./Cargo.toml
              ./Cargo.lock
              ./crates
              ./boats
            ];
          };

          cargoLock.lockFile = ./Cargo.lock;
        in
        {
          _module.args.pkgs = import nixpkgs {
            inherit system;
            overlays = [ rust-overlay.overlays.default ];
          };

          packages = {
            default = self.packages.${system}.vela;

            # Builds the whole workspace. `cargo test` runs in the check phase,
            # so this derivation succeeding means the test suite passed.
            vela = rustPlatform.buildRustPackage {
              pname = "vela";
              version = "0.1.0";
              inherit src cargoLock;

              meta = {
                description = "Sailing yacht physics engine";
                platforms = lib.platforms.unix;
              };
            };

            # Proves the core stays wasm-compatible: no system BLAS, no threads,
            # no renderer deps sneaking in.
            vela-wasm = rustPlatform.buildRustPackage {
              pname = "vela-wasm-check";
              version = "0.1.0";
              inherit src cargoLock;

              buildPhase = ''
                runHook preBuild
                cargo build --release --target wasm32-unknown-unknown --workspace
                runHook postBuild
              '';
              # Cross-compiled artifacts cannot run here.
              doCheck = false;
              # The check is "does the core still cross-compile"; the artifacts
              # are not the deliverable, so nothing is kept.
              installPhase = ''
                runHook preInstall
                mkdir -p $out
                runHook postInstall
              '';
            };
          };

          checks = {
            # cargo build + cargo test, via the package's own check phase.
            vela = self.packages.${system}.vela;
            wasm = self.packages.${system}.vela-wasm;

            fmt = rustPlatform.buildRustPackage {
              pname = "vela-fmt";
              version = "0.1.0";
              inherit src cargoLock;
              buildPhase = "cargo fmt --all --check";
              doCheck = false;
              installPhase = "mkdir -p $out";
            };

            clippy = rustPlatform.buildRustPackage {
              pname = "vela-clippy";
              version = "0.1.0";
              inherit src cargoLock;
              # cargo-clippy ships in buildToolchain, already on PATH.
              buildPhase = "cargo clippy --all-targets --workspace -- -D warnings";
              doCheck = false;
              installPhase = "mkdir -p $out";
            };
          };

          devShells.default = pkgs.mkShell {
            packages = [
              devToolchain
              pkgs.cargo-nextest
            ];

            shellHook = ''
              echo "vela devshell — $(rustc --version)"
            '';
          };
        };
    };
}
