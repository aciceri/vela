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

          # What Bevy links against. Split from the runtime set below because
          # they are different questions: these have to be present to compile
          # and link, those have to be findable when a window opens.
          graphicsBuildInputs = lib.optionals pkgs.stdenv.hostPlatform.isLinux [
            pkgs.alsa-lib
            pkgs.udev
            pkgs.libxkbcommon
            pkgs.wayland
            pkgs.xorg.libX11
            pkgs.xorg.libXcursor
            pkgs.xorg.libXi
            pkgs.xorg.libXrandr
            pkgs.vulkan-loader
          ];

          # Loaded by name at runtime rather than linked, so a plain `cargo run`
          # needs them on the loader path. This is the one piece of NixOS
          # awkwardness the frontend introduces, and it is why `nix develop` sets
          # it rather than leaving it to the reader.
          graphicsRuntime = lib.optionals pkgs.stdenv.hostPlatform.isLinux [
            pkgs.vulkan-loader
            pkgs.libxkbcommon
            pkgs.wayland
            pkgs.xorg.libX11
            pkgs.xorg.libXcursor
            pkgs.xorg.libXi
            pkgs.xorg.libXrandr
          ];
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

              nativeBuildInputs = [ pkgs.pkg-config ];
              buildInputs = graphicsBuildInputs;

              meta = {
                description = "Sailing yacht physics engine";
                platforms = lib.platforms.unix;
              };
            };

            # Proves the engine stays wasm-compatible: no system BLAS, no
            # threads, no renderer deps sneaking in.
            #
            # Scoped to the engine and its headless driver rather than the whole
            # workspace, because the frontend's answer to "does this cross-compile
            # to wasm" is a different and much larger question — it needs a
            # graphics backend feature and `wasm-bindgen` to produce anything
            # runnable. Building it here would turn a fast structural check into
            # a slow one and would stop testing the thing this check is for.
            vela-wasm = rustPlatform.buildRustPackage {
              pname = "vela-wasm-check";
              version = "0.1.0";
              inherit src cargoLock;

              buildPhase = ''
                runHook preBuild
                cargo build --release --target wasm32-unknown-unknown \
                  -p vela-core -p vela-cli
                runHook postBuild
              '';
              # Cross-compiled artifacts cannot run here.
              doCheck = false;
              # The check is "does the engine still cross-compile"; the artifacts
              # are not the deliverable, so nothing is kept.
              installPhase = ''
                runHook preInstall
                mkdir -p $out
                runHook postInstall
              '';
            };

            # The frontend, cross-compiled and bundled for a browser.
            #
            # Separate from `vela-wasm` on purpose: this is the deliverable and
            # that is a structural check. This one links a graphics backend and
            # runs `wasm-bindgen`, so it is slow and its output is a directory
            # someone can serve.
            vela-web = rustPlatform.buildRustPackage {
              pname = "vela-web";
              version = "0.1.0";
              inherit src cargoLock;

              # Version-matched on purpose: `wasm-bindgen` embeds a schema
              # version in the module and the CLI refuses to read a module it did
              # not write. The bound is enforced from the other side too, by an
              # exact `wasm-bindgen` pin in `crates/vela-app/Cargo.toml`; the two
              # move together or the build fails loudly, which is the good case.
              nativeBuildInputs = [
                pkgs.wasm-bindgen-cli_0_2_126
                # `wasm-opt`. Run after `wasm-bindgen` rather than instead of the
                # release profile: the two shrink different things. Cargo drops
                # debug information; binaryen rewrites the code, and on a module
                # this size that is several more megabytes off the download.
                pkgs.binaryen
              ];

              buildPhase = ''
                runHook preBuild
                cargo build --release --target wasm32-unknown-unknown -p vela-app
                runHook postBuild
              '';
              doCheck = false;
              installPhase = ''
                runHook preInstall
                mkdir -p $out
                wasm-bindgen --no-typescript --target web --out-dir $out \
                  target/wasm32-unknown-unknown/release/vela-app.wasm

                # `-O2` rather than `-Oz`: this module contains a 60 Hz physics
                # loop, and trading its speed for bytes is the wrong way round
                # for a simulator. Size still falls substantially.
                wasm-opt -O2 --strip-debug -o $out/vela-app_bg.wasm.opt \
                  $out/vela-app_bg.wasm
                mv $out/vela-app_bg.wasm.opt $out/vela-app_bg.wasm

                # No assets directory: the shaders and the boat model are
                # embedded in the binary, so what a browser needs is the module,
                # the glue and the page. Anything else here would be a file nobody fetches.
                cp crates/vela-app/index.html $out/index.html

                # Printed because it is the number that decides whether anyone
                # waits for the link to load.
                echo "vela-web: $(du -h --apparent-size $out/vela-app_bg.wasm | cut -f1) wasm," \
                  "$(gzip -c $out/vela-app_bg.wasm | wc -c | numfmt --to=iec) gzipped"
                runHook postInstall
              '';
            };
          };

          checks = {
            # cargo build + cargo test, via the package's own check phase.
            vela = self.packages.${system}.vela;
            wasm = self.packages.${system}.vela-wasm;

            # The deliverable itself. In `checks` and not only in `packages`
            # because a frontend that stops cross-compiling is the failure this
            # project would notice last and care about most: the native build
            # keeps working, the tests keep passing, and the link stops existing.
            web = self.packages.${system}.vela-web;

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

              # The frontend is in the workspace, so `--all-targets` links Bevy's
              # windowing crates and needs what they need. Without these the check
              # fails on `wayland-sys` rather than on any lint.
              nativeBuildInputs = [ pkgs.pkg-config ];
              buildInputs = graphicsBuildInputs;

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
              pkgs.pkg-config
              pkgs.wasm-bindgen-cli_0_2_126
            ]
            ++ graphicsBuildInputs;

            # Bevy loads Vulkan and the windowing libraries by name at runtime,
            # which on NixOS means they have to be on the loader path explicitly.
            LD_LIBRARY_PATH = lib.makeLibraryPath graphicsRuntime;

            shellHook = ''
              echo "vela devshell — $(rustc --version)"
            '';
          };
        };
    };
}
