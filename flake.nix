{
  description = "tyger.cpp — realtime turn taking model for voice agents (C++/ggml port)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      forAllSystems = nixpkgs.lib.genAttrs [ "x86_64-linux" "aarch64-linux" ];

      # ggml is a git submodule pinned in the repo (.gitmodules). Nix fetches
      # don't include submodules, so the package build fetches it separately.
      ggmlRev = "a05c9922613c6a2230dbb1d0de679519ecafb4f0";
    in
    {
      # `nix develop` — C++ build environment
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          lib = pkgs.lib;

          # CMake's FindVulkan knows nothing about the nix store layout, so the
          # dev shell points it at the headers and the loader explicitly.
          vulkanCmakePath = lib.makeSearchPath "" [
            (lib.getDev pkgs.vulkan-headers)
            (lib.getDev pkgs.vulkan-loader)
            (lib.getBin pkgs.vulkan-loader)
          ];

          # quick-run wrappers, PATH based instead of shell aliases so they work
          # in any interactive shell (zsh/bash/fish)
          mkRunner =
            wrapperName: binName:
            pkgs.writeShellScriptBin wrapperName ''
              BIN="build/bin/${binName}"
              if [ ! -x "$BIN" ]; then
                echo "${binName} is not built yet — run tybuild first" >&2
                exit 1
              fi
              exec "$BIN" "$@"
            '';

          tybuild = pkgs.writeShellScriptBin "tybuild" ''
            set -e
            if [ ! -f third_party/ggml/CMakeLists.txt ]; then
              echo "tybuild: ggml submodule not present, initializing (shallow) …"
              git submodule update --init --depth 1
            fi
            cmake -S . -B build -G Ninja "$@" && cmake --build build
          '';

          banner = ''
            echo ""
            echo "  tyger.cpp dev shell"
            if [ ! -f third_party/ggml/CMakeLists.txt ]; then
              echo "    first time:   git submodule update --init       # fetch the ggml submodule"
            fi
            echo "    build:        tybuild                          # forwards extra args to cmake configure"
            if [ -z "$(ls models/*.gguf 2>/dev/null || true)" ]; then
              echo "    no model yet: nix develop .#python            # conversion env, see the hints there"
            else
              echo "    run:          tycli -m models/tyger-f16.gguf --user call.wav"
              echo "                  tycli -m models/tyger-f16.gguf --user call.wav --agent agent.wav --fast -t 8"
              echo "    serve:        tyserver -m models/tyger-f16.gguf --port 8765   # test page: http://127.0.0.1:8765/"
              echo "    self-check:   tytest models/tyger-f16.gguf clip.wav          # fast encoder vs offline encode"
              echo "    quantize:     build/bin/tyger-quantize models/tyger-f32.gguf models/tyger-q8_0.gguf q8_0"
            fi
            echo "    handy:        TYGER_PROFILE=1 tycli …   ·   GGML_VK_DISABLE_COOPMAT=1 for more exact vulkan matmuls"
            echo "    quants:       f16 for GPU, q8_0 for CPU (see README)"
            echo ""
          '';

          devShell = {
            default = pkgs.mkShell {
              packages =
                with pkgs;
                [
                  cmake
                  ninja
                  git
                  pkg-config
                  python3
                  pyenv # your .zshrc runs `eval "$(pyenv init -)"` on entry
                ]
                ++ [
                  tybuild
                  (mkRunner "tycli" "tyger-cli")
                  (mkRunner "tyserver" "tyger-server")
                  (mkRunner "tytest" "test-stream")
                ];
              buildInputs = [
                pkgs.vulkan-headers
                pkgs.vulkan-loader
                pkgs.shaderc # provides glslc, used by ggml to compile its vulkan shaders
              ];

              shellHook = ''
                export VULKAN_SDK="${lib.getDev pkgs.vulkan-headers}"
                export CMAKE_PREFIX_PATH="${vulkanCmakePath}:''${CMAKE_PREFIX_PATH:-}"
                ${banner}
              '';
            };

            # `nix develop .#python` — environment for scripts/convert_tyger_to_gguf.py
            python = pkgs.mkShell {
              packages = [
                (pkgs.python3.withPackages (
                  ps: with ps; [
                    numpy
                    gguf
                    librosa
                  ]
                ))
                pkgs.git
              ];

              shellHook = ''
                echo "  tyger.cpp model conversion shell"
                echo "    bundle:   hf download nineninesix/tyger --local-dir tyger-bundle"
                echo "    convert:  python scripts/convert_tyger_to_gguf.py tyger-bundle --outtype f32 --outfile models/tyger-f32.gguf"
                echo "    quantize: build/bin/tyger-quantize models/tyger-f32.gguf models/tyger-q8_0.gguf q8_0   (after tybuild)"
              '';
            };
          };
        in
        devShell
      );

      # `nix build` — the whole thing without a submodule checkout.
      packages = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          lib = pkgs.lib;

          ggmlSource = pkgs.fetchFromGitHub {
            owner = "ggml-org";
            repo = "ggml";
            rev = ggmlRev;
            hash = "sha256-a/NtFGX7jYITnj6EWE6+H2eyhsKutn0k4IiStihOagA=";
          };

          # this flake is also consumed as a plain `path:` input from the
          # config flake, which copies the raw working directory (no git
          # filtering) — keep ignored local output and multi-GB model files
          # out of the store copy
          src = lib.cleanSourceWith {
            src = self;
            filter =
              path: type:
              let
                base = builtins.baseNameOf (toString path);
              in
              !(
                lib.elem base [
                  "build"
                  "models"
                  "tyger-bundle"
                  ".venv"
                  ".git"
                  "models.d"
                ]
                || lib.hasPrefix "result" base
              );
          };
        in
        {
          default = pkgs.stdenv.mkDerivation {
            pname = "tyger";
            version = "0.1.0";

            inherit src;

            nativeBuildInputs = [ pkgs.cmake pkgs.shaderc pkgs.patchelf ];
            buildInputs = [
              pkgs.vulkan-headers
              pkgs.vulkan-loader
            ];

            # keep -march=native out of nix builds so the result stays portable;
            # flip to ON manually if you want a machine-tuned build
            cmakeFlags = [ "-DTYGER_NATIVE=OFF" ];

            postPatch = ''
              mkdir -p third_party
              rm -rf third_party/ggml
              ln -s ${ggmlSource} third_party/ggml
            '';

            # no install() rules in CMakeLists.txt, binaries land in build/bin
            installPhase = ''
              runHook preInstall
              install -Dm755 bin/tyger-cli bin/tyger-quantize bin/tyger-server -t $out/bin
              install -Dm755 bin/libtyger.so $out/lib/libtyger.so
              install -Dm644 ../include/tyger.h $out/include/tyger.h

              # cmake leaves the build-tree rpath (/build/…) on the binaries;
              # swap those entries for the real lib dir
              for f in $out/bin/tyger-cli $out/bin/tyger-quantize $out/bin/tyger-server $out/lib/libtyger.so; do
                clean=$(patchelf --print-rpath "$f" | tr ':' '\n' | grep -v '^/build' | tr '\n' ':')
                clean=''${clean%:}
                patchelf --set-rpath "''${clean:+$clean:}$out/lib" "$f"
              done
              runHook postInstall
            '';

            meta = with lib; {
              description = "Realtime turn taking model for voice agents (C++/ggml port)";
              homepage = "https://huggingface.co/nineninesix/tyger";
              license = licenses.mit;
              mainProgram = "tyger-cli";
              platforms = [ "x86_64-linux" "aarch64-linux" ];
            };
          };
        }
      );
      # NixOS service module — `services.tyger-server`
      nixosModules.tyger-server = import ./nix/module.nix { inherit self; };
      nixosModules.default = self.nixosModules.tyger-server;
    };
}