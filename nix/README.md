# Nix packaging

This directory holds the NixOS service module (`module.nix`, wired up in the
flake as `nixosModules.default`, with `nixosModules.tyger-server` as an alias).
The flake in the repo root also provides a dev shell and a package; the
walkthrough for everything lives here.

Nix only reads files git knows about (tracked or staged), so stage or commit
new files before using any of it.

## Dev shell

```bash
nix develop
```

Comes with cmake, ninja, gcc, git and python3, plus the whole vulkan stack —
`vulkan-headers`, `vulkan-loader`, and `glslc` from shaderc. The shell exports
`CMAKE_PREFIX_PATH` and `VULKAN_SDK` pointing at the store paths, so
`find_package(Vulkan)` works out of the box and no Vulkan SDK install is
needed. The ggml vulkan backend builds with its full shader pipeline here.

Entering the shell prints a short cheat sheet (what it shows depends on the
state of the checkout: submodule present or not, `models/*.gguf` present or
not). The quick commands are small PATH wrappers instead of shell aliases, so
they behave the same in zsh, bash and fish:

| Command | What it does |
|---|---|
| `tybuild` | initializes the ggml submodule when missing (shallow clone), then `cmake -S . -B build -G Ninja` and `cmake --build build`. Extra args forward to configure, for example `tybuild -DTYGER_VULKAN=OFF` |
| `tycli` | runs `build/bin/tyger-cli` |
| `tyserver` | runs `build/bin/tyger-server` |
| `tytest` | runs `build/bin/test-stream`, the fast encoder self check |

By hand it is the same two cmake commands from the root README, after
`git submodule update --init`.

## Conversion shell

```bash
nix develop .#python
```

The model converter needs `numpy`, `gguf` and `librosa`; this shell carries
all three. It prints the `hf download` / `convert_tyger_to_gguf.py` /
quantize sequence on entry.

## Package

```bash
nix build
```

Builds everything without a submodule checkout — ggml is fetched at the rev
pinned in `.gitmodules`. The output has `bin/tyger-cli`, `bin/tyger-quantize`
and `bin/tyger-server` in `result/bin`, `libtyger.so` in `result/lib` and
`tyger.h` in `result/include`. The package builds with `TYGER_NATIVE=OFF` so
the binaries stay portable; machine specific tuning happens in the dev shell
instead. The install also rewrites the rpath of the binaries, which comes out
of cmake pointing at the build tree.

This flake is often consumed as a plain `path:` input from config flakes,
which copies the raw working directory — git filtering or not. The package
source is therefore filtered to keep ignored local output (`build/`), the
model files (`models/`) and other local junk out of the store copy.

## NixOS module: `services.tyger-server`

`nixosModules.default` declares `services.tyger-server`: the websocket server
as a hardened systemd unit under a dynamic user. From a system config's
flake:

```nix
inputs.tyger.url = "github:HoppouAI/tyger.cpp";
```

and in the system module:

```nix
{
  imports = [ tyger.nixosModules.default ];

  services.tyger-server = {
    enable = true;
    model = "/var/lib/tyger/tyger-f16.gguf";
  };
}
```

The options mirror the server's CLI:

| Option | Default | Maps to |
|---|---|---|
| `model` | — | `-m` (path to a gguf) |
| `host` | `127.0.0.1` | `--host` |
| `port` | `8765` | `--port` |
| `fast` | `false` | `--fast` |
| `cpuOnly` | `false` | `--cpu` |
| `noAsr` | `false` | `--no-asr` |
| `verbose` | `false` | `-v` |
| `threads` | — | `-t N` |
| `vulkanDevice` | — | `--device N` |
| `maxBatch` | — | `--max-batch N` |
| `openFirewall` | `false` | adds `port` to `networking.firewall.allowedTCPPorts` |
| `package` | this flake's `tyger` | which binary the unit runs |
| `extraArgs`, `extraEnvironment` | `[]`, `{}` | raw passthrough to the unit |

Notes:

- On GPU the host needs `hardware.graphics.enable` — the unit points
  `XDG_DATA_DIRS` at `/run/opengl-driver/share` for the vulkan ICD manifests
  and adds the `render` and `video` groups for the device nodes. With
  `cpuOnly = true` none of that is set.
- The service reads `model` as a dynamic user, so the file must be world
  readable. A nix store path or somewhere like `/var/lib` works; a locked
  home dir does not (the unit sets `ProtectHome = read-only`).
- It binds loopback and does not authenticate — for LAN access set
  `host = "0.0.0.0"` and `openFirewall = true`.
- Per session options (thresholds, `encoder`, `frames`, `format`, `channels`
  …) stay on the websocket, they are not server flags. The mic test page is
  served on the same port.
- The unit waits for `network.target`, restarts on failure and gets a 5 min
  start timeout for large models. A port below 1024 gets
  `CAP_NET_BIND_SERVICE` automatically.

Check it with `systemctl status tyger-server`, watch the journal with
`journalctl -u tyger-server -f`, and confirm the GPU came up with
`curl http://127.0.0.1:8765/info` — `"gpu":true` means the vulkan backend
initialized.