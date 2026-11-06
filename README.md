# tyger.cpp

A C++ and GGUF port of [nineninesix/tyger](https://huggingface.co/nineninesix/tyger), the realtime turn taking model for voice agents. It listens to both sides of a conversation (the user's mic and the agent's own playback) and tells you, every 80 ms, whether the user is done talking, whether they are trying to cut the agent off, and when a short "mm hm" style backchannel fits. It also transcribes the user's turn with the bundled Nemotron streaming ASR.

Everything runs on [ggml](https://github.com/ggml-org/ggml) with a CPU backend and a Vulkan backend. No Python, no PyTorch and no CUDA at runtime.

## What is in here

- **`tyger` shared library** with a plain C API (`include/tyger.h`). Load a model, open sessions, push audio, get events back.
- **The full streaming runtime**, not just the network. The turn rules from the reference runtime (warmup, debounce, endpoint, barge in, backchannel refractory, 15 s segment cuts, 20 s turn cap, rolling partial transcripts) live in the library, so every integration gets the same behaviour.
- **Two encoder modes.** `exact` re-encodes the rolling 8 s window every 320 ms hop, the same way the reference does it. `fast` keeps a cache per layer and only encodes the new frames.
- **Batching.** Many conversations can share one encode per hop (`tyger_feed_batch`), which is what the server and the Python wrapper do.
- **`tyger-server`**, a WebSocket server with a built in mic test page.
- **Python bindings** (`python/tyger_cpp`), a small ctypes layer plus an async wrapper shaped like the reference runtime's ears plugin.
- **`tyger-cli`** for running wav files, benchmarking and dumping debug output.
- **`tyger-quantize`** for F16, Q8_0, Q6_K, Q5_K and Q4_K files.

## Models and quants

The Nemotron encoder, prompt projector and RNNT decoder are quantized. The turn taker GRU and its heads always stay F32, they are tiny and they are what the decisions come from.

| File | Size | Notes |
|---|---|---|
| `tyger-f32.gguf` | 2.6 GB | reference precision |
| `tyger-f16.gguf` | 1.33 GB | recommended for GPU, no decision differences from F32 in testing |
| `tyger-q8_0.gguf` | 737 MB | recommended for CPU, transcripts match F32 apart from a comma |
| `tyger-q6_k.gguf` | 589 MB | small punctuation and filler word differences, close to Q8_0 on decisions |
| `tyger-q4_k.gguf` | 461 MB | Q4_K_M style mix, worse transcripts and noticeably different decisions, only use it when memory is tight |

How often each quant's turn decisions differ from F32, across 188 ENDPOINT, BARGE and BACKCHANNEL events in four test runs (about 5 minutes of audio, exact mode, Vulkan). An event that moved by one frame counts as one missing and one extra.

| Quant | Missing | Extra |
|---|---|---|
| F16 | 0 | 0 |
| Q8_0 | 1 | 2 |
| Q6_K | 2 | 1 |
| Q4_K | 12 | 9 |

## Turn taking without ASR

The turn decisions never use the ASR, they read the output of encoder layer 15 plus the audio energy. So you can run them on their own, with identical results and less memory.

The quick way is to turn ASR off when loading any model: `--no-asr` on the CLI and server, `asr=False` in Python, or `enable_asr = false` in the C API. Layers 16 to 24, the decoder and the tokenizer are not loaded, and hops no longer stall on transcripts. ENDPOINT, BARGE, BACKCHANNEL and FRAME all still fire, the text fields are just empty and there are no PARTIAL events.

To also get a smaller file, export a turn only GGUF with `--turn-only`:

```bash
build/bin/Release/tyger-quantize models/tyger-f32.gguf models/tyger-turn-q8_0.gguf q8_0 --turn-only
```

| File | Size |
|---|---|
| `tyger-turn-f32.gguf` | 1.59 GB |
| `tyger-turn-f16.gguf` | 822 MB |
| `tyger-turn-q8_0.gguf` | 464 MB |
| `tyger-turn-q6_k.gguf` | 371 MB |
| `tyger-turn-q4_k.gguf` | 291 MB |

These must be loaded with ASR off, otherwise the load fails with an error saying so. Their events are byte identical to the matching full file with ASR off. The Nemotron encoder can not be removed, the GRU needs its first 15 layers.

Q4_K loses the most on decisions, and a turn only file is all decisions, so Q8_0 or Q6_K are the better picks here.

## Building

You need CMake 3.18 or newer and a C++17 compiler (Visual Studio 2022 on Windows, gcc or clang elsewhere). For the GPU backend install the [Vulkan SDK](https://vulkan.lunarg.com/) first so CMake can find `glslc`.

```bash
git clone --recursive https://github.com/HoppouAI/tyger.cpp
```

```bash
cmake -S . -B build
```

```bash
cmake --build build --config Release
```

Binaries land in `build/bin/Release` with the Visual Studio generator and `build/bin` with the others.

CMake options:

| Option | Default | What it does |
|---|---|---|
| `TYGER_VULKAN` | ON | build the Vulkan backend, turn it off for a CPU only build |
| `TYGER_NATIVE` | ON | optimize for the build machine's CPU (AVX2 on MSVC) |
| `TYGER_BUILD_TOOLS` | ON | cli, quantize and server |
| `TYGER_BUILD_TESTS` | ON | `test-stream`, checks the fast encoder against an offline encode |

## Getting a GGUF

Download the original bundle, convert it, then quantize. The converter needs `numpy`, `gguf` and `librosa` (librosa is only used to build the mel filterbank that gets stored in the file).

```bash
hf download nineninesix/tyger --local-dir tyger-bundle
```

```bash
python scripts/convert_tyger_to_gguf.py tyger-bundle --outtype f32 --outfile models/tyger-f32.gguf
```

```bash
build/bin/Release/tyger-quantize models/tyger-f32.gguf models/tyger-q8_0.gguf q8_0
```

Quantize types are `f32`, `f16`, `q8_0`, `q6_k`, `q5_k`, `q4_k` and `q4_k_s`. Converting straight to F16 also works with `--outtype f16`. Add `--turn-only` to leave out everything the ASR needs, see [Turn taking without ASR](#turn-taking-without-asr).

## Command line

```bash
tyger-cli -m models/tyger-f16.gguf --user call.wav
```

Prints every ENDPOINT, BARGE, BACKCHANNEL and PARTIAL with its time, plus hop latency and the real time factor at the end. Useful flags:

- `--agent agent.wav` the agent's own audio as the second ear, and `--speaking 6-12,20-26` for the seconds where the agent was talking (barge in only fires in speaking mode).
- `--fast` for the incremental encoder, `--cpu` to skip the GPU, `-t 8` for CPU threads.
- `--realtime` paces the feed like a live call.
- `--json` prints each hop's events as one JSON line, `--frames-csv out.csv` dumps every frame's head outputs.
- `--transcribe` just transcribes the whole file and exits.

Input wavs can be any sample rate and channel count, they are mixed down and resampled to 16 kHz mono (`--channel N` picks one channel instead).

## WebSocket server

```bash
tyger-server -m models/tyger-f16.gguf --port 8765
```

Open `http://127.0.0.1:8765/` in a browser for a test page that streams your mic and shows the live state, meters, partials and endpoints. `GET /info` returns the model info as JSON.

For your own client, connect a WebSocket to the same port, for example `ws://127.0.0.1:8765/ws?format=s16&channels=1&encoder=exact`.

Query options (all optional):

| Name | Values | Default |
|---|---|---|
| `format` | `s16` (16 bit little endian) or `f32` | `s16` |
| `channels` | `1` for the user only, `2` for user and agent interleaved | `1` |
| `encoder` | `exact` or `fast` | `exact`, or `fast` when the server runs with `--fast` |
| `frames` | `1` to get a FRAME event every 80 ms | `0` |
| thresholds | `ep_thr`, `ep_deb`, `barge_thr`, `barge_deb`, `bc_thr`, `bc_deb`, `bc_refractory`, `theta`, `partial_refresh_s`, `fallback_sil_frames`, `prompt_id` | model defaults |

Then:

- Send 16 kHz PCM as **binary** messages, any size. Every full 320 ms hop is processed.
- Send **text** JSON to control the session:
  - `{"type":"mode","mode":"speaking"}` while the agent is talking, `{"type":"mode","mode":"listen"}` when it stops.
  - `{"type":"reset"}` to start a new turn. After a `BARGE` the turn starts with the speech that triggered it. Add `"keep":true` when the client decides on a barge itself, for example after the user talked over the agent for a while, and the last 2 s of user audio from speaking mode start the turn instead of being dropped.
  - `{"type":"config","ep_thr":0.9}` to change thresholds or the encoder on the fly.
- The server answers with `{"type":"ready",...}` once, then `{"type":"events","events":[...]}` for every hop that produced something, and `{"type":"error","message":"..."}` if something is wrong.

All connected sessions with a hop ready are encoded together in one batch. The server warms up the GPU pipelines at start so the first call does not stall.

## Events

| Kind | When | Fields |
|---|---|---|
| `FRAME` | every 80 ms frame, only when frames are enabled | `state`, `p_end`, `yield_prob`, `bc_prob`, `hazard` |
| `ENDPOINT` | the user finished their turn | `transcript`, `turn_s`, `segments`, `by` |
| `BARGE` | the user is talking over the agent (speaking mode) | `yield_prob`, `onset_s` |
| `BACKCHANNEL` | a good moment for a short acknowledgement | `partial`, `bc_prob`, `turn_s` |
| `PARTIAL` | the transcript so far changed | `text` |

Every event has `t`, the session audio time in seconds at the end of the frame.

## C API

```c
#include "tyger.h"

tyger_model_params mp = tyger_model_default_params();
tyger_model * model = tyger_model_load("tyger-f16.gguf", mp);

tyger_session_params sp = tyger_session_default_params(model);
sp.encoder_mode = TYGER_ENCODER_EXACT;
tyger_session * s = tyger_session_open(model, &sp);

// any number of 16 kHz float samples, agent may be NULL
const tyger_event * ev;
int n = tyger_session_feed(s, mic, agent, n_samples, TYGER_MODE_LISTEN, &ev);
for (int i = 0; i < n; i++) {
    if (ev[i].kind == TYGER_EVENT_ENDPOINT) printf("%.2f %s\n", ev[i].t_s, ev[i].text);
}
// or tyger_session_events_json(s) for the same thing as a json array

tyger_session_close(s);
tyger_model_free(model);
```

`tyger_feed_batch` runs exactly one hop for several sessions in one encode, read each session's events afterwards with `tyger_session_events`. `tyger_transcribe` does an offline transcript of a whole clip. Sessions are independent, but feed one session from one thread at a time.

## Python

```bash
pip install -e python
```

The package looks for the shared library next to itself, then in the repo's `build/bin/Release` and `build/bin`, or wherever the `TYGER_LIB` environment variable points.

```python
import tyger_cpp

model = tyger_cpp.Model("models/tyger-f16.gguf")      # backend="cpu" or "vulkan" to force one
with model.session(encoder="exact", ep_thr=0.915) as s:
    for chunk in mic_chunks():                          # float32 or int16, 16 kHz, any length
        for e in s.feed(chunk, agent=None, mode="listen"):
            if e["kind"] == "ENDPOINT":
                print(e["transcript"])

print(model.transcribe(clip))
```

`tyger_cpp.ears.TygerEars` has the same surface as the reference runtime's ears plugin (`open`, `close`, `async feed(session, user_hop, agent_hop, mode)`, `thresholds`, `set_threshold`, `full`, `active`) and gathers hops from all open conversations into batched ticks, so it can stand in for the PyTorch plugin.

## Exact or fast

The exact mode re-encodes the last 8 s every hop. It reproduces the reference runtime's events exactly, and on a GPU it is cheap enough (about 40 ms per hop on an RTX 3060). On a CPU it is too slow for live use, so use fast there.

The fast mode is a cache aware streaming encode: it keeps 56 frames of attention cache and the convolution state per layer and only runs the new frames each hop. Its encoder output matches an offline full clip encode to about 1e-4, but that is a slightly different context than the 8 s window the model's head was run with in the reference, so a few borderline backchannel or barge decisions can land one hop apart. Endpoints and transcripts matched in testing.

## Speed

Measured on a 30 s two speaker clip with ASR on (Intel i5 10400, RTX 3060, Windows 11). The hop budget for real time is 320 ms.

Hop latency, median and 90th percentile, in milliseconds. Exact mode on the GPU, fast mode on the CPU (6 threads).

| File | Vulkan exact, ASR on | CPU fast, ASR on | CPU fast, no ASR |
|---|---|---|---|
| F32 | 58 / 109 | 200 / 1696 | |
| F16 | 44 / 80 | 148 / 1294 | 122 / 141 |
| Q8_0 | 46 / 84 | 123 / 1007 | 101 / 118 |
| Q6_K | 49 / 92 | 113 / 722 | |
| Q4_K | 45 / 89 | 110 / 815 | 88 / 95 |

On the GPU every quant is far inside the budget, and the bigger win from quantizing is memory, not speed. On the CPU the turn taking itself fits easily, but the ASR is the expensive part: partial transcripts re-encode the whole turn so far every 0.8 s, and on this CPU that pushes some hops past the budget. For a CPU only setup either run without ASR, raise `partial_refresh_s` (or set it to 0 to turn partials off and only transcribe at the endpoint), or use the GPU.

## Accuracy

Checked against the PyTorch reference (`scripts/reference.py`, `scripts/compare.py`, `scripts/compare_stream.py`):

- CPU F32 encoder output after 15 layers is within 1.4e-4 of PyTorch, transcripts are identical.
- Streaming in exact mode reproduces every event of the reference, times and transcripts, on CPU and Vulkan at F32 and F16, both in listen only clips and with agent audio and speaking spans (barge ins included).

## Nix and NixOS

The repo is also a flake. `nix develop` drops into a build shell with cmake, ninja, gcc and the whole vulkan stack (headers, loader and `glslc`), plus the quick commands `tybuild` (initializes the submodule if needed, then configures and builds), `tycli`, `tyserver` and `tytest` for running the binaries in `build/bin`. The model conversion environment (numpy, gguf, librosa) is `nix develop .#python`, and `nix build` produces the same install layout the NixOS module ships, without needing the submodule checked out — ggml comes from the pinned submodule rev and the output is a portable build.

The flake also carries a NixOS module (`nixosModules.default`) that runs `tyger-server` as a hardened systemd service (`services.tyger-server`). Its options mirror the server's CLI flags — the full walkthrough with the option table and host requirements lives in [nix/README.md](nix/README.md).

## Notes

- Vulkan uses F32 accumulation for every matmul here, which keeps F16 weights in line with F32. If a GPU still drifts, setting the environment variable `GGML_VK_DISABLE_COOPMAT=1` (and `GGML_VK_DISABLE_COOPMAT2=1`) makes Vulkan matmuls more exact at some speed cost on quantized files.
- `TYGER_PROFILE=1` prints per hop timings for each stage.
- The silence fallback endpoint from the original bundle is off by default, like in the reference runtime. Set `fallback_sil_frames` to turn it on.

## License

The model weights are released by nineninesix under the OpenMDW 1.1 license, with the Nemotron parts under NVIDIA's terms, see the model card. ggml is MIT.
