#!/usr/bin/env python3
"""PyTorch reference for validating tyger.cpp, built on tyger-s2s's own runtime modules and HF transformers."""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def load_runtime(s2s: Path):
    # import tyger-s2s's ears/tyger modules without its package __init__ (which pulls in the whole app)
    d = s2s / "app" / "plugins" / "ears" / "tyger"
    pkg = types.ModuleType("tyref")
    pkg.__path__ = [str(d)]
    sys.modules["tyref"] = pkg
    import tyref.bundle, tyref.constants, tyref.rules, tyref.session  # noqa: E401
    return tyref


def read_wav(path, channel=-1):
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(1) if channel < 0 else x[:, channel]
    if sr != 16000:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    return np.ascontiguousarray(x, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["encode", "transcribe", "stream", "mel"])
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--s2s", type=Path, required=True, help="tyger-s2s checkout")
    ap.add_argument("--user", type=Path, required=True)
    ap.add_argument("--agent", type=Path)
    ap.add_argument("--channel", type=int, default=-1)
    ap.add_argument("--layers", type=int, default=15)
    ap.add_argument("--lookahead", type=int, default=0)
    ap.add_argument("--speaking", default="", help="A-B,C-D seconds where the agent talks")
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    rt = load_runtime(a.s2s)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    nemotron, proc, tt, cfg = rt.bundle.load(a.bundle)
    dev = a.device
    nemotron = nemotron.to(dev)
    tt = tt.to(dev)
    SR = rt.constants.SR
    u = read_wav(a.user, a.channel)
    if a.seconds > 0:
        u = u[: int(a.seconds * SR)]

    if a.cmd == "mel":
        inp = proc(u, sampling_rate=SR, return_tensors="pt")
        np.save(a.out, inp["input_features"][0].numpy())
        return

    if a.cmd == "encode":
        inp = proc(u, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            eo = nemotron.encoder(input_features=inp["input_features"].to(dev), attention_mask=inp["attention_mask"].to(dev),
                                  num_lookahead_tokens=a.lookahead, output_hidden_states=True)
        np.save(a.out, eo.hidden_states[a.layers][0].float().cpu().numpy())
        return

    if a.cmd == "transcribe":
        with torch.no_grad():
            out = nemotron.generate(**proc(u, sampling_rate=SR, return_tensors="pt").to(dev))
        text = proc.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
        a.out.write_text(text, encoding="utf-8")
        print(text)
        return

    # stream: the plain single-session reference runtime, recording every frame's raw head outputs
    ag = read_wav(a.agent, -1) if a.agent else np.zeros_like(u)
    spans = []
    for part in filter(None, a.speaking.split(",")):
        x, y = part.split("-")
        spans.append((float(x), float(y)))
    rules, C = rt.rules, rt.constants
    sess = rt.session.EarsSession(0, ep_thr=0.915, ep_deb=2, barge_thr=0.4, barge_deb=2, bc_thr=0.4, bc_deb=2)
    state = [None]

    @torch.no_grad()
    def l15(w):
        inp = proc(w, sampling_rate=SR, return_tensors="pt")
        eo = nemotron.encoder(input_features=inp["input_features"].to(dev), attention_mask=inp["attention_mask"].to(dev),
                              num_lookahead_tokens=0, output_hidden_states=True)
        return eo.hidden_states[C.L15][0].float()

    @torch.no_grad()
    def step(hu, ha, act_db, prev):
        act = ((torch.tensor([[float(act_db)]], device=dev) + 50) / 25).clamp(-3, 3)
        pv = torch.tensor([[prev]], dtype=torch.long, device=dev)
        o, state[0] = tt(hu[None, None], ha[None, None], act, pv, state[0])
        return o[0, 0].cpu().numpy()

    @torch.no_grad()
    def transcribe(w):
        out = nemotron.generate(**proc(w, sampling_rate=SR, return_tensors="pt").to(dev))
        return proc.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()

    frames, events = [], []
    n_hops = len(u) // C.HOP_N
    for h in range(n_hops):
        t_end = (h + 1) * C.HOP_N / SR
        mode = "speaking" if any(x < t_end <= y for x, y in spans) else "listen"
        uh = u[h * C.HOP_N:(h + 1) * C.HOP_N]
        ah = ag[h * C.HOP_N:(h + 1) * C.HOP_N]
        if len(ah) < C.HOP_N:
            ah = np.pad(ah, (0, C.HOP_N - len(ah)))
        if rules.begin_hop(sess, uh, ah, mode):
            state[0] = None
        hu, ha = l15(sess.win_u), l15(sess.win_a)
        act = rules.frame_energy(sess.win_u)
        if rules.switch_mode(sess, mode):
            state[0] = None
        n = min(C.HOP_FRAMES, len(hu), len(ha), len(act))
        for i in range(-n, 0):
            o = step(hu[i], ha[i], act[i], sess.prev_state)
            t = h * C.HOP_N / SR + (C.HOP_FRAMES + i + 1) * C.FRAME_S
            frames.append([round(t, 2), mode, *[float(v) for v in o], float(act[i])])
            evs, final, reset = rules.on_frame(sess, mode, o, float(act[i]), left=-i - 1)
            if final is not None:
                ev, wav = final
                ev.pop("seg_texts", None)
                ev["transcript"] = transcribe(wav)
            for e in evs:
                if e["kind"] != "FRAME":
                    e["t"] = round(t, 2)
                    events.append(e)
            if reset:
                state[0] = None
    a.out.write_text(json.dumps({"frames": frames, "events": events}, indent=1), encoding="utf-8")
    for e in events:
        print(e)


if __name__ == "__main__":
    main()
