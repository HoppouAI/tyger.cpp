#!/usr/bin/env python3
"""Convert the nineninesix/tyger bundle (model.safetensors + configs) into a single tyger GGUF (F32 or F16)."""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("needs the gguf package: pip install gguf")

ARCH = "tyger"
FILE_TYPE = {"f32": 0, "f16": 1}

# nemotron matmul weights that the f16 and quant variants shrink; everything else stays f32
QUANT_SUFFIXES = (
    "feed_forward1.linear1.weight", "feed_forward1.linear2.weight",
    "feed_forward2.linear1.weight", "feed_forward2.linear2.weight",
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
    "self_attn.o_proj.weight", "self_attn.relative_k_proj.weight",
    "conv.pointwise_conv1.weight", "conv.pointwise_conv2.weight",
    "encoder.subsampling.linear.weight", "encoder_projector.weight",
    "prompt_projector.linear_1.weight", "prompt_projector.linear_2.weight",
    "decoder.lstm.weight_ih_l0", "decoder.lstm.weight_hh_l0",
    "decoder.lstm.weight_ih_l1", "decoder.lstm.weight_hh_l1",
    "decoder.decoder_projector.weight", "decoder.embedding.weight", "joint.head.weight",
)


def is_quantizable(name: str) -> bool:
    return not name.startswith("turn_taker.") and name.endswith(QUANT_SUFFIXES)


def load_safetensors(path: Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, v in header.items():
            if k == "__metadata__":
                continue
            if v["dtype"] != "F32":
                raise ValueError(f"{k}: expected F32, got {v['dtype']}")
            a, b = v["data_offsets"]
            f.seek(base + a)
            out[k] = np.frombuffer(f.read(b - a), dtype=np.float32).reshape(v["shape"]).copy()
    return out


def mel_filters(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    import librosa
    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=sr / 2, norm="slaney").astype(np.float32)


def remap(weights: dict[str, np.ndarray], num_prompts: int) -> dict[str, np.ndarray]:
    out = {}
    for k, v in weights.items():
        if k.startswith("nemotron."):
            k = k[len("nemotron."):]
        if k.endswith("pointwise_conv1.weight") or k.endswith("pointwise_conv2.weight"):
            if k.startswith("encoder.layers."):
                v = v[:, :, 0]
        if k.startswith("encoder.layers.") and k.endswith("conv.depthwise_conv.weight"):
            v = np.ascontiguousarray(v[:, 0, :])
        if k == "prompt_projector.linear_1.weight":
            h = v.shape[1] - num_prompts
            out["prompt_projector.linear_1.prompt_weight"] = np.ascontiguousarray(v[:, h:].T)
            v = np.ascontiguousarray(v[:, :h])
        if k.startswith("encoder.subsampling.layers.") and k.endswith("pointwise_conv.weight"):
            v = v[:, :, 0, 0]
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle", type=Path, help="local folder of nineninesix/tyger (config.json, model.safetensors, nemotron/)")
    ap.add_argument("--outtype", choices=list(FILE_TYPE), default="f16")
    ap.add_argument("--outfile", type=Path)
    args = ap.parse_args()

    root = args.bundle
    cfg = json.loads((root / "config.json").read_text())
    ncfg = json.loads((root / "nemotron" / "config.json").read_text())
    pcfg = json.loads((root / "nemotron" / "processor_config.json").read_text())
    tok = json.loads((root / "nemotron" / "tokenizer.json").read_text(encoding="utf-8"))
    ecfg = ncfg["encoder_config"]
    fe = pcfg["feature_extractor"]
    if cfg.get("model_type") != "tyger_nemotron_gru" or cfg.get("format_version") != 1:
        sys.exit("not a tyger_nemotron_gru bundle of format 1")

    outfile = args.outfile or Path(f"tyger-{args.outtype}.gguf")
    print(f"loading {root / 'model.safetensors'}")
    weights = remap(load_safetensors(root / "model.safetensors"), ncfg["num_prompts"])

    w = gguf.GGUFWriter(str(outfile), ARCH)
    w.add_name("tyger")
    w.add_description("TYGER speech decision model (nemotron-3.5-asr-streaming-0.6b + GRU turn-taker)")
    w.add_file_type(FILE_TYPE[args.outtype])
    w.add_uint32("general.quantization_version", 2)

    def u32(k, v): w.add_uint32(f"{ARCH}.{k}", int(v))
    def f32(k, v): w.add_float32(f"{ARCH}.{k}", float(v))

    # audio frontend
    u32("audio.sample_rate", fe["sampling_rate"])
    u32("audio.n_fft", fe["n_fft"])
    u32("audio.hop_length", fe["hop_length"])
    u32("audio.win_length", fe["win_length"])
    u32("audio.n_mels", fe["feature_size"])
    f32("audio.preemphasis", fe["preemphasis"])
    # encoder
    u32("encoder.hidden_size", ecfg["hidden_size"])
    u32("encoder.intermediate_size", ecfg["intermediate_size"])
    u32("encoder.num_layers", ecfg["num_hidden_layers"])
    u32("encoder.num_heads", ecfg["num_attention_heads"])
    u32("encoder.conv_kernel_size", ecfg["conv_kernel_size"])
    u32("encoder.subsampling_factor", ecfg["subsampling_factor"])
    u32("encoder.subsampling_channels", ecfg["subsampling_conv_channels"])
    u32("encoder.subsampling_kernel", ecfg["subsampling_conv_kernel_size"])
    u32("encoder.subsampling_stride", ecfg["subsampling_conv_stride"])
    u32("encoder.sliding_window", ecfg["sliding_window"])
    u32("encoder.default_lookahead", ecfg["default_num_lookahead_tokens"])
    u32("encoder.max_positions", ecfg["max_position_embeddings"])
    # rnnt
    u32("rnnt.decoder_hidden", ncfg["decoder_hidden_size"])
    u32("rnnt.decoder_layers", ncfg["num_decoder_layers"])
    u32("rnnt.vocab_size", ncfg["vocab_size"])
    u32("rnnt.blank_id", ncfg["blank_token_id"])
    u32("rnnt.pad_id", ncfg["pad_token_id"])
    u32("rnnt.max_symbols_per_step", ncfg["max_symbols_per_step"])
    u32("rnnt.num_prompts", ncfg["num_prompts"])
    u32("rnnt.default_prompt_id", ncfg["default_prompt_id"])
    # turn taker
    fr, arch = cfg["frontend"], cfg["controller"]["architecture"]
    u32("turn_taker.width", arch["width"])
    u32("turn_taker.layers", arch["layers"])
    u32("turn_taker.feature_layer", fr["layer"])
    f32("turn_taker.frame_period_s", fr["frame_period_s"])
    f32("turn_taker.live_window_s", fr["live_window_s"])
    f32("turn_taker.live_hop_s", fr["live_hop_s"])
    pol = cfg["default_policy"]
    for k in ("ep_thr", "barge_thr", "bc_thr", "vad_db", "min_speech_ms"):
        f32(f"policy.{k}", pol[k])
    for k in ("ep_deb", "barge_deb", "bc_deb", "bc_refractory_frames", "fallback_sil_frames", "warmup_frames"):
        u32(f"policy.{k}", pol[k])
    w.add_array(f"{ARCH}.turn_taker.state_names", cfg["state_names"])
    w.add_string(f"{ARCH}.source.weights_sha256", cfg["weights_sha256"])
    w.add_string(f"{ARCH}.source.frontend_revision", fr["revision"])

    # tokenizer: id -> piece, special flags (decode only)
    n_vocab = ncfg["vocab_size"]
    pieces = [""] * n_vocab
    special = [0] * n_vocab
    for p, i in tok["model"]["vocab"].items():
        if i < n_vocab:
            pieces[i] = p
    for a in tok.get("added_tokens", []):
        if a["id"] < n_vocab:
            pieces[a["id"]] = a["content"]
            special[a["id"]] = 1 if a.get("special") else 0
    special[ncfg["blank_token_id"]] = 1
    if not pieces[ncfg["blank_token_id"]]:
        pieces[ncfg["blank_token_id"]] = "<blank>"
    w.add_tokenizer_model("bpe-metaspace")
    w.add_token_list(pieces)
    w.add_array("tokenizer.ggml.token_special", special)

    weights["frontend.mel_filters"] = mel_filters(fe["sampling_rate"], fe["n_fft"], fe["feature_size"])

    n_q = 0
    for name in sorted(weights):
        v = weights[name]
        if args.outtype == "f16" and is_quantizable(name) and v.ndim == 2:
            v = v.astype(np.float16)
            n_q += 1
        if len(name) >= 64:
            sys.exit(f"tensor name too long for gguf: {name}")
        w.add_tensor(name, v)
    print(f"{len(weights)} tensors, {n_q} stored as f16")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main()
