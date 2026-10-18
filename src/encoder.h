#pragma once

#include "mel.h"
#include "model.h"

#include <memory>
#include <vector>

// one compute context (its own backends and scheduler) over the shared weights
struct enc_runner {
    tyger_model_impl & m;
    ggml_backend_t dev = nullptr;
    ggml_backend_t cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> meta;
    std::unique_ptr<mel_extractor> mel;
    const mel_extractor & mel_ex() const { return *mel; }

    explicit enc_runner(tyger_model_impl & m);
    ~enc_runner();
    enc_runner(const enc_runner &) = delete;
    enc_runner & operator=(const enc_runner &) = delete;
};

// batch of equal-length mel clips ([n_frames][n_mels] each) through `n_layers` conformer layers.
// asr_head adds the prompt and encoder projectors (640 wide). out is [B][T][C], valid per clip
bool encode_offline(enc_runner & r, const std::vector<const float *> & mels, int n_frames,
                    const std::vector<int> & mel_valid, int n_layers, int lookahead, bool asr_head, int prompt_id,
                    std::vector<float> & out, int & T_out, int & C_out, std::vector<int> & out_valid);

// cache-aware incremental encoder state for one audio stream (fast mode)
struct enc_stream {
    mel_stream mel;
    int64_t frames_out = 0;
    int     n_cache = 0;
    std::vector<std::vector<float>> k, v, conv; // per layer: [left][C], [left][C], [kernel-1][C]
    void reset() { *this = enc_stream(); }
};

int enc_stream_ready(const tyger_model_impl & m, const enc_stream & s, int n_new);

// advance every stream by n_new encoder frames; out[i] gets n_new * C floats of layer `feature_layer`
bool encode_stream(enc_runner & r, const std::vector<enc_stream *> & streams, int n_new,
                   std::vector<std::vector<float>> & out);
