#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

struct tyger_hparams {
    int   sample_rate = 16000;
    int   n_fft       = 512;
    int   hop_length  = 160;
    int   win_length  = 400;
    int   n_mels      = 128;
    float preemph     = 0.97f;

    int enc_hidden       = 1024;
    int enc_ff           = 4096;
    int enc_layers       = 24;
    int enc_heads        = 8;
    int conv_kernel      = 9;
    int sub_factor       = 8;
    int sub_channels     = 256;
    int sub_kernel       = 3;
    int sub_stride       = 2;
    int sliding_window   = 57;
    int default_lookahead = 3;
    int max_positions    = 5000;

    int dec_hidden   = 640;
    int dec_layers   = 2;
    int vocab        = 13088;
    int blank_id     = 13087;
    int pad_id       = 0;
    int max_symbols  = 10;
    int num_prompts  = 128;
    int default_prompt = 101;

    int   tt_width      = 768;
    int   tt_layers     = 3;
    int   feature_layer = 15;
    float frame_s       = 0.08f;
    float window_s      = 8.0f;
    float hop_s         = 0.32f;

    float ep_thr = 0.91f, barge_thr = 0.5f, bc_thr = 0.4f, vad_db = -50.0f, min_speech_ms = 350.0f;
    int   ep_deb = 2, barge_deb = 2, bc_deb = 2, bc_refractory = 15, fallback_sil_frames = 15, warmup_frames = 6;

    int frame_samples() const { return (int) (frame_s * sample_rate + 0.5f); }
    int hop_frames() const { return (int) (hop_s / frame_s + 0.5f); }
    int hop_samples() const { return hop_frames() * frame_samples(); }
    int window_samples() const { return (int) (window_s * sample_rate + 0.5f); }
    int head_dim() const { return enc_hidden / enc_heads; }
};

struct enc_layer {
    ggml_tensor * ff1_norm_w, * ff1_norm_b, * ff1_up, * ff1_down;
    ggml_tensor * att_norm_w, * att_norm_b;
    ggml_tensor * wq, * wk, * wv, * wo, * wpos, * bias_u, * bias_v;
    ggml_tensor * conv_norm_w, * conv_norm_b, * pw1, * dw, * conv_ln_w, * conv_ln_b, * pw2;
    ggml_tensor * ff2_norm_w, * ff2_norm_b, * ff2_up, * ff2_down;
    ggml_tensor * out_norm_w, * out_norm_b;
};

// a weight kept on the host for the cpu-side parts (gru, rnnt). stored in its file type
struct host_mat {
    ggml_type          type = GGML_TYPE_F32;
    int64_t            n_in = 0, n_out = 0; // ne0, ne1
    size_t             row_bytes = 0;
    std::vector<uint8_t> data;
    const void * row(int64_t r) const { return data.data() + r * row_bytes; }
};

struct host_vec {
    std::vector<float> v;
    const float * data() const { return v.data(); }
};

struct turn_taker_weights {
    host_mat proj_user, proj_agent;
    host_vec proj_user_b, proj_agent_b, proj_energy_w, proj_energy_b;
    host_vec state_emb; // (5, width)
    host_vec in_norm_w, in_norm_b, out_norm_w, out_norm_b;
    host_mat w_ih[4], w_hh[4];
    host_vec b_ih[4], b_hh[4];
    host_vec state_head_w, state_head_b, hazard_w, hazard_b;
    host_mat yield0, bc0, ep0;
    host_vec yield0_b, yield2_w, yield2_b, bc0_b, bc2_w, bc2_b, ep0_b, ep2_w, ep2_b;
};

struct rnnt_weights {
    host_mat embedding; // (vocab, hidden) rows
    host_mat w_ih[2], w_hh[2];
    host_vec b_ih[2], b_hh[2];
    host_mat proj;
    host_vec proj_b;
    host_mat head;
    host_vec head_b;
};

struct tyger_model_impl {
    tyger_hparams hp;
    std::string   path;
    ggml_type     ftype = GGML_TYPE_F32;
    std::string   ftype_name;

    ggml_context          * ctx_w = nullptr;
    ggml_backend_buffer_t   buf_w = nullptr;
    ggml_backend_dev_t      dev   = nullptr; // compute device for the encoder
    bool                    gpu   = false;
    int                     n_threads = 4;
    bool                    has_asr = true;
    bool                    verbose = false;

    // encoder (device)
    ggml_tensor * sub_conv_in_w, * sub_conv_in_b;
    ggml_tensor * sub_dw_w[2], * sub_dw_b[2], * sub_pw_w[2], * sub_pw_b[2];
    ggml_tensor * sub_lin_w, * sub_lin_b;
    std::vector<enc_layer> layers;
    ggml_tensor * prompt_l1_w, * prompt_l1_b, * prompt_l1_p, * prompt_l2_w, * prompt_l2_b;
    ggml_tensor * enc_proj_w, * enc_proj_b;

    // host
    std::vector<float> mel_filters; // (n_mels, n_fft/2+1)
    turn_taker_weights tt;
    rnnt_weights       rnnt;
    std::vector<std::string> pieces;
    std::vector<uint8_t>     special;

    std::mutex compute_mutex; // one graph at a time on the device
    std::string info_json;

    ~tyger_model_impl();
};

bool tyger_model_load_impl(tyger_model_impl & m, const char * path, int backend, int device, int n_threads,
                           bool enable_asr, bool verbose, std::string & err);

void tyger_log(const char * fmt, ...);
