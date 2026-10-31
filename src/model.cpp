#include "model.h"

#include "ggml-cpu.h"
#include "gguf.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <map>
#include <thread>

static bool g_verbose = false;

void tyger_log(const char * fmt, ...) {
    if (!g_verbose) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "tyger: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}

static void ggml_log_cb(ggml_log_level level, const char * text, void * user) {
    if (g_verbose || level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

tyger_model_impl::~tyger_model_impl() {
    if (buf_w) {
        ggml_backend_buffer_free(buf_w);
    }
    if (ctx_w) {
        ggml_free(ctx_w);
    }
}

namespace {

struct gguf_reader {
    gguf_context * g = nullptr;
    ggml_context * meta = nullptr;
    FILE * f = nullptr;
    size_t data_off = 0;

    ~gguf_reader() {
        if (f) fclose(f);
        if (g) gguf_free(g);
        if (meta) ggml_free(meta);
    }

    bool open(const char * path, std::string & err) {
        gguf_init_params p = { true, &meta };
        g = gguf_init_from_file(path, p);
        if (!g) {
            err = std::string("failed to read gguf: ") + path;
            return false;
        }
        f = fopen(path, "rb");
        if (!f) {
            err = std::string("failed to open: ") + path;
            return false;
        }
        data_off = gguf_get_data_offset(g);
        return true;
    }

    int64_t key(const char * k) const { return gguf_find_key(g, k); }

    void u32(const char * k, int & dst) const {
        int64_t i = key(k);
        if (i >= 0) dst = (int) gguf_get_val_u32(g, i);
    }
    void f32(const char * k, float & dst) const {
        int64_t i = key(k);
        if (i >= 0) dst = gguf_get_val_f32(g, i);
    }

    bool read(const char * name, void * dst, size_t nbytes, std::string & err) {
        int64_t ti = gguf_find_tensor(g, name);
        if (ti < 0) {
            err = std::string("missing tensor ") + name;
            return false;
        }
        size_t off = data_off + gguf_get_tensor_offset(g, ti);
#ifdef _WIN32
        if (_fseeki64(f, (long long) off, SEEK_SET) != 0) {
#else
        if (fseeko(f, (off_t) off, SEEK_SET) != 0) {
#endif
            err = "seek failed";
            return false;
        }
        if (fread(dst, 1, nbytes, f) != nbytes) {
            err = std::string("short read on ") + name;
            return false;
        }
        return true;
    }

    ggml_tensor * meta_tensor(const char * name) const { return ggml_get_tensor(meta, name); }
};

bool load_host_mat(gguf_reader & r, const std::string & name, host_mat & out, std::string & err) {
    ggml_tensor * t = r.meta_tensor(name.c_str());
    if (!t) {
        err = "missing tensor " + name;
        return false;
    }
    out.type = t->type;
    out.n_in = t->ne[0];
    out.n_out = ggml_nrows(t);
    out.row_bytes = ggml_row_size(t->type, t->ne[0]);
    out.data.resize(ggml_nbytes(t));
    return r.read(name.c_str(), out.data.data(), out.data.size(), err);
}

bool load_host_vec(gguf_reader & r, const std::string & name, host_vec & out, std::string & err) {
    host_mat tmp;
    if (!load_host_mat(r, name, tmp, err)) {
        return false;
    }
    int64_t n = tmp.n_in * tmp.n_out;
    out.v.resize(n);
    if (tmp.type == GGML_TYPE_F32) {
        memcpy(out.v.data(), tmp.data.data(), n * sizeof(float));
    } else {
        const ggml_type_traits * tr = ggml_get_type_traits(tmp.type);
        if (!tr->to_float) {
            err = "cannot dequantize " + name;
            return false;
        }
        for (int64_t r0 = 0; r0 < tmp.n_out; r0++) {
            tr->to_float(tmp.row(r0), out.v.data() + r0 * tmp.n_in, tmp.n_in);
        }
    }
    return true;
}

bool is_device_tensor(const std::string & name, const tyger_hparams & hp, bool asr) {
    if (name.rfind("encoder.layers.", 0) == 0) {
        int il = atoi(name.c_str() + strlen("encoder.layers."));
        return asr || il < hp.feature_layer;
    }
    if (name.rfind("encoder.", 0) == 0) {
        return true;
    }
    if (name.rfind("prompt_projector.", 0) == 0 || name.rfind("encoder_projector.", 0) == 0) {
        return asr;
    }
    return false;
}

} // namespace

bool tyger_model_load_impl(tyger_model_impl & m, const char * path, int backend, int device, int n_threads,
                           bool enable_asr, bool verbose, std::string & err) {
    g_verbose = verbose;
    m.verbose = verbose;
    ggml_log_set(ggml_log_cb, nullptr);
    m.path = path;
    m.has_asr = enable_asr;
    m.n_threads = n_threads > 0 ? n_threads : std::max(1u, std::min(8u, std::thread::hardware_concurrency()));

    gguf_reader r;
    if (!r.open(path, err)) {
        return false;
    }
    {
        int64_t ai = r.key("general.architecture");
        if (ai < 0 || std::string(gguf_get_val_str(r.g, ai)) != "tyger") {
            err = "not a tyger gguf (general.architecture != tyger)";
            return false;
        }
    }

    const int64_t turn_key = r.key("tyger.turn_only");
    const bool turn_only = turn_key >= 0 && gguf_get_val_bool(r.g, turn_key);
    if (turn_only && enable_asr) {
        err = "this gguf is turn taking only, it has no asr weights. load it with asr off (--no-asr, or enable_asr = false in the model params)";
        return false;
    }

    tyger_hparams & hp = m.hp;
    r.u32("tyger.audio.sample_rate", hp.sample_rate);
    r.u32("tyger.audio.n_fft", hp.n_fft);
    r.u32("tyger.audio.hop_length", hp.hop_length);
    r.u32("tyger.audio.win_length", hp.win_length);
    r.u32("tyger.audio.n_mels", hp.n_mels);
    r.f32("tyger.audio.preemphasis", hp.preemph);
    r.u32("tyger.encoder.hidden_size", hp.enc_hidden);
    r.u32("tyger.encoder.intermediate_size", hp.enc_ff);
    r.u32("tyger.encoder.num_layers", hp.enc_layers);
    r.u32("tyger.encoder.num_heads", hp.enc_heads);
    r.u32("tyger.encoder.conv_kernel_size", hp.conv_kernel);
    r.u32("tyger.encoder.subsampling_factor", hp.sub_factor);
    r.u32("tyger.encoder.subsampling_channels", hp.sub_channels);
    r.u32("tyger.encoder.subsampling_kernel", hp.sub_kernel);
    r.u32("tyger.encoder.subsampling_stride", hp.sub_stride);
    r.u32("tyger.encoder.sliding_window", hp.sliding_window);
    r.u32("tyger.encoder.default_lookahead", hp.default_lookahead);
    r.u32("tyger.encoder.max_positions", hp.max_positions);
    r.u32("tyger.rnnt.decoder_hidden", hp.dec_hidden);
    r.u32("tyger.rnnt.decoder_layers", hp.dec_layers);
    r.u32("tyger.rnnt.vocab_size", hp.vocab);
    r.u32("tyger.rnnt.blank_id", hp.blank_id);
    r.u32("tyger.rnnt.pad_id", hp.pad_id);
    r.u32("tyger.rnnt.max_symbols_per_step", hp.max_symbols);
    r.u32("tyger.rnnt.num_prompts", hp.num_prompts);
    r.u32("tyger.rnnt.default_prompt_id", hp.default_prompt);
    r.u32("tyger.turn_taker.width", hp.tt_width);
    r.u32("tyger.turn_taker.layers", hp.tt_layers);
    r.u32("tyger.turn_taker.feature_layer", hp.feature_layer);
    r.f32("tyger.turn_taker.frame_period_s", hp.frame_s);
    r.f32("tyger.turn_taker.live_window_s", hp.window_s);
    r.f32("tyger.turn_taker.live_hop_s", hp.hop_s);
    r.f32("tyger.policy.ep_thr", hp.ep_thr);
    r.f32("tyger.policy.barge_thr", hp.barge_thr);
    r.f32("tyger.policy.bc_thr", hp.bc_thr);
    r.f32("tyger.policy.vad_db", hp.vad_db);
    r.f32("tyger.policy.min_speech_ms", hp.min_speech_ms);
    r.u32("tyger.policy.ep_deb", hp.ep_deb);
    r.u32("tyger.policy.barge_deb", hp.barge_deb);
    r.u32("tyger.policy.bc_deb", hp.bc_deb);
    r.u32("tyger.policy.bc_refractory_frames", hp.bc_refractory);
    r.u32("tyger.policy.fallback_sil_frames", hp.fallback_sil_frames);
    r.u32("tyger.policy.warmup_frames", hp.warmup_frames);
    if (hp.tt_layers > 4 || hp.dec_layers > 2) {
        err = "unsupported layer count";
        return false;
    }

    {
        int64_t fi = r.key("general.file_type");
        int ft = fi >= 0 ? (int) gguf_get_val_u32(r.g, fi) : 0;
        static const std::map<int, std::pair<ggml_type, const char *>> names = {
            {0, {GGML_TYPE_F32, "F32"}}, {1, {GGML_TYPE_F16, "F16"}}, {7, {GGML_TYPE_Q8_0, "Q8_0"}},
            {18, {GGML_TYPE_Q6_K, "Q6_K"}}, {15, {GGML_TYPE_Q4_K, "Q4_K_M"}}, {14, {GGML_TYPE_Q4_K, "Q4_K_S"}},
            {17, {GGML_TYPE_Q5_K, "Q5_K_M"}},
        };
        auto it = names.find(ft);
        m.ftype = it != names.end() ? it->second.first : GGML_TYPE_F32;
        m.ftype_name = it != names.end() ? it->second.second : "unknown";
    }

    // pick the compute device
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    m.dev = cpu_dev;
    m.gpu = false;
    if (backend != 1) {
        int gi = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            auto t = ggml_backend_dev_type(d);
            if (t != GGML_BACKEND_DEVICE_TYPE_GPU && t != GGML_BACKEND_DEVICE_TYPE_IGPU) {
                continue;
            }
            if (device < 0 || gi == device) {
                m.dev = d;
                m.gpu = true;
                break;
            }
            gi++;
        }
        if (!m.gpu && backend == 2) {
            err = "no vulkan device found";
            return false;
        }
    }
    tyger_log("compute device: %s (%s)", ggml_backend_dev_name(m.dev), ggml_backend_dev_description(m.dev));

    // device tensors
    const int64_t n_tensors = gguf_get_n_tensors(r.g);
    ggml_init_params ip = { (size_t) (n_tensors + 8) * ggml_tensor_overhead(), nullptr, true };
    m.ctx_w = ggml_init(ip);
    std::vector<std::pair<ggml_tensor *, std::string>> dev_tensors;
    for (int64_t i = 0; i < n_tensors; i++) {
        std::string name = gguf_get_tensor_name(r.g, i);
        if (!is_device_tensor(name, hp, enable_asr)) {
            continue;
        }
        ggml_tensor * mt = r.meta_tensor(name.c_str());
        ggml_tensor * t = ggml_dup_tensor(m.ctx_w, mt);
        ggml_set_name(t, name.c_str());
        dev_tensors.emplace_back(t, name);
    }
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(m.dev);
    m.buf_w = ggml_backend_alloc_ctx_tensors_from_buft(m.ctx_w, buft);
    if (!m.buf_w) {
        err = "failed to allocate weight buffer";
        return false;
    }
    ggml_backend_buffer_set_usage(m.buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    {
        std::vector<uint8_t> tmp;
        for (auto & [t, name] : dev_tensors) {
            tmp.resize(ggml_nbytes(t));
            if (!r.read(name.c_str(), tmp.data(), tmp.size(), err)) {
                return false;
            }
            ggml_backend_tensor_set(t, tmp.data(), 0, tmp.size());
        }
    }
    tyger_log("weights on device: %.1f MiB (%zu tensors)", ggml_backend_buffer_get_size(m.buf_w) / 1048576.0,
              dev_tensors.size());

    auto T = [&](const std::string & name) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(m.ctx_w, name.c_str());
        if (!t) {
            err = "missing tensor " + name;
        }
        return t;
    };

    m.sub_conv_in_w = T("encoder.subsampling.conv_in.weight");
    m.sub_conv_in_b = T("encoder.subsampling.conv_in.bias");
    for (int i = 0; i < 2; i++) {
        std::string p = "encoder.subsampling.layers." + std::to_string(i) + ".";
        m.sub_dw_w[i] = T(p + "depthwise_conv.weight");
        m.sub_dw_b[i] = T(p + "depthwise_conv.bias");
        m.sub_pw_w[i] = T(p + "pointwise_conv.weight");
        m.sub_pw_b[i] = T(p + "pointwise_conv.bias");
    }
    m.sub_lin_w = T("encoder.subsampling.linear.weight");
    m.sub_lin_b = T("encoder.subsampling.linear.bias");
    const int n_dev_layers = enable_asr ? hp.enc_layers : hp.feature_layer;
    m.layers.resize(n_dev_layers);
    for (int il = 0; il < n_dev_layers; il++) {
        std::string p = "encoder.layers." + std::to_string(il) + ".";
        enc_layer & L = m.layers[il];
        L.ff1_norm_w = T(p + "norm_feed_forward1.weight");
        L.ff1_norm_b = T(p + "norm_feed_forward1.bias");
        L.ff1_up = T(p + "feed_forward1.linear1.weight");
        L.ff1_down = T(p + "feed_forward1.linear2.weight");
        L.att_norm_w = T(p + "norm_self_att.weight");
        L.att_norm_b = T(p + "norm_self_att.bias");
        L.wq = T(p + "self_attn.q_proj.weight");
        L.wk = T(p + "self_attn.k_proj.weight");
        L.wv = T(p + "self_attn.v_proj.weight");
        L.wo = T(p + "self_attn.o_proj.weight");
        L.wpos = T(p + "self_attn.relative_k_proj.weight");
        L.bias_u = T(p + "self_attn.bias_u");
        L.bias_v = T(p + "self_attn.bias_v");
        L.conv_norm_w = T(p + "norm_conv.weight");
        L.conv_norm_b = T(p + "norm_conv.bias");
        L.pw1 = T(p + "conv.pointwise_conv1.weight");
        L.dw = T(p + "conv.depthwise_conv.weight");
        L.conv_ln_w = T(p + "conv.norm.weight");
        L.conv_ln_b = T(p + "conv.norm.bias");
        L.pw2 = T(p + "conv.pointwise_conv2.weight");
        L.ff2_norm_w = T(p + "norm_feed_forward2.weight");
        L.ff2_norm_b = T(p + "norm_feed_forward2.bias");
        L.ff2_up = T(p + "feed_forward2.linear1.weight");
        L.ff2_down = T(p + "feed_forward2.linear2.weight");
        L.out_norm_w = T(p + "norm_out.weight");
        L.out_norm_b = T(p + "norm_out.bias");
    }
    if (enable_asr) {
        m.prompt_l1_w = T("prompt_projector.linear_1.weight");
        m.prompt_l1_b = T("prompt_projector.linear_1.bias");
        m.prompt_l1_p = T("prompt_projector.linear_1.prompt_weight");
        m.prompt_l2_w = T("prompt_projector.linear_2.weight");
        m.prompt_l2_b = T("prompt_projector.linear_2.bias");
        m.enc_proj_w = T("encoder_projector.weight");
        m.enc_proj_b = T("encoder_projector.bias");
    }
    if (!err.empty()) {
        return false;
    }

    // host side
    {
        host_vec mel;
        if (!load_host_vec(r, "frontend.mel_filters", mel, err)) return false;
        m.mel_filters = std::move(mel.v);
    }
    turn_taker_weights & tt = m.tt;
    const std::string tp = "turn_taker.";
    bool ok = load_host_mat(r, tp + "proj_user.weight", tt.proj_user, err)
           && load_host_mat(r, tp + "proj_agent.weight", tt.proj_agent, err)
           && load_host_vec(r, tp + "proj_user.bias", tt.proj_user_b, err)
           && load_host_vec(r, tp + "proj_agent.bias", tt.proj_agent_b, err)
           && load_host_vec(r, tp + "proj_energy.weight", tt.proj_energy_w, err)
           && load_host_vec(r, tp + "proj_energy.bias", tt.proj_energy_b, err)
           && load_host_vec(r, tp + "state_emb.weight", tt.state_emb, err)
           && load_host_vec(r, tp + "in_norm.weight", tt.in_norm_w, err)
           && load_host_vec(r, tp + "in_norm.bias", tt.in_norm_b, err)
           && load_host_vec(r, tp + "out_norm.weight", tt.out_norm_w, err)
           && load_host_vec(r, tp + "out_norm.bias", tt.out_norm_b, err)
           && load_host_vec(r, tp + "state_head.weight", tt.state_head_w, err)
           && load_host_vec(r, tp + "state_head.bias", tt.state_head_b, err)
           && load_host_vec(r, tp + "hazard_head.weight", tt.hazard_w, err)
           && load_host_vec(r, tp + "hazard_head.bias", tt.hazard_b, err)
           && load_host_mat(r, tp + "yield_head.0.weight", tt.yield0, err)
           && load_host_vec(r, tp + "yield_head.0.bias", tt.yield0_b, err)
           && load_host_vec(r, tp + "yield_head.2.weight", tt.yield2_w, err)
           && load_host_vec(r, tp + "yield_head.2.bias", tt.yield2_b, err)
           && load_host_mat(r, tp + "bc_head.0.weight", tt.bc0, err)
           && load_host_vec(r, tp + "bc_head.0.bias", tt.bc0_b, err)
           && load_host_vec(r, tp + "bc_head.2.weight", tt.bc2_w, err)
           && load_host_vec(r, tp + "bc_head.2.bias", tt.bc2_b, err)
           && load_host_mat(r, tp + "ep_head.0.weight", tt.ep0, err)
           && load_host_vec(r, tp + "ep_head.0.bias", tt.ep0_b, err)
           && load_host_vec(r, tp + "ep_head.2.weight", tt.ep2_w, err)
           && load_host_vec(r, tp + "ep_head.2.bias", tt.ep2_b, err);
    for (int l = 0; ok && l < hp.tt_layers; l++) {
        std::string s = std::to_string(l);
        ok = load_host_mat(r, tp + "core.weight_ih_l" + s, tt.w_ih[l], err)
          && load_host_mat(r, tp + "core.weight_hh_l" + s, tt.w_hh[l], err)
          && load_host_vec(r, tp + "core.bias_ih_l" + s, tt.b_ih[l], err)
          && load_host_vec(r, tp + "core.bias_hh_l" + s, tt.b_hh[l], err);
    }
    if (!ok) return false;

    if (enable_asr) {
        rnnt_weights & rw = m.rnnt;
        ok = load_host_mat(r, "decoder.embedding.weight", rw.embedding, err)
          && load_host_mat(r, "decoder.decoder_projector.weight", rw.proj, err)
          && load_host_vec(r, "decoder.decoder_projector.bias", rw.proj_b, err)
          && load_host_mat(r, "joint.head.weight", rw.head, err)
          && load_host_vec(r, "joint.head.bias", rw.head_b, err);
        for (int l = 0; ok && l < hp.dec_layers; l++) {
            std::string s = std::to_string(l);
            ok = load_host_mat(r, "decoder.lstm.weight_ih_l" + s, rw.w_ih[l], err)
              && load_host_mat(r, "decoder.lstm.weight_hh_l" + s, rw.w_hh[l], err)
              && load_host_vec(r, "decoder.lstm.bias_ih_l" + s, rw.b_ih[l], err)
              && load_host_vec(r, "decoder.lstm.bias_hh_l" + s, rw.b_hh[l], err);
        }
        if (!ok) return false;

        int64_t ki = r.key("tokenizer.ggml.tokens");
        if (ki < 0) {
            err = "missing tokenizer";
            return false;
        }
        size_t n = gguf_get_arr_n(r.g, ki);
        m.pieces.resize(n);
        for (size_t i = 0; i < n; i++) {
            m.pieces[i] = gguf_get_arr_str(r.g, ki, i);
        }
        m.special.assign(n, 0);
        int64_t si = r.key("tokenizer.ggml.token_special");
        if (si >= 0) {
            size_t ns = std::min(n, gguf_get_arr_n(r.g, si));
            const void * d = gguf_get_arr_data(r.g, si);
            enum gguf_type at = gguf_get_arr_type(r.g, si);
            for (size_t i = 0; i < ns; i++) {
                if (at == GGUF_TYPE_INT32) m.special[i] = ((const int32_t *) d)[i] != 0;
                else if (at == GGUF_TYPE_UINT8 || at == GGUF_TYPE_INT8 || at == GGUF_TYPE_BOOL) m.special[i] = ((const uint8_t *) d)[i] != 0;
            }
        }
    }

    char buf[1024];
    snprintf(buf, sizeof(buf),
             "{\"arch\":\"tyger\",\"file_type\":\"%s\",\"device\":\"%s\",\"gpu\":%s,\"asr\":%s,\"turn_only\":%s,"
             "\"sample_rate\":%d,\"hop_samples\":%d,\"frame_s\":%.3f,\"window_s\":%.1f,\"encoder_layers\":%d,"
             "\"feature_layer\":%d,\"gru_width\":%d,\"gru_layers\":%d,\"vocab\":%d,\"weights_mib\":%.1f}",
             m.ftype_name.c_str(), ggml_backend_dev_name(m.dev), m.gpu ? "true" : "false",
             enable_asr ? "true" : "false", turn_only ? "true" : "false", hp.sample_rate, hp.hop_samples(), hp.frame_s, hp.window_s, hp.enc_layers,
             hp.feature_layer, hp.tt_width, hp.tt_layers, hp.vocab, ggml_backend_buffer_get_size(m.buf_w) / 1048576.0);
    m.info_json = buf;
    return true;
}
