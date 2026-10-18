#include "encoder.h"

#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstring>
#include <limits>

static const int kGraphSize = 16384;

namespace {
struct prof {
    bool on = getenv("TYGER_PROFILE") != nullptr;
    std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
    std::string line;
    void mark(const char * what) {
        if (!on) return;
        auto now = std::chrono::steady_clock::now();
        char b[64];
        snprintf(b, sizeof(b), " %s %.1f", what, std::chrono::duration<double, std::milli>(now - t).count());
        line += b;
        t = now;
    }
    void done(const char * tag, ggml_backend_sched_t sched, ggml_cgraph * gf) {
        if (!on) return;
        fprintf(stderr, "[prof %s] nodes %d splits %d copies %d |%s ms\n", tag, ggml_graph_n_nodes(gf),
                ggml_backend_sched_get_n_splits(sched), ggml_backend_sched_get_n_copies(sched), line.c_str());
        static bool shown = false;
        if (shown) return;
        shown = true;
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * n = ggml_graph_node(gf, i);
            ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched, n);
            if (b && ggml_backend_is_cpu(b)) {
                fprintf(stderr, "  cpu op: %s %s", ggml_op_desc(n), n->name);
                for (int k = 0; k < 2 && n->src[k]; k++) {
                    const ggml_tensor * t = n->src[k];
                    fprintf(stderr, " | src%d %s [%lld %lld %lld %lld] nb [%zu %zu %zu %zu] cont %d", k,
                            ggml_type_name(t->type), (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2],
                            (long long) t->ne[3], t->nb[0], t->nb[1], t->nb[2], t->nb[3], ggml_is_contiguous(t));
                }
                fprintf(stderr, "\n");
            }
        }
    }
};
}

enc_runner::enc_runner(tyger_model_impl & m_) : m(m_) {
    cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_cpu_set_n_threads(cpu, m.n_threads);
    std::vector<ggml_backend_t> bes;
    if (m.gpu) {
        dev = ggml_backend_dev_init(m.dev, nullptr);
        bes.push_back(dev);
    }
    bes.push_back(cpu);
    sched = ggml_backend_sched_new(bes.data(), nullptr, (int) bes.size(), kGraphSize, false, true);
    meta.resize(ggml_tensor_overhead() * kGraphSize + ggml_graph_overhead_custom(kGraphSize, false));
    mel = std::make_unique<mel_extractor>(m);
}

enc_runner::~enc_runner() {
    if (sched) ggml_backend_sched_free(sched);
    if (dev) ggml_backend_free(dev);
    if (cpu) ggml_backend_free(cpu);
}

namespace {

ggml_tensor * mm(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    ggml_tensor * t = ggml_mul_mat(ctx, a, b);
    ggml_mul_mat_set_prec(t, GGML_PREC_F32);
    return t;
}

ggml_tensor * norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, 1e-5f), w), b);
}

ggml_tensor * ffn(ggml_context * ctx, ggml_tensor * x, ggml_tensor * up, ggml_tensor * down) {
    return mm(ctx, down, ggml_silu(ctx, mm(ctx, up, x)));
}

// x [W, H, C, N] -> 1x1 conv over C
ggml_tensor * pointwise2d(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    const int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2], N = x->ne[3];
    ggml_tensor * t = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3)); // [C, W, H, N]
    t = ggml_reshape_2d(ctx, t, C, W * H * N);
    t = ggml_add(ctx, mm(ctx, w, t), b);
    t = ggml_reshape_4d(ctx, t, w->ne[1], W, H, N);
    return ggml_cont(ctx, ggml_permute(ctx, t, 2, 0, 1, 3)); // back to [W, H, C, N]
}

struct sub_masks {
    ggml_tensor * m[3] = {nullptr, nullptr, nullptr};
};

// mel [F, T, 1, N] -> [C, T/8, N]
ggml_tensor * subsample(ggml_context * ctx, const tyger_model_impl & m, ggml_tensor * mel, const sub_masks & sm) {
    ggml_tensor * x = ggml_pad_ext(ctx, mel, 2, 1, 2, 1, 0, 0, 0, 0);
    x = getenv("TYGER_CONV_DIRECT") ? ggml_conv_2d_direct(ctx, m.sub_conv_in_w, x, 2, 2, 0, 0, 1, 1)
                                    : ggml_conv_2d(ctx, m.sub_conv_in_w, x, 2, 2, 0, 0, 1, 1);
    x = ggml_add(ctx, x, ggml_reshape_4d(ctx, m.sub_conv_in_b, 1, 1, m.sub_conv_in_b->ne[0], 1));
    if (sm.m[0]) x = ggml_mul(ctx, x, sm.m[0]);
    x = ggml_relu(ctx, x);
    for (int i = 0; i < 2; i++) {
        x = ggml_pad_ext(ctx, x, 2, 1, 2, 1, 0, 0, 0, 0);
        x = ggml_conv_2d_dw_direct(ctx, m.sub_dw_w[i], x, 2, 2, 0, 0, 1, 1);
        x = ggml_add(ctx, x, ggml_reshape_4d(ctx, m.sub_dw_b[i], 1, 1, m.sub_dw_b[i]->ne[0], 1));
        x = pointwise2d(ctx, x, m.sub_pw_w[i], m.sub_pw_b[i]);
        if (sm.m[i + 1]) x = ggml_mul(ctx, x, sm.m[i + 1]);
        x = ggml_relu(ctx, x);
    }
    // [F', T', C, N] -> features ordered c * F' + f
    const int64_t F = x->ne[0], T = x->ne[1], C = x->ne[2], N = x->ne[3];
    x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3)); // [F', C, T', N]
    x = ggml_reshape_3d(ctx, x, F * C, T, N);
    return ggml_add(ctx, mm(ctx, m.sub_lin_w, x), m.sub_lin_b);
}

struct layer_io {
    ggml_tensor * k_cache = nullptr; // [C, Lc, B] or null
    ggml_tensor * v_cache = nullptr;
    ggml_tensor * conv_cache = nullptr; // [C, K-1, B] or null (zero padding)
    ggml_tensor * k_new = nullptr;      // outputs for the cache update
    ggml_tensor * v_new = nullptr;
    ggml_tensor * glu = nullptr;
};

// x [C, Tq, B]; pos [C, 2L-1]; mask [L, Tq, 1, B]
ggml_tensor * conformer(ggml_context * ctx, const tyger_model_impl & m, const enc_layer & L, ggml_tensor * x,
                        ggml_tensor * pos, ggml_tensor * mask, layer_io & io) {
    const tyger_hparams & hp = m.hp;
    const int64_t C = x->ne[0], Tq = x->ne[1], B = x->ne[2];
    const int64_t H = hp.enc_heads, D = hp.head_dim();

    ggml_tensor * ff = ffn(ctx, norm(ctx, x, L.ff1_norm_w, L.ff1_norm_b), L.ff1_up, L.ff1_down);
    x = ggml_add(ctx, x, ggml_scale(ctx, ff, 0.5f));

    // relative position multi-head attention
    {
        ggml_tensor * t = norm(ctx, x, L.att_norm_w, L.att_norm_b);
        ggml_tensor * q = ggml_reshape_4d(ctx, mm(ctx, L.wq, t), D, H, Tq, B);
        ggml_tensor * k = mm(ctx, L.wk, t);
        ggml_tensor * v = mm(ctx, L.wv, t);
        io.k_new = k;
        io.v_new = v;
        if (io.k_cache) {
            k = ggml_concat(ctx, io.k_cache, k, 1);
            v = ggml_concat(ctx, io.v_cache, v, 1);
        }
        const int64_t Lk = k->ne[1];
        const int64_t P = 2 * Lk - 1;
        GGML_ASSERT(pos->ne[1] == P);

        ggml_tensor * qu = ggml_permute(ctx, ggml_add(ctx, q, ggml_reshape_4d(ctx, L.bias_u, D, H, 1, 1)), 0, 2, 1, 3);
        ggml_tensor * qv = ggml_permute(ctx, ggml_add(ctx, q, ggml_reshape_4d(ctx, L.bias_v, D, H, 1, 1)), 0, 2, 1, 3);
        ggml_tensor * kp = ggml_permute(ctx, ggml_reshape_4d(ctx, k, D, H, Lk, B), 0, 2, 1, 3);   // [D, Lk, H, B]
        ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, v, D, H, Lk, B), 1, 2, 0, 3)); // [Lk, D, H, B]

        ggml_tensor * r = mm(ctx, L.wpos, pos);                                        // [C, P]
        r = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, r, D, H, P), 0, 2, 1, 3));     // [D, P, H]
        if (B > 1) {
            r = ggml_repeat_4d(ctx, r, D, P, H, B); // vulkan matmul does not broadcast dim 3
        }

        ggml_tensor * ac = mm(ctx, kp, qu);   // [Lk, Tq, H, B]
        ggml_tensor * bd = mm(ctx, r, qv);    // [P, Tq, H, B]
        // transformer-xl shift: out[i][j] = bd[i][j + Tq - 1 - i]
        ggml_tensor * bds = ggml_view_4d(ctx, bd, Lk, Tq, H, B, bd->nb[1] - bd->nb[0], bd->nb[2], bd->nb[3],
                                         (Tq - 1) * bd->nb[0]);
        ggml_tensor * sc = ggml_add(ctx, ac, ggml_cont(ctx, bds));
        sc = ggml_soft_max_ext(ctx, sc, mask, 1.0f / std::sqrt((float) D), 0.0f);
        ggml_tensor * o = mm(ctx, vt, sc);    // [D, Tq, H, B]
        o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));
        o = ggml_reshape_3d(ctx, o, C, Tq, B);
        x = ggml_add(ctx, x, mm(ctx, L.wo, o));
    }

    // convolution module
    {
        ggml_tensor * t = norm(ctx, x, L.conv_norm_w, L.conv_norm_b);
        const int64_t C2 = L.pw1->ne[1] / 2;
        ggml_tensor * wa = ggml_view_2d(ctx, L.pw1, L.pw1->ne[0], C2, L.pw1->nb[1], 0);
        ggml_tensor * wg = ggml_view_2d(ctx, L.pw1, L.pw1->ne[0], C2, L.pw1->nb[1], C2 * L.pw1->nb[1]);
        ggml_tensor * g = ggml_mul(ctx, mm(ctx, wa, t), ggml_sigmoid(ctx, mm(ctx, wg, t)));
        io.glu = g;
        const int pad = (int) L.dw->ne[0] - 1;
        ggml_tensor * gp = io.conv_cache ? ggml_concat(ctx, io.conv_cache, g, 1)
                                         : ggml_pad_ext(ctx, g, 0, 0, pad, 0, 0, 0, 0, 0);
        gp = ggml_cont(ctx, ggml_permute(ctx, gp, 1, 0, 2, 3)); // [Tq+pad, C, B]
        ggml_tensor * c = ggml_ssm_conv(ctx, gp, L.dw);           // [C, Tq, B]
        c = norm(ctx, c, L.conv_ln_w, L.conv_ln_b);
        c = mm(ctx, L.pw2, ggml_silu(ctx, c));
        x = ggml_add(ctx, x, c);
    }

    ff = ffn(ctx, norm(ctx, x, L.ff2_norm_w, L.ff2_norm_b), L.ff2_up, L.ff2_down);
    x = ggml_add(ctx, x, ggml_scale(ctx, ff, 0.5f));
    return norm(ctx, x, L.out_norm_w, L.out_norm_b);
}

void rel_pos(const tyger_hparams & hp, int64_t Lk, std::vector<float> & pe) {
    const int C = hp.enc_hidden;
    const int64_t P = 2 * Lk - 1;
    pe.assign((size_t) P * C, 0.0f);
    std::vector<float> inv(C / 2);
    for (int i = 0; i < C / 2; i++) {
        inv[i] = 1.0f / std::pow(10000.0f, (float) (2 * i) / (float) C);
    }
    for (int64_t p = 0; p < P; p++) {
        const float pos = (float) (Lk - 1 - p);
        float * row = pe.data() + p * C;
        for (int i = 0; i < C / 2; i++) {
            const float f = inv[i] * pos;
            row[2 * i] = std::sin(f);
            row[2 * i + 1] = std::cos(f);
        }
    }
}

struct graph_ctx {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    graph_ctx(enc_runner & r) {
        ggml_init_params ip = { r.meta.size(), r.meta.data(), true };
        ctx = ggml_init(ip);
        gf = ggml_new_graph_custom(ctx, kGraphSize, false);
    }
    ~graph_ctx() { ggml_free(ctx); }
};

int out_len(int n) { return n / 2 + 1; }

} // namespace

bool encode_offline(enc_runner & r, const std::vector<const float *> & mels, int n_frames,
                    const std::vector<int> & mel_valid, int n_layers, int lookahead, bool asr_head, int prompt_id,
                    std::vector<float> & out, int & T_out, int & C_out, std::vector<int> & out_valid) {
    tyger_model_impl & m = r.m;
    const tyger_hparams & hp = m.hp;
    const int B = (int) mels.size();
    const int F = hp.n_mels;
    if (n_layers > (int) m.layers.size() || (asr_head && !m.has_asr)) {
        return false;
    }

    // frame validity through the three stride-2 stages
    int T1 = out_len(n_frames), T2 = out_len(T1), T3 = out_len(T2);
    std::vector<int> l1(B), l2(B), l3(B);
    bool masked = false;
    for (int b = 0; b < B; b++) {
        l1[b] = out_len(mel_valid[b]);
        l2[b] = out_len(l1[b]);
        l3[b] = out_len(l2[b]);
        masked |= l1[b] < T1 || l2[b] < T2;
    }
    const int T = T3;

    prof pf;
    graph_ctx g(r);
    ggml_context * ctx = g.ctx;
    ggml_tensor * mel = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, F, n_frames, 1, B);
    ggml_set_input(mel);
    sub_masks sm;
    if (masked) {
        sm.m[0] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, T1, 1, B);
        sm.m[1] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, T2, 1, B);
        sm.m[2] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, T3, 1, B);
        for (auto * t : sm.m) ggml_set_input(t);
    }
    ggml_tensor * pos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.enc_hidden, 2 * T - 1);
    ggml_set_input(pos);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, T, T, 1, B);
    ggml_set_input(mask);
    ggml_tensor * prompt = nullptr;

    ggml_tensor * x = subsample(ctx, m, mel, sm);
    for (int il = 0; il < n_layers; il++) {
        layer_io io;
        x = conformer(ctx, m, m.layers[il], x, pos, mask, io);
    }
    if (asr_head) {
        const int pid = prompt_id >= 0 && prompt_id < hp.num_prompts ? prompt_id : hp.default_prompt;
        prompt = ggml_view_1d(ctx, m.prompt_l1_p, m.prompt_l1_p->ne[0], pid * m.prompt_l1_p->nb[1]);
        x = ggml_add(ctx, ggml_add(ctx, mm(ctx, m.prompt_l1_w, x), m.prompt_l1_b), prompt);
        x = ggml_add(ctx, mm(ctx, m.prompt_l2_w, ggml_relu(ctx, x)), m.prompt_l2_b);
        x = ggml_add(ctx, mm(ctx, m.enc_proj_w, x), m.enc_proj_b);
    }
    ggml_set_output(x);
    ggml_build_forward_expand(g.gf, x);

    std::lock_guard<std::mutex> lock(m.compute_mutex);
    pf.mark("build");
    ggml_backend_sched_reset(r.sched);
    if (!ggml_backend_sched_alloc_graph(r.sched, g.gf)) {
        return false;
    }
    pf.mark("alloc");
    {
        std::vector<float> buf((size_t) F * n_frames * B);
        for (int b = 0; b < B; b++) {
            memcpy(buf.data() + (size_t) b * F * n_frames, mels[b], sizeof(float) * F * n_frames);
        }
        ggml_backend_tensor_set(mel, buf.data(), 0, buf.size() * sizeof(float));
    }
    if (masked) {
        const int lens[3][2] = {{T1, 0}, {T2, 1}, {T3, 2}};
        for (auto & li : lens) {
            const int Ts = li[0];
            const std::vector<int> & lv = li[1] == 0 ? l1 : li[1] == 1 ? l2 : l3;
            std::vector<float> mv((size_t) Ts * B);
            for (int b = 0; b < B; b++)
                for (int t = 0; t < Ts; t++) mv[(size_t) b * Ts + t] = t < lv[b] ? 1.0f : 0.0f;
            ggml_backend_tensor_set(sm.m[li[1]], mv.data(), 0, mv.size() * sizeof(float));
        }
    }
    if (pos->buffer) {
        std::vector<float> pe;
        rel_pos(hp, T, pe);
        ggml_backend_tensor_set(pos, pe.data(), 0, pe.size() * sizeof(float));
    }
    if (mask->buffer) {
        // chunked_limited: key chunk within [q chunk - left chunks, q chunk], keys past the clip masked
        const int chunk = lookahead + 1;
        const int left_chunks = (hp.sliding_window - 1) / chunk;
        const float ninf = -std::numeric_limits<float>::infinity();
        std::vector<float> mv((size_t) T * T * B);
        for (int b = 0; b < B; b++) {
            for (int q = 0; q < T; q++) {
                float * row = mv.data() + ((size_t) b * T + q) * T;
                for (int k = 0; k < T; k++) {
                    const int d = q / chunk - k / chunk;
                    row[k] = (d >= 0 && d <= left_chunks && k < l3[b]) ? 0.0f : ninf;
                }
            }
        }
        ggml_backend_tensor_set(mask, mv.data(), 0, mv.size() * sizeof(float));
    }
    pf.mark("upload");
    if (ggml_backend_sched_graph_compute(r.sched, g.gf) != GGML_STATUS_SUCCESS) {
        return false;
    }
    pf.mark("compute");
    T_out = T;
    C_out = (int) x->ne[0];
    out.resize((size_t) C_out * T * B);
    ggml_backend_tensor_get(x, out.data(), 0, out.size() * sizeof(float));
    out_valid = l3;
    pf.mark("download");
    pf.done(asr_head ? "asr" : "exact", r.sched, g.gf);
    return true;
}

int enc_stream_ready(const tyger_model_impl & m, const enc_stream & s, int n_new) {
    // encoder frame c needs mel frames up to sub_factor * c
    const int64_t last = (s.frames_out + n_new - 1) * m.hp.sub_factor;
    return s.mel.n_frames > last ? 1 : 0;
}

bool encode_stream(enc_runner & r, const std::vector<enc_stream *> & streams, int n_new,
                   std::vector<std::vector<float>> & out) {
    tyger_model_impl & m = r.m;
    const tyger_hparams & hp = m.hp;
    const int C = hp.enc_hidden, F = hp.n_mels, NL = hp.feature_layer;
    const int left = hp.sliding_window - 1;
    const int kpad = hp.conv_kernel - 1;
    const int sf = hp.sub_factor;
    out.assign(streams.size(), {});
    // one more query than committed frames: the newest is computed with a zero-padded tail like the windowed path and never cached
    const int n_q = n_new + 1;

    // first chunks start at mel 0; later ones carry 2 extra encoder frames of context that are dropped
    std::vector<int> firsts, rest;
    for (int i = 0; i < (int) streams.size(); i++) {
        (streams[i]->frames_out == 0 ? firsts : rest).push_back(i);
    }
    for (int pass = 0; pass < 2; pass++) {
        const std::vector<int> & idx = pass == 0 ? firsts : rest;
        if (idx.empty()) continue;
        const int B = (int) idx.size();
        const bool first = pass == 0;
        const int ctx_frames = first ? 0 : 2;
        const int n_mel = sf * (n_q - 1 + ctx_frames) + 1;
        const int T_sub = out_len(out_len(out_len(n_mel)));
        GGML_ASSERT(T_sub == n_q + ctx_frames);

        prof pf;
        graph_ctx g(r);
        ggml_context * ctx = g.ctx;
        ggml_tensor * mel = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, F, n_mel, 1, B);
        ggml_set_input(mel);
        const int Lk = left + n_q;
        ggml_tensor * pos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, 2 * Lk - 1);
        ggml_set_input(pos);
        ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, Lk, n_q, 1, B);
        ggml_set_input(mask);

        ggml_tensor * x = subsample(ctx, m, mel, sub_masks());
        if (ctx_frames > 0) {
            x = ggml_cont(ctx, ggml_view_3d(ctx, x, C, n_q, B, x->nb[1], x->nb[2], ctx_frames * x->nb[1]));
        }
        std::vector<layer_io> io(NL);
        std::vector<ggml_tensor *> kc(NL), vc(NL), cc(NL);
        for (int il = 0; il < NL; il++) {
            kc[il] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, left, B);
            vc[il] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, left, B);
            cc[il] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, kpad, B);
            ggml_set_input(kc[il]);
            ggml_set_input(vc[il]);
            ggml_set_input(cc[il]);
            io[il].k_cache = kc[il];
            io[il].v_cache = vc[il];
            io[il].conv_cache = cc[il];
            x = conformer(ctx, m, m.layers[il], x, pos, mask, io[il]);
            ggml_set_output(io[il].k_new);
            ggml_set_output(io[il].v_new);
            ggml_set_output(io[il].glu);
            ggml_build_forward_expand(g.gf, io[il].k_new);
            ggml_build_forward_expand(g.gf, io[il].v_new);
            ggml_build_forward_expand(g.gf, io[il].glu);
        }
        ggml_set_output(x);
        ggml_build_forward_expand(g.gf, x);

        std::lock_guard<std::mutex> lock(m.compute_mutex);
        pf.mark("build");
        ggml_backend_sched_reset(r.sched);
        if (!ggml_backend_sched_alloc_graph(r.sched, g.gf)) {
            return false;
        }
        pf.mark("alloc");
        {
            std::vector<float> buf((size_t) F * n_mel * B);
            for (int b = 0; b < B; b++) {
                const enc_stream & s = *streams[idx[b]];
                const int64_t m0 = first ? 0 : sf * (s.frames_out - ctx_frames);
                const int64_t off = m0 - s.mel.frames_base;
                const int64_t have = (int64_t) (s.mel.frames.size() / F) - off;
                const int64_t n_real = std::min<int64_t>(have, n_mel);
                GGML_ASSERT(off >= 0 && n_real > 0);
                float * dst = buf.data() + (size_t) b * F * n_mel;
                memcpy(dst, s.mel.frames.data() + off * F, sizeof(float) * F * n_real);
                for (int64_t f = n_real; f < n_mel; f++) {
                    s.mel.speculative(r.mel_ex(), m0 + f, dst + f * F);
                }
            }
            ggml_backend_tensor_set(mel, buf.data(), 0, buf.size() * sizeof(float));
        }
        {
            std::vector<float> pe;
            rel_pos(hp, Lk, pe);
            ggml_backend_tensor_set(pos, pe.data(), 0, pe.size() * sizeof(float));
        }
        {
            const float ninf = -std::numeric_limits<float>::infinity();
            std::vector<float> mv((size_t) Lk * n_q * B);
            for (int b = 0; b < B; b++) {
                const int nc = streams[idx[b]]->n_cache;
                for (int q = 0; q < n_q; q++) {
                    float * row = mv.data() + ((size_t) b * n_q + q) * Lk;
                    for (int k = 0; k < Lk; k++) {
                        const int dist = (left + q) - k; // in frames, cache slot k is absolute q0 - left + k
                        const bool ok = dist >= 0 && dist <= left && k >= left - nc;
                        row[k] = ok ? 0.0f : ninf;
                    }
                }
            }
            ggml_backend_tensor_set(mask, mv.data(), 0, mv.size() * sizeof(float));
        }
        {
            std::vector<float> kb((size_t) C * left * B), vb((size_t) C * left * B), cb((size_t) C * kpad * B);
            for (int il = 0; il < NL; il++) {
                for (int b = 0; b < B; b++) {
                    enc_stream & s = *streams[idx[b]];
                    if (s.k.empty()) {
                        s.k.assign(NL, std::vector<float>((size_t) left * C, 0.0f));
                        s.v.assign(NL, std::vector<float>((size_t) left * C, 0.0f));
                        s.conv.assign(NL, std::vector<float>((size_t) kpad * C, 0.0f));
                    }
                    memcpy(kb.data() + (size_t) b * left * C, s.k[il].data(), sizeof(float) * left * C);
                    memcpy(vb.data() + (size_t) b * left * C, s.v[il].data(), sizeof(float) * left * C);
                    memcpy(cb.data() + (size_t) b * kpad * C, s.conv[il].data(), sizeof(float) * kpad * C);
                }
                ggml_backend_tensor_set(kc[il], kb.data(), 0, kb.size() * sizeof(float));
                ggml_backend_tensor_set(vc[il], vb.data(), 0, vb.size() * sizeof(float));
                ggml_backend_tensor_set(cc[il], cb.data(), 0, cb.size() * sizeof(float));
            }
        }
        pf.mark("upload");
        if (ggml_backend_sched_graph_compute(r.sched, g.gf) != GGML_STATUS_SUCCESS) {
            return false;
        }
        pf.mark("compute");
        std::vector<float> res((size_t) C * n_q * B);
        ggml_backend_tensor_get(x, res.data(), 0, res.size() * sizeof(float));
        for (int b = 0; b < B; b++) {
            // the newest n_new frames, the speculative one last
            const size_t o = (size_t) b * C * n_q + (size_t) (n_q - n_new) * C;
            out[idx[b]].assign(res.begin() + o, res.begin() + o + (size_t) C * n_new);
        }
        // roll the caches
        std::vector<float> kn(res.size()), vn(res.size()), gn(res.size());
        auto roll = [&](std::vector<float> & cache, int len, const float * fresh) {
            if (n_new >= len) {
                memcpy(cache.data(), fresh + (size_t) (n_new - len) * C, sizeof(float) * len * C);
            } else {
                memmove(cache.data(), cache.data() + (size_t) n_new * C, sizeof(float) * (len - n_new) * C);
                memcpy(cache.data() + (size_t) (len - n_new) * C, fresh, sizeof(float) * n_new * C);
            }
        };
        for (int il = 0; il < NL; il++) {
            ggml_backend_tensor_get(io[il].k_new, kn.data(), 0, kn.size() * sizeof(float));
            ggml_backend_tensor_get(io[il].v_new, vn.data(), 0, vn.size() * sizeof(float));
            ggml_backend_tensor_get(io[il].glu, gn.data(), 0, gn.size() * sizeof(float));
            for (int b = 0; b < B; b++) {
                enc_stream & s = *streams[idx[b]];
                const size_t o = (size_t) b * C * n_q;
                roll(s.k[il], left, kn.data() + o);
                roll(s.v[il], left, vn.data() + o);
                roll(s.conv[il], kpad, gn.data() + o);
            }
        }
        pf.mark("download");
        pf.done("fast", r.sched, g.gf);
        for (int b = 0; b < B; b++) {
            enc_stream & s = *streams[idx[b]];
            s.frames_out += n_new;
            s.n_cache = std::min(left, s.n_cache + n_new);
            // drop mel frames no later chunk will read
            const int64_t keep = sf * (s.frames_out - 2);
            if (keep > s.mel.frames_base) {
                const int64_t drop = keep - s.mel.frames_base;
                s.mel.frames.erase(s.mel.frames.begin(), s.mel.frames.begin() + (size_t) (drop * F));
                s.mel.frames_base = keep;
            }
        }
    }
    return true;
}
