#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

static const char * kQuantSuffixes[] = {
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
};

static bool ends_with(const std::string & s, const char * suf) {
    size_t n = strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

static bool quantizable(const std::string & name, const ggml_tensor * t) {
    if (name.rfind("turn_taker.", 0) == 0 || ggml_n_dims(t) != 2) return false;
    for (const char * s : kQuantSuffixes) {
        if (ends_with(name, s)) return true;
    }
    return false;
}

static int layer_of(const std::string & name) {
    const char * p = "encoder.layers.";
    if (name.rfind(p, 0) != 0) return -1;
    return atoi(name.c_str() + strlen(p));
}

// llama.cpp's q4_k_m rule for which layers get more bits
static bool use_more_bits(int i, int n) { return i < n / 8 || i >= 7 * n / 8 || (i - n / 8) % 3 == 2; }

struct qtype {
    const char * name;
    ggml_type type;
    int file_type;
};

static const qtype kTypes[] = {
    {"f16", GGML_TYPE_F16, 1}, {"q8_0", GGML_TYPE_Q8_0, 7}, {"q6_k", GGML_TYPE_Q6_K, 18},
    {"q5_k", GGML_TYPE_Q5_K, 17}, {"q4_k", GGML_TYPE_Q4_K, 15}, {"q4_k_s", GGML_TYPE_Q4_K, 14},
};

static ggml_type pick(const qtype & q, const std::string & name, const ggml_tensor * t, int n_layers) {
    ggml_type ty = q.type;
    if (q.type == GGML_TYPE_Q4_K && q.file_type == 15) {
        const int il = layer_of(name);
        if ((ends_with(name, "self_attn.v_proj.weight") || ends_with(name, "linear2.weight")) && il >= 0 &&
            use_more_bits(il, n_layers)) {
            ty = GGML_TYPE_Q6_K;
        }
        if (name == "encoder.subsampling.linear.weight" || name == "joint.head.weight" ||
            name == "encoder_projector.weight") {
            ty = GGML_TYPE_Q6_K;
        }
    }
    // rows that don't fit k-quant blocks (the 640-wide rnnt decoder) fall back to q8_0
    if (t->ne[0] % ggml_blck_size(ty) != 0) {
        ty = t->ne[0] % ggml_blck_size(GGML_TYPE_Q8_0) == 0 ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    }
    return ty;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: tyger-quantize in.gguf out.gguf TYPE [threads]\n  TYPE: f16 q8_0 q6_k q5_k q4_k q4_k_s\n");
        return 1;
    }
    const std::string in = argv[1], out = argv[2], tname = argv[3];
    const int nth = argc > 4 ? atoi(argv[4]) : (int) std::thread::hardware_concurrency();
    const qtype * q = nullptr;
    for (auto & t : kTypes) {
        if (tname == t.name) q = &t;
    }
    if (!q) {
        fprintf(stderr, "unknown type %s\n", tname.c_str());
        return 1;
    }
    ggml_cpu_init();

    ggml_context * ctx_in = nullptr;
    gguf_init_params ip = {false, &ctx_in};
    gguf_context * g = gguf_init_from_file(in.c_str(), ip);
    if (!g) {
        fprintf(stderr, "cannot read %s\n", in.c_str());
        return 1;
    }
    int n_layers = 24;
    {
        int64_t k = gguf_find_key(g, "tyger.encoder.num_layers");
        if (k >= 0) n_layers = (int) gguf_get_val_u32(g, k);
    }

    gguf_context * go = gguf_init_empty();
    gguf_set_kv(go, g);
    gguf_set_val_u32(go, "general.file_type", (uint32_t) q->file_type);

    const int64_t nt = gguf_get_n_tensors(g);
    ggml_init_params cp = {(size_t) (nt + 4) * ggml_tensor_overhead(), nullptr, true};
    ggml_context * ctx_out = ggml_init(cp);
    std::vector<std::unique_ptr<std::vector<uint8_t>>> bufs;
    size_t bytes_in = 0, bytes_out = 0;
    int n_quant = 0;
    for (int64_t i = 0; i < nt; i++) {
        const std::string name = gguf_get_tensor_name(g, i);
        ggml_tensor * t = ggml_get_tensor(ctx_in, name.c_str());
        bytes_in += ggml_nbytes(t);
        ggml_type ty = quantizable(name, t) ? pick(*q, name, t, n_layers) : t->type;
        if (!quantizable(name, t) && t->type == GGML_TYPE_F16) ty = GGML_TYPE_F32;
        ggml_tensor * o = ggml_new_tensor(ctx_out, ty, ggml_n_dims(t), t->ne);
        ggml_set_name(o, name.c_str());
        auto buf = std::make_unique<std::vector<uint8_t>>(ggml_nbytes(o));
        const int64_t n_per_row = t->ne[0], nrows = ggml_nrows(t);
        if (ty == t->type) {
            memcpy(buf->data(), t->data, ggml_nbytes(t));
        } else {
            std::vector<float> f32((size_t) n_per_row * nrows);
            if (t->type == GGML_TYPE_F32) {
                memcpy(f32.data(), t->data, f32.size() * sizeof(float));
            } else {
                ggml_get_type_traits(t->type)->to_float(t->data, f32.data(), (int64_t) f32.size());
            }
            if (ty == GGML_TYPE_F32) {
                memcpy(buf->data(), f32.data(), f32.size() * sizeof(float));
            } else {
                // split rows across threads
                std::vector<std::thread> th;
                const int64_t per = (nrows + nth - 1) / nth;
                const size_t row_sz = ggml_row_size(ty, n_per_row);
                for (int k = 0; k < nth; k++) {
                    const int64_t r0 = k * per, r1 = std::min(nrows, r0 + per);
                    if (r0 >= r1) break;
                    th.emplace_back([&, r0, r1] {
                        ggml_quantize_chunk(ty, f32.data() + r0 * n_per_row, buf->data() + r0 * row_sz, 0, r1 - r0,
                                            n_per_row, nullptr);
                    });
                }
                for (auto & x : th) x.join();
                n_quant++;
            }
        }
        o->data = buf->data();
        bytes_out += buf->size();
        bufs.push_back(std::move(buf));
        gguf_add_tensor(go, o);
    }
    fprintf(stderr, "%s: %lld tensors, %d converted to %s, %.1f MiB -> %.1f MiB\n", out.c_str(), (long long) nt,
            n_quant, q->name, bytes_in / 1048576.0, bytes_out / 1048576.0);
    if (!gguf_write_to_file(go, out.c_str(), false)) {
        fprintf(stderr, "write failed\n");
        return 1;
    }
    gguf_free(go);
    gguf_free(g);
    ggml_free(ctx_out);
    ggml_free(ctx_in);
    return 0;
}
