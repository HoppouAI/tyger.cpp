#pragma once

#include "model.h"

#include "ggml-cpu.h"

#include <cmath>
#include <vector>

// y[o] = W[o] . x (+ b[o]) for a host weight in any ggml type, rows split across threads
struct host_linear {
    std::vector<uint8_t> xq;

    void run(const host_mat & w, const float * x, const float * b, float * y, int n_threads = 1) {
        const ggml_type_traits_cpu * tr = ggml_get_type_traits_cpu(w.type);
        const void * xin = x;
        if (tr->vec_dot_type != GGML_TYPE_F32) {
            xq.resize(ggml_row_size(tr->vec_dot_type, w.n_in));
            ggml_get_type_traits_cpu(tr->vec_dot_type)->from_float(x, xq.data(), w.n_in);
            xin = xq.data();
        }
        const int64_t n = w.n_out;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) schedule(static) if (n_threads > 1 && n >= 256)
#endif
        for (int64_t o = 0; o < n; o++) {
            float s = 0.0f;
            tr->vec_dot((int) w.n_in, &s, 0, w.row(o), 0, xin, 0, 1);
            y[o] = b ? s + b[o] : s;
        }
    }
};

inline float sigmoidf_(float x) { return 1.0f / (1.0f + std::exp(-x)); }

inline void layer_norm(float * x, int n, const float * w, const float * b, float eps = 1e-5f) {
    double mean = 0.0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    double var = 0.0;
    for (int i = 0; i < n; i++) {
        double d = x[i] - mean;
        var += d * d;
    }
    var /= n;
    const float inv = (float) (1.0 / std::sqrt(var + eps));
    for (int i = 0; i < n; i++) {
        x[i] = (float) ((x[i] - mean) * inv) * w[i] + b[i];
    }
}

inline float dot_f32(const float * a, const float * b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}
