// checks the fast encoder's incremental frames against one offline encode of the whole clip
#include "encoder.h"
#include "mel.h"
#include "model.h"
#include "wav.h"

#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: test-stream model.gguf clip.wav [cpu]\n");
        return 2;
    }
    ggml_cpu_init();
    tyger_model_impl m;
    std::string err;
    const bool cpu = argc > 3 && strcmp(argv[3], "cpu") == 0;
    if (!tyger_model_load_impl(m, argv[1], cpu ? 1 : 0, -1, 0, false, false, err)) {
        fprintf(stderr, "load: %s\n", err.c_str());
        return 1;
    }
    wav_data w;
    if (!wav_read(argv[2], w, err)) {
        fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    std::vector<float> pcm = wav_mono(w, -1, 16000);
    const int hop = m.hp.hop_samples(), HF = m.hp.hop_frames(), C = m.hp.enc_hidden;
    const int n_hops = (int) (pcm.size() / hop);
    pcm.resize((size_t) n_hops * hop);

    mel_extractor mel(m);
    enc_runner r(m);

    // offline over the whole clip
    std::vector<float> feats, off;
    int nf, nv, T, Co;
    std::vector<int> ov;
    mel.offline(pcm.data(), (int) pcm.size(), feats, nf, nv);
    if (!encode_offline(r, {feats.data()}, nf, {nv}, m.hp.feature_layer, 0, false, -1, off, T, Co, ov)) {
        fprintf(stderr, "offline encode failed\n");
        return 1;
    }

    // streamed hop by hop
    enc_stream s;
    std::vector<float> got;
    for (int h = 0; h < n_hops; h++) {
        s.mel.push(mel, pcm.data() + (size_t) h * hop, hop);
        if (!enc_stream_ready(m, s, HF)) {
            fprintf(stderr, "hop %d not ready\n", h);
            return 1;
        }
        std::vector<std::vector<float>> out;
        if (!encode_stream(r, {&s}, HF, out)) {
            fprintf(stderr, "stream encode failed\n");
            return 1;
        }
        got.insert(got.end(), out[0].begin(), out[0].end());
    }
    const int n = (int) (got.size() / C);
    // streamed index i is absolute frame i + 1; every hop's newest frame is speculative, the last offline ones see padding
    const int lim = std::min(n, T - 1) - 2;
    double max_abs = 0, sum_abs = 0, min_cos = 1, spec_cos = 1;
    int cmp = 0;
    for (int i = 0; i < lim; i++) {
        const float * a = off.data() + (size_t) (i + 1) * C;
        const float * b = got.data() + (size_t) i * C;
        if ((i + 1) % HF == 0) {
            double dot = 0, na = 0, nb = 0;
            for (int k = 0; k < C; k++) {
                dot += a[k] * b[k];
                na += a[k] * a[k];
                nb += b[k] * b[k];
            }
            spec_cos = std::min(spec_cos, dot / std::sqrt(na * nb + 1e-30));
            continue;
        }
        cmp++;
        double dot = 0, na = 0, nb = 0;
        for (int k = 0; k < C; k++) {
            double d = std::fabs(a[k] - b[k]);
            max_abs = std::max(max_abs, d);
            sum_abs += d;
            dot += a[k] * b[k];
            na += a[k] * a[k];
            nb += b[k] * b[k];
        }
        min_cos = std::min(min_cos, dot / std::sqrt(na * nb + 1e-30));
    }
    printf("offline %d frames, streamed %d, compared %d: max abs %.3e mean abs %.3e min cos %.6f\n", T, n, cmp, max_abs,
           sum_abs / ((double) cmp * C), min_cos);
    return min_cos > 0.999 ? 0 : 1;
}
