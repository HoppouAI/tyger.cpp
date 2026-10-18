#include "mel.h"

#include <cmath>
#include <cstring>

static const double kPi = 3.14159265358979323846;
static const float kLogGuard = 5.960464477539063e-08f; // 2^-24

mel_extractor::mel_extractor(const tyger_model_impl & m) {
    const tyger_hparams & hp = m.hp;
    n_fft_ = hp.n_fft;
    hop_ = hp.hop_length;
    n_mels_ = hp.n_mels;
    n_bins_ = n_fft_ / 2 + 1;
    preemph_ = hp.preemph;
    filters_ = m.mel_filters.data();

    // torch.hann_window(win, periodic=False) centered inside n_fft
    window_.assign(n_fft_, 0.0);
    const int win = hp.win_length;
    const int off = (n_fft_ - win) / 2;
    for (int i = 0; i < win; i++) {
        window_[off + i] = 0.5 - 0.5 * std::cos(2.0 * kPi * i / (win - 1));
    }

    int bits = 0;
    while ((1 << bits) < n_fft_) bits++;
    bitrev_.resize(n_fft_);
    for (int i = 0; i < n_fft_; i++) {
        int r = 0;
        for (int b = 0; b < bits; b++) {
            if (i & (1 << b)) r |= 1 << (bits - 1 - b);
        }
        bitrev_[i] = r;
    }
    twiddle_.resize(n_fft_ / 2);
    for (int i = 0; i < n_fft_ / 2; i++) {
        twiddle_[i] = std::polar(1.0, -2.0 * kPi * i / n_fft_);
    }
}

void mel_extractor::frame(const float * seg, float * out) const {
    std::vector<std::complex<double>> a(n_fft_);
    for (int i = 0; i < n_fft_; i++) {
        a[bitrev_[i]] = std::complex<double>(seg[i] * window_[i], 0.0);
    }
    for (int len = 2; len <= n_fft_; len <<= 1) {
        const int half = len / 2, step = n_fft_ / len;
        for (int i = 0; i < n_fft_; i += len) {
            for (int j = 0; j < half; j++) {
                std::complex<double> t = twiddle_[j * step] * a[i + j + half];
                a[i + j + half] = a[i + j] - t;
                a[i + j] += t;
            }
        }
    }
    double power[1024];
    for (int k = 0; k < n_bins_; k++) {
        power[k] = std::norm(a[k]);
    }
    for (int mI = 0; mI < n_mels_; mI++) {
        const float * f = filters_ + (size_t) mI * n_bins_;
        double s = 0.0;
        for (int k = 0; k < n_bins_; k++) {
            s += f[k] * power[k];
        }
        out[mI] = std::log((float) s + kLogGuard);
    }
}

void mel_extractor::offline(const float * x, int n, std::vector<float> & out, int & n_frames, int & n_valid) const {
    const int pad = n_fft_ / 2;
    std::vector<float> y(n + 2 * pad, 0.0f);
    if (n > 0) {
        y[pad] = x[0];
        for (int i = 1; i < n; i++) {
            y[pad + i] = x[i] - preemph_ * x[i - 1];
        }
    }
    n_frames = n / hop_ + 1;
    n_valid = n / hop_;
    out.assign((size_t) n_frames * n_mels_, 0.0f);
    for (int t = 0; t < n_valid; t++) {
        frame(y.data() + (size_t) t * hop_, out.data() + (size_t) t * n_mels_);
    }
}

void mel_stream::push(const mel_extractor & mel, const float * x, int n) {
    const int nf = mel.n_fft(), hop = mel.hop(), nm = mel.n_mels();
    const float a = mel.preemph();
    for (int i = 0; i < n; i++) {
        float v = n_in == 0 ? x[i] : x[i] - a * last_x;
        last_x = x[i];
        pre.push_back(v);
        n_in++;
    }
    std::vector<float> seg(nf);
    // frame t covers [t*hop - nf/2, t*hop + nf/2), zeros before the stream start
    while (n_frames * hop + nf / 2 <= n_in) {
        const int64_t s0 = n_frames * hop - nf / 2;
        for (int i = 0; i < nf; i++) {
            int64_t ai = s0 + i;
            seg[i] = ai < 0 ? 0.0f : pre[(size_t) (ai - base)];
        }
        size_t o = frames.size();
        frames.resize(o + nm);
        mel.frame(seg.data(), frames.data() + o);
        n_frames++;
    }
    // keep what the next frame still needs
    const int64_t keep_from = std::max<int64_t>(0, n_frames * hop - nf / 2);
    if (keep_from > base) {
        pre.erase(pre.begin(), pre.begin() + (size_t) (keep_from - base));
        base = keep_from;
    }
}
