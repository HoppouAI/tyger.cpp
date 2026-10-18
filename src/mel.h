#pragma once

#include "model.h"

#include <complex>
#include <cstdint>
#include <vector>

// log-mel features as nemotron's processor computes them (preemphasis, centered stft, slaney mel, log)
class mel_extractor {
public:
    explicit mel_extractor(const tyger_model_impl & m);

    // whole clip, center=True: n/hop + 1 frames of which n/hop are valid, the rest zeroed
    void offline(const float * x, int n, std::vector<float> & out, int & n_frames, int & n_valid) const;

    // one frame from n_fft preemphasized samples starting at frame*hop - n_fft/2
    void frame(const float * seg, float * out) const;

    int n_mels() const { return n_mels_; }
    int n_fft() const { return n_fft_; }
    int hop() const { return hop_; }
    float preemph() const { return preemph_; }

private:
    int n_fft_, hop_, n_mels_, n_bins_;
    float preemph_;
    const float * filters_;
    std::vector<double> window_;
    std::vector<std::complex<double>> twiddle_;
    std::vector<int> bitrev_;
};

// incremental mel for the fast encoder: continuous over the whole stream like one center=True pass
struct mel_stream {
    std::vector<float> pre;     // preemphasized samples, pre[0] is absolute sample `base`
    int64_t base = 0;
    int64_t n_in = 0;
    float   last_x = 0.0f;
    int64_t n_frames = 0;       // frames produced so far
    std::vector<float> frames;  // every produced frame, n_mels each (trimmed by the caller)
    int64_t frames_base = 0;    // absolute index of frames[0]

    void reset() { *this = mel_stream(); }
    void push(const mel_extractor & mel, const float * x, int n);
};
