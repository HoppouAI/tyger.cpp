#pragma once

#include <string>
#include <vector>

struct wav_data {
    int sample_rate = 0;
    int channels = 0;
    std::vector<std::vector<float>> ch; // per channel, [-1, 1]
};

bool wav_read(const std::string & path, wav_data & out, std::string & err);
bool wav_write(const std::string & path, const std::vector<float> & pcm, int sample_rate);

// one channel (-1 mixes all down) resampled to `sr`
std::vector<float> wav_mono(const wav_data & w, int channel, int sr);
std::vector<float> resample_linear(const std::vector<float> & x, int from, int to);
