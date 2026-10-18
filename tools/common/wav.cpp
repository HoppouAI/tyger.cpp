#include "wav.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

static uint32_t rd32(const uint8_t * p) { return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t) p[3] << 24); }
static uint16_t rd16(const uint8_t * p) { return (uint16_t) (p[0] | (p[1] << 8)); }

bool wav_read(const std::string & path, wav_data & out, std::string & err) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    std::vector<uint8_t> d;
    uint8_t buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) d.insert(d.end(), buf, buf + n);
    fclose(f);
    if (d.size() < 12 || memcmp(d.data(), "RIFF", 4) || memcmp(d.data() + 8, "WAVE", 4)) {
        err = path + ": not a RIFF/WAVE file";
        return false;
    }
    int fmt = 0, ch = 0, sr = 0, bits = 0;
    const uint8_t * data = nullptr;
    size_t data_len = 0;
    for (size_t p = 12; p + 8 <= d.size();) {
        uint32_t len = rd32(&d[p + 4]);
        const uint8_t * body = &d[p + 8];
        size_t avail = d.size() - (p + 8);
        if (!memcmp(&d[p], "fmt ", 4) && len >= 16) {
            fmt = rd16(body);
            ch = rd16(body + 2);
            sr = (int) rd32(body + 4);
            bits = rd16(body + 14);
            if (fmt == 0xFFFE && len >= 26) fmt = rd16(body + 24);
        } else if (!memcmp(&d[p], "data", 4)) {
            data = body;
            data_len = len > avail ? avail : len;
        }
        p += 8 + len + (len & 1);
    }
    if (!data || ch <= 0 || sr <= 0) {
        err = path + ": missing fmt or data chunk";
        return false;
    }
    const int bps = bits / 8;
    if (!((fmt == 1 && (bits == 16 || bits == 24 || bits == 32)) || (fmt == 3 && bits == 32))) {
        err = path + ": only 16/24/32-bit pcm and 32-bit float wav are supported";
        return false;
    }
    const size_t frames = data_len / (size_t) (bps * ch);
    out.sample_rate = sr;
    out.channels = ch;
    out.ch.assign(ch, std::vector<float>(frames));
    for (size_t i = 0; i < frames; i++) {
        for (int c = 0; c < ch; c++) {
            const uint8_t * s = data + (i * ch + c) * bps;
            float v;
            if (fmt == 3) {
                memcpy(&v, s, 4);
            } else if (bits == 16) {
                v = (int16_t) rd16(s) / 32768.0f;
            } else if (bits == 24) {
                int32_t x = (int32_t) ((s[0] << 8) | (s[1] << 16) | ((uint32_t) s[2] << 24)) >> 8;
                v = x / 8388608.0f;
            } else {
                v = (int32_t) rd32(s) / 2147483648.0f;
            }
            out.ch[c][i] = v;
        }
    }
    return true;
}

bool wav_write(const std::string & path, const std::vector<float> & pcm, int sample_rate) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    const uint32_t n = (uint32_t) pcm.size();
    auto w32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); };
    auto w16 = [&](uint16_t v) { fwrite(&v, 2, 1, f); };
    fwrite("RIFF", 1, 4, f);
    w32(36 + n * 2);
    fwrite("WAVEfmt ", 1, 8, f);
    w32(16);
    w16(1);
    w16(1);
    w32(sample_rate);
    w32(sample_rate * 2);
    w16(2);
    w16(16);
    fwrite("data", 1, 4, f);
    w32(n * 2);
    for (float v : pcm) {
        float c = v < -1.0f ? -1.0f : v > 1.0f ? 1.0f : v;
        int16_t s = (int16_t) (c * 32767.0f);
        fwrite(&s, 2, 1, f);
    }
    fclose(f);
    return true;
}

std::vector<float> resample_linear(const std::vector<float> & x, int from, int to) {
    if (from == to || x.empty()) return x;
    const double ratio = (double) from / to;
    const size_t n = (size_t) ((double) x.size() / ratio);
    std::vector<float> y(n);
    for (size_t i = 0; i < n; i++) {
        double p = i * ratio;
        size_t a = (size_t) p;
        double f = p - a;
        float va = x[a], vb = a + 1 < x.size() ? x[a + 1] : x[a];
        y[i] = (float) (va + (vb - va) * f);
    }
    return y;
}

std::vector<float> wav_mono(const wav_data & w, int channel, int sr) {
    std::vector<float> m;
    if (w.channels == 0) return m;
    if (channel >= 0 && channel < w.channels) {
        m = w.ch[channel];
    } else {
        m.assign(w.ch[0].size(), 0.0f);
        for (int c = 0; c < w.channels; c++)
            for (size_t i = 0; i < m.size(); i++) m[i] += w.ch[c][i] / w.channels;
    }
    return resample_linear(m, w.sample_rate, sr);
}
