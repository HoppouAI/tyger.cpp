#include "rnnt.h"

#include <algorithm>
#include <cmath>
#include <cstring>

rnnt_decoder::rnnt_decoder(const tyger_model_impl & m) : m_(m), w_(m.rnnt) {
    H_ = m.hp.dec_hidden;
    V_ = (int) w_.head.n_out;
    n_threads_ = m.n_threads;
    emb_.resize(H_);
    gates_.resize(4 * H_);
    gates2_.resize(4 * H_);
    x_.resize(H_);
    joint_.resize(H_);
    logits_.resize(V_);
}

void rnnt_decoder::run_decoder(int tok, std::vector<float> & h, std::vector<float> & c, std::vector<float> & out) {
    const host_mat & E = w_.embedding;
    const ggml_type_traits * tr = ggml_get_type_traits(E.type);
    if (E.type == GGML_TYPE_F32) {
        memcpy(emb_.data(), E.row(tok), sizeof(float) * H_);
    } else {
        tr->to_float(E.row(tok), emb_.data(), H_);
    }
    const float * in = emb_.data();
    for (int l = 0; l < m_.hp.dec_layers; l++) {
        float * hl = h.data() + (size_t) l * H_;
        float * cl = c.data() + (size_t) l * H_;
        lin_.run(w_.w_ih[l], in, w_.b_ih[l].data(), gates_.data());
        lin_.run(w_.w_hh[l], hl, w_.b_hh[l].data(), gates2_.data());
        // torch lstm gate order i, f, g, o
        for (int i = 0; i < H_; i++) {
            const float ig = sigmoidf_(gates_[i] + gates2_[i]);
            const float fg = sigmoidf_(gates_[H_ + i] + gates2_[H_ + i]);
            const float gg = std::tanh(gates_[2 * H_ + i] + gates2_[2 * H_ + i]);
            const float og = sigmoidf_(gates_[3 * H_ + i] + gates2_[3 * H_ + i]);
            cl[i] = fg * cl[i] + ig * gg;
            x_[i] = og * std::tanh(cl[i]);
        }
        memcpy(hl, x_.data(), sizeof(float) * H_);
        in = hl;
    }
    out.resize(H_);
    lin_.run(w_.proj, in, w_.proj_b.data(), out.data());
}

std::vector<int> rnnt_decoder::decode(const float * enc, int T) {
    const int blank = m_.hp.blank_id, max_sym = m_.hp.max_symbols;
    const int L = m_.hp.dec_layers;
    std::vector<float> h((size_t) L * H_, 0.0f), c((size_t) L * H_, 0.0f), d, h2, c2, d2;
    run_decoder(blank, h, c, d);
    std::vector<int> toks;
    int t = 0, sym = 0;
    // one step per joint eval: a blank or max_sym tokens in a row move to the next frame
    while (t < T) {
        const float * e = enc + (size_t) t * H_;
        for (int i = 0; i < H_; i++) {
            joint_[i] = std::max(0.0f, e[i] + d[i]);
        }
        lin_.run(w_.head, joint_.data(), w_.head_b.data(), logits_.data(), n_threads_);
        int best = 0;
        float bv = logits_[0];
        for (int i = 1; i < V_; i++) {
            if (logits_[i] > bv) {
                bv = logits_[i];
                best = i;
            }
        }
        if (best == blank) {
            sym = 0;
            t++;
            continue;
        }
        toks.push_back(best);
        sym++;
        if (sym >= max_sym) {
            sym = 0;
            t++;
        }
        run_decoder(best, h, c, d);
    }
    return toks;
}

std::string rnnt_decoder::detokenize(const std::vector<int> & ids) const {
    std::string s;
    for (int id : ids) {
        if (id < 0 || id >= (int) m_.pieces.size() || m_.special[id]) {
            continue;
        }
        const std::string & p = m_.pieces[id];
        // byte fallback pieces look like <0x41>
        if (p.size() == 6 && p[0] == '<' && p[1] == '0' && p[2] == 'x' && p[5] == '>') {
            s.push_back((char) strtol(p.substr(3, 2).c_str(), nullptr, 16));
            continue;
        }
        s += p;
    }
    // metaspace: the marker becomes a space
    static const std::string mark = "\xE2\x96\x81";
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        if (s.compare(i, mark.size(), mark) == 0) {
            out.push_back(' ');
            i += mark.size();
        } else {
            out.push_back(s[i++]);
        }
    }
    size_t a = out.find_first_not_of(' '), b = out.find_last_not_of(' ');
    return a == std::string::npos ? std::string() : out.substr(a, b - a + 1);
}
