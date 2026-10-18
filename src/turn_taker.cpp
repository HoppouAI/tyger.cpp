#include "turn_taker.h"

#include <algorithm>
#include <cmath>

turn_taker::turn_taker(const tyger_model_impl & m) : m_(m), w_(m.tt) {
    width_ = m.hp.tt_width;
    layers_ = m.hp.tt_layers;
    n_threads_ = std::min(m.n_threads, 8);
    x_.resize(width_);
    a_.resize(width_);
    gi_.resize(3 * width_);
    gh_.resize(3 * width_);
    t_.resize(width_);
    z_.resize(width_);
    hid_.resize(width_);
}

static float mlp_head(host_linear & lin, const host_mat & w0, const host_vec & b0, const host_vec & w2,
                      const host_vec & b2, const float * z, std::vector<float> & hid, int nth) {
    lin.run(w0, z, b0.data(), hid.data(), nth);
    float s = b2.v[0];
    for (size_t i = 0; i < hid.size(); i++) {
        const float h = hid[i];
        s += w2.v[i] * (h / (1.0f + std::exp(-h)));
    }
    return sigmoidf_(s);
}

tt_out turn_taker::step(std::vector<float> & h, const float * hu, const float * ha, float act_db, int prev_state) {
    const int d = width_;
    const float act = std::min(3.0f, std::max(-3.0f, (act_db + 50.0f) / 25.0f));

    // fuse: in_norm(proj_user(hu) + proj_agent(ha) + proj_energy(act) + state_emb(prev)), summed in that order
    lin_.run(w_.proj_user, hu, w_.proj_user_b.data(), x_.data(), n_threads_);
    lin_.run(w_.proj_agent, ha, w_.proj_agent_b.data(), a_.data(), n_threads_);
    const float * se = w_.state_emb.data() + (size_t) prev_state * d;
    for (int i = 0; i < d; i++) {
        const float e = w_.proj_energy_w.v[i] * act + w_.proj_energy_b.v[i];
        x_[i] = ((x_[i] + a_[i]) + e) + se[i];
    }
    layer_norm(x_.data(), d, w_.in_norm_w.data(), w_.in_norm_b.data());

    // gru, gate order r, z, n
    const float * in = x_.data();
    for (int l = 0; l < layers_; l++) {
        float * hl = h.data() + (size_t) l * d;
        lin_.run(w_.w_ih[l], in, w_.b_ih[l].data(), gi_.data(), n_threads_);
        lin_.run(w_.w_hh[l], hl, w_.b_hh[l].data(), gh_.data(), n_threads_);
        for (int i = 0; i < d; i++) {
            const float r = sigmoidf_(gi_[i] + gh_[i]);
            const float z = sigmoidf_(gi_[d + i] + gh_[d + i]);
            const float n = std::tanh(gi_[2 * d + i] + r * gh_[2 * d + i]);
            t_[i] = (1.0f - z) * n + z * hl[i];
        }
        std::copy(t_.begin(), t_.end(), hl);
        in = hl;
    }

    z_.assign(in, in + d);
    layer_norm(z_.data(), d, w_.out_norm_w.data(), w_.out_norm_b.data());
    tt_out o;
    for (int k = 0; k < 4; k++) {
        o.v[k] = w_.state_head_b.v[k] + dot_f32(w_.state_head_w.data() + (size_t) k * d, z_.data(), d);
    }
    o.v[4] = sigmoidf_(w_.hazard_b.v[0] + dot_f32(w_.hazard_w.data(), z_.data(), d));
    o.v[5] = mlp_head(lin_, w_.yield0, w_.yield0_b, w_.yield2_w, w_.yield2_b, z_.data(), hid_, n_threads_);
    o.v[6] = mlp_head(lin_, w_.bc0, w_.bc0_b, w_.bc2_w, w_.bc2_b, z_.data(), hid_, n_threads_);
    o.v[7] = mlp_head(lin_, w_.ep0, w_.ep0_b, w_.ep2_w, w_.ep2_b, z_.data(), hid_, n_threads_);
    return o;
}
