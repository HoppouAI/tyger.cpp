#pragma once

#include "model.h"
#include "ops.h"

#include <vector>

enum { TT_LISTEN = 0, TT_HOLD = 1, TT_SPEAK = 2, TT_BACKCHANNEL = 3 };

// outputs per frame: state logits x4, then hazard, yield, backchannel and endpoint probabilities
struct tt_out {
    float v[8];
};

class turn_taker {
public:
    explicit turn_taker(const tyger_model_impl & m);

    int state_size() const { return width_ * layers_; }
    void reset(std::vector<float> & h) const { h.assign(state_size(), 0.0f); }

    // one frame from layer-15 features of both ears, the user's frame energy in dB and the previous state
    tt_out step(std::vector<float> & h, const float * hu, const float * ha, float act_db, int prev_state);

private:
    const tyger_model_impl & m_;
    const turn_taker_weights & w_;
    int width_, layers_, n_threads_;
    host_linear lin_;
    std::vector<float> x_, a_, gi_, gh_, t_, z_, hid_;
};
