#pragma once

#include "model.h"
#include "ops.h"

#include <string>
#include <vector>

// greedy rnn-t over encoder-projected frames (enc: [T][dec_hidden]), the same steps as hf generate()
class rnnt_decoder {
public:
    explicit rnnt_decoder(const tyger_model_impl & m);
    std::vector<int> decode(const float * enc, int T);
    std::string detokenize(const std::vector<int> & ids) const;

private:
    void run_decoder(int tok, std::vector<float> & h, std::vector<float> & c, std::vector<float> & out);

    const tyger_model_impl & m_;
    const rnnt_weights & w_;
    int H_, V_, n_threads_;
    host_linear lin_;
    std::vector<float> emb_, gates_, gates2_, x_, joint_, logits_;
};
