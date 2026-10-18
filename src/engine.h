#pragma once

#include "encoder.h"
#include "mel.h"
#include "rnnt.h"
#include "rules.h"
#include "turn_taker.h"

#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

struct asr_job {
    enum kind_t { FINAL, SEGMENT, PARTIAL } kind;
    std::shared_ptr<session_impl> sess;
    int gen = 0;
    std::vector<float> wav;
    int prompt_id = -1;
    int segs = 0;
    std::function<void(const std::string &)> done;
};

struct hop_item {
    std::shared_ptr<session_impl> s;
    const float * uh;
    const float * ah;
    int mode;
};

struct engine {
    explicit engine(tyger_model_impl & m);
    ~engine();

    // one batched tick; every item is exactly one hop. fills each session's events
    bool tick(std::vector<hop_item> & items, std::string & err);

    // blocking offline transcription on the caller's thread
    std::string transcribe(const float * pcm, int n, int prompt_id);
    bool encode_debug(const float * pcm, int n, int n_layers, int lookahead, std::vector<float> & out, int & T);

    tyger_model_impl & m;
    mel_extractor mel;

private:
    void asr_loop();
    void cut_segment(session_impl & s);
    void maybe_partial(const std::shared_ptr<session_impl> & s);
    bool encode_exact(std::vector<hop_item> & items, std::vector<std::vector<float>> & hu,
                      std::vector<std::vector<float>> & ha, std::vector<int> & n_frames);

    enc_runner tick_runner;
    std::unique_ptr<enc_runner> asr_runner;
    std::unique_ptr<rnnt_decoder> dec;
    std::mutex asr_dec_mu;
    turn_taker tt;
    std::mutex tick_mu;
    std::map<int, std::vector<float>> zero_cache; // window samples -> last hop frames of an all-zero window

    std::thread asr_thread;
    std::mutex q_mu;
    std::condition_variable q_cv;
    std::deque<asr_job> q;
    bool stop = false;
};
