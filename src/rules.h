#pragma once

#include "encoder.h"
#include "tyger.h"
#include "turn_taker.h"

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct engine;

// fixed turn-rule numbers from the reference runtime
struct rule_consts {
    float seg_s = 15.0f;        // long turns are transcribed in blocks past this
    float seg_search_s = 4.0f;  // cut at the quietest frame of the block's last seconds
    float turn_max_s = 20.0f;   // longest audio one transcription gets
    float barge_pre_s = 2.0f;   // user audio kept while speaking, seeds the turn a barge starts
    int   onset_gap_frames = 8; // a pause longer than this ends a stretch of user speech
    float onset_max_s = 4.0f;
};

struct seg_list {
    std::mutex mu;
    std::vector<std::string> texts;
    std::vector<bool> done;
};

struct session_impl {
    tyger_model_impl * m = nullptr;
    engine * eng = nullptr;
    tyger_session_params p{};
    std::mutex params_mu;

    // audio and turn state, the reference session's fields
    std::vector<float> win_u, win_a, turn_u, pre_u, seed_u;
    bool has_seed = false;
    double seg_s = 0.0;
    std::shared_ptr<seg_list> segs;
    int talk_frames = 0, quiet_frames = 0;
    int prev_state = TT_LISTEN, yield_run = 0;
    double voiced_ms = 0.0;
    bool in_turn = false;
    int warm = 0;
    int mode = TYGER_MODE_LISTEN;
    int bc_run = 0, bc_cooldown = 0, ep_run = 0, sil_run = 0;
    double partial_at_s = 0.0;
    std::string words_sent;
    std::atomic<bool> pending_reset{true};

    // written by the asr thread
    std::mutex asr_mu;
    std::string partial_asr, words;
    bool partial_inflight = false;
    int turn_gen = 0;

    // models
    std::vector<float> gru_h;
    enc_stream es_u, es_a;

    // feed buffering and output
    std::vector<float> buf_u, buf_a;
    int64_t hops = 0;
    std::vector<tyger_event> events;
    std::deque<std::string> strs;
    std::string json;
    bool closed = false;

    void clear_turn_state();
    double turn_len_s() const;
    const char * keep(std::string s);
};

// one frame's outputs as events; the session's model state is reset by the caller when reset is set
struct frame_result {
    bool final = false;          // the turn ended: transcribe final_wav
    std::vector<float> final_wav;
    std::shared_ptr<seg_list> final_segs;
    size_t final_event = 0;      // index into session events
    bool reset = false;
};

void   frame_energy(const float * wav, size_t n, int hop, std::vector<float> & out);
bool   rules_begin_hop(session_impl & s, const float * uh, const float * ah, int n, int mode);
bool   rules_switch_mode(session_impl & s, int mode);
void   rules_words_event(session_impl & s, int mode, double t_s);
frame_result rules_on_frame(session_impl & s, int mode, const tt_out & out, float act_db, int left, double t_s);
