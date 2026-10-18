#include "rules.h"

#include <algorithm>
#include <cmath>

static const rule_consts kRules;
static const char * kStateNames[4] = {"LISTEN", "HOLD", "SPEAK", "BACKCHANNEL"};

void session_impl::clear_turn_state() {
    win_u.clear();
    win_a.clear();
    turn_u.clear();
    seg_s = 0.0;
    segs = std::make_shared<seg_list>();
    pre_u.clear();
    seed_u.clear();
    has_seed = false;
    talk_frames = quiet_frames = 0;
    prev_state = TT_LISTEN;
    yield_run = 0;
    voiced_ms = 0.0;
    in_turn = false;
    warm = 0;
    mode = TYGER_MODE_LISTEN;
    bc_run = bc_cooldown = ep_run = sil_run = 0;
    partial_at_s = 0.0;
    words_sent.clear();
    std::lock_guard<std::mutex> lk(asr_mu);
    partial_asr.clear();
    words.clear();
}

double session_impl::turn_len_s() const {
    return seg_s + (double) turn_u.size() / m->hp.sample_rate;
}

const char * session_impl::keep(std::string s) {
    strs.push_back(std::move(s));
    return strs.back().c_str();
}

void frame_energy(const float * wav, size_t n, int hop, std::vector<float> & out) {
    const size_t nf = n / hop;
    out.resize(nf);
    for (size_t f = 0; f < nf; f++) {
        const float * s = wav + f * hop;
        double acc = 0.0;
        for (int i = 0; i < hop; i++) acc += (double) s[i] * s[i];
        const float rms = std::sqrt((float) (acc / hop) + 1e-12f);
        out[f] = 20.0f * std::log10(rms + 1e-9f);
    }
}

static int state_decision(const float * logits, float hazard, int prev, float theta, float bc_thresh = 0.5f) {
    float mx = std::max(std::max(logits[0], logits[1]), std::max(logits[2], logits[3]));
    float p[4], sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        p[i] = std::exp(logits[i] - mx);
        sum += p[i];
    }
    for (float & v : p) v /= sum;
    const float pmax = std::max(std::max(p[0], p[1]), std::max(p[2], p[3]));
    if ((prev == TT_LISTEN || prev == TT_HOLD) && p[TT_BACKCHANNEL] >= bc_thresh && p[TT_BACKCHANNEL] == pmax) {
        return TT_BACKCHANNEL;
    }
    if (prev == TT_LISTEN || prev == TT_HOLD) {
        if (hazard >= theta) return TT_SPEAK;
        return p[TT_HOLD] > p[TT_LISTEN] ? TT_HOLD : TT_LISTEN;
    }
    if (prev == TT_SPEAK) {
        return p[TT_SPEAK] >= std::max(p[TT_LISTEN], p[TT_HOLD]) ? TT_SPEAK : TT_LISTEN;
    }
    if (prev == TT_BACKCHANNEL) {
        return p[TT_BACKCHANNEL] == pmax ? TT_BACKCHANNEL : TT_LISTEN;
    }
    return TT_LISTEN;
}

static void append_tail(std::vector<float> & v, const float * x, int n, size_t cap) {
    v.insert(v.end(), x, x + n);
    if (v.size() > cap) {
        v.erase(v.begin(), v.begin() + (v.size() - cap));
    }
}

bool rules_begin_hop(session_impl & s, const float * uh, const float * ah, int n, int mode) {
    const tyger_hparams & hp = s.m->hp;
    const int sr = hp.sample_rate;
    const int fs = hp.frame_samples();
    bool reset = s.pending_reset.exchange(false);
    if (reset) {
        std::vector<float> seed;
        bool had_seed = s.has_seed;
        if (had_seed) seed = s.seed_u;
        s.clear_turn_state();
        {
            std::lock_guard<std::mutex> lk(s.asr_mu);
            s.turn_gen++;
        }
        if (had_seed) {
            // a barge-in: the user started talking over the agent, keep that start
            s.turn_u = std::move(seed);
            std::vector<float> e;
            frame_energy(s.turn_u.data(), s.turn_u.size(), fs, e);
            int voiced = 0;
            for (float v : e) voiced += v > hp.vad_db;
            s.voiced_ms = voiced * hp.frame_s * 1e3;
            s.in_turn = s.voiced_ms > 0;
        }
    }
    s.has_seed = false;
    s.seed_u.clear();
    const size_t wcap = (size_t) hp.window_samples();
    append_tail(s.win_u, uh, n, wcap);
    append_tail(s.win_a, ah, n, wcap);
    if (mode == TYGER_MODE_LISTEN) {
        s.turn_u.insert(s.turn_u.end(), uh, uh + n);
    } else {
        append_tail(s.pre_u, uh, n, (size_t) (kRules.barge_pre_s * sr));
    }
    return reset;
}

bool rules_switch_mode(session_impl & s, int mode) {
    if (mode == s.mode) {
        return false;
    }
    s.warm = 0;
    s.yield_run = 0;
    s.bc_run = 0;
    s.bc_cooldown = 0;
    s.ep_run = 0;
    s.sil_run = 0;
    s.talk_frames = s.quiet_frames = 0;
    s.prev_state = mode == TYGER_MODE_SPEAKING ? TT_SPEAK : TT_LISTEN;
    s.mode = mode;
    return true;
}

void rules_words_event(session_impl & s, int mode, double t_s) {
    std::string words;
    {
        std::lock_guard<std::mutex> lk(s.asr_mu);
        words = s.words;
    }
    if (mode != TYGER_MODE_LISTEN || words == s.words_sent) {
        return;
    }
    s.words_sent = words;
    tyger_event ev{};
    ev.kind = TYGER_EVENT_PARTIAL;
    ev.t_s = t_s;
    ev.text = s.keep(words);
    s.events.push_back(ev);
}

static void track_talk(session_impl & s, bool voiced) {
    if (s.talk_frames) {
        s.talk_frames++;
        s.quiet_frames = voiced ? 0 : s.quiet_frames + 1;
        if (s.quiet_frames > kRules.onset_gap_frames) {
            s.talk_frames = s.quiet_frames = 0;
        }
    } else if (voiced) {
        s.talk_frames = 1;
        s.quiet_frames = 0;
    }
}


frame_result rules_on_frame(session_impl & s, int mode, const tt_out & out, float act_db, int left, double t_s) {
    const tyger_hparams & hp = s.m->hp;
    tyger_session_params p;
    {
        std::lock_guard<std::mutex> lk(s.params_mu);
        p = s.p;
    }
    frame_result res;
    const float haz = out.v[4], yp = out.v[5], bcp = out.v[6], pep = out.v[7];
    const int st = state_decision(out.v, haz, s.prev_state, p.theta);
    const bool voiced = act_db > hp.vad_db;

    auto frame_ev = [&](const char * state) {
        if (!p.emit_frames) return;
        tyger_event ev{};
        ev.kind = TYGER_EVENT_FRAME;
        ev.t_s = t_s;
        ev.state = state;
        ev.yield_prob = yp;
        ev.hazard = haz;
        ev.p_end = pep;
        ev.bc_prob = bcp;
        s.events.push_back(ev);
    };

    if (mode == TYGER_MODE_SPEAKING) {
        s.warm++;
        track_talk(s, voiced);
        if (s.warm > hp.warmup_frames) {
            s.yield_run = yp >= p.barge_thr ? s.yield_run + 1 : 0;
            if (s.yield_run >= p.barge_deb) {
                float onset = 0.0f;
                if (s.talk_frames) {
                    onset = std::min((s.talk_frames + left) * hp.frame_s, kRules.onset_max_s);
                }
                tyger_event ev{};
                ev.kind = TYGER_EVENT_BARGE;
                ev.t_s = t_s;
                ev.yield_prob = yp;
                ev.onset_s = onset;
                s.events.push_back(ev);
                s.yield_run = 0;
                s.seed_u = s.pre_u;
                s.has_seed = true;
            }
        }
        frame_ev("SPEAK");
        s.prev_state = TT_SPEAK;
        return res;
    }

    s.warm++;
    if (voiced) {
        s.voiced_ms += hp.frame_s * 1e3;
        s.in_turn = true;
    }
    s.sil_run = voiced ? 0 : s.sil_run + 1;
    if (s.bc_cooldown > 0) {
        s.bc_cooldown--;
        s.bc_run = 0;
    } else if (s.in_turn && s.voiced_ms >= hp.min_speech_ms && s.warm > hp.warmup_frames) {
        s.bc_run = bcp >= p.bc_thr ? s.bc_run + 1 : 0;
        if (s.bc_run >= p.bc_deb) {
            tyger_event ev{};
            ev.kind = TYGER_EVENT_BACKCHANNEL;
            ev.t_s = t_s;
            {
                std::lock_guard<std::mutex> lk(s.asr_mu);
                ev.text = s.keep(s.partial_asr);
            }
            ev.bc_prob = bcp;
            ev.turn_s = (float) s.turn_len_s();
            s.events.push_back(ev);
            s.bc_run = 0;
            s.bc_cooldown = p.bc_refractory;
        }
    }
    if (s.warm > hp.warmup_frames) {
        s.ep_run = pep >= p.ep_thr ? s.ep_run + 1 : 0;
    } else {
        s.ep_run = 0;
    }
    const bool spoke = s.in_turn && s.voiced_ms >= hp.min_speech_ms;
    const bool by_head = spoke && s.ep_run >= p.ep_deb;
    const bool by_sil = spoke && !by_head && p.fallback_sil_frames > 0 && s.warm > hp.warmup_frames &&
                        s.sil_run >= p.fallback_sil_frames;
    if (by_head || by_sil) {
        tyger_event ev{};
        ev.kind = TYGER_EVENT_ENDPOINT;
        ev.t_s = t_s;
        ev.turn_s = (float) s.turn_len_s();
        ev.segments = (int) s.segs->texts.size();
        ev.by = by_head ? "head" : "silence";
        ev.p_end = pep;
        ev.text = s.keep("");
        res.final = true;
        res.final_wav = s.turn_u;
        res.final_segs = s.segs;
        res.final_event = s.events.size();
        s.events.push_back(ev);

        s.turn_u.clear();
        s.voiced_ms = 0.0;
        s.in_turn = false;
        s.seg_s = 0.0;
        s.segs = std::make_shared<seg_list>();
        s.prev_state = TT_LISTEN;
        s.warm = 0;
        s.bc_run = s.bc_cooldown = s.ep_run = s.sil_run = 0;
        s.partial_at_s = 0.0;
        {
            std::lock_guard<std::mutex> lk(s.asr_mu);
            s.partial_asr.clear();
            s.turn_gen++;
            s.words.clear();
        }
        s.words_sent.clear();
        frame_ev("idle");
        res.reset = true;
        return res;
    }
    const char * disp = s.in_turn ? kStateNames[st] : (!voiced ? "idle" : "LISTEN");
    frame_ev(disp);
    s.prev_state = TT_LISTEN;
    return res;
}
