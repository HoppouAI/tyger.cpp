#include "tyger.h"

#include "engine.h"
#include "model.h"

#include "ggml-cpu.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

struct tyger_model {
    std::unique_ptr<tyger_model_impl> m;
    std::unique_ptr<engine> eng;
};

struct tyger_session {
    tyger_model * model = nullptr;
    std::shared_ptr<session_impl> s;
};

static thread_local std::string g_err;

static void set_err(const std::string & e) { g_err = e; }

extern "C" {

const char * tyger_last_error(void) { return g_err.c_str(); }

const char * tyger_version(void) { return "0.1.0"; }

tyger_model_params tyger_model_default_params(void) {
    tyger_model_params p;
    p.backend = TYGER_BACKEND_AUTO;
    p.device = -1;
    p.n_threads = 0;
    p.enable_asr = true;
    p.verbose = false;
    return p;
}

tyger_model * tyger_model_load(const char * path, tyger_model_params params) {
    if (!path) {
        set_err("null path");
        return nullptr;
    }
    ggml_cpu_init();
    auto mdl = std::make_unique<tyger_model>();
    mdl->m = std::make_unique<tyger_model_impl>();
    std::string err;
    if (!tyger_model_load_impl(*mdl->m, path, params.backend, params.device, params.n_threads, params.enable_asr,
                               params.verbose, err)) {
        set_err(err);
        return nullptr;
    }
    try {
        mdl->eng = std::make_unique<engine>(*mdl->m);
    } catch (const std::exception & e) {
        set_err(e.what());
        return nullptr;
    }
    return mdl.release();
}

void tyger_model_free(tyger_model * model) {
    if (!model) return;
    model->eng.reset();
    model->m.reset();
    delete model;
}

const char * tyger_model_info(const tyger_model * model) { return model ? model->m->info_json.c_str() : "{}"; }

int tyger_sample_rate(const tyger_model * model) { return model ? model->m->hp.sample_rate : 0; }

int tyger_hop_samples(const tyger_model * model) { return model ? model->m->hp.hop_samples() : 0; }

tyger_session_params tyger_session_default_params(const tyger_model * model) {
    tyger_session_params p;
    // the reference runtime's live defaults
    p.ep_thr = 0.915f;
    p.ep_deb = 2;
    p.barge_thr = 0.4f;
    p.barge_deb = 2;
    p.bc_thr = 0.4f;
    p.bc_deb = 2;
    p.bc_refractory = model ? model->m->hp.bc_refractory : 15;
    p.theta = 0.2f;
    p.partial_refresh_s = 0.8f;
    p.fallback_sil_frames = 0;
    p.encoder_mode = TYGER_ENCODER_EXACT;
    p.emit_frames = true;
    p.prompt_id = -1;
    return p;
}

static bool valid_params(const tyger_session_params * p) {
    return p && p->ep_deb >= 1 && p->barge_deb >= 1 && p->bc_deb >= 1 && p->bc_refractory >= 0 &&
           (p->encoder_mode == TYGER_ENCODER_EXACT || p->encoder_mode == TYGER_ENCODER_FAST);
}

tyger_session * tyger_session_open(tyger_model * model, const tyger_session_params * params) {
    if (!model) {
        set_err("null model");
        return nullptr;
    }
    tyger_session_params p = params ? *params : tyger_session_default_params(model);
    if (!valid_params(&p)) {
        set_err("invalid session params");
        return nullptr;
    }
    auto * ses = new tyger_session();
    ses->model = model;
    ses->s = std::make_shared<session_impl>();
    ses->s->m = model->m.get();
    ses->s->eng = model->eng.get();
    ses->s->p = p;
    ses->s->clear_turn_state();
    return ses;
}

void tyger_session_close(tyger_session * session) {
    if (!session) return;
    session->s->closed = true;
    delete session;
}

void tyger_session_reset(tyger_session * session) {
    if (session) session->s->pending_reset = true;
}

void tyger_session_reset_keep(tyger_session * session) {
    if (!session) return;
    session->s->keep_pre = true;
    session->s->pending_reset = true;
}

int tyger_session_set_params(tyger_session * session, const tyger_session_params * params) {
    if (!session || !valid_params(params)) {
        set_err("invalid session params");
        return -1;
    }
    std::lock_guard<std::mutex> lk(session->s->params_mu);
    if (params->encoder_mode != session->s->p.encoder_mode) {
        session->s->pending_reset = true;
    }
    session->s->p = *params;
    return 0;
}

int tyger_session_get_params(const tyger_session * session, tyger_session_params * params) {
    if (!session || !params) return -1;
    std::lock_guard<std::mutex> lk(session->s->params_mu);
    *params = session->s->p;
    return 0;
}

static void begin_events(session_impl & s) {
    s.events.clear();
    s.strs.clear();
    s.json.clear();
}

int tyger_session_feed(tyger_session * session, const float * user, const float * agent, int n_samples, int mode,
                       const tyger_event ** events) {
    if (!session || n_samples < 0 || (n_samples > 0 && !user) || (mode != TYGER_MODE_LISTEN && mode != TYGER_MODE_SPEAKING)) {
        set_err("bad feed arguments");
        return -1;
    }
    session_impl & s = *session->s;
    begin_events(s);
    s.buf_u.insert(s.buf_u.end(), user, user + n_samples);
    if (agent) {
        s.buf_a.insert(s.buf_a.end(), agent, agent + n_samples);
    } else {
        s.buf_a.insert(s.buf_a.end(), (size_t) n_samples, 0.0f);
    }
    const size_t hop = (size_t) session->model->m->hp.hop_samples();
    size_t off = 0;
    while (s.buf_u.size() - off >= hop) {
        std::vector<hop_item> items = {{session->s, s.buf_u.data() + off, s.buf_a.data() + off, mode}};
        std::string err;
        if (!session->model->eng->tick(items, err)) {
            set_err(err);
            return -1;
        }
        off += hop;
    }
    s.buf_u.erase(s.buf_u.begin(), s.buf_u.begin() + off);
    s.buf_a.erase(s.buf_a.begin(), s.buf_a.begin() + off);
    if (events) *events = s.events.empty() ? nullptr : s.events.data();
    return (int) s.events.size();
}

int tyger_feed_batch(tyger_model * model, int n, tyger_session ** sessions, const float * const * user,
                     const float * const * agent, const int * modes) {
    if (!model || n < 0 || (n > 0 && (!sessions || !user || !modes))) {
        set_err("bad batch arguments");
        return -1;
    }
    const int hop = model->m->hp.hop_samples();
    std::vector<std::vector<float>> zeros;
    std::vector<hop_item> items;
    for (int i = 0; i < n; i++) {
        if (!sessions[i] || sessions[i]->model != model || !user[i]) {
            set_err("bad session in batch");
            return -1;
        }
        begin_events(*sessions[i]->s);
        const float * a = agent ? agent[i] : nullptr;
        if (!a) {
            zeros.emplace_back(hop, 0.0f);
            a = zeros.back().data();
        }
        items.push_back({sessions[i]->s, user[i], a, modes[i]});
    }
    std::string err;
    if (!model->eng->tick(items, err)) {
        set_err(err);
        return -1;
    }
    return 0;
}

int tyger_session_events(const tyger_session * session, const tyger_event ** events) {
    if (!session) return -1;
    if (events) *events = session->s->events.empty() ? nullptr : session->s->events.data();
    return (int) session->s->events.size();
}

static void json_str(std::string & o, const char * s) {
    o += '"';
    for (const char * p = s ? s : ""; *p; p++) {
        unsigned char c = (unsigned char) *p;
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) {
                    char b[8];
                    snprintf(b, sizeof(b), "\\u%04x", c);
                    o += b;
                } else {
                    o += (char) c;
                }
        }
    }
    o += '"';
}

static void json_num(std::string & o, const char * k, double v, int prec = 3) {
    char b[64];
    snprintf(b, sizeof(b), ",\"%s\":%.*f", k, prec, v);
    o += b;
}

const char * tyger_session_events_json(tyger_session * session) {
    if (!session) return "[]";
    session_impl & s = *session->s;
    static const char * kinds[] = {"FRAME", "ENDPOINT", "BARGE", "BACKCHANNEL", "PARTIAL"};
    std::string o = "[";
    for (size_t i = 0; i < s.events.size(); i++) {
        const tyger_event & e = s.events[i];
        if (i) o += ",";
        o += "{\"kind\":\"";
        o += kinds[e.kind];
        o += "\"";
        json_num(o, "t", e.t_s, 2);
        switch (e.kind) {
            case TYGER_EVENT_FRAME:
                o += ",\"state\":";
                json_str(o, e.state);
                json_num(o, "yield_prob", e.yield_prob);
                json_num(o, "hazard", e.hazard);
                json_num(o, "bc_prob", e.bc_prob);
                json_num(o, "p_end", e.p_end);
                break;
            case TYGER_EVENT_ENDPOINT:
                o += ",\"transcript\":";
                json_str(o, e.text);
                json_num(o, "turn_s", e.turn_s, 2);
                o += ",\"segments\":" + std::to_string(e.segments) + ",\"by\":";
                json_str(o, e.by);
                json_num(o, "p_end", e.p_end);
                break;
            case TYGER_EVENT_BARGE:
                json_num(o, "yield_prob", e.yield_prob);
                json_num(o, "onset_s", e.onset_s, 2);
                break;
            case TYGER_EVENT_BACKCHANNEL:
                o += ",\"partial\":";
                json_str(o, e.text);
                json_num(o, "bc_prob", e.bc_prob);
                json_num(o, "turn_s", e.turn_s, 2);
                break;
            case TYGER_EVENT_PARTIAL:
                o += ",\"text\":";
                json_str(o, e.text);
                break;
        }
        o += "}";
    }
    o += "]";
    s.json = std::move(o);
    return s.json.c_str();
}

static int quietest_cut(const float * x, int n, int sr, int fs, int lo_s, int hi_s) {
    const int lo = lo_s * sr / fs, hi = std::min(n, hi_s * sr) / fs;
    int best = hi, bi = hi;
    float bv = 1e30f;
    for (int f = lo; f < hi; f++) {
        double acc = 0.0;
        for (int i = 0; i < fs; i++) acc += (double) x[f * fs + i] * x[f * fs + i];
        float db = 20.0f * std::log10(std::sqrt((float) (acc / fs) + 1e-12f) + 1e-9f);
        if (db < bv) {
            bv = db;
            bi = f + 1;
        }
    }
    best = bi;
    return best * fs;
}

char * tyger_transcribe(tyger_model * model, const float * pcm, int n_samples, int prompt_id) {
    if (!model || !pcm || n_samples < 0) {
        set_err("bad transcribe arguments");
        return nullptr;
    }
    if (!model->m->has_asr) {
        set_err("model loaded without asr");
        return nullptr;
    }
    const int sr = model->m->hp.sample_rate, fs = model->m->hp.frame_samples();
    std::string all;
    int off = 0;
    // long clips go in blocks of up to 20 s, cut at the quietest frame of 15..19 s
    while (off < n_samples) {
        int len = n_samples - off;
        if (len > 20 * sr) {
            len = quietest_cut(pcm + off, len, sr, fs, 15, 19);
        }
        std::string t = model->eng->transcribe(pcm + off, len, prompt_id);
        if (!t.empty()) {
            if (!all.empty()) all += " ";
            all += t;
        }
        off += len;
    }
    char * out = (char *) malloc(all.size() + 1);
    memcpy(out, all.c_str(), all.size() + 1);
    return out;
}

void tyger_free(void * p) { free(p); }

int tyger_encode(tyger_model * model, const float * pcm, int n_samples, int n_layers, int lookahead, float * out,
                 int max_frames) {
    if (!model || !pcm || n_samples <= 0) {
        set_err("bad encode arguments");
        return -1;
    }
    std::vector<float> res;
    int T = 0;
    if (!model->eng->encode_debug(pcm, n_samples, n_layers, lookahead, res, T)) {
        set_err("encode failed");
        return -1;
    }
    const int C = model->m->hp.enc_hidden;
    if (out) {
        memcpy(out, res.data(), sizeof(float) * (size_t) C * std::min(T, max_frames));
    }
    return T;
}

} // extern "C"
