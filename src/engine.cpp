#include "engine.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>

static const rule_consts kRules;

engine::engine(tyger_model_impl & m_) : m(m_), mel(m_), tick_runner(m_), tt(m_) {
    if (m.has_asr) {
        asr_runner = std::make_unique<enc_runner>(m);
        dec = std::make_unique<rnnt_decoder>(m);
        asr_thread = std::thread([this] { asr_loop(); });
    }
}

engine::~engine() {
    {
        std::lock_guard<std::mutex> lk(q_mu);
        stop = true;
    }
    q_cv.notify_all();
    if (asr_thread.joinable()) {
        asr_thread.join();
    }
}

std::string engine::transcribe(const float * pcm, int n, int prompt_id) {
    if (!m.has_asr || n <= 0) {
        return "";
    }
    std::lock_guard<std::mutex> lk(asr_dec_mu);
    std::vector<float> feats;
    int nf = 0, nv = 0;
    mel.offline(pcm, n, feats, nf, nv);
    if (nv <= 0) {
        return "";
    }
    std::vector<float> enc;
    int T = 0, C = 0;
    std::vector<int> valid;
    if (!encode_offline(*asr_runner, {feats.data()}, nf, {nv}, m.hp.enc_layers, m.hp.default_lookahead, true,
                        prompt_id, enc, T, C, valid)) {
        tyger_log("asr encode failed");
        return "";
    }
    std::vector<int> ids = dec->decode(enc.data(), std::min(T, valid[0]));
    return dec->detokenize(ids);
}

bool engine::encode_debug(const float * pcm, int n, int n_layers, int lookahead, std::vector<float> & out, int & T) {
    std::vector<float> feats;
    int nf = 0, nv = 0;
    mel.offline(pcm, n, feats, nf, nv);
    int C = 0;
    std::vector<int> valid;
    std::lock_guard<std::mutex> lk(tick_mu);
    return encode_offline(tick_runner, {feats.data()}, nf, {nv}, n_layers, lookahead, false, -1, out, T, C, valid);
}

void engine::asr_loop() {
    for (;;) {
        std::vector<asr_job> jobs;
        {
            std::unique_lock<std::mutex> lk(q_mu);
            q_cv.wait(lk, [this] { return stop || !q.empty(); });
            if (stop) {
                for (auto & j : q) {
                    if (j.done && j.kind != asr_job::PARTIAL) j.done("");
                }
                q.clear();
                return;
            }
            while (!q.empty()) {
                jobs.push_back(std::move(q.front()));
                q.pop_front();
            }
        }
        // a turn's blocks and its final come before the rolling partials
        std::stable_partition(jobs.begin(), jobs.end(), [](const asr_job & j) { return j.kind != asr_job::PARTIAL; });
        for (auto & j : jobs) {
            std::string text = transcribe(j.wav.data(), (int) j.wav.size(), j.prompt_id);
            if (j.done) j.done(text);
        }
    }
}

void engine::cut_segment(session_impl & s) {
    const int sr = m.hp.sample_rate, hop = m.hp.frame_samples();
    const int lo = (int) ((kRules.seg_s - kRules.seg_search_s) * sr) / hop;
    const size_t end = (size_t) (kRules.seg_s * sr);
    std::vector<float> e;
    if ((size_t) lo * hop < s.turn_u.size()) {
        const size_t hi = std::min(end, s.turn_u.size());
        frame_energy(s.turn_u.data() + (size_t) lo * hop, hi - (size_t) lo * hop, hop, e);
    }
    size_t cut = end;
    if (!e.empty()) {
        cut = (size_t) (lo + (int) (std::min_element(e.begin(), e.end()) - e.begin()) + 1) * hop;
    }
    cut = std::min(cut, s.turn_u.size());
    asr_job j;
    j.kind = asr_job::SEGMENT;
    j.wav.assign(s.turn_u.begin(), s.turn_u.begin() + cut);
    s.turn_u.erase(s.turn_u.begin(), s.turn_u.begin() + cut);
    std::shared_ptr<seg_list> segs = s.segs;
    size_t idx;
    {
        std::lock_guard<std::mutex> lk(segs->mu);
        idx = segs->texts.size();
        segs->texts.emplace_back();
        segs->done.push_back(false);
    }
    s.seg_s += (double) cut / sr;
    {
        std::lock_guard<std::mutex> lk(s.params_mu);
        j.prompt_id = s.p.prompt_id;
    }
    j.done = [segs, idx](const std::string & t) {
        std::lock_guard<std::mutex> lk(segs->mu);
        segs->texts[idx] = t;
        segs->done[idx] = true;
    };
    std::lock_guard<std::mutex> lk(q_mu);
    q.push_back(std::move(j));
    q_cv.notify_one();
}

void engine::maybe_partial(const std::shared_ptr<session_impl> & sp) {
    session_impl & s = *sp;
    float refresh;
    int prompt;
    {
        std::lock_guard<std::mutex> lk(s.params_mu);
        refresh = s.p.partial_refresh_s;
        prompt = s.p.prompt_id;
    }
    if (!m.has_asr || refresh <= 0.0f) {
        return;
    }
    const double cur = s.turn_len_s();
    asr_job j;
    {
        std::lock_guard<std::mutex> lk(s.asr_mu);
        if (!s.in_turn || s.partial_inflight || cur - s.partial_at_s < refresh) {
            return;
        }
        if (s.turn_u.size() < (size_t) (0.3 * m.hp.sample_rate)) {
            return;
        }
        s.partial_at_s = cur;
        s.partial_inflight = true;
        j.gen = s.turn_gen;
    }
    j.kind = asr_job::PARTIAL;
    j.sess = sp;
    j.wav = s.turn_u;
    j.prompt_id = prompt;
    std::shared_ptr<seg_list> segs = s.segs;
    size_t nseg;
    {
        std::lock_guard<std::mutex> lk(segs->mu);
        nseg = segs->texts.size();
    }
    std::weak_ptr<session_impl> wp = sp;
    const int gen = j.gen;
    j.sess.reset();
    j.done = [wp, segs, nseg, gen](const std::string & t) {
        auto s2 = wp.lock();
        if (!s2) return;
        std::string words;
        {
            std::lock_guard<std::mutex> lk(segs->mu);
            for (size_t i = 0; i < nseg && i < segs->texts.size(); i++) {
                if (segs->done[i] && !segs->texts[i].empty()) {
                    if (!words.empty()) words += " ";
                    words += segs->texts[i];
                }
            }
        }
        if (!t.empty()) {
            if (!words.empty()) words += " ";
            words += t;
        }
        std::lock_guard<std::mutex> lk(s2->asr_mu);
        if (s2->turn_gen == gen) {
            s2->partial_asr = t;
            s2->words = words;
        }
        s2->partial_inflight = false;
    };
    std::lock_guard<std::mutex> lk(q_mu);
    q.push_back(std::move(j));
    q_cv.notify_one();
}

bool engine::encode_exact(std::vector<hop_item> & items, std::vector<std::vector<float>> & hu,
                          std::vector<std::vector<float>> & ha, std::vector<int> & n_frames) {
    const int C = m.hp.enc_hidden, HF = m.hp.hop_frames();
    const size_t n = items.size();
    hu.assign(n, {});
    ha.assign(n, {});
    n_frames.assign(2 * n, 0);
    struct win_ref { size_t item; bool agent; const std::vector<float> * w; mel_window_cache * mc; int64_t start; };
    std::map<size_t, std::vector<win_ref>> groups;
    for (size_t i = 0; i < n; i++) {
        session_impl & s = *items[i].s;
        for (int a = 0; a < 2; a++) {
            const std::vector<float> & w = a ? s.win_a : s.win_u;
            const bool zero = std::all_of(w.begin(), w.end(), [](float v) { return v == 0.0f; });
            auto zc = zero_cache.find((int) w.size());
            if (zero && zc != zero_cache.end()) {
                (a ? ha : hu)[i] = zc->second;
                n_frames[2 * i + a] = (int) (zc->second.size() / C);
                continue;
            }
            groups[w.size()].push_back({i, a != 0, &w, a ? &s.mel_a : &s.mel_u, s.win_end - (int64_t) w.size()});
        }
    }
    for (auto & [len, refs] : groups) {
        auto tm0 = std::chrono::steady_clock::now();
        std::vector<std::vector<float>> feats(refs.size());
        std::vector<const float *> ptrs;
        std::vector<int> valid;
        int nf = 0;
        for (size_t k = 0; k < refs.size(); k++) {
            int nv = 0;
            mel.window(refs[k].w->data(), (int) len, refs[k].start, *refs[k].mc, feats[k], nf, nv);
            ptrs.push_back(feats[k].data());
            valid.push_back(nv);
        }
        if (getenv("TYGER_PROFILE")) {
            fprintf(stderr, "[prof mel] %zu windows %.1f ms\n", refs.size(),
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tm0).count());
        }
        std::vector<float> out;
        int T = 0, Co = 0;
        std::vector<int> ov;
        if (!encode_offline(tick_runner, ptrs, nf, valid, m.hp.feature_layer, 0, false, -1, out, T, Co, ov)) {
            return false;
        }
        const int keep = std::min(T, HF);
        for (size_t k = 0; k < refs.size(); k++) {
            const float * base = out.data() + (size_t) k * T * C + (size_t) (T - keep) * C;
            std::vector<float> last(base, base + (size_t) keep * C);
            const win_ref & r = refs[k];
            if (std::all_of(r.w->begin(), r.w->end(), [](float v) { return v == 0.0f; })) {
                zero_cache[(int) len] = last;
            }
            n_frames[2 * r.item + (r.agent ? 1 : 0)] = keep;
            (r.agent ? ha : hu)[r.item] = std::move(last);
        }
    }
    return true;
}

bool engine::tick(std::vector<hop_item> & items, std::string & err) {
    std::lock_guard<std::mutex> lk(tick_mu);
    const tyger_hparams & hp = m.hp;
    const int hop = hp.hop_samples(), HF = hp.hop_frames(), C = hp.enc_hidden, sr = hp.sample_rate;
    const size_t n = items.size();
    std::vector<double> t0(n);
    std::vector<tyger_session_params> ps(n);

    for (size_t i = 0; i < n; i++) {
        session_impl & s = *items[i].s;
        {
            std::lock_guard<std::mutex> pl(s.params_mu);
            ps[i] = s.p;
        }
        t0[i] = s.hops * (double) hop / sr;
        if (rules_begin_hop(s, items[i].uh, items[i].ah, hop, items[i].mode) || s.gru_h.empty()) {
            tt.reset(s.gru_h);
            s.es_u.reset();
            s.es_a.reset();
        }
        if (items[i].mode == TYGER_MODE_LISTEN) {
            if (m.has_asr && s.turn_u.size() > (size_t) (kRules.seg_s * sr)) {
                cut_segment(s);
            }
            const size_t cap = (size_t) (kRules.turn_max_s * sr);
            if (s.turn_u.size() > cap) {
                s.turn_u.erase(s.turn_u.begin(), s.turn_u.end() - cap);
            }
        }
        rules_words_event(s, items[i].mode, t0[i]);
    }

    std::vector<std::vector<float>> hu, ha;
    std::vector<int> nfr(2 * n, 0);
    std::vector<hop_item> exact;
    std::vector<size_t> exact_idx;
    std::vector<enc_stream *> streams;
    std::vector<size_t> stream_idx;
    hu.assign(n, {});
    ha.assign(n, {});
    for (size_t i = 0; i < n; i++) {
        session_impl & s = *items[i].s;
        if (ps[i].encoder_mode == TYGER_ENCODER_FAST) {
            s.es_u.mel.push(mel, items[i].uh, hop);
            s.es_a.mel.push(mel, items[i].ah, hop);
            if (enc_stream_ready(m, s.es_u, HF) && enc_stream_ready(m, s.es_a, HF)) {
                streams.push_back(&s.es_u);
                streams.push_back(&s.es_a);
                stream_idx.push_back(i);
            }
        } else {
            exact.push_back(items[i]);
            exact_idx.push_back(i);
        }
    }
    if (!exact.empty()) {
        std::vector<std::vector<float>> eu, ea;
        std::vector<int> en;
        if (!encode_exact(exact, eu, ea, en)) {
            err = "encoder failed";
            return false;
        }
        for (size_t k = 0; k < exact.size(); k++) {
            hu[exact_idx[k]] = std::move(eu[k]);
            ha[exact_idx[k]] = std::move(ea[k]);
            nfr[2 * exact_idx[k]] = en[2 * k];
            nfr[2 * exact_idx[k] + 1] = en[2 * k + 1];
        }
    }
    if (!streams.empty()) {
        std::vector<std::vector<float>> out;
        if (!encode_stream(tick_runner, streams, HF, out)) {
            err = "streaming encoder failed";
            return false;
        }
        for (size_t k = 0; k < stream_idx.size(); k++) {
            const size_t i = stream_idx[k];
            hu[i] = std::move(out[2 * k]);
            ha[i] = std::move(out[2 * k + 1]);
            nfr[2 * i] = nfr[2 * i + 1] = HF;
        }
    }

    std::vector<std::vector<float>> act(n);
    std::vector<int> nlive(n);
    for (size_t i = 0; i < n; i++) {
        session_impl & s = *items[i].s;
        std::vector<float> e;
        const size_t fs = (size_t) hp.frame_samples();
        const size_t nwin = s.win_u.size() / fs;
        const size_t take = std::min<size_t>(nwin, HF);
        frame_energy(s.win_u.data() + (nwin - take) * fs, take * fs, (int) fs, act[i]);
        if (items[i].mode == TYGER_MODE_LISTEN) {
            maybe_partial(items[i].s);
        }
        if (rules_switch_mode(s, items[i].mode)) {
            tt.reset(s.gru_h);
        }
        nlive[i] = std::min({HF, nfr[2 * i], nfr[2 * i + 1], (int) act[i].size()});
    }

    auto tg0 = std::chrono::steady_clock::now();
    struct pending { size_t item; frame_result fr; };
    std::vector<pending> finals;
    for (int j = 0; j < HF; j++) {
        for (size_t i = 0; i < n; i++) {
            if (j < HF - nlive[i]) continue;
            session_impl & s = *items[i].s;
            const int back = HF - j; // 1..HF frames from the end
            const float * fu = hu[i].data() + (size_t) (nfr[2 * i] - back) * C;
            const float * fa = ha[i].data() + (size_t) (nfr[2 * i + 1] - back) * C;
            const float adb = act[i][act[i].size() - back];
            tt_out o = tt.step(s.gru_h, fu, fa, adb, s.prev_state);
            const double t = t0[i] + (j + 1) * hp.frame_s;
            frame_result fr = rules_on_frame(s, items[i].mode, o, adb, back - 1, t);
            if (fr.reset) {
                tt.reset(s.gru_h);
            }
            if (fr.final) {
                finals.push_back({i, std::move(fr)});
            }
        }
    }
    for (size_t i = 0; i < n; i++) {
        items[i].s->hops++;
    }
    if (getenv("TYGER_PROFILE")) {
        fprintf(stderr, "[prof gru] %zu sessions %.1f ms\n", n,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tg0).count());
    }

    // endpoint transcripts: the hop resolves once they are ready
    if (!finals.empty() && m.has_asr) {
        std::vector<std::future<std::string>> futs;
        for (auto & f : finals) {
            auto prom = std::make_shared<std::promise<std::string>>();
            futs.push_back(prom->get_future());
            asr_job jb;
            jb.kind = asr_job::FINAL;
            jb.wav = std::move(f.fr.final_wav);
            jb.prompt_id = ps[f.item].prompt_id;
            jb.done = [prom](const std::string & t) { prom->set_value(t); };
            std::lock_guard<std::mutex> ql(q_mu);
            q.push_back(std::move(jb));
            q_cv.notify_one();
        }
        for (size_t k = 0; k < finals.size(); k++) {
            std::string text = futs[k].get();
            session_impl & s = *items[finals[k].item].s;
            std::string full;
            {
                std::lock_guard<std::mutex> sl(finals[k].fr.final_segs->mu);
                for (auto & t : finals[k].fr.final_segs->texts) {
                    if (t.empty()) continue;
                    if (!full.empty()) full += " ";
                    full += t;
                }
            }
            if (!text.empty()) {
                if (!full.empty()) full += " ";
                full += text;
            }
            s.events[finals[k].fr.final_event].text = s.keep(full);
        }
    }
    return true;
}
