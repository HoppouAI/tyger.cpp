#include "tyger.h"
#include "wav.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static void usage() {
    fprintf(stderr,
            "usage: tyger-cli -m model.gguf --user user.wav [options]\n"
            "\n"
            "  -m, --model PATH        tyger gguf\n"
            "  --user PATH             the user's mic (16 kHz mono, other rates are resampled)\n"
            "  --agent PATH            the agent's playback, silence if omitted\n"
            "  --channel N             wav channel for --user, -1 mixes down (default -1)\n"
            "  --agent-channel N       wav channel for --agent (default -1)\n"
            "  --speaking A-B,C-D      seconds where the agent is talking (speaking mode)\n"
            "  --fast                  cache-aware incremental encoder instead of the exact 8 s window\n"
            "  --cpu                   cpu only\n"
            "  --device N              vulkan device index\n"
            "  -t, --threads N         cpu threads\n"
            "  --no-asr                skip transcripts\n"
            "  --frames                print every FRAME event\n"
            "  --json                  print each hop's events as a json line\n"
            "  --frames-csv PATH       write every frame's head outputs to a csv\n"
            "  --realtime              pace the feed at real time\n"
            "  --ep-thr F --barge-thr F --bc-thr F --fallback-sil N   thresholds\n"
            "  --transcribe            offline transcript of --user and exit\n"
            "  --encode PATH --layers N --lookahead N   dump offline encoder output (f32 [T][1024]) and exit\n"
            "  -v, --verbose\n");
}

struct span { double a, b; };

int main(int argc, char ** argv) {
    std::string model, user, agent, csv, encode_out;
    int channel = -1, agent_channel = -1, device = -1, threads = 0, layers = 15, lookahead = 0;
    bool fast = false, cpu = false, no_asr = false, frames = false, json = false, realtime = false;
    bool transcribe = false, verbose = false;
    float ep_thr = -1, barge_thr = -1, bc_thr = -1;
    int fallback = -1;
    std::vector<span> speaking;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", a.c_str());
                exit(1);
            }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") model = next();
        else if (a == "--user") user = next();
        else if (a == "--agent") agent = next();
        else if (a == "--channel") channel = atoi(next());
        else if (a == "--agent-channel") agent_channel = atoi(next());
        else if (a == "--speaking") {
            std::string s = next();
            size_t p = 0;
            while (p < s.size()) {
                size_t c = s.find(',', p);
                std::string part = s.substr(p, c == std::string::npos ? std::string::npos : c - p);
                size_t d = part.find('-');
                if (d != std::string::npos) speaking.push_back({atof(part.substr(0, d).c_str()), atof(part.substr(d + 1).c_str())});
                if (c == std::string::npos) break;
                p = c + 1;
            }
        }
        else if (a == "--fast") fast = true;
        else if (a == "--cpu") cpu = true;
        else if (a == "--device") device = atoi(next());
        else if (a == "-t" || a == "--threads") threads = atoi(next());
        else if (a == "--no-asr") no_asr = true;
        else if (a == "--frames") frames = true;
        else if (a == "--json") json = true;
        else if (a == "--frames-csv") csv = next();
        else if (a == "--realtime") realtime = true;
        else if (a == "--ep-thr") ep_thr = (float) atof(next());
        else if (a == "--barge-thr") barge_thr = (float) atof(next());
        else if (a == "--bc-thr") bc_thr = (float) atof(next());
        else if (a == "--fallback-sil") fallback = atoi(next());
        else if (a == "--transcribe") transcribe = true;
        else if (a == "--encode") encode_out = next();
        else if (a == "--layers") layers = atoi(next());
        else if (a == "--lookahead") lookahead = atoi(next());
        else if (a == "-v" || a == "--verbose") verbose = true;
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else {
            fprintf(stderr, "unknown argument %s\n", a.c_str());
            usage();
            return 1;
        }
    }
    if (model.empty() || user.empty()) {
        usage();
        return 1;
    }

    tyger_model_params mp = tyger_model_default_params();
    mp.backend = cpu ? TYGER_BACKEND_CPU : TYGER_BACKEND_AUTO;
    mp.device = device;
    mp.n_threads = threads;
    mp.enable_asr = encode_out.empty() ? !no_asr : layers > 15;
    mp.verbose = verbose;
    auto t_load = std::chrono::steady_clock::now();
    tyger_model * m = tyger_model_load(model.c_str(), mp);
    if (!m) {
        fprintf(stderr, "load failed: %s\n", tyger_last_error());
        return 1;
    }
    const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_load).count();
    fprintf(stderr, "loaded %s in %.0f ms: %s\n", model.c_str(), load_ms, tyger_model_info(m));
    const int sr = tyger_sample_rate(m), hop = tyger_hop_samples(m);

    std::string err;
    wav_data wu, wa;
    if (!wav_read(user, wu, err)) {
        fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    std::vector<float> u = wav_mono(wu, channel, sr), ag;
    if (!agent.empty()) {
        if (!wav_read(agent, wa, err)) {
            fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        ag = wav_mono(wa, agent_channel, sr);
    }

    if (!encode_out.empty()) {
        std::vector<float> out((size_t) (u.size() / 1280 + 8) * 1024);
        int T = tyger_encode(m, u.data(), (int) u.size(), layers, lookahead, out.data(), (int) (out.size() / 1024));
        if (T < 0) {
            fprintf(stderr, "encode failed: %s\n", tyger_last_error());
            return 1;
        }
        FILE * f = fopen(encode_out.c_str(), "wb");
        fwrite(out.data(), sizeof(float), (size_t) T * 1024, f);
        fclose(f);
        fprintf(stderr, "wrote %d frames to %s\n", T, encode_out.c_str());
        tyger_model_free(m);
        return 0;
    }
    if (transcribe) {
        auto t0 = std::chrono::steady_clock::now();
        char * text = tyger_transcribe(m, u.data(), (int) u.size(), -1);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (!text) {
            fprintf(stderr, "transcribe failed: %s\n", tyger_last_error());
            return 1;
        }
        printf("%s\n", text);
        fprintf(stderr, "%.2f s of audio in %.0f ms\n", u.size() / (double) sr, ms);
        tyger_free(text);
        tyger_model_free(m);
        return 0;
    }

    tyger_session_params sp = tyger_session_default_params(m);
    sp.encoder_mode = fast ? TYGER_ENCODER_FAST : TYGER_ENCODER_EXACT;
    if (ep_thr >= 0) sp.ep_thr = ep_thr;
    if (barge_thr >= 0) sp.barge_thr = barge_thr;
    if (bc_thr >= 0) sp.bc_thr = bc_thr;
    if (fallback >= 0) sp.fallback_sil_frames = fallback;
    tyger_session * s = tyger_session_open(m, &sp);
    if (!s) {
        fprintf(stderr, "session failed: %s\n", tyger_last_error());
        return 1;
    }
    FILE * fcsv = csv.empty() ? nullptr : fopen(csv.c_str(), "w");
    if (fcsv) fprintf(fcsv, "t,mode,state,p_end,yield,hazard,bc\n");

    const size_t n_hops = u.size() / hop;
    std::vector<float> zeros(hop, 0.0f), abuf(hop);
    std::vector<double> ms;
    auto wall0 = std::chrono::steady_clock::now();
    for (size_t h = 0; h < n_hops; h++) {
        const double t_end = (h + 1) * (double) hop / sr;
        int mode = TYGER_MODE_LISTEN;
        for (auto & sp2 : speaking) {
            if (t_end > sp2.a && t_end <= sp2.b) mode = TYGER_MODE_SPEAKING;
        }
        const float * ah = zeros.data();
        if (!ag.empty()) {
            for (int i = 0; i < hop; i++) {
                size_t k = h * hop + i;
                abuf[i] = k < ag.size() ? ag[k] : 0.0f;
            }
            ah = abuf.data();
        }
        auto t0 = std::chrono::steady_clock::now();
        const tyger_event * ev = nullptr;
        int ne = tyger_session_feed(s, u.data() + h * hop, ah, hop, mode, &ev);
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        if (ne < 0) {
            fprintf(stderr, "feed failed: %s\n", tyger_last_error());
            return 1;
        }
        if (json) {
            printf("%s\n", tyger_session_events_json(s));
        }
        for (int k = 0; k < ne; k++) {
            const tyger_event & e = ev[k];
            if (fcsv && e.kind == TYGER_EVENT_FRAME) {
                fprintf(fcsv, "%.2f,%s,%s,%.6f,%.6f,%.6f,%.6f\n", e.t_s, mode ? "speaking" : "listen", e.state, e.p_end,
                        e.yield_prob, e.hazard, e.bc_prob);
            }
            if (json) continue;
            switch (e.kind) {
                case TYGER_EVENT_FRAME:
                    if (frames) printf("%7.2f  FRAME       %-11s p_end=%.3f yield=%.3f bc=%.3f hazard=%.3f\n", e.t_s, e.state,
                                       e.p_end, e.yield_prob, e.bc_prob, e.hazard);
                    break;
                case TYGER_EVENT_ENDPOINT:
                    printf("%7.2f  ENDPOINT    p_end=%.3f turn=%.2fs by=%s  \"%s\"\n", e.t_s, e.p_end, e.turn_s, e.by,
                           e.text ? e.text : "");
                    break;
                case TYGER_EVENT_BARGE:
                    printf("%7.2f  BARGE       yield=%.3f onset=%.2fs\n", e.t_s, e.yield_prob, e.onset_s);
                    break;
                case TYGER_EVENT_BACKCHANNEL:
                    printf("%7.2f  BACKCHANNEL bc=%.3f turn=%.2fs  \"%s\"\n", e.t_s, e.bc_prob, e.turn_s, e.text ? e.text : "");
                    break;
                case TYGER_EVENT_PARTIAL:
                    printf("%7.2f  PARTIAL     \"%s\"\n", e.t_s, e.text ? e.text : "");
                    break;
            }
        }
        fflush(stdout);
        if (realtime) {
            auto target = wall0 + std::chrono::microseconds((long long) (t_end * 1e6));
            std::this_thread::sleep_until(target);
        }
    }
    if (fcsv) fclose(fcsv);
    if (!ms.empty()) {
        std::vector<double> sorted = ms;
        std::sort(sorted.begin(), sorted.end());
        double sum = 0;
        for (double v : ms) sum += v;
        const double audio_s = n_hops * (double) hop / sr;
        fprintf(stderr, "%zu hops (%.1f s): hop p50 %.1f ms, p90 %.1f ms, max %.1f ms, rtf %.3f (budget 320 ms/hop)\n",
                n_hops, audio_s, sorted[sorted.size() / 2], sorted[sorted.size() * 9 / 10], sorted.back(),
                sum / 1000.0 / audio_s);
    }
    tyger_session_close(s);
    tyger_model_free(m);
    return 0;
}
