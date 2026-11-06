#include "tyger.h"
#include "ws.h"
#include "page.h"

#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

static void usage() {
    fprintf(stderr,
            "usage: tyger-server -m model.gguf [options]\n"
            "\n"
            "  -m, --model PATH     tyger gguf\n"
            "  --host ADDR          listen address (default 127.0.0.1)\n"
            "  --port N             listen port (default 8765)\n"
            "  --fast               default sessions to the incremental encoder\n"
            "  --cpu                cpu only\n"
            "  --device N           vulkan device index\n"
            "  -t, --threads N      cpu threads\n"
            "  --no-asr             skip transcripts\n"
            "  --max-batch N        sessions per batched hop (default 8)\n"
            "  -v, --verbose\n"
            "\n"
            "websocket on any path: send 16 kHz pcm as binary messages, get json events back.\n"
            "query or a config message sets format (s16, f32), channels (1 user, 2 user+agent interleaved),\n"
            "encoder (exact, fast), frames (0, 1) and any session threshold.\n");
}

using kv = std::map<std::string, std::string>;

static std::string err_json(const std::string & m) {
    std::string o = "{\"type\":\"error\",\"message\":\"";
    for (char c : m) {
        if (c == '"' || c == '\\') o += '\\';
        if ((unsigned char) c >= 0x20) o += c;
    }
    return o + "\"}";
}

// flat json object parser, enough for control messages
static bool parse_flat_json(const std::string & s, kv & out) {
    size_t i = 0;
    auto ws = [&]() { while (i < s.size() && isspace((unsigned char) s[i])) i++; };
    auto str = [&](std::string & v) -> bool {
        if (i >= s.size() || s[i] != '"') return false;
        i++;
        v.clear();
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) {
                char c = s[++i];
                v += c == 'n' ? '\n' : c == 't' ? '\t' : c;
            } else {
                v += s[i];
            }
            i++;
        }
        if (i >= s.size()) return false;
        i++;
        return true;
    };
    ws();
    if (i >= s.size() || s[i] != '{') return false;
    i++;
    for (;;) {
        ws();
        if (i < s.size() && s[i] == '}') return true;
        std::string k, v;
        if (!str(k)) return false;
        ws();
        if (i >= s.size() || s[i] != ':') return false;
        i++;
        ws();
        if (i < s.size() && s[i] == '"') {
            if (!str(v)) return false;
        } else {
            size_t b = i;
            while (i < s.size() && s[i] != ',' && s[i] != '}' && !isspace((unsigned char) s[i])) i++;
            v = s.substr(b, i - b);
        }
        out[k] = v;
        ws();
        if (i < s.size() && s[i] == ',') { i++; continue; }
        if (i < s.size() && s[i] == '}') return true;
        return false;
    }
}

static kv parse_query(const std::string & path) {
    kv out;
    size_t q = path.find('?');
    if (q == std::string::npos) return out;
    std::string s = path.substr(q + 1);
    size_t p = 0;
    while (p <= s.size()) {
        size_t a = s.find('&', p);
        std::string part = s.substr(p, a == std::string::npos ? std::string::npos : a - p);
        size_t e = part.find('=');
        if (!part.empty()) out[part.substr(0, e)] = e == std::string::npos ? "1" : part.substr(e + 1);
        if (a == std::string::npos) break;
        p = a + 1;
    }
    return out;
}

static bool truthy(const std::string & v) { return v == "1" || v == "true" || v == "yes" || v == "on"; }

// applies known keys onto params, returns false with a message on a bad value
static bool apply_params(const kv & m, tyger_session_params & p, std::string & err) {
    for (const auto & it : m) {
        const std::string & k = it.first;
        const char * v = it.second.c_str();
        if (k == "ep_thr") p.ep_thr = (float) atof(v);
        else if (k == "ep_deb") p.ep_deb = atoi(v);
        else if (k == "barge_thr") p.barge_thr = (float) atof(v);
        else if (k == "barge_deb") p.barge_deb = atoi(v);
        else if (k == "bc_thr") p.bc_thr = (float) atof(v);
        else if (k == "bc_deb") p.bc_deb = atoi(v);
        else if (k == "bc_refractory") p.bc_refractory = atoi(v);
        else if (k == "theta") p.theta = (float) atof(v);
        else if (k == "partial_refresh_s") p.partial_refresh_s = (float) atof(v);
        else if (k == "fallback_sil_frames") p.fallback_sil_frames = atoi(v);
        else if (k == "prompt_id") p.prompt_id = atoi(v);
        else if (k == "frames" || k == "emit_frames") p.emit_frames = truthy(it.second);
        else if (k == "encoder") {
            if (it.second == "fast") p.encoder_mode = TYGER_ENCODER_FAST;
            else if (it.second == "exact") p.encoder_mode = TYGER_ENCODER_EXACT;
            else { err = "encoder must be exact or fast"; return false; }
        }
    }
    return true;
}

struct client {
    std::unique_ptr<ws_conn> ws;
    tyger_session * sess = nullptr;
    std::string peer;
    int id = 0;
    bool f32 = false;
    int channels = 1;

    std::mutex mu; // guards everything below
    std::vector<float> u, a;
    int mode = TYGER_MODE_LISTEN;
    bool reset = false;
    bool reset_keep = false;
    bool lag_warned = false;

    ~client() { tyger_session_close(sess); }
};

struct server {
    tyger_model * model = nullptr;
    int hop = 5120;
    int sr = 16000;
    int max_batch = 8;
    bool fast = false;
    bool verbose = false;

    std::mutex mu;
    std::condition_variable cv;
    std::set<std::shared_ptr<client>> clients;
    std::atomic<int> next_id{1};

    void notify() {
        std::lock_guard<std::mutex> lk(mu);
        cv.notify_one();
    }

    bool has_work_locked() {
        for (const auto & c : clients) {
            std::lock_guard<std::mutex> lk(c->mu);
            if ((int) c->u.size() >= hop) return true;
        }
        return false;
    }

    void worker() {
        std::vector<std::shared_ptr<client>> batch;
        std::vector<std::vector<float>> bu, ba;
        std::vector<tyger_session *> ss;
        std::vector<const float *> pu, pa;
        std::vector<int> modes;
        size_t rr = 0;
        for (;;) {
            batch.clear();
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&]() { return has_work_locked(); });
                std::vector<std::shared_ptr<client>> all(clients.begin(), clients.end());
                // rotate the start so one busy client cannot hog the batch
                for (size_t k = 0; k < all.size() && (int) batch.size() < max_batch; k++) {
                    auto & c = all[(rr + k) % all.size()];
                    std::lock_guard<std::mutex> cl(c->mu);
                    if ((int) c->u.size() >= hop) batch.push_back(c);
                }
                rr++;
            }
            const size_t n = batch.size();
            bu.resize(n);
            ba.resize(n);
            ss.resize(n);
            pu.resize(n);
            pa.resize(n);
            modes.resize(n);
            for (size_t i = 0; i < n; i++) {
                client & c = *batch[i];
                std::lock_guard<std::mutex> cl(c.mu);
                bu[i].assign(c.u.begin(), c.u.begin() + hop);
                ba[i].assign(c.a.begin(), c.a.begin() + hop);
                c.u.erase(c.u.begin(), c.u.begin() + hop);
                c.a.erase(c.a.begin(), c.a.begin() + hop);
                if (c.reset) {
                    if (c.reset_keep) tyger_session_reset_keep(c.sess);
                    else tyger_session_reset(c.sess);
                    c.reset = c.reset_keep = false;
                }
                const double backlog = (double) c.u.size() / sr;
                if (backlog > 2.0 && !c.lag_warned) {
                    fprintf(stderr, "[%d] falling behind real time, %.1f s queued\n", c.id, backlog);
                    c.lag_warned = true;
                } else if (backlog < 0.5) {
                    c.lag_warned = false;
                }
                ss[i] = c.sess;
                pu[i] = bu[i].data();
                pa[i] = ba[i].data();
                modes[i] = c.mode;
            }
            const auto t0 = std::chrono::steady_clock::now();
            if (tyger_feed_batch(model, (int) n, ss.data(), pu.data(), pa.data(), modes.data()) != 0) {
                std::string msg = err_json(tyger_last_error());
                for (auto & c : batch) c->ws->send_text(msg);
                continue;
            }
            if (verbose) {
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                fprintf(stderr, "hop batch=%zu %.1f ms\n", n, ms);
            }
            for (auto & c : batch) {
                if (tyger_session_events(c->sess, nullptr) <= 0) continue;
                std::string msg = "{\"type\":\"events\",\"events\":";
                msg += tyger_session_events_json(c->sess);
                msg += "}";
                c->ws->send_text(msg);
            }
        }
    }

    void push_audio(client & c, const std::string & data) {
        const size_t bps = c.f32 ? 4 : 2;
        const size_t n = data.size() / (bps * c.channels);
        std::lock_guard<std::mutex> lk(c.mu);
        const size_t base = c.u.size();
        c.u.resize(base + n);
        c.a.resize(base + n, 0.0f);
        const uint8_t * p = (const uint8_t *) data.data();
        for (size_t i = 0; i < n; i++) {
            for (int ch = 0; ch < c.channels && ch < 2; ch++) {
                float v;
                const uint8_t * q = p + (i * c.channels + ch) * bps;
                if (c.f32) {
                    memcpy(&v, q, 4);
                } else {
                    int16_t s;
                    memcpy(&s, q, 2);
                    v = s / 32768.0f;
                }
                (ch == 0 ? c.u : c.a)[base + i] = v;
            }
        }
    }

    bool control(client & c, const std::string & text) {
        kv m;
        if (!parse_flat_json(text, m)) {
            c.ws->send_text("{\"type\":\"error\",\"message\":\"bad json\"}");
            return true;
        }
        const std::string type = m.count("type") ? m["type"] : "config";
        if (type == "mode") {
            const std::string v = m["mode"];
            std::lock_guard<std::mutex> lk(c.mu);
            c.mode = (v == "speaking" || v == "1") ? TYGER_MODE_SPEAKING : TYGER_MODE_LISTEN;
        } else if (type == "reset") {
            std::lock_guard<std::mutex> lk(c.mu);
            c.reset = true;
            c.reset_keep = m.count("keep") && (m["keep"] == "true" || m["keep"] == "1");
        } else if (type == "config") {
            tyger_session_params p;
            tyger_session_get_params(c.sess, &p);
            std::string err;
            if (!apply_params(m, p, err) || tyger_session_set_params(c.sess, &p) != 0) {
                if (err.empty()) err = tyger_last_error();
                c.ws->send_text(err_json(err));
            }
        } else if (type == "close") {
            return false;
        } else {
            c.ws->send_text("{\"type\":\"error\",\"message\":\"unknown type\"}");
        }
        return true;
    }

    void handle(sock_t s, std::string peer) {
        auto c = std::make_shared<client>();
        c->ws.reset(new ws_conn(s));
        c->peer = peer;
        http_request req;
        if (!c->ws->read_request(req)) return;
        const std::string path = req.path.substr(0, req.path.find('?'));
        if (!req.headers.count("upgrade")) {
            if (req.method == "GET" && path == "/") {
                c->ws->send_http(200, "OK", "text/html; charset=utf-8", k_page_html);
            } else if (req.method == "GET" && path == "/info") {
                c->ws->send_http(200, "OK", "application/json", tyger_model_info(model));
            } else {
                c->ws->send_http(404, "Not Found", "text/plain", "not found\n");
            }
            return;
        }
        kv q = parse_query(req.path);
        c->f32 = q.count("format") && q["format"] == "f32";
        c->channels = q.count("channels") && q["channels"] == "2" ? 2 : 1;
        tyger_session_params p = tyger_session_default_params(model);
        if (fast) p.encoder_mode = TYGER_ENCODER_FAST;
        std::string err;
        if (!apply_params(q, p, err)) {
            c->ws->send_http(400, "Bad Request", "text/plain", err + "\n");
            return;
        }
        if (!c->ws->accept_upgrade(req)) return;
        c->sess = tyger_session_open(model, &p);
        if (!c->sess) {
            c->ws->send_text(err_json(tyger_last_error()));
            c->ws->close(1011);
            return;
        }
        c->id = next_id++;
        {
            std::lock_guard<std::mutex> lk(mu);
            clients.insert(c);
        }
        fprintf(stderr, "[%d] connected %s %s ch=%d %s\n", c->id, peer.c_str(), c->f32 ? "f32" : "s16", c->channels,
                p.encoder_mode == TYGER_ENCODER_FAST ? "fast" : "exact");
        char hello[256];
        snprintf(hello, sizeof(hello), "{\"type\":\"ready\",\"session\":%d,\"sample_rate\":%d,\"hop_samples\":%d}", c->id,
                 sr, hop);
        c->ws->send_text(hello);

        std::string msg;
        for (;;) {
            ws_conn::msg_type t = c->ws->recv(msg);
            if (t == ws_conn::CLOSED) break;
            if (t == ws_conn::BINARY) {
                if (msg.size() % ((c->f32 ? 4 : 2) * c->channels) != 0) {
                    c->ws->send_text("{\"type\":\"error\",\"message\":\"audio message is not a whole number of frames\"}");
                    continue;
                }
                push_audio(*c, msg);
                notify();
            } else if (t == ws_conn::TEXT) {
                if (!control(*c, msg)) {
                    c->ws->close();
                    break;
                }
            }
        }
        {
            std::lock_guard<std::mutex> lk(mu);
            clients.erase(c);
        }
        fprintf(stderr, "[%d] disconnected\n", c->id);
    }
};

int main(int argc, char ** argv) {
    std::string model_path, host = "127.0.0.1";
    int port = 8765, device = -1, threads = 0, max_batch = 8;
    bool cpu = false, no_asr = false, fast = false, verbose = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", a.c_str());
                exit(1);
            }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") model_path = next();
        else if (a == "--host") host = next();
        else if (a == "--port") port = atoi(next());
        else if (a == "--fast") fast = true;
        else if (a == "--cpu") cpu = true;
        else if (a == "--device") device = atoi(next());
        else if (a == "-t" || a == "--threads") threads = atoi(next());
        else if (a == "--no-asr") no_asr = true;
        else if (a == "--max-batch") max_batch = atoi(next());
        else if (a == "-v" || a == "--verbose") verbose = true;
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); usage(); return 1; }
    }
    if (model_path.empty()) {
        usage();
        return 1;
    }
    if (!net_init()) {
        fprintf(stderr, "network init failed\n");
        return 1;
    }

    tyger_model_params mp = tyger_model_default_params();
    mp.backend = cpu ? TYGER_BACKEND_CPU : TYGER_BACKEND_AUTO;
    mp.device = device;
    mp.n_threads = threads;
    mp.enable_asr = !no_asr;
    mp.verbose = verbose;
    tyger_model * m = tyger_model_load(model_path.c_str(), mp);
    if (!m) {
        fprintf(stderr, "load failed: %s\n", tyger_last_error());
        return 1;
    }
    fprintf(stderr, "loaded %s: %s\n", model_path.c_str(), tyger_model_info(m));

    // vulkan compiles pipelines on first use, pay that here instead of mid stream
    {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<float> noise(tyger_sample_rate(m) * 2);
        uint32_t r = 1;
        for (float & v : noise) v = ((r = r * 1664525u + 1013904223u) >> 8) / 16777216.0f * 0.02f - 0.01f;
        for (int mode : {TYGER_ENCODER_EXACT, TYGER_ENCODER_FAST}) {
            tyger_session_params p = tyger_session_default_params(m);
            p.encoder_mode = mode;
            tyger_session * s = tyger_session_open(m, &p);
            if (s) tyger_session_feed(s, noise.data(), nullptr, (int) noise.size(), TYGER_MODE_LISTEN, nullptr);
            tyger_session_close(s);
        }
        if (!no_asr) tyger_free(tyger_transcribe(m, noise.data(), (int) noise.size(), -1));
        fprintf(stderr, "warmed up in %.0f ms\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }

    server srv;
    srv.model = m;
    srv.hop = tyger_hop_samples(m);
    srv.sr = tyger_sample_rate(m);
    srv.max_batch = max_batch < 1 ? 1 : max_batch;
    srv.fast = fast;
    srv.verbose = verbose;

    std::string err;
    sock_t ls = net_listen(host, port, err);
    if (ls == (sock_t) -1) {
        fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    fprintf(stderr, "listening on http://%s:%d (websocket on the same port)\n", host.c_str(), port);

    std::thread(&server::worker, &srv).detach();
    for (;;) {
        std::string peer;
        sock_t c = net_accept(ls, peer);
        if (c == (sock_t) -1) continue;
        std::thread(&server::handle, &srv, c, peer).detach();
    }
}
