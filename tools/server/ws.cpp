#include "ws.h"

#include <algorithm>
#include <cctype>
#include <cstring>

#ifndef _WIN32
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#define INVALID_SOCKET (-1)
#define closesocket close
#endif

bool net_init() {
#ifdef _WIN32
    WSADATA w;
    return WSAStartup(MAKEWORD(2, 2), &w) == 0;
#else
    return true;
#endif
}

sock_t net_listen(const std::string & host, int port, std::string & err) {
    addrinfo hints{}, * res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    std::string ps = std::to_string(port);
    if (getaddrinfo(host.empty() ? nullptr : host.c_str(), ps.c_str(), &hints, &res) != 0 || !res) {
        err = "cannot resolve " + host;
        return INVALID_SOCKET;
    }
    sock_t s = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (s == INVALID_SOCKET) {
        freeaddrinfo(res);
        err = "socket() failed";
        return INVALID_SOCKET;
    }
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char *) &one, sizeof(one));
    if (bind(s, res->ai_addr, (int) res->ai_addrlen) != 0 || listen(s, 16) != 0) {
        freeaddrinfo(res);
        closesocket(s);
        err = "cannot listen on " + host + ":" + ps;
        return INVALID_SOCKET;
    }
    freeaddrinfo(res);
    return s;
}

sock_t net_accept(sock_t s, std::string & peer) {
    sockaddr_storage sa{};
    socklen_t len = sizeof(sa);
    sock_t c = accept(s, (sockaddr *) &sa, &len);
    if (c == INVALID_SOCKET) return c;
    int one = 1;
    setsockopt(c, IPPROTO_TCP, TCP_NODELAY, (const char *) &one, sizeof(one));
    char host[128] = {0};
    getnameinfo((sockaddr *) &sa, len, host, sizeof(host), nullptr, 0, NI_NUMERICHOST);
    peer = host;
    return c;
}

void net_close(sock_t s) {
    if (s != INVALID_SOCKET) closesocket(s);
}

namespace {

struct sha1 {
    uint32_t h[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    static uint32_t rol(uint32_t v, int b) { return (v << b) | (v >> (32 - b)); }
    void block(const uint8_t * p) {
        uint32_t w[80];
        for (int i = 0; i < 16; i++) w[i] = (p[4 * i] << 24) | (p[4 * i + 1] << 16) | (p[4 * i + 2] << 8) | p[4 * i + 3];
        for (int i = 16; i < 80; i++) w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        for (int i = 0; i < 80; i++) {
            uint32_t f, k;
            if (i < 20) { f = (b & c) | (~b & d); k = 0x5A827999; }
            else if (i < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
            else { f = b ^ c ^ d; k = 0xCA62C1D6; }
            uint32_t t = rol(a, 5) + f + e + k + w[i];
            e = d; d = c; c = rol(b, 30); b = a; a = t;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
    }
    std::string digest(const std::string & msg) {
        std::string m = msg;
        uint64_t bits = (uint64_t) msg.size() * 8;
        m.push_back((char) 0x80);
        while (m.size() % 64 != 56) m.push_back(0);
        for (int i = 7; i >= 0; i--) m.push_back((char) (bits >> (8 * i)));
        for (size_t i = 0; i < m.size(); i += 64) block((const uint8_t *) m.data() + i);
        std::string out;
        for (uint32_t v : h)
            for (int i = 3; i >= 0; i--) out.push_back((char) (v >> (8 * i)));
        return out;
    }
};

std::string base64(const std::string & in) {
    static const char * t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string o;
    size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        uint32_t v = ((uint8_t) in[i] << 16) | ((uint8_t) in[i + 1] << 8) | (uint8_t) in[i + 2];
        o += t[v >> 18]; o += t[(v >> 12) & 63]; o += t[(v >> 6) & 63]; o += t[v & 63];
    }
    if (i + 1 == in.size()) {
        uint32_t v = (uint8_t) in[i] << 16;
        o += t[v >> 18]; o += t[(v >> 12) & 63]; o += "==";
    } else if (i + 2 == in.size()) {
        uint32_t v = ((uint8_t) in[i] << 16) | ((uint8_t) in[i + 1] << 8);
        o += t[v >> 18]; o += t[(v >> 12) & 63]; o += t[(v >> 6) & 63]; o += '=';
    }
    return o;
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s;
}

std::string trim(const std::string & s) {
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

} // namespace

ws_conn::~ws_conn() { net_close(s_); }

bool ws_conn::read_exact(void * dst, size_t n) {
    uint8_t * p = (uint8_t *) dst;
    if (!pending_.empty()) {
        size_t k = std::min(n, pending_.size());
        memcpy(p, pending_.data(), k);
        pending_.erase(0, k);
        p += k;
        n -= k;
    }
    while (n > 0) {
        int r = ::recv(s_, (char *) p, (int) std::min<size_t>(n, 1 << 20), 0);
        if (r <= 0) return false;
        p += r;
        n -= (size_t) r;
    }
    return true;
}

bool ws_conn::send_all(const void * d, size_t n) {
    const char * p = (const char *) d;
    while (n > 0) {
        int r = ::send(s_, p, (int) std::min<size_t>(n, 1 << 20), 0);
        if (r <= 0) return false;
        p += r;
        n -= (size_t) r;
    }
    return true;
}

bool ws_conn::read_request(http_request & req) {
    std::string buf;
    char tmp[4096];
    while (buf.find("\r\n\r\n") == std::string::npos) {
        int r = ::recv(s_, tmp, sizeof(tmp), 0);
        if (r <= 0 || buf.size() > 65536) return false;
        buf.append(tmp, r);
    }
    size_t end = buf.find("\r\n\r\n");
    pending_ = buf.substr(end + 4);
    std::string head = buf.substr(0, end);
    size_t p = head.find("\r\n");
    std::string first = head.substr(0, p);
    size_t a = first.find(' '), b = first.rfind(' ');
    if (a == std::string::npos || b == a) return false;
    req.method = first.substr(0, a);
    req.path = first.substr(a + 1, b - a - 1);
    while (p != std::string::npos && p < head.size()) {
        size_t q = head.find("\r\n", p + 2);
        std::string line = head.substr(p + 2, q == std::string::npos ? std::string::npos : q - p - 2);
        size_t c = line.find(':');
        if (c != std::string::npos) req.headers[lower(trim(line.substr(0, c)))] = trim(line.substr(c + 1));
        p = q;
    }
    return true;
}

bool ws_conn::accept_upgrade(const http_request & req) {
    auto it = req.headers.find("sec-websocket-key");
    auto up = req.headers.find("upgrade");
    if (it == req.headers.end() || up == req.headers.end() || lower(up->second) != "websocket") {
        return false;
    }
    std::string accept = base64(sha1().digest(it->second + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"));
    std::string resp = "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                       "Sec-WebSocket-Accept: " + accept + "\r\n\r\n";
    return send_all(resp.data(), resp.size());
}

bool ws_conn::send_http(int code, const char * status, const std::string & ctype, const std::string & body) {
    std::string r = "HTTP/1.1 " + std::to_string(code) + " " + status + "\r\nContent-Type: " + ctype +
                    "\r\nContent-Length: " + std::to_string(body.size()) +
                    "\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n" + body;
    return send_all(r.data(), r.size());
}

bool ws_conn::send_frame(int op, const void * d, size_t n) {
    std::lock_guard<std::mutex> lk(send_mu_);
    if (closed_) return false;
    uint8_t h[10];
    size_t hl = 2;
    h[0] = (uint8_t) (0x80 | op);
    if (n < 126) {
        h[1] = (uint8_t) n;
    } else if (n < 65536) {
        h[1] = 126;
        h[2] = (uint8_t) (n >> 8);
        h[3] = (uint8_t) n;
        hl = 4;
    } else {
        h[1] = 127;
        for (int i = 0; i < 8; i++) h[2 + i] = (uint8_t) ((uint64_t) n >> (8 * (7 - i)));
        hl = 10;
    }
    return send_all(h, hl) && (n == 0 || send_all(d, n));
}

ws_conn::msg_type ws_conn::recv(std::string & payload) {
    payload.clear();
    int msg_op = 0;
    for (;;) {
        uint8_t h[2];
        if (!read_exact(h, 2)) return CLOSED;
        const bool fin = h[0] & 0x80;
        const int op = h[0] & 0x0F;
        const bool masked = h[1] & 0x80;
        uint64_t len = h[1] & 0x7F;
        if (len == 126) {
            uint8_t e[2];
            if (!read_exact(e, 2)) return CLOSED;
            len = (e[0] << 8) | e[1];
        } else if (len == 127) {
            uint8_t e[8];
            if (!read_exact(e, 8)) return CLOSED;
            len = 0;
            for (int i = 0; i < 8; i++) len = (len << 8) | e[i];
        }
        if (len > (64u << 20)) return CLOSED;
        uint8_t mask[4] = {0, 0, 0, 0};
        if (masked && !read_exact(mask, 4)) return CLOSED;
        std::string data(len, '\0');
        if (len && !read_exact(&data[0], len)) return CLOSED;
        if (masked)
            for (uint64_t i = 0; i < len; i++) data[i] ^= mask[i & 3];
        if (op == 0x8) {
            close();
            return CLOSED;
        }
        if (op == 0x9) {
            send_frame(0xA, data.data(), data.size());
            continue;
        }
        if (op == 0xA) continue;
        if (op == 0x1 || op == 0x2) msg_op = op;
        payload += data;
        if (fin && msg_op) return msg_op == 0x1 ? TEXT : BINARY;
    }
}

void ws_conn::close(uint16_t code) {
    uint8_t b[2] = {(uint8_t) (code >> 8), (uint8_t) code};
    send_frame(0x8, b, 2);
    std::lock_guard<std::mutex> lk(send_mu_);
    closed_ = true;
}
