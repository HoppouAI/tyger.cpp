#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET sock_t;
#else
typedef int sock_t;
#endif

bool net_init();
sock_t net_listen(const std::string & host, int port, std::string & err);
sock_t net_accept(sock_t s, std::string & peer);
void net_close(sock_t s);

struct http_request {
    std::string method, path;
    std::map<std::string, std::string> headers; // lowercased names
};

// one websocket connection (server side)
class ws_conn {
public:
    explicit ws_conn(sock_t s) : s_(s) {}
    ~ws_conn();

    // reads the http request; returns false on a broken request
    bool read_request(http_request & req);
    bool accept_upgrade(const http_request & req);
    bool send_http(int code, const char * status, const std::string & ctype, const std::string & body);

    enum msg_type { TEXT = 1, BINARY = 2, CLOSED = 0 };
    // blocks for the next complete message; control frames are handled inside
    msg_type recv(std::string & payload);

    bool send_text(const std::string & s) { return send_frame(0x1, s.data(), s.size()); }
    bool send_binary(const void * d, size_t n) { return send_frame(0x2, d, n); }
    void close(uint16_t code = 1000);

private:
    bool send_frame(int op, const void * d, size_t n);
    bool read_exact(void * dst, size_t n);
    bool send_all(const void * d, size_t n);

    sock_t s_;
    std::mutex send_mu_;
    std::string pending_; // bytes read past the http request
    bool closed_ = false;
};
