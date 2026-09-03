/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: fake_model_server.cc
* Date: 26-8-22
************************************************/

// Deterministic model-server stand-in for supervision / gateway tests.
// Behaviour comes from the config file (the supervisor always passes the
// server config as argv[1]) and can be overridden by flags:
//   --port N --mode ready|never-ready|exit-now --exit-after-ms N --exit-code N

#include <arpa/inet.h>
#include <netinet/in.h>
#include <cstdio>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>

namespace {

std::string trim(const std::string& s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r')) ++b;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r')) --e;
    return s.substr(b, e - b);
}

std::map<std::string, std::string> parse_config(const std::string& path) {
    std::map<std::string, std::string> kv;
    std::ifstream in(path);
    if (!in.is_open()) {
        return kv;
    }
    std::string line;
    while (std::getline(in, line)) {
        const std::string t = trim(line);
        if (t.empty() || t[0] == '#' || t.front() == '[') {
            continue;
        }
        const auto eq = t.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        std::string v = trim(t.substr(eq + 1));
        if (v.size() >= 2 && v.front() == '"' && v.back() == '"') {
            v = v.substr(1, v.size() - 2);
        }
        kv[trim(t.substr(0, eq))] = v;
    }
    return kv;
}

void handle_client(int fd, const std::string& mode) {
    std::string req;
    char buf[2048];
    while (req.find("\r\n\r\n") == std::string::npos) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) {
            ::close(fd);
            return;
        }
        req.append(buf, static_cast<size_t>(n));
        if (req.size() > 65536) {
            break;
        }
    }
    const std::string method = req.substr(0, req.find(' '));
    const size_t sp = req.find(' ');
    const size_t sp2 = req.find(' ', sp + 1);
    std::string path = sp2 == std::string::npos ? "/" : req.substr(sp + 1, sp2 - sp - 1);
    const auto q = path.find('?');
    if (q != std::string::npos) {
        path = path.substr(0, q);
    }

    std::string status = "200 OK";
    std::string body = "{\"code\":0,\"msg\":\"success\",\"data\":{\"fake\":true}}";
    if (path == "/ready" && mode == "never-ready") {
        status = "503 Service Unavailable";
        body = "{\"code\":65}";
    } else if (method == "GET") {
        body = "ok";
    }
    std::string extra_headers;
    if (mode == "overloaded" && method == "POST") {
        status = "429 Too Many Requests";
        body = "{\"code\":429}";
        extra_headers = "Retry-After: 2\r\n";
    }
    const std::string resp = "HTTP/1.1 " + status + "\r\n" +
                             "Content-Type: application/json; charset=utf-8\r\n" +
                             "Content-Length: " + std::to_string(body.size()) + "\r\n" +
                             extra_headers +
                             "Connection: close\r\n\r\n" + body;
    ::send(fd, resp.data(), resp.size(), 0);
    ::close(fd);
}

}  // namespace

int main(int argc, char** argv) {
    ::signal(SIGPIPE, SIG_IGN);
    if (const char* dump = std::getenv("MORTRED_ARGV_FILE"); dump != nullptr && *dump != '\0') {
        std::ofstream out(dump);
        for (int i = 0; i < argc; ++i) {
            out << argv[i] << '\n';
        }
    }
    std::map<std::string, std::string> cfg;
    int port = 0;
    std::string mode = "ready";
    int exit_after_ms = -1;
    int exit_code = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--exit-after-ms" && i + 1 < argc) {
            exit_after_ms = std::atoi(argv[++i]);
        } else if (arg == "--exit-code" && i + 1 < argc) {
            exit_code = std::atoi(argv[++i]);
        } else if (arg == "--model" && i + 1 < argc) {
            ++i;
        } else if (!arg.empty() && arg[0] != '-') {
            cfg = parse_config(arg);
        }
    }
    if (cfg.count("fake_port") != 0 && port == 0) {
        port = std::atoi(cfg["fake_port"].c_str());
    }
    if (cfg.count("fake_mode") != 0 && mode == "ready") {
        mode = cfg["fake_mode"];
    }
    if (cfg.count("fake_exit_after_ms") != 0) {
        const int v = std::atoi(cfg["fake_exit_after_ms"].c_str());
        // 0 / negative means "no timed exit" (a timer of 0ms would kill the
        // server instantly and make "ready" mode undistinguishable from crash)
        if (v > 0) {
            exit_after_ms = v;
        }
    }
    if (cfg.count("fake_exit_code") != 0 && exit_code == 0) {
        exit_code = std::atoi(cfg["fake_exit_code"].c_str());
    }
    if (port <= 0) {
        std::fprintf(stderr, "fake_model_server: invalid port %d\n", port);
        return 2;
    }
    if (mode == "exit-now") {
        return exit_code;
    }

    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return 2;
    }
    int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
        ::listen(fd, 16) != 0) {
        std::fprintf(stderr, "fake_model_server: cannot bind 127.0.0.1:%d\n", port);
        return 2;
    }

    std::atomic<bool> stopped{false};
    if (exit_after_ms > 0) {
        std::thread([&stopped, exit_after_ms, exit_code]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(exit_after_ms));
            stopped.store(true);
            ::_exit(exit_code);
        }).detach();
    }

    while (!stopped.load()) {
        const int client = ::accept(fd, nullptr, nullptr);
        if (client < 0) {
            if (stopped.load()) {
                break;
            }
            continue;
        }
        handle_client(client, mode);
    }
    return exit_code;
}
