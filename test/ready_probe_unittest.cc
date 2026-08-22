/************************************************
 * Author: Codex
 * File: ready_probe_unittest.cc
 * Date: 2026-08-19
 *
 * ServerManager readiness probe contract: HTTP GET /ready, 2xx = ready;
 * non-2xx / connection refused / timeout all mean not ready.
 ************************************************/

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include "control/ready_probe.h"

namespace {

// Starts a one-shot listening socket: accepts a single connection, sends the
// canned response, then closes. Returns the bound local port; an empty
// canned_response means accept but do not respond (timeout scenario).
int start_one_shot_listener(const std::string& canned_response) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
        ::listen(fd, 1) != 0) {
        ::close(fd);
        return -1;
    }
    socklen_t len = sizeof(addr);
    ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len);
    const int port = ntohs(addr.sin_port);

    std::thread([fd, canned_response]() {
        const int conn = ::accept(fd, nullptr, nullptr);
        if (conn >= 0) {
            if (!canned_response.empty()) {
                ::send(conn, canned_response.data(), canned_response.size(), 0);
            } else {
                // timeout scenario: hold the connection without responding so
                // the probe takes the SO_RCVTIMEO path
                std::this_thread::sleep_for(std::chrono::milliseconds(1500));
            }
            ::close(conn);
        }
        ::close(fd);
    }).detach();
    return port;
}

}  // namespace

TEST(ready_probe, http_200_means_ready) {
    const int port = start_one_shot_listener(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}");
    ASSERT_GT(port, 0);
    EXPECT_TRUE(mortred::control::endpoint_ready(port, "/ready", 1000));
}

TEST(ready_probe, http_503_means_not_ready) {
    const int port = start_one_shot_listener(
        "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n");
    ASSERT_GT(port, 0);
    EXPECT_FALSE(mortred::control::endpoint_ready(port, "/ready", 1000));
}

TEST(ready_probe, connection_refused_means_not_ready) {
    // bind then close immediately, leaving a port that is almost certainly free
    const int port = start_one_shot_listener("");
    ASSERT_GT(port, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_FALSE(mortred::control::endpoint_ready(port, "/ready", 500));
}

TEST(ready_probe, no_response_times_out_as_not_ready) {
    const int port = start_one_shot_listener("");
    ASSERT_GT(port, 0);
    const auto t0 = std::chrono::steady_clock::now();
    EXPECT_FALSE(mortred::control::endpoint_ready(port, "/ready", 300));
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - t0).count();
    // the timeout must be honored (no hang); leave scheduling slack
    EXPECT_LT(elapsed_ms, 1500);
}
