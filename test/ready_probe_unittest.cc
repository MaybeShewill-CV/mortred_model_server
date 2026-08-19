/************************************************
 * Author: Codex
 * File: ready_probe_unittest.cc
 * Date: 2026-08-19
 *
 * ServerManager 就绪探测契约：HTTP GET /ready，2xx 即就绪；
 * 非 2xx / 连接被拒 / 超时均视为未就绪。
 ************************************************/

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include "apps/web_console/backend/ready_probe.h"

namespace {

// 起一次性监听 socket：接受单个连接并发送 canned 响应后关闭。
// 返回绑定的本地端口；canned_response 为空表示接受后不响应（超时剧本）。
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
                // 超时剧本：挂住连接不响应，让探测端走 SO_RCVTIMEO 路径
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
    EXPECT_TRUE(mortred_web::endpoint_ready(port, "/ready", 1000));
}

TEST(ready_probe, http_503_means_not_ready) {
    const int port = start_one_shot_listener(
        "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n");
    ASSERT_GT(port, 0);
    EXPECT_FALSE(mortred_web::endpoint_ready(port, "/ready", 1000));
}

TEST(ready_probe, connection_refused_means_not_ready) {
    // 绑定后立即关闭，留下一个几乎必然空闲的端口
    const int port = start_one_shot_listener("");
    ASSERT_GT(port, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_FALSE(mortred_web::endpoint_ready(port, "/ready", 500));
}

TEST(ready_probe, no_response_times_out_as_not_ready) {
    const int port = start_one_shot_listener("");
    ASSERT_GT(port, 0);
    const auto t0 = std::chrono::steady_clock::now();
    EXPECT_FALSE(mortred_web::endpoint_ready(port, "/ready", 300));
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - t0).count();
    // 超时必须被尊重（不会挂死）；留出调度余量
    EXPECT_LT(elapsed_ms, 1500);
}
