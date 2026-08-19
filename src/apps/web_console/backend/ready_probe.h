/************************************************
 * Author: Codex
 * File: ready_probe.h
 * Date: 2026-08-19
 *
 * HTTP GET readiness probe for managed model servers. Replaces the former
 * log-grep heuristic ("server init successfully" string scan) with the
 * servers' real /ready endpoint: 2xx = ready, anything else (connection
 * refused / timeout / non-2xx) = not ready.
 ************************************************/

#ifndef MORTRED_WEB_READY_PROBE_H
#define MORTRED_WEB_READY_PROBE_H

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <string>

namespace mortred_web {

/***
 * Probe 127.0.0.1:port with a short-timeout HTTP GET.
 * @param port target port (managed model server)
 * @param path request path, e.g. "/ready"
 * @param timeout_ms budget for connect + response, must be > 0
 * @return true iff an HTTP response with a 2xx status line arrives in time
 */
inline bool endpoint_ready(int port, const char* path, int timeout_ms) {
    if (port <= 0 || path == nullptr || path[0] != '/' || timeout_ms <= 0) {
        return false;
    }

    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return false;
    }

    // non-blocking connect + poll：连接阶段也不能超过预算
    const int flags = ::fcntl(fd, F_GETFL, 0);
    ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    const int rc = ::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    if (rc != 0 && errno != EINPROGRESS) {
        ::close(fd);
        return false;  // 连接被拒等：服务未监听，即未就绪
    }
    if (rc != 0) {
        pollfd pfd{fd, POLLOUT, 0};
        if (::poll(&pfd, 1, timeout_ms) != 1 || (pfd.revents & POLLOUT) == 0) {
            ::close(fd);
            return false;  // 连接超时
        }
        int so_error = 0;
        socklen_t optlen = sizeof(so_error);
        if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_error, &optlen) != 0 ||
            so_error != 0) {
            ::close(fd);
            return false;
        }
    }

    // 恢复阻塞模式并对读设置剩余预算为超时
    ::fcntl(fd, F_SETFL, flags);
    timeval recv_timeout{};
    recv_timeout.tv_usec = static_cast<suseconds_t>(timeout_ms) * 1000;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &recv_timeout, sizeof(recv_timeout));

    const std::string request = std::string("GET ") + path +
                                " HTTP/1.1\r\n"
                                "Host: 127.0.0.1\r\n"
                                "Connection: close\r\n"
                                "\r\n";
    if (::send(fd, request.data(), request.size(), 0) !=
        static_cast<ssize_t>(request.size())) {
        ::close(fd);
        return false;
    }

    // 状态行足够判定就绪，无需读完整响应体
    char status_line[64] = {0};
    const ssize_t n = ::recv(fd, status_line, sizeof(status_line) - 1, 0);
    ::close(fd);
    if (n <= 0) {
        return false;  // 无响应 / 超时 / 对端关闭
    }
    // 形如 "HTTP/1.1 200 OK"：校验版本前缀 + 三位 2xx 状态码
    const std::string line(status_line, static_cast<size_t>(n));
    if (line.compare(0, 7, "HTTP/1.") != 0 || line.size() < 12) {
        return false;
    }
    // "HTTP/1.1 200 OK"：状态码三字节位于 [9..11]（[8] 是空格）
    return line[9] == '2' && line[10] >= '0' && line[10] <= '9' &&
           line[11] >= '0' && line[11] <= '9';
}

}  // namespace mortred_web

#endif  // MORTRED_WEB_READY_PROBE_H
