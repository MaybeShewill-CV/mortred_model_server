/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: mortredctl.cpp
* Date: 26-8-22
************************************************/

// Thin REST client: management commands talk to mortred-supervisor; infer
// smoke tests post the data-plane envelope to mortred-gateway.

#include <filesystem>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <workflow/WFFacilities.h>
#include <workflow/WFTaskFactory.h>
#include <workflow/Workflow.h>

#include "common/base64.h"
#include "common/request_envelope.h"

#include <rapidjson/document.h>

#include "control/cli/cli_app.h"

namespace {

struct Options {
    std::string addr;          // supervisor http://host:port
    std::string gateway_addr;  // gateway http://host:port (infer only)
    std::string token;
};

struct HttpResult {
    int status = 0;
    std::string body;
};

HttpResult http_request(const Options& opt, const std::string& method, const std::string& path,
                        const std::string& body) {
    HttpResult out;
    const std::string url = opt.addr + path;
    WFFacilities::WaitGroup wg(1);
    auto* task = WFTaskFactory::create_http_task(
        url, 0, 0, [&wg, &out](WFHttpTask* t) {
            if (t->get_state() == WFT_STATE_SUCCESS) {
                out.status = std::atoi(t->get_resp()->get_status_code());
                const void* data = nullptr;
                size_t size = 0;
                t->get_resp()->get_parsed_body(&data, &size);
                out.body.assign(static_cast<const char*>(data), size);
            } else {
                out.status = -1;
                out.body = std::string("transport failure: state ") +
                           std::to_string(t->get_state()) + ", errno " +
                           std::to_string(t->get_error());
            }
            wg.done();
        });
    task->get_req()->set_method(method.c_str());
    if (!body.empty()) {
        task->get_req()->append_output_body(body.data(), body.size());
        task->get_req()->add_header_pair("Content-Type", "application/json; charset=utf-8");
    }
    if (!opt.token.empty()) {
        const std::string auth = "Bearer " + opt.token;
        task->get_req()->add_header_pair("Authorization", auth.c_str());
    }
    task->set_receive_timeout(300000);  // infer can take a while
    task->start();
    wg.wait();
    return out;
}

std::string read_file_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return "";
    }
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void usage() {
    std::fprintf(stderr,
                 "usage: mortredctl [--addr URL] [--gateway URL] [--token T] <command> [args]\n"
                 "  commands: status [id] | catalog | start <id> | stop <id> | restart <id>\n"
                 "            logs <id> [--offset N] [--limit N]\n"
                 "            infer <id> --image <path>\n"
                 "            init [--profile cpu|gpu] | doctor [--strict] | prepare [--pack FILE]\n"
                 "            calibrate [--pack FILE] | upgrade [version]\n"
                 "  env: MORTREDCTL_ADDR (default http://127.0.0.1:8787), MORTREDCTL_TOKEN,\n"
                 "       MORTREDCTL_GATEWAY_ADDR (default http://127.0.0.1:8080)\n");
}

}  // namespace

namespace mortred {
namespace control {

int run_cli(int argc, char** argv) {
    Options opt;
    if (const char* env = std::getenv("MORTREDCTL_ADDR"); env != nullptr && *env != '\0') {
        opt.addr = env;
    } else {
        opt.addr = "http://127.0.0.1:8787";
    }
    if (const char* env = std::getenv("MORTREDCTL_GATEWAY_ADDR"); env != nullptr && *env != '\0') {
        opt.gateway_addr = env;
    } else {
        opt.gateway_addr = "http://127.0.0.1:8080";
    }
    if (!opt.gateway_addr.empty() && opt.gateway_addr.back() == '/') {
        opt.gateway_addr.pop_back();
    }
    if (const char* env = std::getenv("MORTREDCTL_TOKEN"); env != nullptr && *env != '\0') {
        opt.token = env;
    }

    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }
    size_t index = 0;
    auto next = [&](const char* what) -> std::string {
        if (index >= args.size()) {
            std::fprintf(stderr, "missing value for %s\n", what);
            usage();
            std::exit(2);
        }
        return args[index++];
    };
    while (index < args.size() &&
           (args[index] == "--addr" || args[index] == "--token" || args[index] == "--gateway")) {
        if (args[index] == "--addr") {
            ++index;
            opt.addr = next("--addr");
        } else if (args[index] == "--gateway") {
            ++index;
            opt.gateway_addr = next("--gateway");
            if (!opt.gateway_addr.empty() && opt.gateway_addr.back() == '/') {
                opt.gateway_addr.pop_back();
            }
        } else {
            ++index;
            opt.token = next("--token");
        }
    }
    if (index >= args.size()) {
        usage();
        return 2;
    }
    const std::string cmd = next("command");

    // local ops commands: thin dispatchers to the scripts/ core
    // (single source of truth shared with bootstrap.sh and the docs)
    if (cmd == "init" || cmd == "doctor" || cmd == "upgrade" || cmd == "prepare" ||
        cmd == "calibrate") {
        const std::string root = []() {
            if (const char* env = std::getenv("MORTRED_PROJECT_ROOT"); env != nullptr && *env != '\0') {
                return std::string(env);
            }
            // resolve from the executable itself: <root>/bin/mortredctl.out
            // in both the installed tree and the source-tree _bin layout
            // (never relative to the caller's cwd)
            char buf[4096] = {0};
            const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
            if (n > 0) {
                std::filesystem::path exe(buf);
                auto dir = exe.parent_path();
                if (dir.filename() == "bin" || dir.filename() == "_bin") {
                    return dir.parent_path().string();
                }
                return dir.string();
            }
            return std::string(".");
        }();
        const std::string script = root + "/scripts/mortredctl_" + cmd + ".sh";
        // fork/execvp (NOT std::system): the consistency checker bans
        // shell-spawning calls, and direct execvp passes arguments without
        // any shell re-parsing
        std::vector<char*> child_argv;
        child_argv.push_back(const_cast<char*>(script.c_str()));
        // remaining args (e.g. upgrade v0.2.0 / init --profile cpu)
        while (index < args.size()) {
            child_argv.push_back(const_cast<char*>(args[index++].c_str()));
        }
        child_argv.push_back(nullptr);
        const pid_t pid = ::fork();
        if (pid < 0) {
            std::fprintf(stderr, "mortredctl: fork failed\n");
            return 1;
        }
        if (pid == 0) {
            ::execvp(child_argv[0], child_argv.data());
            std::fprintf(stderr, "mortredctl: cannot execute %s\n", script.c_str());
            ::_exit(127);
        }
        int status = 0;
        ::waitpid(pid, &status, 0);
        return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
    }

    HttpResult r;
    if (cmd == "status" || cmd == "catalog") {
        r = http_request(opt, "GET", "/api/v1/" + cmd, "");
    } else if (cmd == "start" || cmd == "stop" || cmd == "restart") {
        const std::string id = next("server id");
        r = http_request(opt, "POST", "/api/v1/servers/" + id + "/" + cmd, "{}");
    } else if (cmd == "logs") {
        const std::string id = next("server id");
        size_t offset = 0;
        size_t limit = 200;
        while (index < args.size()) {
            const std::string flag = next("flag");
            if (flag == "--offset") {
                offset = static_cast<size_t>(std::stoull(next("--offset")));
            } else if (flag == "--limit") {
                limit = static_cast<size_t>(std::stoull(next("--limit")));
            } else {
                std::fprintf(stderr, "unknown logs flag: %s\n", flag.c_str());
                return 2;
            }
        }
        r = http_request(opt, "GET",
                         "/api/v1/servers/" + id + "/logs?offset=" + std::to_string(offset) +
                             "&limit=" + std::to_string(limit),
                         "");
    } else if (cmd == "infer") {
        const std::string id = next("server id");
        std::string image_path;
        while (index < args.size()) {
            const std::string flag = next("flag");
            if (flag == "--image") {
                image_path = next("--image");
            } else {
                std::fprintf(stderr, "unknown infer flag: %s\n", flag.c_str());
                return 2;
            }
        }
        if (image_path.empty()) {
            std::fprintf(stderr, "infer requires --image <path>\n");
            return 2;
        }
        const std::string bytes = read_file_bytes(image_path);
        if (bytes.empty()) {
            std::fprintf(stderr, "cannot read image file: %s\n", image_path.c_str());
            return 2;
        }
        const std::string b64 = jinq::common::base64::encode(
            reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size());
        const HttpResult cat = http_request(opt, "GET", "/api/v1/catalog", "");
        if (cat.status < 200 || cat.status >= 300) {
            std::fwrite(cat.body.data(), 1, cat.body.size(), stdout);
            if (!cat.body.empty() && cat.body.back() != '\n') {
                std::fputc('\n', stdout);
            }
            return 1;
        }
        rapidjson::Document doc;
        doc.Parse(cat.body.c_str());
        bool found = false;
        if (!doc.HasParseError() && doc.IsObject() && doc.HasMember("servers") &&
            doc["servers"].IsArray()) {
            for (const auto& server : doc["servers"].GetArray()) {
                if (server.IsObject() && server.HasMember("id") && server["id"].IsString() &&
                    id == server["id"].GetString()) {
                    found = true;
                    break;
                }
            }
        }
        if (!found) {
            std::fprintf(stderr, "unknown server id in catalog: %s\n", id.c_str());
            return 1;
        }
        jinq::common::envelope::Request envelope;
        envelope.images.push_back(b64);
        const std::string body = jinq::common::envelope::encode(envelope);
        Options gateway_opt = opt;
        gateway_opt.addr = opt.gateway_addr;
        r = http_request(gateway_opt, "POST", "/v1/models/" + id + "/infer", body);
    } else {
        std::fprintf(stderr, "unknown command: %s\n", cmd.c_str());
        usage();
        return 2;
    }

    std::fwrite(r.body.data(), 1, r.body.size(), stdout);
    if (!r.body.empty() && r.body.back() != '\n') {
        std::fputc('\n', stdout);
    }
    return r.status >= 200 && r.status < 300 ? 0 : 1;
}

}  // namespace control
}  // namespace mortred
