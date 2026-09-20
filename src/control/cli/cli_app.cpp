/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: mortredctl.cpp
* Date: 26-8-22
************************************************/

// Thin REST client: management commands talk to mortred-supervisor; infer
// smoke tests post the data-plane envelope to mortred-gateway.

#include <chrono>
#include <cmath>
#include <filesystem>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <workflow/WFFacilities.h>
#include <workflow/WFTaskFactory.h>
#include <workflow/Workflow.h>

#include "common/base64.h"
#include "common/request_envelope.h"
#include "control/project_root.h"

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
                        const std::string& body, int timeout_ms = 300000,
                        const char* content_type = "application/json; charset=utf-8") {
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
        task->get_req()->add_header_pair("Content-Type", content_type);
    }
    if (!opt.token.empty()) {
        const std::string auth = "Bearer " + opt.token;
        task->get_req()->add_header_pair("Authorization", auth.c_str());
    }
    task->set_receive_timeout(timeout_ms);  // infer can take a while; probes stay short
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
                 "            profile <id> --image <path> [--n N] [--warmup N] [--encoding raw|json]\n"
                 "                     [--log-dir DIR] [--keep-flag]   (per-stage latency table)\n"
                 "            ps | down [--id a,b] [--all] [--yes]\n"
                 "            init [--profile cpu|gpu] | init-trust [--force] | init-edge --mode lan|acme|files\n"
                 "            doctor [--strict] | prepare [--pack FILE]\n"
                 "            calibrate [--pack FILE] [--write-pack] | next | upgrade [version]\n"
                 "  env: MORTREDCTL_ADDR (default http://127.0.0.1:8787), MORTREDCTL_TOKEN,\n"
                 "       MORTREDCTL_GATEWAY_ADDR (default http://127.0.0.1:8080)\n");
}

/* ---------------- ps / down: control-plane probe and shutdown ---------------- */

const char* col_ok()   { static const char* c = ::isatty(STDOUT_FILENO) ? "\033[32m" : ""; return c; }
const char* col_err()  { static const char* c = ::isatty(STDOUT_FILENO) ? "\033[31m" : ""; return c; }
const char* col_warn() { static const char* c = ::isatty(STDOUT_FILENO) ? "\033[33m" : ""; return c; }
const char* col_dim()  { static const char* c = ::isatty(STDOUT_FILENO) ? "\033[2m" : ""; return c; }
const char* col_bold() { static const char* c = ::isatty(STDOUT_FILENO) ? "\033[1m" : ""; return c; }
const char* col_off()  { static const char* c = ::isatty(STDOUT_FILENO) ? "\033[0m" : ""; return c; }

std::string pad(const std::string& s, size_t n) {
    return s.size() >= n ? s : s + std::string(n - s.size(), ' ');
}

std::string fmt_uptime(long long started_ms) {
    if (started_ms <= 0) {
        return "--";
    }
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch()).count();
    long long sec = std::max(0LL, (now_ms - started_ms) / 1000);
    const long long d = sec / 86400; sec %= 86400;
    const long long h = sec / 3600;  sec %= 3600;
    const long long m = sec / 60;
    const long long s = sec % 60;
    char buf[32];
    if (d > 0) {
        std::snprintf(buf, sizeof buf, "%lldd%lldh", d, h);
    } else if (h > 0) {
        std::snprintf(buf, sizeof buf, "%lld:%02lld:%02lld", h, m, s);
    } else {
        std::snprintf(buf, sizeof buf, "%lld:%02lld", m, s);
    }
    return buf;
}

std::string json_str(const rapidjson::Value& obj, const char* key, const char* fallback = "--") {
    if (obj.IsObject() && obj.HasMember(key) && obj[key].IsString()) {
        return obj[key].GetString();
    }
    return fallback;
}

long long json_i64(const rapidjson::Value& obj, const char* key, long long fallback = 0) {
    if (obj.IsObject() && obj.HasMember(key) && obj[key].IsInt64()) {
        return obj[key].GetInt64();
    }
    return fallback;
}

bool parse_json(const std::string& body, rapidjson::Document* doc) {
    doc->Parse(body.c_str());
    return !doc->HasParseError() && doc->IsObject();
}

const char* state_color(const std::string& state) {
    if (state == "running") return col_ok();
    if (state == "failed") return col_err();
    if (state == "starting" || state == "backoff") return col_warn();
    return col_dim();
}

int cmd_ps(const Options& opt) {
    std::printf("%smortred ps%s\n", col_bold(), col_off());

    // supervisor liveness (public endpoint, no auth)
    const HttpResult health = http_request(opt, "GET", "/api/v1/health", "", 2500);
    if (health.status == 200) {
        std::printf("  SUPERVISOR  %s● up%s     %s\n", col_ok(), col_off(), opt.addr.c_str());
    } else {
        std::printf("  SUPERVISOR  %s● down%s   %s\n", col_err(), col_off(), opt.addr.c_str());
        std::printf("%s    hint: ssh -L 8787:<host>:8787 … / systemctl status mortred-supervisor%s\n",
                    col_dim(), col_off());
    }

    // gateway: truth = direct healthz probe
    Options gw_opt = opt;
    gw_opt.addr = opt.gateway_addr;
    gw_opt.token.clear();  // healthz is public
    const HttpResult gw = http_request(gw_opt, "GET", "/healthz", "", 2500);
    const bool gw_alive = gw.status == 200;
    std::printf("  GATEWAY     %s● %s%s   %s  healthz=%s\n",
                gw_alive ? col_ok() : col_err(), gw_alive ? "up" : "down", col_off(),
                opt.gateway_addr.c_str(), gw_alive ? "ok" : "no response");

    // supervisor's view + per-server detail (needs token)
    const HttpResult st = http_request(opt, "GET", "/api/v1/status", "", 3000);
    const HttpResult cat = http_request(opt, "GET", "/api/v1/catalog", "", 3000);
    const HttpResult gpu = http_request(opt, "GET", "/api/v1/gpu", "", 3000);

    if (st.status == 401 || cat.status == 401) {
        std::printf("%s    ⚠ management API needs a token: --token T / MORTREDCTL_TOKEN "
                    "(server detail hidden)%s\n", col_warn(), col_off());
        return (health.status == 200 && gw_alive) ? 0 : 1;
    }

    rapidjson::Document st_doc, cat_doc, gpu_doc;
    const bool st_ok = st.status == 200 && parse_json(st.body, &st_doc);
    const bool cat_ok = cat.status == 200 && parse_json(cat.body, &cat_doc);
    if (!st_ok) {
        if (health.status == 200) {
            std::printf("%s    ⚠ /api/v1/status unreachable (HTTP %d)%s\n", col_warn(), st.status, col_off());
        }
        return (health.status == 200 && gw_alive) ? 0 : 1;
    }

    if (st_doc.HasMember("gateway") && st_doc["gateway"].IsObject()) {
        const auto& g = st_doc["gateway"];
        std::printf("%s    supervisor-view: state=%s", col_dim(), json_str(g, "state", "?").c_str());
        if (json_i64(g, "pid") > 0) {
            std::printf(" pid=%lld", json_i64(g, "pid"));
        }
        std::printf(" restarts=%lld", json_i64(g, "restart_count"));
        if (g.HasMember("address") && g["address"].IsObject()) {
            const auto& a = g["address"];
            std::printf(" bind=%s:%lld", json_str(a, "host", "").c_str(), json_i64(a, "port"));
        }
        std::printf("%s\n", col_off());
    }

    if (gpu.status == 200 && parse_json(gpu.body, &gpu_doc) && gpu_doc.HasMember("samples") &&
        gpu_doc["samples"].IsArray() && !gpu_doc["samples"].Empty()) {
        const auto& last = gpu_doc["samples"][gpu_doc["samples"].Size() - 1];
        const long long mem_t = json_i64(last, "mem_total_mib");
        const long long mem_u = json_i64(last, "mem_used_mib");
        char mem[48] = "--";
        if (mem_t > 0) {
            std::snprintf(mem, sizeof mem, "%.1fG/%.1fG (%lld%%)",
                          mem_u / 1024.0, mem_t / 1024.0, mem_u * 100 / mem_t);
        }
        std::printf("  GPU         %s●%s %s  util %lld%%  vram %s  temp %lld°C  pwr %lldW\n",
                    col_ok(), col_off(), json_str(gpu_doc, "name", "?").c_str(),
                    json_i64(last, "util", -1), mem,
                    json_i64(last, "temp_c", -1), json_i64(last, "power_w", -1));
    }

    // catalog x status merge table
    if (!cat_ok) {
        return (health.status == 200 && gw_alive) ? 0 : 1;
    }
    struct Row {
        std::string id, state, category, type, port, pid, uptime, restarts, info;
        bool ready = false, has_status = false;
    };
    std::vector<Row> rows;
    if (cat_doc.HasMember("servers") && cat_doc["servers"].IsArray()) {
        for (const auto& c : cat_doc["servers"].GetArray()) {
            Row r;
            r.id = json_str(c, "id");
            r.category = json_str(c, "category");
            r.type = json_str(c, "type");
            r.port = std::to_string(json_i64(c, "port", -1));
            if (st_doc.HasMember("servers") && st_doc["servers"].IsArray()) {
                for (const auto& s : st_doc["servers"].GetArray()) {
                    if (json_str(s, "id") == r.id) {
                        r.has_status = true;
                        r.state = json_str(s, "state", "stopped");
                        r.ready = s.HasMember("ready") && s["ready"].IsBool() && s["ready"].GetBool();
                        r.pid = std::to_string(json_i64(s, "pid", -1));
                        r.uptime = fmt_uptime(json_i64(s, "started_at_ms"));
                        r.restarts = std::to_string(json_i64(s, "restart_count"));
                        if (s.HasMember("error") && s["error"].IsString()) {
                            r.info = std::string("err: ") + s["error"].GetString();
                        } else if (s.HasMember("last_exit_status") && s["last_exit_status"].IsInt64() &&
                                   r.state != "running") {
                            r.info = "last_exit=" + std::to_string(s["last_exit_status"].GetInt64());
                        }
                        break;
                    }
                }
            }
            if (!r.has_status) {
                r.state = "stopped";
                r.pid = "--";
                r.uptime = "--";
                r.restarts = "0";
                r.info = "not in supervisor status (never started)";
            }
            rows.push_back(r);
        }
    }
    long long live = 0;
    for (const auto& r : rows) {
        if (r.state == "running" || r.state == "starting" || r.state == "backoff") {
            ++live;
        }
    }
    std::printf("  SERVERS     %lld live / %zu total\n", live, rows.size());
    std::printf("%s    %-18s%-12s%-7s%-20s%-6s%-7s%-8s%-10s%-4sinfo%s\n",
                col_dim(), "id", "state", "ready", "category", "type", "port", "pid",
                "uptime", "↻", col_off());
    for (const auto& r : rows) {
        std::printf("    %-18s%s%-12s%s%-6s%-20s%-6s%-7s%-8s%-10s%-4s",
                    r.id.c_str(),
                    state_color(r.state), pad(r.state, 12).c_str(), col_off(),
                    r.ready ? "yes" : "no",
                    r.category.c_str(), r.type.c_str(), r.port.c_str(),
                    r.pid.c_str(), r.uptime.c_str(), r.restarts.c_str());
        if (!r.info.empty()) {
            const bool is_err = r.info.rfind("err: ", 0) == 0;
            std::printf("%s%s%s", is_err ? col_err() : col_dim(), r.info.substr(0, 60).c_str(), col_off());
        }
        std::printf("\n");
    }
    return (health.status == 200 && gw_alive) ? 0 : 1;
}

bool confirm(const std::string& question, bool yes_flag) {
    if (yes_flag) {
        return true;
    }
    const char* env = std::getenv("MORTREDCTL_YES");
    if (env != nullptr && std::string(env) == "1") {
        return true;
    }
    if (::isatty(STDIN_FILENO) != 1) {
        std::fprintf(stderr, "refusing interactive confirm without a tty (use --yes)\n");
        return false;
    }
    std::fprintf(stderr, "%s [y/N] ", question.c_str());
    std::string line;
    if (!std::getline(std::cin, line)) {
        return false;
    }
    return line == "y" || line == "Y" || line == "yes";
}

int cmd_down(const Options& opt, const std::vector<std::string>& rest) {
    std::string id_list;
    bool all = false, yes = false;
    for (size_t i = 0; i < rest.size(); ++i) {
        if (rest[i] == "--id" && i + 1 < rest.size()) {
            id_list = rest[++i];
        } else if (rest[i] == "--all") {
            all = true;
        } else if (rest[i] == "--yes") {
            yes = true;
        } else {
            std::fprintf(stderr, "unknown down flag: %s\n", rest[i].c_str());
            return 2;
        }
    }
    // resolve targets
    std::vector<std::string> targets, known;
    const HttpResult st = http_request(opt, "GET", "/api/v1/status", "", 3000);
    rapidjson::Document doc;
    if (st.status != 200 || !parse_json(st.body, &doc) || !doc.HasMember("servers") ||
        !doc["servers"].IsArray()) {
        std::fprintf(stderr, "cannot list servers (supervisor %s HTTP %d); nothing stopped\n",
                     opt.addr.c_str(), st.status);
        if (st.status == 401) {
            std::fprintf(stderr,
                         "401 = supervisor api token missing/wrong. Set MORTREDCTL_TOKEN (or pass "
                         "--token) to the MORTRED_API_TOKEN value, e.g.:\n"
                         "  export MORTREDCTL_TOKEN=$(grep -oP '(?<=^MORTRED_API_TOKEN=).*' "
                         "/etc/mortred/supervisor.env | tr -d '\"')\n"
                         "or bypass the api entirely: sudo systemctl stop mortred-supervisor\n");
        }
        return 1;
    }
    std::vector<std::string> running;
    for (const auto& s : doc["servers"].GetArray()) {
        const std::string id = json_str(s, "id");
        known.push_back(id);
        const std::string state = json_str(s, "state");
        if (state == "running" || state == "starting" || state == "backoff") {
            running.push_back(id);
        }
    }
    if (!id_list.empty()) {
        std::stringstream ss(id_list);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) {
                targets.push_back(item);
            }
        }
        for (const auto& id : targets) {
            if (std::find(known.begin(), known.end(), id) == known.end()) {
                std::fprintf(stderr, "unknown server ids: %s\n", id.c_str());
                return 2;
            }
        }
    } else {
        targets = running;
    }
    if (targets.empty()) {
        std::printf("no running model servers — nothing to stop\n");
    } else {
        std::string label = targets.size() <= 8
            ? std::accumulate(targets.begin(), targets.end(), std::string(),
                              [](const std::string& a, const std::string& b) {
                                  return a.empty() ? b : a + ", " + b;
                              })
            : std::to_string(targets.size()) + " servers";
        if (!confirm("stop " + label + "?", yes)) {
            std::printf("aborted\n");
            return 1;
        }
        int rc = 0;
        for (const auto& id : targets) {
            const HttpResult r = http_request(opt, "POST", "/api/v1/servers/" + id + "/stop", "{}", 10000);
            if (r.status >= 200 && r.status < 300) {
                std::printf("  %-18s ✓ stopped\n", id.c_str());
            } else {
                std::printf("  %-18s %s✗ HTTP %d %s%s\n", id.c_str(), col_err(), r.status,
                            r.body.substr(0, 80).c_str(), col_off());
                rc = 1;
            }
        }
        if (!all) {
            return rc;
        }
    }
    if (!all) {
        return 0;
    }
    if (!confirm("stop ALL servers + gateway + supervisor?", yes)) {
        std::printf("aborted\n");
        return 1;
    }
    // gateway+supervisor: no management API for this — local process control only
    std::printf("\nstopping gateway + supervisor (local host only)…\n");
    auto run = [](const std::vector<std::string>& argv) -> int {
        std::vector<char*> av;
        for (const auto& a : argv) {
            av.push_back(const_cast<char*>(a.c_str()));
        }
        av.push_back(nullptr);
        const pid_t pid = ::fork();
        if (pid < 0) {
            return -1;
        }
        if (pid == 0) {
            ::execvp(av[0], av.data());
            ::_exit(127);  // not installed
        }
        int status = 0;
        ::waitpid(pid, &status, 0);
        return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    };
    const int sys_rc = run({"systemctl", "stop", "mortred-supervisor"});
    if (sys_rc == 0) {
        std::printf("  ✓ systemctl stop mortred-supervisor (kills the whole tree)\n");
    } else {
        bool killed_any = false;
        for (const char* pattern : {"mortred-supervisor", "mortred-gateway"}) {
            if (run({"pkill", "-TERM", "-f", pattern}) == 0) {
                std::printf("  ✓ pkill -TERM -f %s\n", pattern);
                killed_any = true;
            }
        }
        if (!killed_any) {
            std::printf("%s  no local mortred processes found — supervisor/gateway run elsewhere?%s\n",
                        col_warn(), col_off());
            std::printf("%s    (API has no gateway-stop endpoint; on the host: "
                        "systemctl stop mortred-supervisor)%s\n", col_dim(), col_off());
        }
    }
    return 0;
}

}  // namespace

namespace mortred {
namespace control {

/* ------------- profile: per-stage latency of one model server ------------- */

using StageMap = std::map<std::string, std::vector<double>>;

const char* const k_stage_timing_flag = "/tmp/mortred_stage_timing";

size_t count_log_lines(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return 0;
    }
    size_t lines = 0;
    std::string line;
    while (std::getline(in, line)) {
        ++lines;
    }
    return lines;
}

// parses "stage_trace svc=... id=... key=value ..." lines that appear after
// line `from` (exclusive) in the log; numeric fields land in the map
void parse_stage_window(const std::string& path, size_t from, StageMap* out, size_t* found) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return;
    }
    std::string line;
    size_t at = 0;
    *found = 0;
    while (std::getline(in, line)) {
        ++at;
        if (at <= from || line.find("stage_trace") == std::string::npos) {
            continue;
        }
        ++*found;
        std::istringstream tokens(line);
        std::string token;
        while (tokens >> token) {
            const size_t eq = token.find('=');
            if (eq == std::string::npos) {
                continue;
            }
            const std::string key = token.substr(0, eq);
            if (key == "svc" || key == "id") {
                continue;
            }
            try {
                (*out)[key].push_back(std::stod(token.substr(eq + 1)));
            } catch (const std::exception&) {
                // non-numeric field - skip
            }
        }
    }
}

double percentile(std::vector<double> samples, double q) {
    if (samples.empty()) {
        return 0.0;
    }
    std::sort(samples.begin(), samples.end());
    const double index = static_cast<double>(samples.size() - 1) * q;
    const size_t lo = static_cast<size_t>(std::floor(index));
    const size_t hi = static_cast<size_t>(std::ceil(index));
    const double frac = index - static_cast<double>(lo);
    return samples[lo] * (1.0 - frac) + samples[hi] * frac;
}

double stage_p50(const StageMap& stages, const std::string& key) {
    const auto it = stages.find(key);
    return it == stages.end() ? -1.0 : percentile(it->second, 0.5);
}

double stage_mean(const StageMap& stages, const std::string& key) {
    const auto it = stages.find(key);
    if (it == stages.end() || it->second.empty()) {
        return -1.0;
    }
    const double sum = std::accumulate(it->second.begin(), it->second.end(), 0.0);
    return sum / static_cast<double>(it->second.size());
}

double sum_p50(const StageMap& stages, const std::vector<std::string>& keys) {
    double total = 0.0;
    for (const auto& key : keys) {
        const double v = stage_p50(stages, key);
        if (v > 0) {
            total += v;
        }
    }
    return total;
}

std::string fmt_ms(double v, int width = 8, int prec = 3) {
    if (v < 0) {
        return std::string(width, '-');
    }
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.*f", prec, v);
    std::string s(buf);
    return s.size() >= static_cast<size_t>(width) ? s : std::string(static_cast<size_t>(width) - s.size(), ' ') + s;
}

/*** terminal display width of a UTF-8 string: CJK codepoints occupy two
 * columns but three bytes, so byte-based %-70s padding misaligns table
 * columns that mix Chinese and ASCII */
size_t display_width(const std::string& s) {
    size_t width = 0;
    size_t at = 0;
    while (at < s.size()) {
        const unsigned char lead = static_cast<unsigned char>(s[at]);
        size_t bytes = 1;
        uint32_t codepoint = lead;
        if ((lead & 0xE0) == 0xC0) {
            bytes = 2;
            codepoint = lead & 0x1F;
        } else if ((lead & 0xF0) == 0xE0) {
            bytes = 3;
            codepoint = lead & 0x0F;
        } else if ((lead & 0xF8) == 0xF0) {
            bytes = 4;
            codepoint = lead & 0x07;
        }
        for (size_t i = 1; i < bytes && at + i < s.size(); ++i) {
            codepoint = (codepoint << 6) | (static_cast<unsigned char>(s[at + i]) & 0x3F);
        }
        width += (codepoint >= 0x1100 && codepoint <= 0xFFFD) ? 2 : 1;
        at += bytes;
    }
    return width;
}

std::string pad_display(const std::string& s, size_t target_width) {
    const size_t current = display_width(s);
    return current >= target_width ? s : s + std::string(target_width - current, ' ');
}

std::string image_media_type(const std::string& path) {
    const size_t dot = path.rfind('.');
    std::string ext = dot == std::string::npos ? "" : path.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == "jpg" || ext == "jpeg") return "image/jpeg";
    if (ext == "png") return "image/png";
    if (ext == "bmp") return "image/bmp";
    if (ext == "webp") return "image/webp";
    return "application/octet-stream";
}

int cmd_profile(const Options& opt, const std::string& id, const std::string& image_path,
                size_t requests, size_t warmup, const std::string& encoding, bool keep_flag,
                const std::string& log_dir_override) {
    const std::string bytes = read_file_bytes(image_path);
    if (bytes.empty()) {
        std::fprintf(stderr, "cannot read image file: %s\n", image_path.c_str());
        return 2;
    }
    const HttpResult cat = http_request(opt, "GET", "/api/v1/catalog", "");
    if (cat.status < 200 || cat.status >= 300) {
        std::fprintf(stderr, "catalog unavailable (status %d)\n", cat.status);
        return 1;
    }
    rapidjson::Document doc;
    doc.Parse(cat.body.c_str());
    bool found = false;
    if (!doc.HasParseError() && doc.IsObject() && doc.HasMember("servers") && doc["servers"].IsArray()) {
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

    const std::string root = resolve_project_root();
    std::string log_dir = log_dir_override.empty() ? "logs" : log_dir_override;
    if (log_dir_override.empty()) {
        // one-key scan of conf/mortred.toml for log_dir (default "logs")
        std::ifstream conf(root + "/conf/mortred.toml");
        std::string line;
        while (std::getline(conf, line)) {
            const size_t at = line.find("log_dir");
            if (at != std::string::npos) {
                const size_t q1 = line.find('"');
                const size_t q2 = q1 == std::string::npos ? std::string::npos : line.find('"', q1 + 1);
                if (q1 != std::string::npos && q2 != std::string::npos && q2 > q1 + 1) {
                    log_dir = line.substr(q1 + 1, q2 - q1 - 1);
                }
                break;
            }
        }
    }
    const std::string model_log = root + "/" + log_dir + "/" + id + ".log";
    const std::string gw_log = root + "/" + log_dir + "/__gateway.log";
    if (!std::filesystem::exists(model_log)) {
        std::fprintf(stderr, "model log not found: %s (override with --log-dir)\n", model_log.c_str());
        return 1;
    }

    const bool flag_existed = std::filesystem::exists(k_stage_timing_flag);
    if (!flag_existed) {
        std::ofstream touch(k_stage_timing_flag);
    }

    Options gw_opt = opt;
    gw_opt.addr = opt.gateway_addr;
    const std::string path = "/v1/models/" + id + "/infer";
    const bool raw = encoding == "raw";
    const std::string body = raw ? bytes : [&bytes] {
        jinq::common::envelope::Request envelope;
        envelope.images.push_back(jinq::common::base64::encode(
            reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size()));
        return jinq::common::envelope::encode(envelope);
    }();
    const std::string media_type = image_media_type(image_path);
    const char* ctype = raw ? media_type.c_str() : "application/json; charset=utf-8";

    auto send = [&]() { return http_request(gw_opt, "POST", path, body, 300000, ctype); };

    int failures = 0;
    for (size_t i = 0; i < warmup; ++i) {
        const HttpResult r = send();
        if (r.status < 200 || r.status >= 300) {
            std::fprintf(stderr, "warmup request failed (status %d): %s\n", r.status, r.body.substr(0, 200).c_str());
            if (!flag_existed) {
                std::remove(k_stage_timing_flag);
            }
            return 1;
        }
    }

    // snapshot AFTER warmup so warmup trace lines stay out of the window
    const size_t model_from = count_log_lines(model_log);
    const size_t gw_from = std::filesystem::exists(gw_log) ? count_log_lines(gw_log) : 0;

    std::vector<double> latency_ms;
    latency_ms.reserve(requests);
    for (size_t i = 0; i < requests; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        const HttpResult r = send();
        const auto t1 = std::chrono::steady_clock::now();
        if (r.status < 200 || r.status >= 300) {
            std::fprintf(stderr, "request %zu failed (status %d): %s\n", i, r.status, r.body.substr(0, 200).c_str());
            ++failures;
            if (failures > 3) {
                if (!flag_existed) {
                    std::remove(k_stage_timing_flag);
                }
                return 1;
            }
            continue;
        }
        latency_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    if (!flag_existed && !keep_flag) {
        std::remove(k_stage_timing_flag);
    }

    StageMap model, gw;
    size_t model_traces = 0;
    size_t gw_traces = 0;
    parse_stage_window(model_log, model_from, &model, &model_traces);
    parse_stage_window(gw_log, gw_from, &gw, &gw_traces);
    if (model_traces == 0) {
        std::fprintf(stderr,
                     "no stage_trace lines found in %s - the running server binary lacks stage "
                     "tracing (perf/stage-timing instrumentation)\n",
                     model_log.c_str());
        return 1;
    }

    const double client_p50 = percentile(latency_ms, 0.5);
    const double client_sum = std::accumulate(latency_ms.begin(), latency_ms.end(), 0.0);
    const double client_mean = latency_ms.empty() ? -1.0 : client_sum / static_cast<double>(latency_ms.size());
    const double gw_total_p50 = stage_p50(gw, "total");
    const double model_total_p50 = stage_p50(model, "total");
    const double gw_total_mean = stage_mean(gw, "total");
    const double model_total_mean = stage_mean(model, "total");
    const double shell1_p50 = (gw_total_p50 > 0 && client_p50 > 0) ? client_p50 - gw_total_p50 : -1;
    const double shell3_p50 = (gw_total_p50 > 0 && model_total_p50 > 0) ? gw_total_p50 - model_total_p50 : -1;
    const double shell1_mean = (gw_total_mean > 0) ? client_mean - gw_total_mean : -1;
    const double shell3_mean = (gw_total_mean > 0 && model_total_mean > 0) ? gw_total_mean - model_total_mean : -1;

    auto pct_of = [&client_p50](double v) -> std::string {
        if (client_p50 <= 0 || v < 0) {
            return "    --";
        }
        char p[16];
        std::snprintf(p, sizeof p, "%5.1f%%", v / client_p50 * 100.0);
        return std::string(p);
    };

    struct Row {
        const char* idx;
        std::string name;
        double p50;
        double mean;
        std::string note;
    };
    char label[192];
    std::snprintf(label, sizeof label, "route %.3f / auth %.3f / fwd %.3f", stage_p50(gw, "route"),
                  stage_p50(gw, "auth"), stage_p50(gw, "fwd"));
    const std::string note2(label);
    std::snprintf(label, sizeof label, "env %.3f / admit %.3f / go %.3f / wkr %.3f / rpl %.3f / ser %.3f",
                  stage_p50(model, "envelope"), stage_p50(model, "admit"), stage_p50(model, "go"),
                  stage_p50(model, "worker"), stage_p50(model, "reply"), stage_p50(model, "serialize"));
    const std::string note4(label);
    const double row2 = sum_p50(gw, {"route", "auth", "fwd"});
    const double row2_mean = stage_mean(gw, "route") + stage_mean(gw, "auth") + stage_mean(gw, "fwd");
    const double row4 = sum_p50(model, {"envelope", "admit", "go", "worker", "reply", "serialize"});
    const double row4_mean = stage_mean(model, "envelope") + stage_mean(model, "admit") + stage_mean(model, "go") +
                             stage_mean(model, "worker") + stage_mean(model, "reply") + stage_mean(model, "serialize");
    const std::vector<Row> rows = {
        {"1", "客户端 + socket① + 网关框架壳（残差）", shell1_p50, shell1_mean, "客户端−网关total"},
        {"2", "网关处理（route+auth+fwd）", row2, row2_mean, note2},
        {"3", "socket② 往返 + 网关回调（残差）", shell3_p50, shell3_mean, "网关total−模型total"},
        {"4", "模型框架与调度", row4, row4_mean, note4},
        {"5", "decode（图片解码）", stage_p50(model, "decode"), stage_mean(model, "decode"), "libjpeg-turbo"},
        {"6", "pre（letterbox+归一化）", stage_p50(model, "pre"), stage_mean(model, "pre"), ""},
        {"7", "h2d（输入上传）", stage_p50(model, "h2d"), stage_mean(model, "h2d"), ""},
        {"8", "exec（GPU 推理+sync）", stage_p50(model, "exec"), stage_mean(model, "exec"), "共享GPU噪声敏感"},
        {"9", "d2h（输出回传）", stage_p50(model, "d2h"), stage_mean(model, "d2h"), ""},
        {"10", "post（解码+NMS+坐标还原）", stage_p50(model, "post"), stage_mean(model, "post"), ""},
    };

    std::printf("%smortred profile%s %s  n=%zu ok=%zu warmup=%zu encoding=%s image=%s\n", col_bold(),
                col_off(), id.c_str(), requests, latency_ms.size(), warmup, encoding.c_str(),
                image_path.c_str());
    std::printf("%s %s %s %s %s  %s\n", pad_display("#", 4).c_str(), pad_display("环节", 52).c_str(),
                pad_display("P50(ms)", 9).c_str(), pad_display("mean(ms)", 9).c_str(),
                pad_display("占比", 7).c_str(), "说明");
    for (const auto& row : rows) {
        std::printf("%s %s %s %s %s  %s\n", pad_display(row.idx, 4).c_str(), pad_display(row.name, 52).c_str(),
                    fmt_ms(row.p50, 9).c_str(), fmt_ms(row.mean, 9).c_str(), pad_display(pct_of(row.p50), 7).c_str(),
                    row.note.c_str());
    }
    std::printf("%s %s %s %s %s  %s\n", pad_display("", 4).c_str(), pad_display("客户端合计（实测）", 52).c_str(),
                fmt_ms(client_p50, 9, 2).c_str(), fmt_ms(client_mean, 9, 2).c_str(), pad_display("100.0%", 7).c_str(),
                "");
    std::printf("\ntraces: model=%zu gateway=%zu (requests=%zu)", model_traces, gw_traces, latency_ms.size());
    if (model_traces != latency_ms.size() || gw_traces != latency_ms.size()) {
        std::printf("  %s[warn] window polluted (concurrent traffic?)%s", col_warn(), col_off());
    }
    if (keep_flag || flag_existed) {
        std::printf("  (trace flag left ON)");
    }
    std::printf("\n");
    return failures == 0 ? 0 : 1;
}

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
    } else if (const char* env = std::getenv("MORTRED_API_TOKEN"); env != nullptr && *env != '\0') {
        opt.token = env;  // same token the console uses; MORTREDCTL_TOKEN wins if both set
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
        cmd == "calibrate" || cmd == "init-trust" || cmd == "init-edge" ||
        cmd == "next") {
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
    if (cmd == "ps") {
        return cmd_ps(opt);
    }
    if (cmd == "down") {
        std::vector<std::string> rest(args.begin() + static_cast<long>(index), args.end());
        return cmd_down(opt, rest);
    }
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
    } else if (cmd == "profile") {
        const std::string id = next("server id");
        std::string image_path;
        size_t requests = 200;
        size_t warmup = 10;
        std::string encoding = "raw";
        bool keep_flag = false;
        std::string log_dir;
        while (index < args.size()) {
            const std::string flag = next("flag");
            if (flag == "--image") {
                image_path = next("--image");
            } else if (flag == "--n") {
                requests = static_cast<size_t>(std::stoull(next("--n")));
            } else if (flag == "--warmup") {
                warmup = static_cast<size_t>(std::stoull(next("--warmup")));
            } else if (flag == "--encoding") {
                encoding = next("--encoding");
                if (encoding != "raw" && encoding != "json") {
                    std::fprintf(stderr, "--encoding must be raw|json\n");
                    return 2;
                }
            } else if (flag == "--keep-flag") {
                keep_flag = true;
            } else if (flag == "--log-dir") {
                log_dir = next("--log-dir");
            } else {
                std::fprintf(stderr, "unknown profile flag: %s\n", flag.c_str());
                return 2;
            }
        }
        if (image_path.empty()) {
            std::fprintf(stderr, "profile requires --image <path>\n");
            return 2;
        }
        return cmd_profile(opt, id, image_path, requests, warmup, encoding, keep_flag, log_dir);
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
