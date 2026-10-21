// Phase-6 pre-flight: where does the envelope stage actually go for a
// 1.6MB base64 body? Measures the current pipeline (body copy -> doc parse
// with string copies -> vector<string> images -> base64 decode) against the
// candidate (owned buffer -> ParseInsitu -> views -> base64 decode from view).
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <rapidjson/document.h>

#include "common/base64.h"
#include "common/request_envelope.h"

using mortred::common::envelope::Request;

static std::string read_file(const char *path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::printf("usage: %s <image-file> [reps]\n", argv[0]);
        return 1;
    }
    const std::string image = read_file(argv[1]);
    const int reps = argc > 2 ? std::atoi(argv[2]) : 200;
    const std::string b64 = mortred::common::base64::encode(image.data(), image.size());
    std::string body;
    body.reserve(b64.size() + 64);
    body += "{\"images\":[\"";
    body += b64;
    body += "\"]}";
    std::printf("image=%zu bytes, body=%zu bytes\n", image.size(), body.size());

    double t_chunk = 0, t_parse = 0, t_images = 0, t_decode = 0;
    double t_insitu = 0, t_insitu_decode = 0;
    size_t check_current = 0, check_insitu = 0;
    for (int r = 0; r < reps; ++r) {
        // current pipeline, stage by stage
        double t0 = now_ms();
        std::string req_body = body;  // stands in for decode_chunked_body's copy
        t_chunk += now_ms() - t0;

        t0 = now_ms();
        rapidjson::Document doc;
        doc.Parse(req_body.data(), req_body.size());
        const rapidjson::Value &images = doc["images"];
        t_parse += now_ms() - t0;

        t0 = now_ms();
        Request req;
        for (rapidjson::SizeType i = 0; i < images.Size(); ++i) {
            req.images.emplace_back(images[i].GetString(), images[i].GetStringLength());
        }
        t_images += now_ms() - t0;

        t0 = now_ms();
        for (const auto &img : req.images) {
            auto decoded = mortred::common::base64::decode(img);
            check_current += decoded.size();
        }
        t_decode += now_ms() - t0;

        // candidate: one owned mutable buffer, in-situ parse, views, decode
        t0 = now_ms();
        std::vector<char> arena(body.begin(), body.end());
        arena.push_back('\0');
        rapidjson::Document doc2;
        doc2.ParseInsitu(arena.data());
        const rapidjson::Value &images2 = doc2["images"];
        t_insitu += now_ms() - t0;

        t0 = now_ms();
        for (rapidjson::SizeType i = 0; i < images2.Size(); ++i) {
            const std::string_view view(images2[i].GetString(),
                                        static_cast<size_t>(images2[i].GetStringLength()));
            auto decoded = mortred::common::base64::decode(view);
            check_insitu += decoded.size();
        }
        t_insitu_decode += now_ms() - t0;
    }
    std::printf("current : chunk_copy=%.3fms parse=%.3fms images_copy=%.3fms b64_decode=%.3fms | sum=%.3fms\n",
                t_chunk / reps, t_parse / reps, t_images / reps, t_decode / reps,
                (t_chunk + t_parse + t_images + t_decode) / reps);
    std::printf("insitu  : parse=%.3fms (arena copy incl) b64_decode=%.3fms | sum=%.3fms\n",
                t_insitu / reps, t_insitu_decode / reps,
                (t_insitu + t_insitu_decode) / reps);
    std::printf("decoded bytes equal: %s\n", check_current == check_insitu ? "yes" : "NO");
    return check_current == check_insitu ? 0 : 2;
}
