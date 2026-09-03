/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: cli_flags.h
 * Date: 26-9-3
 ************************************************/

#ifndef MORTRED_APPS_CLI_FLAGS_H
#define MORTRED_APPS_CLI_FLAGS_H

#include <cstdlib>
#include <string>
#include <vector>

namespace jinq {
namespace apps {

/***
 * Shared `--list` / `--model` parsing for the unified server and benchmark
 * entrypoints. Remaining positional arguments (config path, optional image,
 * diffusion knobs, `--batch` / `--loops`) stay in `rest` so family drivers
 * and `run_benchmark` keep seeing their original argv shape.
 *
 * CLI `--model` wins over the `MORTRED_MODEL` environment fallback.
 */
struct ModelCli {
    bool list = false;
    bool help = false;
    std::string model;
    std::vector<char *> rest;

    int rest_argc() const {
        return static_cast<int>(rest.size());
    }
};

inline bool parse_model_cli(int argc, char **argv, ModelCli *out, std::string *err) {
    if (out == nullptr) {
        return false;
    }
    *out = ModelCli{};
    if (argc < 1 || argv == nullptr) {
        if (err != nullptr) {
            *err = "empty argv";
        }
        return false;
    }
    out->rest.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--list") {
            out->list = true;
            continue;
        }
        if (arg == "--help" || arg == "-h") {
            out->help = true;
            continue;
        }
        if (arg == "--model") {
            if (i + 1 >= argc) {
                if (err != nullptr) {
                    *err = "--model requires a catalog id";
                }
                return false;
            }
            out->model = argv[++i];
            continue;
        }
        if (arg.rfind("--model=", 0) == 0) {
            out->model = arg.substr(std::string("--model=").size());
            continue;
        }
        out->rest.push_back(argv[i]);
    }
    if (out->model.empty()) {
        if (const char *env = std::getenv("MORTRED_MODEL"); env != nullptr && *env != '\0') {
            out->model = env;
        }
    }
    return true;
}

} // namespace apps
} // namespace jinq

#endif // MORTRED_APPS_CLI_FLAGS_H
