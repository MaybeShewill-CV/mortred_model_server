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

namespace mortred {
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
    bool version = false;
    std::string model;
    std::vector<char *> rest;

    int rest_argc() const {
        return static_cast<int>(rest.size());
    }
};

/*** One catalog id, given exactly once: an empty value is a typo, and two
 * different ids mean the operator expected the last one to win. */
inline bool set_model(const std::string &value, ModelCli *out, std::string *err) {
    if (value.empty()) {
        if (err != nullptr) {
            *err = "--model requires a non-empty catalog id (see --list)";
        }
        return false;
    }
    if (!out->model.empty() && out->model != value) {
        if (err != nullptr) {
            *err = "--model given twice ('" + out->model + "' then '" + value + "')";
        }
        return false;
    }
    out->model = value;
    return true;
}

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
        if (arg == "--") {
            // everything after the separator belongs to the family driver
            // verbatim, even if it looks like a flag we know
            for (++i; i < argc; ++i) {
                out->rest.push_back(argv[i]);
            }
            break;
        }
        if (arg == "--list") {
            out->list = true;
            continue;
        }
        if (arg == "--help" || arg == "-h") {
            out->help = true;
            continue;
        }
        if (arg == "--version" || arg == "-v") {
            out->version = true;
            continue;
        }
        if (arg == "--model") {
            if (i + 1 >= argc) {
                if (err != nullptr) {
                    *err = "--model requires a catalog id";
                }
                return false;
            }
            if (!set_model(argv[++i], out, err)) {
                return false;
            }
            continue;
        }
        if (arg.rfind("--model=", 0) == 0) {
            if (!set_model(arg.substr(std::string("--model=").size()), out, err)) {
                return false;
            }
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
} // namespace mortred

#endif // MORTRED_APPS_CLI_FLAGS_H
