/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: model_server_main.cpp
 * Date: 26-9-3
 ************************************************/

#include <cstdio>
#include <string>

#include <glog/logging.h>
#include <toml/toml.hpp>

#include "apps/cli_flags.h"
#include "apps/common/model_server_main.h"
#include "apps/product_index.h"

namespace {

void print_usage(const char *exe) {
    std::fprintf(stderr,
                 "usage:\n"
                 "  %s --list\n"
                 "  %s --model <ID> <server_config.toml>\n"
                 "\n"
                 "  --model is case-sensitive and must match a catalog model_section.\n"
                 "  MORTRED_MODEL may supply the id when --model is omitted.\n",
                 exe, exe);
}

} // namespace

int main(int argc, char **argv) {
    jinq::apps::ModelCli cli;
    std::string err;
    if (!jinq::apps::parse_model_cli(argc, argv, &cli, &err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        print_usage(argv[0]);
        return -1;
    }
    if (cli.help) {
        print_usage(argv[0]);
        return 0;
    }
    if (cli.list) {
        jinq::apps::ProductIndex::print_list(stdout);
        return 0;
    }
    if (cli.model.empty() || cli.rest_argc() != 2) {
        print_usage(argv[0]);
        return -1;
    }

    const auto *entry = jinq::apps::ProductIndex::find(cli.model);
    if (entry == nullptr) {
        std::fprintf(stderr, "unknown --model '%s' (see --list)\n", cli.model.c_str());
        return -1;
    }
    if (!entry->http || !entry->make_server) {
        std::fprintf(stderr, "model '%s' is not served over HTTP (bench-only; see --list)\n",
                     cli.model.c_str());
        return -1;
    }

    const std::string config_path = cli.rest[1];
    auto parsed = toml::parse_file(config_path);
    if (!parsed) {
        std::fprintf(stderr, "parse toml config file failed: %s\n",
                     std::string(parsed.error().description()).c_str());
        return -1;
    }
    const auto config = std::move(parsed).table();
    if (!config.contains(cli.model)) {
        std::fprintf(stderr, "--model %s does not match a [%s] table in %s\n", cli.model.c_str(),
                     cli.model.c_str(), config_path.c_str());
        return -1;
    }
    if (!entry->server_section.empty() && !config.contains(entry->server_section)) {
        std::fprintf(stderr, "server section [%s] missing from %s\n", entry->server_section.c_str(),
                     config_path.c_str());
        return -1;
    }
    if (config.contains(entry->server_section)) {
        const auto declared = config[entry->server_section]["model"].value<std::string>();
        if (declared && *declared != cli.model) {
            std::fprintf(stderr, "config model=\"%s\" does not match --model %s\n", declared->c_str(),
                         cli.model.c_str());
            return -1;
        }
    }

    return jinq::apps::run_model_server_main(cli.rest_argc(), cli.rest.data(), entry->server_section,
                                             entry->make_server);
}
