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
#include "apps/common/exit_codes.h"
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
    mortred::apps::ModelCli cli;
    std::string err;
    if (!mortred::apps::parse_model_cli(argc, argv, &cli, &err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        print_usage(argv[0]);
        return mortred::apps::kExitUsage;
    }
    if (cli.help) {
        print_usage(argv[0]);
        return 0;
    }
    const auto &duplicate_ids = mortred::apps::ProductIndex::duplicate_ids();
    if (!duplicate_ids.empty()) {
        std::fprintf(stderr, "duplicate catalog id(s):");
        for (const auto &id : duplicate_ids) {
            std::fprintf(stderr, " %s", id.c_str());
        }
        std::fprintf(stderr, "\n");
        return mortred::apps::kExitConfig;
    }
    if (cli.list) {
        mortred::apps::ProductIndex::print_list(stdout);
        return 0;
    }
    if (cli.model.empty() || cli.rest_argc() != 2) {
        print_usage(argv[0]);
        return mortred::apps::kExitUsage;
    }

    const auto *entry = mortred::apps::ProductIndex::find(cli.model);
    if (entry == nullptr) {
        std::fprintf(stderr, "unknown --model '%s' (see --list)\n", cli.model.c_str());
        return mortred::apps::kExitUsage;
    }
    if (!entry->http || !entry->make_server) {
        std::fprintf(stderr, "model '%s' is not served over HTTP (bench-only; see --list)\n",
                     cli.model.c_str());
        return mortred::apps::kExitUsage;
    }

    const std::string config_path = cli.rest[1];
    auto parsed = toml::parse_file(config_path);
    if (!parsed) {
        std::fprintf(stderr, "parse toml config file failed: %s\n",
                     std::string(parsed.error().description()).c_str());
        return mortred::apps::kExitConfig;
    }
    const auto config = std::move(parsed).table();
    std::string model_alias;
    if (mortred::apps::find_section_ci(config, cli.model, &model_alias) == nullptr) {
        std::fprintf(stderr, "--model %s does not match a [%s] table in %s\n", cli.model.c_str(),
                     cli.model.c_str(), config_path.c_str());
        return mortred::apps::kExitConfig;
    }
    if (model_alias != cli.model) {
        std::fprintf(stderr,
                     "note: config table [%s] matches --model %s case-insensitively\n",
                     model_alias.c_str(), cli.model.c_str());
    }
    std::string section_alias;
    const auto* server_tbl =
        mortred::apps::find_section_ci(config, entry->server_section, &section_alias);
    if (server_tbl == nullptr) {
        std::fprintf(stderr, "server section [%s] missing from %s\n",
                     entry->server_section.c_str(), config_path.c_str());
        return mortred::apps::kExitConfig;
    }
    const auto declared = server_tbl->get("model");
    if (declared != nullptr) {
        if (auto declared_value = declared->as_string(); declared_value != nullptr) {
            const std::string declared_name = declared_value->get();
            if (declared_name != cli.model) {
                std::fprintf(stderr, "config model=\"%s\" does not match --model %s\n",
                             declared_name.c_str(), cli.model.c_str());
                return mortred::apps::kExitConfig;
            }
        }
    }

    return mortred::apps::run_model_server_main(cli.rest_argc(), cli.rest.data(), entry->server_section,
                                             entry->make_server);
}
