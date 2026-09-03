/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: model_benchmark_main.cpp
 * Date: 26-9-3
 ************************************************/

#include <cstdio>
#include <string>

#include <glog/logging.h>

#include "apps/cli_flags.h"
#include "apps/product_index.h"

namespace {

void print_usage(const char *exe) {
    std::fprintf(stderr,
                 "usage:\n"
                 "  %s --list\n"
                 "  %s --model <ID> <model_config.toml> [image_or_extra_args...]\n"
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
    if (cli.model.empty() || cli.rest_argc() < 2) {
        print_usage(argv[0]);
        return -1;
    }

    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    google::SetStderrLogging(google::GLOG_INFO);
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;

    const auto *entry = jinq::apps::ProductIndex::find(cli.model);
    if (entry == nullptr) {
        LOG(ERROR) << "unknown --model '" << cli.model << "' (see --list)";
        return -1;
    }
    if (!entry->benchmark || !entry->run_benchmark) {
        LOG(ERROR) << "model '" << cli.model << "' has no benchmark driver";
        return -1;
    }
    return entry->run_benchmark(cli.rest_argc(), cli.rest.data());
}
