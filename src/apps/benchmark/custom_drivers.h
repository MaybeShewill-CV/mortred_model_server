/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: custom_drivers.h
 * Date: 26-9-3
 ************************************************/

#ifndef MORTRED_APPS_BENCHMARK_CUSTOM_DRIVERS_H
#define MORTRED_APPS_BENCHMARK_CUSTOM_DRIVERS_H

#include <string>

namespace jinq {
namespace apps {
namespace benchmark {

int run_openai_clip_benchmark(int argc, char **argv);
int run_sam_predictor_benchmark(int argc, char **argv);
int run_fast_sam_benchmark(int argc, char **argv);
int run_lightglue_benchmark(int argc, char **argv);
int run_byte_track_benchmark(int argc, char **argv);
int run_diffusion_family_benchmark(const std::string &model_section, int argc, char **argv);

} // namespace benchmark
} // namespace apps
} // namespace jinq

#endif // MORTRED_APPS_BENCHMARK_CUSTOM_DRIVERS_H
