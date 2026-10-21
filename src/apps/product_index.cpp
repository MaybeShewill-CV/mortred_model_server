/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: product_index.cpp
 * Date: 26-9-3
 ************************************************/

#include "apps/product_index.h"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <set>
#include <utility>

#ifdef MORTRED_WITH_CUSTOM_DRIVERS
#include "apps/benchmark/custom_drivers.h"
#endif
#include "apps/benchmark/family_hooks.h"
#include "apps/benchmark/image_family.h"
#include "factory/classification_task.h"
#ifdef MORTRED_WITH_CUSTOM_DRIVERS
#include "factory/clip_task.h"
#endif
#include "factory/cv_catalog.h"
#include "factory/diffusion_task.h"
#include "factory/enhancement_task.h"
#include "factory/feature_embedding_task.h"
#include "factory/feature_point_task.h"
#include "factory/matting_task.h"
#include "factory/model_catalog.h"
#include "factory/mono_depth_estimate_task.h"
#include "factory/obj_detection_task.h"
#include "factory/ocr_task.h"
#include "factory/sam_task.h"
#include "factory/scene_segmentation_task.h"

namespace mortred {
namespace apps {
namespace {

using mortred::apps::benchmark::run_image_family_benchmark;
using mortred::factory::cv_catalog::CvModelEntry;
using mortred::factory::model_catalog::ModelCatalogEntry;

void push_unique(std::vector<ProductEntry> *out, std::set<std::string> *seen, ProductEntry entry) {
    if (!seen->insert(entry.id).second) {
        std::fprintf(stderr, "[product_index] duplicate catalog id ignored: %s\n", entry.id.c_str());
        return;
    }
    out->push_back(std::move(entry));
}

template <typename OUTPUT>
void project_http(std::vector<ProductEntry> *out, std::set<std::string> *seen, const char *family,
                  const std::vector<CvModelEntry<OUTPUT>> &catalog,
                  mortred::apps::benchmark::ImageFamilyHooks<OUTPUT> hooks) {
    for (const auto &entry : catalog) {
        ProductEntry product;
        product.id = entry.model_section;
        product.family = family;
        product.display_name = entry.display_name;
        product.server_section = entry.server_section;
        product.http = entry.fill_response != nullptr && !entry.server_section.empty();
        product.benchmark = static_cast<bool>(entry.make_worker);
        const auto captured = entry;
        if (product.http) {
            product.make_server = [captured](const std::string &name) {
                return mortred::factory::cv_catalog::create_server(captured, name);
            };
        }
        if (product.benchmark) {
            auto model_hooks = hooks;
            product.run_benchmark = [captured, model_hooks, family](int argc, char **argv) mutable {
                const std::string &weight = captured.family.empty() ? captured.model_section : captured.family;
                if (family == std::string("enhancement")) {
                    if (weight == "enlightengan") {
                        model_hooks.default_image =
                            "../demo_data/model_test_input/enhancement/low_light/lol_test_1.png";
                    } else if (weight == "realesrgan") {
                        model_hooks.default_image =
                            "../demo_data/model_test_input/enhancement/real_esr/test.jpg";
                    }
                }
                if (family == std::string("scene_segmentation")) {
                    if (weight == "pphuman") {
                        model_hooks.default_image =
                            "../demo_data/model_test_input/scene_segmentation/human_image.jpg";
                    } else if (weight == "hrnet") {
                        model_hooks.default_image =
                            "../demo_data/model_test_input/scene_segmentation/00000_266092.jpg";
                    }
                }
                return run_image_family_benchmark<OUTPUT>(captured.model_section, captured.display_name,
                                                          captured.make_worker, model_hooks, argc, argv);
            };
        }
        push_unique(out, seen, std::move(product));
    }
}

template <typename INPUT, typename OUTPUT, typename Driver>
void project_bench_only(std::vector<ProductEntry> *out, std::set<std::string> *seen, const char *family,
                        const std::vector<ModelCatalogEntry<INPUT, OUTPUT>> &catalog, Driver driver) {
    for (const auto &entry : catalog) {
        ProductEntry product;
        product.id = entry.model_section;
        product.family = family;
        product.display_name = entry.display_name;
        product.http = false;
        product.benchmark = static_cast<bool>(entry.make_model);
        const auto captured = entry;
        if (product.benchmark) {
            product.run_benchmark = [captured, driver](int argc, char **argv) {
                return driver(captured, argc, argv);
            };
        }
        push_unique(out, seen, std::move(product));
    }
}

std::vector<ProductEntry> build_index() {
    std::vector<ProductEntry> out;
    std::set<std::string> seen;

    project_http(&out, &seen, "classification", mortred::factory::classification::catalog(),
                 mortred::apps::benchmark::classification_hooks());
    project_http(&out, &seen, "object_detection", mortred::factory::object_detection::catalog(),
                 mortred::apps::benchmark::object_detection_hooks());
    project_http(&out, &seen, "face_detection", mortred::factory::object_detection::face_catalog(),
                 mortred::apps::benchmark::face_detection_hooks());
    project_http(&out, &seen, "scene_segmentation", mortred::factory::scene_segmentation::catalog(),
                 mortred::apps::benchmark::scene_segmentation_hooks());
    project_http(&out, &seen, "ocr", mortred::factory::ocr::catalog(), mortred::apps::benchmark::ocr_hooks());
    project_http(&out, &seen, "matting", mortred::factory::matting::catalog(), mortred::apps::benchmark::matting_hooks());
    project_http(&out, &seen, "enhancement", mortred::factory::enhancement::catalog(),
                 mortred::apps::benchmark::enhancement_hooks());
    project_http(&out, &seen, "feature_point", mortred::factory::feature_point::catalog(),
                 mortred::apps::benchmark::feature_point_hooks());
    project_http(&out, &seen, "feature_embedding", mortred::factory::feature_embedding::catalog(),
                 mortred::apps::benchmark::feature_embedding_hooks());
    project_http(&out, &seen, "mono_depth_estimation", mortred::factory::mono_depth_estimation::catalog(),
                 mortred::apps::benchmark::mono_depth_hooks());
    project_http(&out, &seen, "segment_anything", mortred::factory::segment_anything::amg_catalog(),
                 mortred::apps::benchmark::sam_amg_hooks());

    for (const auto &entry : mortred::factory::diffusion::catalog()) {
        ProductEntry product;
        product.id = entry.model_section;
        product.family = "diffusion";
        product.display_name = entry.display_name;
        product.server_section = entry.server_section;
        product.http = true;
#ifdef MORTRED_WITH_CUSTOM_DRIVERS
        product.benchmark = true;
#endif
        const auto captured = entry;
        product.make_server = [captured](const std::string &name) {
            return mortred::factory::cv_catalog::create_server(captured, name);
        };
#ifdef MORTRED_WITH_CUSTOM_DRIVERS
        product.run_benchmark = [id = entry.model_section](int argc, char **argv) {
            return mortred::apps::benchmark::run_diffusion_family_benchmark(id, argc, argv);
        };
#endif
        push_unique(&out, &seen, std::move(product));
    }

#ifdef MORTRED_WITH_CUSTOM_DRIVERS
    project_bench_only(&out, &seen, "scene_segmentation", mortred::factory::scene_segmentation::bench_catalog(),
                       [](const auto &entry, int argc, char **argv) {
                           return run_image_family_benchmark<mortred::apps::benchmark::SegOutput>(
                               entry.model_section, entry.display_name, entry.make_model,
                               mortred::apps::benchmark::scene_segmentation_hooks(), argc, argv);
                       });

    project_bench_only(&out, &seen, "clip", mortred::factory::clip::catalog(),
                       [](const auto &, int argc, char **argv) {
                           return mortred::apps::benchmark::run_openai_clip_benchmark(argc, argv);
                       });
    project_bench_only(&out, &seen, "segment_anything", mortred::factory::segment_anything::predictor_catalog(),
                       [](const auto &, int argc, char **argv) {
                           return mortred::apps::benchmark::run_sam_predictor_benchmark(argc, argv);
                       });
    project_bench_only(&out, &seen, "segment_anything", mortred::factory::segment_anything::fast_sam_catalog(),
                       [](const auto &, int argc, char **argv) {
                           return mortred::apps::benchmark::run_fast_sam_benchmark(argc, argv);
                       });
    project_bench_only(&out, &seen, "feature_point", mortred::factory::feature_point::match_catalog(),
                       [](const auto &, int argc, char **argv) {
                           return mortred::apps::benchmark::run_lightglue_benchmark(argc, argv);
                       });

    ProductEntry mot;
    mot.id = "BYTE_TRACK";
    mot.family = "mot";
    mot.display_name = "ByteTrack multi-object tracker";
    mot.http = false;
    mot.benchmark = true;
    mot.run_benchmark = [](int argc, char **argv) {
        return mortred::apps::benchmark::run_byte_track_benchmark(argc, argv);
    };
    push_unique(&out, &seen, std::move(mot));
#endif

    std::sort(out.begin(), out.end(), [](const ProductEntry &a, const ProductEntry &b) { return a.id < b.id; });
    return out;
}

} // namespace

const std::vector<ProductEntry> &ProductIndex::all() {
    static const std::vector<ProductEntry> entries = build_index();
    return entries;
}

const ProductEntry *ProductIndex::find(const std::string &id) {
    for (const auto &entry : all()) {
        if (entry.id == id) {
            return &entry;
        }
    }
    return nullptr;
}

void ProductIndex::print_list(std::FILE *out) {
    if (out == nullptr) {
        out = stdout;
    }
    std::fprintf(out, "%-18s %-22s %-11s %s\n", "id", "family", "surface", "display_name");
    for (const auto &entry : all()) {
        const char *surface = entry.http ? "http" : "bench-only";
        std::fprintf(out, "%-18s %-22s %-11s %s\n", entry.id.c_str(), entry.family.c_str(), surface,
                     entry.display_name.c_str());
    }
}

} // namespace apps
} // namespace mortred
