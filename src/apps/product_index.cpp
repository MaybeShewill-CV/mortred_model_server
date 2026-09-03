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

#include "apps/benchmark/custom_drivers.h"
#include "apps/benchmark/family_hooks.h"
#include "apps/benchmark/image_family.h"
#include "factory/classification_task.h"
#include "factory/clip_task.h"
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

namespace jinq {
namespace apps {
namespace {

using jinq::apps::benchmark::run_image_family_benchmark;
using jinq::factory::cv_catalog::CvModelEntry;
using jinq::factory::model_catalog::ModelCatalogEntry;

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
                  jinq::apps::benchmark::ImageFamilyHooks<OUTPUT> hooks) {
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
                return jinq::factory::cv_catalog::create_server(captured, name);
            };
        }
        if (product.benchmark) {
            auto model_hooks = hooks;
            product.run_benchmark = [captured, model_hooks, family](int argc, char **argv) mutable {
                if (family == std::string("enhancement")) {
                    if (captured.model_section == "ENLIGHTEN_GAN") {
                        model_hooks.default_image =
                            "../demo_data/model_test_input/enhancement/low_light/lol_test_1.png";
                    } else if (captured.model_section == "REAL_ESRGAN") {
                        model_hooks.default_image =
                            "../demo_data/model_test_input/enhancement/real_esr/test.jpg";
                    }
                }
                if (family == std::string("scene_segmentation")) {
                    if (captured.model_section == "PPHUMAN_SEG") {
                        model_hooks.default_image =
                            "../demo_data/model_test_input/scene_segmentation/human_image.jpg";
                    } else if (captured.model_section == "HRNET") {
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

    project_http(&out, &seen, "classification", jinq::factory::classification::catalog(),
                 jinq::apps::benchmark::classification_hooks());
    project_http(&out, &seen, "object_detection", jinq::factory::object_detection::catalog(),
                 jinq::apps::benchmark::object_detection_hooks());
    project_http(&out, &seen, "face_detection", jinq::factory::object_detection::face_catalog(),
                 jinq::apps::benchmark::face_detection_hooks());
    project_http(&out, &seen, "scene_segmentation", jinq::factory::scene_segmentation::catalog(),
                 jinq::apps::benchmark::scene_segmentation_hooks());
    project_http(&out, &seen, "ocr", jinq::factory::ocr::catalog(), jinq::apps::benchmark::ocr_hooks());
    project_http(&out, &seen, "matting", jinq::factory::matting::catalog(), jinq::apps::benchmark::matting_hooks());
    project_http(&out, &seen, "enhancement", jinq::factory::enhancement::catalog(),
                 jinq::apps::benchmark::enhancement_hooks());
    project_http(&out, &seen, "feature_point", jinq::factory::feature_point::catalog(),
                 jinq::apps::benchmark::feature_point_hooks());
    project_http(&out, &seen, "feature_embedding", jinq::factory::feature_embedding::catalog(),
                 jinq::apps::benchmark::feature_embedding_hooks());
    project_http(&out, &seen, "mono_depth_estimation", jinq::factory::mono_depth_estimation::catalog(),
                 jinq::apps::benchmark::mono_depth_hooks());
    project_http(&out, &seen, "segment_anything", jinq::factory::segment_anything::amg_catalog(),
                 jinq::apps::benchmark::sam_amg_hooks());

    for (const auto &entry : jinq::factory::diffusion::catalog()) {
        ProductEntry product;
        product.id = entry.model_section;
        product.family = "diffusion";
        product.display_name = entry.display_name;
        product.server_section = entry.server_section;
        product.http = true;
        product.benchmark = true;
        const auto captured = entry;
        product.make_server = [captured](const std::string &name) {
            return jinq::factory::cv_catalog::create_server(captured, name);
        };
        product.run_benchmark = [id = entry.model_section](int argc, char **argv) {
            return jinq::apps::benchmark::run_diffusion_family_benchmark(id, argc, argv);
        };
        push_unique(&out, &seen, std::move(product));
    }

    project_bench_only(&out, &seen, "scene_segmentation", jinq::factory::scene_segmentation::bench_catalog(),
                       [](const auto &entry, int argc, char **argv) {
                           return run_image_family_benchmark<jinq::apps::benchmark::SegOutput>(
                               entry.model_section, entry.display_name, entry.make_model,
                               jinq::apps::benchmark::scene_segmentation_hooks(), argc, argv);
                       });

    project_bench_only(&out, &seen, "clip", jinq::factory::clip::catalog(),
                       [](const auto &, int argc, char **argv) {
                           return jinq::apps::benchmark::run_openai_clip_benchmark(argc, argv);
                       });
    project_bench_only(&out, &seen, "segment_anything", jinq::factory::segment_anything::predictor_catalog(),
                       [](const auto &, int argc, char **argv) {
                           return jinq::apps::benchmark::run_sam_predictor_benchmark(argc, argv);
                       });
    project_bench_only(&out, &seen, "segment_anything", jinq::factory::segment_anything::fast_sam_catalog(),
                       [](const auto &, int argc, char **argv) {
                           return jinq::apps::benchmark::run_fast_sam_benchmark(argc, argv);
                       });
    project_bench_only(&out, &seen, "feature_point", jinq::factory::feature_point::match_catalog(),
                       [](const auto &, int argc, char **argv) {
                           return jinq::apps::benchmark::run_lightglue_benchmark(argc, argv);
                       });

    ProductEntry mot;
    mot.id = "BYTE_TRACK";
    mot.family = "mot";
    mot.display_name = "ByteTrack multi-object tracker";
    mot.http = false;
    mot.benchmark = true;
    mot.run_benchmark = [](int argc, char **argv) {
        return jinq::apps::benchmark::run_byte_track_benchmark(argc, argv);
    };
    push_unique(&out, &seen, std::move(mot));

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
} // namespace jinq
