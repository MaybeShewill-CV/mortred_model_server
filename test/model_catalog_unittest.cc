#include <algorithm>
#include <filesystem>
#include <map>
#include <set>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <toml/toml.hpp>

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
#include "server/generic_cv_server.h"

namespace {

namespace fs = std::filesystem;

using ServerEntry = mortred::factory::catalog::ServedModelEntry;
using mortred::factory::catalog::model_entry_valid;
using mortred::factory::catalog::served_model_entry_valid;

std::vector<fs::path> collect_toml_files(const fs::path &root) {
    std::vector<fs::path> files;
    std::error_code ec;
    if (!fs::exists(root, ec)) {
        return files;
    }
    for (auto it = fs::recursive_directory_iterator(root, ec); it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (!ec && it->is_regular_file() && it->path().extension() == ".toml") {
            files.push_back(it->path());
        }
    }
    return files;
}

// top-level table sections of one parsed TOML file
std::vector<std::string> top_level_sections(const fs::path &file) {
    std::vector<std::string> sections;
    auto parsed = toml::parse_file(file.string());
    if (!parsed) {
        return sections;
    }
    for (auto &[key, value] : std::move(parsed).table()) {
        if (value.is_table()) {
            sections.emplace_back(key.str());
        }
    }
    return sections;
}

// the server side config refers to model files as "../conf/...": that prefix is
// relative to the server process working directory, so drop it and fall back to
// the server file location when needed
fs::path resolve_model_config(const fs::path &server_file, const std::string &raw_path) {
    auto normalized = fs::path(raw_path).lexically_normal();
    if (!normalized.empty() && normalized.begin()->string() == "..") {
        normalized = normalized.lexically_relative("..");
    }
    std::error_code ec;
    const auto from_root = fs::path(".") / normalized;
    if (fs::exists(from_root, ec)) {
        return fs::canonical(from_root, ec);
    }
    return server_file.parent_path() / raw_path;
}

class ModelCatalogTest : public ::testing::Test {
  protected:
    void SetUp() override {
        register_entries("classification", mortred::factory::classification::catalog());
        register_entries("object_detection", mortred::factory::object_detection::catalog());
        register_entries("face_detection", mortred::factory::object_detection::face_catalog());
        register_entries("scene_segmentation", mortred::factory::scene_segmentation::catalog());
        register_entries("ocr", mortred::factory::ocr::catalog());
        register_entries("matting", mortred::factory::matting::catalog());
        register_entries("enhancement", mortred::factory::enhancement::catalog());
        register_entries("feature_embedding", mortred::factory::feature_embedding::catalog());
        register_entries("feature_point", mortred::factory::feature_point::catalog());
        register_entries("mono_depth_estimation", mortred::factory::mono_depth_estimation::catalog());
        register_entries("diffusion", mortred::factory::diffusion::catalog());
        register_entries("segment_anything_amg", mortred::factory::segment_anything::amg_catalog());
    }

    template <typename OUTPUT>
    void register_entries(const std::string &task, const std::vector<mortred::factory::cv_catalog::CvModelEntry<OUTPUT>> &entries) {
        for (const auto &entry : entries) {
            served_entries_.push_back(&entry);
            entry_tasks_[&entry] = task;
        }
    }

    std::string describe(const ServerEntry &entry) const {
        const auto found = entry_tasks_.find(&entry);
        return (found == entry_tasks_.end() ? std::string("<unknown>") : found->second) + "/" + entry.model_section;
    }

    // constructing the server instantiates CvModelServer<OUTPUT>; init() is
    // intentionally not called
    template <typename OUTPUT>
    void check_create_server(const std::vector<mortred::factory::cv_catalog::CvModelEntry<OUTPUT>> &entries, const std::string &task) {
        for (const auto &entry : entries) {
            const auto label = task + "/" + entry.model_section;
            auto server = mortred::factory::cv_catalog::create_server(entry);
            ASSERT_NE(server, nullptr) << label;
            EXPECT_NE(dynamic_cast<mortred::server::CvModelServer<OUTPUT> *>(server.get()), nullptr) << label;
        }
    }

    std::vector<const ServerEntry *> served_entries_;
    std::map<const ServerEntry *, std::string> entry_tasks_;
};

TEST_F(ModelCatalogTest, ServedEntriesAreFullyPopulated) {
    ASSERT_FALSE(served_entries_.empty());
    for (const auto *entry : served_entries_) {
        ASSERT_TRUE(served_model_entry_valid(*entry)) << describe(*entry);
    }
}

TEST_F(ModelCatalogTest, ModelSectionsAreGloballyUnique) {
    std::map<std::string, int> counts;
    for (const auto *entry : served_entries_) {
        ++counts[entry->model_section];
    }
    for (const auto &[section, count] : counts) {
        ASSERT_EQ(count, 1) << "duplicated model section: " << section;
    }
}

TEST_F(ModelCatalogTest, ServerSectionsResolveToShippedConfigs) {
    // server_section is resolved *inside the config file the model process was
    // started with* (model_server_main.h: config[server_section]), and
    // model_server_main.cpp further requires that section's model = the catalog
    // model_section. Sibling variants therefore legitimately share one server
    // section name (yolov8n/s/l/x all use [YOLOV8_DETECTION_SERVER]), so a
    // global-uniqueness rule is wrong; what must hold is that every factory
    // entry resolves to a shipped conf/server TOML that declares that section
    // for exactly that model.
    std::map<std::string, std::set<std::string>> section_models;
    for (const auto &file : collect_toml_files(fs::path("conf") / "server")) {
        auto parsed = toml::parse_file(file.string());
        if (!parsed) {
            continue;
        }
        for (auto &[key, value] : parsed.table()) {
            const std::string section(key.str());
            if (section.size() <= 7 || section.compare(section.size() - 7, 7, "_SERVER") != 0) {
                continue;
            }
            if (!value.is_table()) {
                continue;
            }
            const auto model = value.as_table()->get("model");
            if (model != nullptr && model->is_string()) {
                section_models[section].insert(model->value<std::string>().value_or(""));
            }
        }
    }
    ASSERT_FALSE(section_models.empty()) << "no *_SERVER section found under conf/server";

    for (const auto *entry : served_entries_) {
        const auto found = section_models.find(entry->server_section);
        ASSERT_NE(found, section_models.end())
            << describe(*entry) << ": server section [" << entry->server_section
            << "] is not declared by any conf/server TOML";
        EXPECT_NE(found->second.count(entry->model_section), 0)
            << describe(*entry) << ": [" << entry->server_section
            << "] never declares model=" << entry->model_section
            << " (declared: " << [&] {
                   std::string all;
                   for (const auto &m : found->second) {
                       all += (all.empty() ? "" : ",") + m;
                   }
                   return all;
               }() << ")";
    }
}

TEST_F(ModelCatalogTest, CreateServerBuildsGenericCvServer) {
    check_create_server(mortred::factory::classification::catalog(), "classification");
    check_create_server(mortred::factory::object_detection::catalog(), "object_detection");
    check_create_server(mortred::factory::object_detection::face_catalog(), "face_detection");
    check_create_server(mortred::factory::scene_segmentation::catalog(), "scene_segmentation");
    check_create_server(mortred::factory::ocr::catalog(), "ocr");
    check_create_server(mortred::factory::matting::catalog(), "matting");
    check_create_server(mortred::factory::enhancement::catalog(), "enhancement");
    check_create_server(mortred::factory::feature_embedding::catalog(), "feature_embedding");
    check_create_server(mortred::factory::feature_point::catalog(), "feature_point");
    check_create_server(mortred::factory::mono_depth_estimation::catalog(), "mono_depth_estimation");
    check_create_server(mortred::factory::diffusion::catalog(), "diffusion");
    check_create_server(mortred::factory::segment_anything::amg_catalog(), "segment_anything_amg");
}

TEST_F(ModelCatalogTest, ServerConfigSectionsAndModelFilesExist) {
    const auto files = collect_toml_files("conf/server");
    ASSERT_FALSE(files.empty()) << "conf/server not found; run tests from the repository root";

    for (const auto *entry : served_entries_) {
        bool server_section_found = false;
        bool model_section_found = false;
        fs::path model_config;
        for (const auto &file : files) {
            const auto sections = top_level_sections(file);
            const auto has_server = std::find(sections.begin(), sections.end(), entry->server_section) != sections.end();
            if (!has_server) {
                continue;
            }
            server_section_found = true;
            const auto has_model = std::find(sections.begin(), sections.end(), entry->model_section) != sections.end();
            if (!has_model) {
                continue;
            }
            model_section_found = true;
            auto parsed = toml::parse_file(file.string());
            ASSERT_TRUE(parsed) << file;
            const auto value = std::move(parsed).table()[entry->model_section]["model_config_file_path"];
            if (const auto *path = value.as_string()) {
                model_config = resolve_model_config(file, path->get());
            }
        }
        ASSERT_TRUE(server_section_found) << describe(*entry) << " missing server section " << entry->server_section;
        ASSERT_TRUE(model_section_found) << describe(*entry) << " missing model section " << entry->model_section << " next to "
                                         << entry->server_section;
        ASSERT_FALSE(model_config.empty()) << describe(*entry) << " has no model_config_file_path";
        std::error_code ec;
        ASSERT_TRUE(fs::exists(model_config, ec)) << describe(*entry) << " model config missing: " << model_config.string();
    }
}

TEST_F(ModelCatalogTest, CatalogCoversEveryServedModel) {
    EXPECT_EQ(mortred::factory::classification::catalog().size(), 3U);
    EXPECT_EQ(mortred::factory::object_detection::catalog().size(), 10U);
    EXPECT_EQ(mortred::factory::object_detection::face_catalog().size(), 2U);
    EXPECT_EQ(mortred::factory::scene_segmentation::catalog().size(), 6U);
    EXPECT_EQ(mortred::factory::ocr::catalog().size(), 1U);
    EXPECT_EQ(mortred::factory::matting::catalog().size(), 5U);
    EXPECT_EQ(mortred::factory::enhancement::catalog().size(), 3U);
    EXPECT_EQ(mortred::factory::feature_embedding::catalog().size(), 3U);
    EXPECT_EQ(mortred::factory::feature_point::catalog().size(), 1U);
    EXPECT_EQ(mortred::factory::mono_depth_estimation::catalog().size(), 5U);
    EXPECT_EQ(mortred::factory::diffusion::catalog().size(), 4U);
    EXPECT_EQ(mortred::factory::segment_anything::amg_catalog().size(), 1U);
    EXPECT_EQ(served_entries_.size(), 44U);
}

TEST_F(ModelCatalogTest, FeatureEmbeddingEntryDeclaresParamSpecs) {
    const auto &entries = mortred::factory::feature_embedding::catalog();
    ASSERT_EQ(entries.size(), 3U);
    const auto found = std::find_if(entries.begin(), entries.end(), [](const auto &entry) {
        return entry.model_section == "dinov2_vitb14";
    });
    ASSERT_NE(found, entries.end());
    const auto &entry = *found;
    EXPECT_EQ(entry.family, "dinov2");
    EXPECT_EQ(entry.server_section, "DINOV2_FEATURE_EMBEDDING_SERVER");

    ASSERT_EQ(entry.param_specs.size(), 2U);
    const auto &normalize = entry.param_specs[0];
    EXPECT_EQ(normalize.key, "normalize");
    EXPECT_EQ(normalize.type, mortred::models::backend::ParamSpec::Type::BOOL);
    EXPECT_TRUE(normalize.request_overridable);

    const auto &pooling = entry.param_specs[1];
    EXPECT_EQ(pooling.key, "pooling");
    EXPECT_EQ(pooling.type, mortred::models::backend::ParamSpec::Type::STRING);
    EXPECT_TRUE(pooling.request_overridable);
    ASSERT_EQ(pooling.enum_values.size(), 1U);
    EXPECT_EQ(pooling.enum_values[0], "cls");

    // the request validator round-trips the declared specs
    mortred::models::backend::ParamSet accepted;
    const auto violations = mortred::models::backend::validate_params(
        entry.param_specs,
        {{"normalize", mortred::models::backend::ParamValue::of(true)},
         {"pooling", mortred::models::backend::ParamValue::of(std::string("cls"))}},
        &accepted);
    EXPECT_TRUE(violations.empty());
    EXPECT_TRUE(accepted.get_bool("normalize", false));
    EXPECT_EQ(accepted.get_str("pooling", ""), "cls");

    // values outside the declared enum are rejected before they reach the model
    mortred::models::backend::ParamSet rejected;
    const auto enum_violations = mortred::models::backend::validate_params(
        entry.param_specs, {{"pooling", mortred::models::backend::ParamValue::of(std::string("mean"))}}, &rejected);
    ASSERT_EQ(enum_violations.size(), 1U);
    EXPECT_NE(enum_violations[0].message.find("must be one of"), std::string::npos);
}

TEST(ModelCatalog, UnknownSectionIsRejected) {
    EXPECT_EQ(mortred::factory::cv_catalog::create_server(mortred::factory::classification::catalog(), "NO_SUCH_MODEL"),
              nullptr);
    EXPECT_EQ(mortred::factory::model_catalog::create_model(mortred::factory::clip::catalog(), "NO_SUCH_MODEL"), nullptr);
}

TEST(ModelCatalog, ModelOnlyCatalogsExposeWorkingCreators) {
    EXPECT_NE(mortred::factory::model_catalog::create_model(mortred::factory::clip::catalog(), "OPENAI_CLIP"), nullptr);
    EXPECT_NE(mortred::factory::model_catalog::create_model(mortred::factory::segment_anything::predictor_catalog(), "SAM_PREDICTOR"), nullptr);
    EXPECT_NE(mortred::factory::model_catalog::create_model(mortred::factory::segment_anything::fast_sam_catalog(), "FAST_SAM"), nullptr);

    for (const auto &entry : mortred::factory::clip::catalog()) {
        EXPECT_TRUE(model_entry_valid(entry));
    }
    for (const auto &entry : mortred::factory::segment_anything::predictor_catalog()) {
        EXPECT_TRUE(model_entry_valid(entry));
    }
    for (const auto &entry : mortred::factory::segment_anything::fast_sam_catalog()) {
        EXPECT_TRUE(model_entry_valid(entry));
    }
}

TEST(ModelCatalog, ModelOnlySectionsExistInModelConfigs) {
    const auto files = collect_toml_files("conf/model");
    ASSERT_FALSE(files.empty()) << "conf/model not found; run tests from the repository root";

    const std::vector<std::string> sections = {
        "OPENAI_CLIP",
        "SAM_PREDICTOR",
        "FAST_SAM",
    };
    for (const auto &section : sections) {
        bool found = false;
        for (const auto &file : files) {
            const auto parsed_sections = top_level_sections(file);
            if (std::find(parsed_sections.begin(), parsed_sections.end(), section) != parsed_sections.end()) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "model section " << section << " has no conf/model TOML";
    }
}

} // namespace
