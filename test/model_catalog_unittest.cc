#include <algorithm>
#include <filesystem>
#include <map>
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
#include "factory/feature_point_task.h"
#include "factory/matting_task.h"
#include "factory/model_catalog.h"
#include "factory/mono_depth_estimate_task.h"
#include "factory/obj_detection_task.h"
#include "factory/ocr_task.h"
#include "factory/sam_task.h"
#include "factory/scene_segmentation_task.h"
#include "models/catalog/model_entry.h"
#include "server/generic_cv_server.h"

namespace {

namespace fs = std::filesystem;

using ServerEntry = jinq::models::catalog::ServedModelEntry;
using jinq::models::catalog::model_entry_valid;
using jinq::models::catalog::served_model_entry_valid;

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
        register_entries("classification", jinq::factory::classification::catalog());
        register_entries("object_detection", jinq::factory::object_detection::catalog());
        register_entries("face_detection", jinq::factory::object_detection::face_catalog());
        register_entries("scene_segmentation", jinq::factory::scene_segmentation::catalog());
        register_entries("ocr", jinq::factory::ocr::catalog());
        register_entries("matting", jinq::factory::matting::catalog());
        register_entries("enhancement", jinq::factory::enhancement::catalog());
        register_entries("feature_point", jinq::factory::feature_point::catalog());
        register_entries("mono_depth_estimation", jinq::factory::mono_depth_estimation::catalog());
        register_entries("diffusion", jinq::factory::diffusion::catalog());
        register_entries("segment_anything_amg", jinq::factory::segment_anything::amg_catalog());
    }

    template <typename OUTPUT>
    void register_entries(const std::string &task, const std::vector<jinq::factory::cv_catalog::CvModelEntry<OUTPUT>> &entries) {
        for (const auto &entry : entries) {
            served_entries_.push_back(&entry);
            entry_tasks_[&entry] = task;
        }
    }

    std::string describe(const ServerEntry &entry) const {
        const auto found = entry_tasks_.find(&entry);
        return (found == entry_tasks_.end() ? std::string("<unknown>") : found->second) + "/" + entry.model_section;
    }

    // constructing the server instantiates CvModelServer<OUTPUT> and proves the
    // registration lambda is valid; init() is intentionally not called
    template <typename OUTPUT>
    void check_create_server(const std::vector<jinq::factory::cv_catalog::CvModelEntry<OUTPUT>> &entries, const std::string &task) {
        for (const auto &entry : entries) {
            const auto server_name = "model_catalog_ut_" + task + "_" + entry.model_section;
            auto server = jinq::factory::cv_catalog::create_server(entry, server_name);
            ASSERT_NE(server, nullptr) << server_name;
            EXPECT_NE(dynamic_cast<jinq::server::CvModelServer<OUTPUT> *>(server.get()), nullptr) << server_name;
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

TEST_F(ModelCatalogTest, ServerSectionsAreGloballyUnique) {
    std::map<std::string, int> counts;
    for (const auto *entry : served_entries_) {
        ++counts[entry->server_section];
    }
    for (const auto &[section, count] : counts) {
        ASSERT_EQ(count, 1) << "duplicated server section: " << section;
    }
}

TEST_F(ModelCatalogTest, CreateServerBuildsGenericCvServer) {
    check_create_server(jinq::factory::classification::catalog(), "classification");
    check_create_server(jinq::factory::object_detection::catalog(), "object_detection");
    check_create_server(jinq::factory::object_detection::face_catalog(), "face_detection");
    check_create_server(jinq::factory::scene_segmentation::catalog(), "scene_segmentation");
    check_create_server(jinq::factory::ocr::catalog(), "ocr");
    check_create_server(jinq::factory::matting::catalog(), "matting");
    check_create_server(jinq::factory::enhancement::catalog(), "enhancement");
    check_create_server(jinq::factory::feature_point::catalog(), "feature_point");
    check_create_server(jinq::factory::mono_depth_estimation::catalog(), "mono_depth_estimation");
    check_create_server(jinq::factory::diffusion::catalog(), "diffusion");
    check_create_server(jinq::factory::segment_anything::amg_catalog(), "segment_anything_amg");
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
    EXPECT_EQ(jinq::factory::classification::catalog().size(), 3U);
    EXPECT_EQ(jinq::factory::object_detection::catalog().size(), 5U);
    EXPECT_EQ(jinq::factory::object_detection::face_catalog().size(), 2U);
    EXPECT_EQ(jinq::factory::scene_segmentation::catalog().size(), 3U);
    EXPECT_EQ(jinq::factory::ocr::catalog().size(), 1U);
    EXPECT_EQ(jinq::factory::matting::catalog().size(), 2U);
    EXPECT_EQ(jinq::factory::enhancement::catalog().size(), 3U);
    EXPECT_EQ(jinq::factory::feature_point::catalog().size(), 1U);
    EXPECT_EQ(jinq::factory::mono_depth_estimation::catalog().size(), 2U);
    EXPECT_EQ(jinq::factory::diffusion::catalog().size(), 4U);
    EXPECT_EQ(jinq::factory::segment_anything::amg_catalog().size(), 1U);
    EXPECT_EQ(served_entries_.size(), 27U);
}

TEST(ModelCatalog, UnknownSectionIsRejected) {
    EXPECT_EQ(jinq::factory::cv_catalog::create_server(jinq::factory::classification::catalog(), "NO_SUCH_MODEL", "no_such_server"),
              nullptr);
    EXPECT_EQ(jinq::factory::model_catalog::create_model(jinq::factory::clip::catalog(), "NO_SUCH_MODEL"), nullptr);
}

TEST(ModelCatalog, ModelOnlyCatalogsExposeWorkingCreators) {
    EXPECT_NE(jinq::factory::model_catalog::create_model(jinq::factory::clip::catalog(), "OPENAI_CLIP"), nullptr);
    EXPECT_NE(jinq::factory::model_catalog::create_model(jinq::factory::segment_anything::predictor_catalog(), "SAM_PREDICTOR"), nullptr);
    EXPECT_NE(jinq::factory::model_catalog::create_model(jinq::factory::segment_anything::fast_sam_catalog(), "FAST_SAM"), nullptr);

    for (const auto &entry : jinq::factory::clip::catalog()) {
        EXPECT_TRUE(model_entry_valid(entry));
    }
    for (const auto &entry : jinq::factory::segment_anything::predictor_catalog()) {
        EXPECT_TRUE(model_entry_valid(entry));
    }
    for (const auto &entry : jinq::factory::segment_anything::fast_sam_catalog()) {
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
