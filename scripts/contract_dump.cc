/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: contract_dump.cc
 * Date: 26-8-31
 ************************************************/

// Contract dump: the C++ side of the contract generation chain.
//
//   C++ catalogs (single truth)  ->  contract_dump  ->  docs/contract_dump.json
//   docs/contract_dump.json      ->  gen_openapi.py ->  docs/openapi.json + openapi_doc.h
//
// The committed dump artifact plus `gen_openapi.py --check` make contract
// drift a CI failure: hand-edited docs cannot survive regeneration, and a
// ParamSpec change that is not regenerated fails the gate.

#include <cstdio>
#include <string>
#include <vector>

#include "factory/classification_task.h"
#include "factory/diffusion_task.h"
#include "factory/enhancement_task.h"
#include "factory/feature_embedding_task.h"
#include "factory/feature_point_task.h"
#include "factory/matting_task.h"
#include "factory/mono_depth_estimate_task.h"
#include "factory/obj_detection_task.h"
#include "factory/ocr_task.h"
#include "factory/sam_task.h"
#include "factory/scene_segmentation_task.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "server/output_options.h"

namespace {

using jinq::models::backend::ParamSpec;
using jinq::server::OutputOptions;

rapidjson::Document param_spec_json(const ParamSpec &spec, rapidjson::Document::AllocatorType &allocator) {
    rapidjson::Document value(rapidjson::kObjectType, &allocator);
    value.AddMember("key", rapidjson::Value(spec.key.c_str(), spec.key.size(), value.GetAllocator()), value.GetAllocator());
    value.AddMember("type", rapidjson::Value(spec.type_name(), value.GetAllocator()), value.GetAllocator());
    value.AddMember("request_overridable", spec.request_overridable, value.GetAllocator());
    if (!spec.description.empty()) {
        value.AddMember("description", rapidjson::Value(spec.description.c_str(), spec.description.size(), value.GetAllocator()),
                        value.GetAllocator());
    }
    if (spec.has_range) {
        rapidjson::Document range(rapidjson::kArrayType, &allocator);
        range.PushBack(spec.range_min, range.GetAllocator());
        range.PushBack(spec.range_max, range.GetAllocator());
        value.AddMember("range", range, value.GetAllocator());
    }
    if (!spec.enum_values.empty()) {
        rapidjson::Document values(rapidjson::kArrayType, &allocator);
        for (const auto &entry : spec.enum_values) {
            values.PushBack(rapidjson::Value(entry.c_str(), entry.size(), values.GetAllocator()), values.GetAllocator());
        }
        value.AddMember("values", values, value.GetAllocator());
    }
    return value;
}

template <typename OUTPUT>
rapidjson::Document entry_json(const jinq::factory::cv_catalog::CvModelEntry<OUTPUT> &entry,
                                rapidjson::Document::AllocatorType &allocator) {
    rapidjson::Document value(rapidjson::kObjectType, &allocator);
    auto &a = value.GetAllocator();
    value.AddMember("model_section", rapidjson::Value(entry.model_section.c_str(), entry.model_section.size(), a), a);
    value.AddMember("display_name", rapidjson::Value(entry.display_name.c_str(), entry.display_name.size(), a), a);
    value.AddMember("server_section", rapidjson::Value(entry.server_section.c_str(), entry.server_section.size(), a), a);
    rapidjson::Document params(rapidjson::kArrayType, &allocator);
    for (const auto &spec : entry.param_specs) {
        rapidjson::Document item = param_spec_json(spec, allocator);
        params.PushBack(item, params.GetAllocator());
    }
    value.AddMember("params", params, value.GetAllocator());
    return value;
}

template <typename OUTPUT>
void add_task(rapidjson::Document &doc, rapidjson::Document::AllocatorType &allocator, const char *task,
              const std::vector<jinq::factory::cv_catalog::CvModelEntry<OUTPUT>> &entries) {
    rapidjson::Document task_obj(rapidjson::kObjectType, &allocator);
    task_obj.AddMember("task", rapidjson::Value(task, task_obj.GetAllocator()), task_obj.GetAllocator());
    rapidjson::Document array(rapidjson::kArrayType, &allocator);
    for (const auto &entry : entries) {
        rapidjson::Document item = entry_json(entry, allocator);
        array.PushBack(item, array.GetAllocator());
    }
    task_obj.AddMember("entries", array, task_obj.GetAllocator());
    doc.PushBack(task_obj, doc.GetAllocator());
}

rapidjson::Document output_options_json(rapidjson::Document::AllocatorType &allocator) {
    rapidjson::Document value(rapidjson::kObjectType, &allocator);
    auto &a = value.GetAllocator();
    value.AddMember("encoding", rapidjson::Value("png", a), a);
    value.AddMember("include_image", true, a);
    value.AddMember("max_results", 0, a);
    value.AddMember("echo_params", false, a);
    value.AddMember("additional_properties", false, a);
    return value;
}

} // namespace

int main() {
    rapidjson::Document doc(rapidjson::kObjectType);
    rapidjson::Document::AllocatorType &allocator = doc.GetAllocator();

    rapidjson::Document tasks(rapidjson::kArrayType, &allocator);
    add_task(tasks, allocator, "classification", jinq::factory::classification::catalog());
    add_task(tasks, allocator, "object_detection", jinq::factory::object_detection::catalog());
    add_task(tasks, allocator, "face_detection", jinq::factory::object_detection::face_catalog());
    add_task(tasks, allocator, "scene_segmentation", jinq::factory::scene_segmentation::catalog());
    add_task(tasks, allocator, "ocr", jinq::factory::ocr::catalog());
    add_task(tasks, allocator, "matting", jinq::factory::matting::catalog());
    add_task(tasks, allocator, "enhancement", jinq::factory::enhancement::catalog());
    add_task(tasks, allocator, "feature_point", jinq::factory::feature_point::catalog());
    add_task(tasks, allocator, "feature_embedding", jinq::factory::feature_embedding::catalog());
    add_task(tasks, allocator, "mono_depth_estimation", jinq::factory::mono_depth_estimation::catalog());
    add_task(tasks, allocator, "diffusion", jinq::factory::diffusion::catalog());
    add_task(tasks, allocator, "segment_anything_amg", jinq::factory::segment_anything::amg_catalog());
    doc.AddMember("tasks", tasks, allocator);

    rapidjson::Document options = output_options_json(allocator);
    doc.AddMember("output_options_defaults", options, allocator);

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);
    std::printf("%s\n", buffer.GetString());
    return 0;
}
