#include <gtest/gtest.h>

#include <vector>

#include "models/backend/inference_context.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"
#include "models/object_detection/rtdetr_detector.h"

using jinq::common::StatusCode;
using jinq::models::backend::InferenceContext;
using jinq::models::backend::NamedTensor;
using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::object_detection::std_object_detection_output;
using jinq::models::object_detection::RtdetrDetector;

namespace {

// test-only subclass exposing the protected postprocess hook
class TestRtdetrDetector : public RtdetrDetector<mat_input, std_object_detection_output> {
  public:
    using RtdetrDetector<mat_input, std_object_detection_output>::postprocess;
};

InferenceContext test_context() {
    InferenceContext context;
    context.source_size = cv::Size(20, 20);
    context.network_size = cv::Size(10, 10);
    return context;
}

} // namespace

// the scaffold must fail loudly while the model is unimplemented; flip the
// expectation once a real decoder exists and assert decoded values instead
TEST(RtdetrDetectorOutputContract, UnimplementedPostprocessFailsExplicitly) {
    TestRtdetrDetector model;

    const std::vector<NamedTensor> outputs;
    std_object_detection_output result;
    EXPECT_EQ(model.postprocess(outputs, test_context(), result), StatusCode::MODEL_NOT_IMPLEMENTED);
}

// TODO(new_model): replace the test above with a real output fixture and assert
// the decoded values. Reference: test/object_detection_output_contract_unittest.cc
