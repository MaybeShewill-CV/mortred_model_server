#include "model_contract_test_util.h"

#include "models/io/common_input.h"
#include "models/io/object_detection.h"
#include "models/object_detection/rtdetr_detector.h"

using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::object_detection::std_object_detection_output;
using jinq::models::object_detection::RtdetrDetector;

// one line buys the full seven-variant rejection matrix (missing output /
// wrong dtype / wrong rank / wrong shape / short buffer / NaN / Inf).
// While the model is a scaffold every variant fails with MODEL_NOT_IMPLEMENTED,
// which the harness accepts as an explicit rejection.
POSTPROCESS_CONTRACT_TEST(RtdetrDetector, mat_input, std_object_detection_output, "output", 1, 1)

// TODO(new_model): replace the placeholder shape above with the real output
// shape (it must be concrete, not dynamic), then add a fixture that asserts decoded values.
// Reference: test/object_detection_output_contract_unittest.cc
