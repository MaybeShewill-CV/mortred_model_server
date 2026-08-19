# How To Add New Server

Here is brief instruction about how to add a new server in this framework. Since the registry-driven refactor, adding a server no longer means writing a ~200-line hand-written class: every server is a single `CvServerSpec` entry (two TOML section names, a worker factory and a response filler) registered through a creator closure. The generic implementation lives in [jinq::server::CvModelServer&lt;MODEL_OUTPUT&gt;](../src/server/generic_cv_server.h) on top of [jinq::server::BaseAiServerImpl&lt;WORKER, MODEL_OUTPUT&gt;](../src/server/base_server_impl.h), which keeps providing auth, rate limiting, request validation, per-request timeout, the worker pool, Prometheus metrics and the `/openapi.json` endpoint. The model input uses base64 encoded images uniformly. The example below adds a densenet image classification server; the model itself comes from [how_to_add_new_model.md](../docs/how_to_add_new_model.md).

## Step 1: Define Your Own Output Data Type :monkey_face:

This step is the same as adding a new model. Default model output types for each vision task live in [model_io_define.h](../src/models/model_io_define.h); the types named `std_*_output` are the default outputs. For classification:

```cpp
namespace classification {
    struct cls_output {
        int class_id;
        std::vector<float> scores;
    };
    using std_classification_output = cls_output;
}
```

`class_id` equals the index of the max score in `scores`. If your task needs a new
output shape, define it here first — the server spec and the response serializer both
refer to it.

## Step 2: Register An CvServerSpec Entry

Open the task header of the model's family (`src/factory/classification_task.h` for the example) and write the server create function as a spec-closure registration:

```cpp
// create densenet classification server
inline std::unique_ptr<BaseAiServer> create_densenet_cls_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_creator(server_name, []() -> std::unique_ptr<BaseAiServer> {
        using Output = jinq::models::io_define::classification::std_classification_output;
        jinq::server::CvServerSpec<Output> spec;
        spec.server_section = "DENSENET_CLASSIFICATION_SERVER";  // server TOML section
        spec.model_section = "DENSENET";                          // holds model_config_file_path
        spec.display_name = "Densenet classification";
        spec.make_worker = [](const std::string& name) {
            return create_densenet_classifier<jinq::server::Base64Input, Output>(name);
        };
        spec.fill_response = &jinq::server::response::fill_classification;
        return std::unique_ptr<BaseAiServer>(
            new jinq::server::CvModelServer<Output>(std::move(spec)));
    });
    return server_factory.create(server_name);
}
```

That is the whole per-server footprint. `CvModelServer<Output>::init` reads the server
section (worker pool, timeouts, auth, rate limit — see
[about_model_server_configuration.md](../docs/about_model_server_configuration.md)),
loads the model config referenced by the model section, creates `worker_nums` workers and
assembles the HTTP server — the exact flow the former 22 hand-written classes duplicated.

If your output is a brand-new shape, add the matching serializer in
[src/server/response_serializers.h](../src/server/response_serializers.h) and point
`spec.fill_response` at it; field names and JSON types must follow
`docs/openapi.json` `components.schemas` (regenerate with `python scripts/gen_openapi.py`).

## Step 3: Wire The Executable

Create a thin main under `src/apps/server/<task>/` (the path and executable name are part
of the repository layout contract checked by `scripts/check_consistency.py`) that
delegates to the shared entry:

```cpp
// densenet classification server tool

#include "apps/common/model_server_main.h"
#include "factory/classification_task.h"

int main(int argc, char** argv) {
    return jinq::apps::run_model_server_main(
        argc, argv, "DENSENET_CLASSIFICATION_SERVER",
        [](const std::string& server_name) {
            return jinq::factory::classification::create_densenet_cls_server(server_name);
        });
}
```

Then:

1. add the target in `src/apps/CMakeLists.txt` (`add_server_app(...)`);
2. add a config under `conf/server/<task>/<model>/` (copy a sibling config and adjust
   `server_uri`, `port`, `worker_nums`);
3. add the executable row to `docs/repository-layout.md` (the consistency gate checks
   this mapping and that the referenced `model_config_file_path` exists);
4. declare the new `server_uri` in `docs/openapi.json` via `scripts/gen_openapi.py`.

## Step 4: What The Base Framework Already Does For You

You usually do NOT need to touch request serving. `BaseAiServerImpl` provides
`serve_process` / `do_work` / `do_work_cb`: JSON request parsing (with 400/413/415/405
contract errors), bearer auth, per-IP rate limiting, worker checkout from the blocking
queue (timeout budgeted), model inference, response serialization through
`fill_response`, Prometheus metrics and structured request logs. Override points exist
(`handle_custom_endpoint`) only if you need extra endpoints.