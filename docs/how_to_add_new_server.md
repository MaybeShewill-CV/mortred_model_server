# How To Add New Server

Here is brief instruction about how to add a new server in this framework. Since the registry-driven refactor, adding a served model is one `CvModelEntry` row in the family `catalog()` (two TOML section names, a worker factory and a response filler). `ProductIndex` projects that row onto both unified CLIs. The generic implementation lives in [jinq::server::CvModelServer&lt;MODEL_OUTPUT&gt;](../src/server/generic_cv_server.h) on top of [jinq::server::BaseAiServerImpl&lt;WORKER, MODEL_OUTPUT&gt;](../src/server/base_server_impl.h), which keeps providing auth, rate limiting, request validation, per-request timeout, the worker pool, Prometheus metrics and the `/openapi.json` endpoint. The model input uses base64 encoded images uniformly. The example below adds a densenet image classification server; the model itself comes from [how_to_add_new_model.md](../docs/how_to_add_new_model.md).

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

## Step 2: Add One Catalog Row

Open the task header of the model's family (`src/factory/classification_task.h` for the example) and add a `CvModelEntry` to `catalog()`. That single row is the product identity: `model_section` is the `--model` id, `server_section` names the TOML table, `make_worker` and `fill_response` make it HTTP-servable, and family-default vis/default-image make it benchmarkable.

```cpp
inline const std::vector<Entry> &catalog() {
    static const std::vector<Entry> entries = {
        // ...existing models...
        Entry{"DENSENET", "densenet classification", "DENSENET_CLASSIFICATION_SERVER",
              &create_densenet_classifier<ImageInput, Output>,
              &jinq::server::response::fill_classification,
              classification_param_specs()},
    };
    return entries;
}
```

`ProductIndex` iterates every family `catalog()` at startup. There is no second product table and no `create_densenet_cls_server` wrapper. `CvModelServer<Output>::init` reads the server section (worker pool, timeouts, auth, rate limit — see
[about_model_server_configuration.md](../docs/about_model_server_configuration.md)),
loads the model config referenced by the model section, creates `worker_nums` workers and
assembles the HTTP server.

If your output is a brand-new shape, add the matching serializer in
[src/server/response_serializers.h](../src/server/response_serializers.h) and point
`fill_response` at it; field names and JSON types must follow
`docs/openapi.json` `components.schemas` (regenerate with `python scripts/gen_openapi.py`).

## Step 3: Unified executable, no new ELF

Do **not** add a per-model `src/apps/server/*.cpp` or `add_mortred_app` line.
`ProductIndex` projects every `CvModelEntry` in the family catalog, so a new
catalog row is enough for:

```
mortred-model-server.out --model DENSENET /path/to/densenet_server_config.toml
mortred-model-benchmark.out --model DENSENET /path/to/densenet_config.toml [image]
```

Then:

1. add a config under `conf/server/<task>/<model>/` with `model = "DENSENET"` and
   `server_exe = "mortred-model-server.out"` (copy a sibling and adjust
   `server_uri`, `port`, `worker_nums`);
2. `python3 scripts/check_consistency.py` must stay green (catalog id ↔ conf `model=`,
   and a `catalog_tiers` entry in `conf/ci_hosted_golden.json`: `hosted`, `gpu-smoke`,
   or `nightly`);
3. declare the new `server_uri` in `docs/openapi.json` via `scripts/gen_openapi.py`.

## Step 4: What The Base Framework Already Does For You

You usually do NOT need to touch request serving. `BaseAiServerImpl` provides
`serve_process` / `do_work` / `do_work_cb`: JSON request parsing (with 400/413/415/405
contract errors), bearer auth, per-IP rate limiting, worker checkout from the blocking
queue (timeout budgeted), model inference, response serialization through
`fill_response`, Prometheus metrics and structured request logs.