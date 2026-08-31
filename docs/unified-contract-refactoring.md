# 统一请求契约改造（M1–M6）技术总结

> 版本：main `8848a8a`（2026-08-31）
> 执行日志：见仓库根目录 `CONTRACT_V1_REFACTOR_PLAN.md`（M0–M6 全部勾选记录）
> 本文面向：服务端维护者、业务接入方、运维。

---

## 1. 这次改造要解决的问题

### 1.1 旧契约（img_data 时代）的五个硬伤

| # | 旧契约行为 | 业务后果 |
|---|---|---|
| 1 | 请求体只有 `{"img_data": "<base64>", "req_id": "..."}` 两个字段 | 检测阈值、NMS、top-k 全部写死在 TOML 配置里，业务方"按请求调阈值"这类最基本诉求无法满足；改一次参数 = 改配置 + 重启进程 |
| 2 | 单请求 = 单图 | 相册批量审核、视频关键帧等场景只能串行发 N 个请求，无法利用一次网络往返 |
| 3 | 未知字段**静默忽略** | 客户端拼错字段名（如 `score_treshold`）不报错，静默无效果，排查极其困难 |
| 4 | base64-in-JSON 传输 | 体积膨胀 ~33%，JSON 解析 + base64 解码 + imdecode 三段拷贝，直接吃掉动态批处理的收益 |
| 5 | 超时是"取 worker 的等待上限"，与排队/批处理不共享预算 | 多租户排队下超时语义不正确：请求可能在队列里耗尽用户耐心却仍占用引擎 |

### 1.2 目标定位

本项目 P0 清单第 1 项（SME 视角）：**请求级参数 + 输入输出契约升级**——把"模型服务器"变成"业务可用服务"的分界线。

### 1.3 设计立场（五条铁律）

整个改造期间以下五条被测试逐一钉死，任何后续演进不得违反：

1. **`images` 恒为数组**：单图也是 `images[0]`，永不出现两种表达
2. **`img_data` 永不成功**：legacy 字段触发 422 + 迁移提示，即使同请求携带合法 `images`
3. **未知键永远 422**：请求字段、参数键、选项键全部严格校验，带 JSON Pointer 定位
4. **严格性 + 附加演进**：新字段必须可选有默认；客户端必须忽略未知响应字段——保证契约永不再分叉出 v2
5. **编码不改变语义**：JSON+base64 与 raw 字节是同一契约的两种字节形态，同一错误必须产生逐字节一致的 422

配套的分层原则：**契约 = 信封语义（编译期/启动期确定），编码 = 字节形态（可扩展）**。HTTP 层不知道后端类型，模型层不知道引擎类型，新增编码/后端/模型互不牵连。

---

## 2. 改造总览

### 2.1 里程碑

| 里程碑 | 内容 | 关键产出 |
|---|---|---|
| M0 | 基线固化 | v1 契约快照（`test/contract_baseline/`）、golden 基线（26/27，确认 realesrgan 为 MNN 抖动） |
| M1 | 类型层 | `byte_source`/`image_input`、`ParamSpec`/`ParamSet`/校验器、状态码 66/67/68、`load_image(image_input)`、`InferenceContext.params` |
| M2 | 解析层 | `request_envelope.h`（严格信封解析 + `img_data` 快速失败）、`output_options.h`、统一响应信封结构 |
| M3 | 装配层 | worker 自述输入类型（`BaseAiModel::input_type`）、11 个任务目录切 `image_input`、检测族参数覆盖（`finalize_detections(..., context)`） |
| M4 | **内核翻转** | `task_request` 改形（items/params/options/deadline）、逐项隔离、按 item 背压、deadline 贯通、批路径请求级闩、网关透传修复、全部 HTTP 测试重写 |
| M5 | 契约出口 | `contract_dump`（C++ 目录 → 契约清单）→ `gen_openapi.py` → OpenAPI，CI 同步门禁 `check_contract_sync.py` |
| M6 | raw 编码 | 三分支编码判定、`X-Mortred-*` 头部、双编码等价断言、真实 yolov8/TRT 基准 |

### 2.2 改造后请求流转

```
客户端（JSON 或 raw）
   │  Authorization / Content-Type / X-Mortred-*
   ▼
mortred-gateway（:8080，Content-Type/Accept 原样透传 + 内部 token）
   ▼
模型服务进程（loopback）
   ├─ serve_process：限流→鉴权→编码三分支→严格信封解析(422)→item 上限(413)
   │   →按 item 背压(429+Retry-After)→deadline 打点→WFGoTask
   ├─ 单请求路径：一次 worker 领取跑完全部 items（逐项查预算）
   ├─ 批路径：每 item 一个 batch_entry + request_state 闩（跨请求打包同一引擎批）
   └─ 异步 /jobs：同一信封提交，同一 run_items 执行核，/result 返回同一响应信封
   ▼
BackendCvModel（preprocess/postprocess 钩子 + context.params 覆盖）
   ▼
InferenceSession（MNN / ONNX Runtime / TensorRT 统一 NamedTensor 契约）
```

### 2.3 验证结论

- **CI 档（FULL=OFF）**：36 个测试目标全绿（含 `request_envelope_unittest` 18 用例、`server_e2e_contract_test` 31 用例）
- **FULL 档**：45 个测试目标全绿（含 golden 27 用例、`model_catalog_unittest` 实例化全部目录）
- **契约门禁负测试**：篡改 `contract_dump.json` 一个字段 → 三连报错（dump / openapi.json / openapi_doc.h 过期）exit 1
- **双编码等价**：同一参数错误走 JSON 与 raw 头部，422 的 pointer 与 message 逐字节一致（单测断言）

---

## 3. 新契约规范

### 3.1 请求信封（模型端点，POST）

```json
{
  "req_id": "client-trace-id",
  "images": ["<base64>", "<base64>"],
  "params": { "score_threshold": 0.35, "top_k": 100 },
  "options": { "encoding": "png", "include_image": false }
}
```

| 字段 | 必填 | 语义 |
|---|---|---|
| `req_id` | 否 | 追踪 id，回显为响应 `task_id`；缺省服务端生成 16 位 hex |
| `images` | **是** | base64 字符串数组，≥1 张；响应 `results[]` 与之**下标对齐** |
| `params` | 否 | **请求级**参数（作用于全部 images），按模型声明的白名单严格校验 |
| `options` | 否 | 输出选项（见下） |

`options` 已知键（`additionalProperties: false`）：

| 键 | 类型/取值 | 默认 | 说明 |
|---|---|---|---|
| `encoding` | `png` \| `jpeg` \| `webp` | `png` | 内嵌图片输出的编码 |
| `include_image` | bool | `true` | 是否内嵌图片类结果 |
| `max_results` | int ≥ 0 | `0` | 0 = 不限量 |
| `echo_params` | bool | `false` | 响应是否回显参数 |

### 3.2 响应信封（模型端点 + `/jobs/{id}/result`）

```json
{
  "status": 0,
  "status_str": "OK",
  "task_id": "client-trace-id",
  "model": { "name": "YOLOV8", "version": "" },
  "results": [ { "status": 0, "data": { "...": "任务载荷" } } ],
  "server_time_ms": 41.2,
  "partial": false
}
```

- `results[i]` 与请求 `images[i]` 下标对齐；`status=0` 时 `data` 为任务载荷，否则 `data: null`
- 任务载荷形状不变（分类 `{class_id,category,scores}`、检测 `[{class_id,score,category,bbox,detail_infos}]` 等，详见 `response_serializers.h` / OpenAPI）
- `errors[]`（仅 422 出现）：`[{ "pointer": "/params/score_threshold", "message": "..." }]`

> **注意**：`/healthz` `/ready` `/welcome` 等基础设施端点仍使用旧版运维信封 `{req_id, code, msg, data}`——统一契约只覆盖模型推理面。

### 3.3 错误模型与 HTTP 语义

新增状态码（wire 值为稳定契约）：

| StatusCode | wire | HTTP | 场景 |
|---|---|---|---|
| `INVALID_REQUEST_PARAMETER` | 66 | 422 | 信封/参数/选项任何严格校验失败，`errors[]` 带 JSON Pointer |
| `REQUEST_ITEM_LIMIT` | 67 | 413 | `images` 超过 `max_request_items`（默认 16），pointer `/images` |
| `DEADLINE_EXCEEDED_PARTIAL` | 68 | **200** | deadline 中途耗尽：已完成项照常返回，`partial: true` |

关键语义裁定：

- **任一非超时项失败 → HTTP 500** + 聚合 status，但 `results[]` 逐项枚举（错误必须在 HTTP 层对监控可见）
- **部分超时（≥1 项完成）→ HTTP 200 + status=68 + partial=true**（部分结果优于全有全无）
- **全部超时 → 504**，`results: []`
- 背压：**按 item 计数**（16 图请求 = 16 个队列槽位），429 携带 `Retry-After`（EWMA 估算）
- deadline：`model_run_timeout` 是**绝对预算**，覆盖 排队等待 + worker 等待 + 全部 items 推理

### 3.4 双编码

| | JSON 编码 | raw 编码 |
|---|---|---|
| Content-Type | `application/json` | `image/*` 或 `application/octet-stream` |
| 图片 | `images: ["<b64>", ...]`（可多图） | **body 即 `images[0]`**（恒单图） |
| 追踪 id | `req_id` 字段 | `X-Request-ID` 头 |
| params | `params` 对象 | `X-Mortred-Params` 头（值为紧凑 JSON 对象） |
| options | `options` 对象 | `X-Mortred-Options` 头（值为紧凑 JSON 对象） |

两种编码走**同一个** ParamSpec 校验器，错误 pointer/message 完全一致。类型仅作提示，实际格式由解码器嗅探。

### 3.5 请求级参数（当前已声明：13 个模型）

| 任务族 | 参数 |
|---|---|
| object_detection + face（7 模型） | `score_threshold` f32 [0,1]、`nms_threshold` f32 [0.1,1]、`top_k` i32 [1,10000] |
| classification（3 模型） | `top_k` i32 [1,1000]（保留前 k 高分，降序，缩减载荷） |
| ocr / DBNET | `score_threshold` f32 [0.1,0.9]、`top_k` i32 [1,10000] |
| feature_point / SuperPoint | `score_threshold` f32 [0.001,1]、`nms_radius` i32 [1,50]（像素半径，非 IoU，故意异名） |
| SAM AMG | `points_per_side` i32 [1,64]、`pred_iou_thresh`、`stability_score_thresh`、`box_nms_thresh` f32 [0,1]、`min_mask_region_area` i32 [0,100000] |

解析优先级（唯一且确定）：**请求 `params` > `[MODEL.params]` TOML > 代码默认值**。环境变量永不进入推理语义。scene_segmentation/matting/enhancement/depth 四族（10 模型）声明为"无请求参数"（稠密逐像素输出无逐请求阈值语义，空声明是正确终态）；diffusion（4 模型）因适配器忽略请求体而延后（见 §7）。

---

## 4. 服务端使用流程与注意事项

### 4.1 新增/修改配置项

`[*_SERVER]` 节新增（已进入 schema 校验白名单）：

```toml
max_request_items = 16   # 单请求最多图片数；背压/Retry-After/队列深度均按 item 计
```

注意事项：

- `model_run_timeout` 语义升级为**全链路 deadline**（旧版只覆盖取 worker 的等待）——压测时按"端到端预算"理解
- 队列深度指标（`mortred_queue_depth` 等）与日志里的 `waiting_jobs/received_jobs/finished_jobs` 现在都是 **item 口径**
- 动态批处理的 `mortred_batch_size` 直方图统计的是**引擎批内的 item 数**（跨请求混批）

### 4.2 启动与部署注意事项（重要）

1. **工作目录**：模型配置里的 `model_config_file_path = "../conf/..."` 是**相对 cwd** 解析的。服务进程必须以**仓库根的直接子目录**为 cwd 启动（源码树用 `_bin`，安装树用 `bin`）——supervisor/容器布局即如此。从仓库根直接启动会报 `model config file not exist`
2. **库路径**：运行期 `LD_LIBRARY_PATH` 需同时包含构建产物 `lib/` 与 `3rd_party/libs`（e2e 依赖 `libssl.so.1.1` 就在其中）
3. 编译并行度：本机约定 **`make -j6` 上限**

### 4.3 为模型声明请求参数（服务端开发者）

在任务目录（如 `src/factory/obj_detection_task.h`）的 entry 上挂 `ParamSpec`：

```cpp
inline const std::vector<jinq::models::backend::ParamSpec> &detection_param_specs() {
    static const std::vector<jinq::models::backend::ParamSpec> specs = {
        jinq::models::backend::ParamSpec::f32("score_threshold").range(0.0, 1.0).desc("confidence threshold"),
        jinq::models::backend::ParamSpec::f32("nms_threshold").range(0.1, 1.0).desc("per-class NMS IoU threshold"),
        jinq::models::backend::ParamSpec::i32("top_k").range(1, 10000).desc("keep at most k detections"),
    };
    return specs;
}
```

在模型后处理读取（配置值为缺省）：

```cpp
float thr = context.params->get_f32("score_threshold", _m_detection_params.score_threshold);
```

**改完 ParamSpec 必须走契约再生成（见 4.5），否则 CI 门禁失败。**

### 4.4 新增一个受服务模型的最短路径

1. 模型类继承 `BackendCvModel<INPUT, OUTPUT>`，实现 `preprocess` / `postprocess` / `on_init`（参数从 `context.params` 读，缺省回落 TOML）
2. 任务目录登记 entry：`{model_section, display_name, server_section, make_worker, fill_response, param_specs}`
3. `conf/model/<task>/<m>/*.toml`：`[MODEL.backend]`（mnn/onnx/tensorrt）+ `[MODEL.params]`（默认值）
4. `conf/server/<task>/<m>/*.toml`：端口/线程/worker_nums/server_uri/server_exe
5. 走 4.5 再生成契约产物，提交

### 4.5 契约生成链与维护流程（M5 核心）

```
C++ catalogs（唯一事实源）
    │  cmake --build <full-build> --target contract_dump
    ▼  <full-build>/bin/contract_dump > docs/contract_dump.json     （提交）
    │  python scripts/gen_openapi.py
    ▼  docs/openapi.json + src/server/openapi_doc.h                 （提交）
```

- CI（cpu-profile job）运行 `scripts/check_contract_sync.py --dump-bin ...`：**改 C++ 不再生成、或手改文档，两方向漂移都是构建失败**
- `openapi.json` 的 `info.x-contract-hash` 是 dump 的规范 SHA-256：任何契约面变更必然改变文档字节
- 门禁对 dump 读取做了 BOM 容忍，但生成物必须用工具再生，不要手编

### 4.6 指标变化

新增：`mortred_requests_by_encoding_total{encoding="json|raw"}`（仅两个取值，参数值**永不**进 label——基数纪律）。

其余既有指标语义迁移到 item 口径（见 4.1）。`/metrics` `/openapi.json` `/healthz` `/ready` 仍为公开端点。

### 4.7 升级注意事项（从 img_data 版本升级）

- **破坏性变更**：`img_data` 立即 422。若有存量客户端，先改客户端再升级服务（迁移表见 §5.4）
- 同步/异步响应信封变化：旧解析 `.code/.msg/.data` 的代码需切 `.status/.status_str/.results[N].data`
- infra 端点（health/ready）信封**未变**，探针无需修改
- `test/contract_baseline/v1_http_contract.md` 保留了旧契约完整快照，可作迁移对照

---

## 5. 客户端使用流程与注意事项

### 5.1 JSON 编码（默认，互操作/调试友好）

```bash
IMG=$(base64 -w0 demo_data/model_test_input/object_detection/bus.jpg)
curl -s -X POST \
  http://localhost:8080/mortred_ai_server_v1/obj_detection/yolov8 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
        \"req_id\": \"demo-1\",
        \"images\": [\"$IMG\"],
        \"params\": {\"score_threshold\": 0.35, \"top_k\": 100},
        \"options\": {\"include_image\": false}
      }"
```

### 5.2 raw 编码（性能优先，单图）

```bash
curl -s -X POST \
  http://localhost:8080/mortred_ai_server_v1/obj_detection/yolov8 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: image/jpeg" \
  -H "X-Request-ID: demo-raw-1" \
  -H 'X-Mortred-Params: {"score_threshold":0.35}' \
  --data-binary @demo_data/model_test_input/object_detection/bus.jpg
```

Python（纯标准库）示例：

```python
import base64, http.client, json

def infer_json(host, port, uri, token, image_path, params=None):
    b64 = base64.b64encode(open(image_path, "rb").read()).decode()
    body = json.dumps({"req_id": "py-1", "images": [b64], "params": params or {}})
    conn = http.client.HTTPConnection(host, port, timeout=30)
    conn.request("POST", uri, body=body, headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer %s" % token,
    })
    resp = conn.getresponse()
    return resp.status, json.loads(resp.read())

def infer_raw(host, port, uri, token, image_path, params=None):
    data = open(image_path, "rb").read()
    headers = {"Content-Type": "application/octet-stream",
               "Authorization": "Bearer %s" % token,
               "X-Request-ID": "py-raw-1"}
    if params:
        headers["X-Mortred-Params"] = json.dumps(params, separators=(",", ":"))
    conn = http.client.HTTPConnection(host, port, timeout=30)
    conn.request("POST", uri, body=data, headers=headers)
    resp = conn.getresponse()
    return resp.status, json.loads(resp.read())
```

### 5.3 响应处理要点

1. **先看 HTTP，再看 `status`，最后逐项看 `results[i].status`**
2. `results[i].data` 在该项失败时为 `null`——逐项判空，不要假设整包一致
3. `partial: true` 表示 deadline 中途耗尽：已完成项可用，未完成项 `status=4`；业务可凭 `task_id` 重试缺失项
4. 429 时读 `Retry-After` 头（秒）；注意**全端口按 IP 限流**（含健康检查）
5. 422 时解析 `errors[].pointer` 定位字段——这是机器可读的排障入口
6. **忽略未知响应字段**（客户端第一守则）：服务端只做附加演进，新字段随时可能出现

### 5.4 迁移表（img_data → 统一契约）

| 旧 | 新 |
|---|---|
| `"img_data": "<b64>"` | `"images": ["<b64>"]`（恒数组） |
| （无参数概念） | `"params": {...}`（按模型白名单） |
| （无输出选项） | `"options": {...}` |
| 未知字段被忽略 | **422** + pointer |
| 响应 `.req_id` | `.task_id` |
| 响应 `.code` / `.msg` | `.status` / `.status_str` |
| 响应 `.data` | `.results[0].data`（先查 `.results[0].status`） |
| 单图单结果 | 1–16 图，逐项结果与隔离 |

发送 `img_data` 会立即收到带迁移指引的 422：

```json
{ "status": 66, "errors": [ { "pointer": "/img_data",
    "message": "field 'img_data' was removed; use images: [\"<base64>\"] (migration: img_data -> images[0])" } ] }
```

### 5.5 编码选择与已知边界

- 图片 > ~100KB 优先 raw（实测 476KB JPG：载荷 −25%、p50 −10%、吞吐 +23%）
- raw **恒单图**（body 即 `images[0]`）；多图二进制（multipart）为后续附加演进，未实现
- 按图独立参数（`images[i].params`）明确不做——请求级 `params` 作用于全部图
- `X-Mortred-*` 头仅在 raw Content-Type 下有意义
- URL 输入源在类型层预留（`byte_source` 槽位），当前解析器不接受——SSRF/egress 策略层就绪前 fail-closed

### 5.6 异步长任务（扩散 / SAM AMG）

提交与同步同一信封（可多图）；流转：

```
POST /jobs            → 202 {"job_id","state":"pending","poll_url","result_url"} + Location
GET  /jobs/{id}       → {"job_id","state","elapsed_ms"}
GET  /jobs/{id}/wait?timeout=N   → 长轮询（默认 30s，上限 300s）
GET  /jobs/{id}/result → 统一响应信封（未完成 409）
```

注意：`/jobs` 提交**会创建服务端状态**（与同步推理的无状态可重试不同）——重试提交前先确认是否已成功，避免重复建 job。

### 5.7 SDK 建议

契约已 100% 由 OpenAPI 描述（`/openapi.json` 实时可取，`docs/openapi.json` 为提交版）。SDK 应：

1. 由 OpenAPI 生成 typed 模型（`results[]` 的 per-item status 映射为联合返回类型）
2. 内建 422 pointer → 异常字段的映射、429 + Retry-After 退避
3. 默认 raw 编码发送（客户端拿到的就是字节），调试模式切 JSON

---

## 6. 实测收益（真实 yolov8 / TensorRT / bus.jpg 476KB / 100 请求）

| encoding | payload | p50 | p99 | mean | rps | errors |
|---|---|---|---|---|---|---|
| json+base64 | 649,958 B | 26.4 ms | 37.0 ms | 30.0 ms | 33.3 | 0 |
| raw body | 487,438 B | **23.7 ms** | 34.1 ms | **24.4 ms** | **41.1** | 0 |

载荷 −25%、p50 −10%、吞吐 +23%。复现：`scripts/server/bench_encoding.py`（纯标准库，`--out` 输出 markdown）。

---

## 7. 已知限制与后续路线

| 项 | 状态 | 说明 |
|---|---|---|
| multipart 多图二进制 | 未做 | raw 恒单图；有需求时按附加演进加入 |
| 每图独立参数 | 明确不做 | 参数矩阵爆炸，无真实场景支撑 |
| diffusion 请求参数 | 延后 | 适配器目前忽略请求体（参数全在 TOML 构造的采样输入里）；需"每请求组装 sampler input"的适配器改造，单独立项 |
| URL/object_ref 输入源 | 类型层预留 | 解析器不接受；待 SSRF/egress 策略层 |
| `model.version` | 恒空串 | 待版本指纹（权重/引擎/配置 hash）里程碑 |
| TLS / 每 Key 限流 / 用量计量 | 不在本次范围 | SME P0 清单 #6 |
| request-id 贯通 / 访问日志 / GPU 指标 | 不在本次范围 | SME P0 清单 #2（soak/容量基线的前置） |

---

## 8. 附录

### 8.1 关键文件索引

| 领域 | 文件 |
|---|---|
| 输入类型 | `src/models/io/common_input.h`（byte_source/image_input） |
| 参数系统 | `src/models/backend/param_spec.h`（ParamSpec/ParamValue/ParamSet/validate_params） |
| 图像解码 | `src/models/cv_image_input.h`（base64/raw 双分支，ImageInputLimits） |
| 请求上下文 | `src/models/backend/inference_context.h`（params 视图）、`backend_cv_model.h`（接线） |
| 信封解析 | `src/server/request_envelope.h`（JSON + raw 两入口，共享校验器） |
| 输出选项 | `src/server/output_options.h` |
| 服务内核 | `src/server/base_server_impl.h`（编码三分支/背压/deadline/batch/async） |
| 任务目录 | `src/factory/*_task.h`（11 个，含 param_specs） |
| 响应序列化 | `src/server/response_serializers.h`（fill_* + options） |
| 响应信封 | `src/common/http_response.h`（UnifiedResponse/ResponseItem/ResponseError） |
| 契约生成 | `scripts/contract_dump.cc` → `docs/contract_dump.json` → `scripts/gen_openapi.py` → `docs/openapi.json` + `src/server/openapi_doc.h` |
| 同步门禁 | `scripts/check_contract_sync.py`（CI: cpu-profile job） |
| 基准 | `scripts/server/bench_encoding.py`、`docs/bench/encoding_benchmark.md` |
| 旧契约快照 | `test/contract_baseline/v1_http_contract.md` |

### 8.2 测试资产

| 测试 | 覆盖 |
|---|---|
| `request_envelope_unittest`（18 用例） | 信封矩阵 + raw 头部 + **双编码 422 逐字节一致** |
| `param_spec_unittest` | 类型/范围/枚举/重复/容量护栏 |
| `server_e2e_contract_test`（31 用例） | 422/413/多图对齐/逐项隔离/item 上限/异步/双编码等价 |
| `object_detection_output_contract_unittest` | 参数阈值扫描单调性 + legacy 路径钉死 |
| `openapi_consistency_test` | 统一信封 schema、全部 Request_* 严格性 |
| `model_golden_test`（27 用例） | 数值零漂移证明（M1–M6 每步全绿） |

### 8.3 FAQ

**Q：为什么不用两套契约（v1/v2 并存）平滑迁移？**
A：项目当时没有规模化存量用户，双契约是给不存在的客户买保险。统一窗口只在首个外部用户接入之前——现在契约已"一次做对"，靠严格性 + 附加纪律保证永不再分叉（详见 §1.3）。

**Q：为什么参数校验这么严格？拼错键直接 422 不友好。**
A：静默忽略拼错的参数才是最不友好的——业务以为生效了。422 + pointer 让错误在第一秒暴露，且是 SDK 自动化的基础。

**Q：多图请求里一张图坏了会怎样？**
A：只有那一张失败：`results[k].status != 0` 且 `data: null`，其余项正常返回；HTTP 为 500（错误对监控可见）。

**Q：`model_run_timeout` 内请求排队也算吗？**
A：算。deadline 在进入服务那一刻打点，排队/取 worker/推理共享同一预算；预算耗尽时已完成项照常返回（`partial: true`）。
