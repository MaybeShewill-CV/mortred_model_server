# 技术方案

- 问题: P1-2
- 版本: v2
- 对应列表: cowork/ledger/round-010/02-p0-p1-final.md v2

## 边界

- 覆盖 CLIP、LightGlue、SAM、diffusion 的缺段点名，以及至少一种错会话绑定。
- 一个 catalog id 一个 OS 进程。不推翻 HTTP 契约和现有后端。不新增 StatusCode。不做 OCR 识别、MOT HTTP。不把 RTDETR 当已上线检测器。不改 P0-1、P1-1。
- 无权重、无 vocab、无 engine。失败详情必须含被点名的键或段名。状态码用已有 `MODEL_INIT_FAILED`。

## 做法

### 推荐

1. 在 `MultiSessionModel::init_sessions` 创建 session 之前，按声明的 `backend_key` 预检。缺表，或声明键存在但不是本会话的 backend 表，立即 `MODEL_INIT_FAILED`，详情含声明键字面量。CLIP 用 `text_backend`，LightGlue 用 `matcher_backend` 或 `extractor_backend`，SAM 用 `decoder_backend` 或 `encoder_backend`。入口与生产路径相同。
2. diffusion 复用 `seed_ddpm_input` / `seed_sampler_input`。缺 `[DDPM_SAMPLER]` 时返回 `MODEL_INIT_FAILED`，可读详情含 `DDPM_SAMPLER`。不加载权重，不把缺 unet 写成验收。
3. 错会话：配置里声明的 `backend_key` 指向另一会话的表，或键存在但不是本会话 table。详情含声明键名。

### 否决

1. 只验收 CLIP 缺 `text_backend`。
2. 用真实权重或 engine IO 形状差发现错绑。
3. 让 diffusion 走 MultiSession，或把缺 `[DDPM_UNET]` 当验收。
4. 只打日志，或新增 StatusCode。

## 改动面

- `src/models/backend/multi_session_model.h`：建会话前缺表和错绑预检。
- diffusion 的 `seed_*` 仅在现有失败详情还不能被断言时补上可断言文案。
- `test/multi_session_model_unittest.cc` 与 `test/diffusion_param_unittest.cc`。

## 明确不改

产品 toml、HTTP JSON、`status_code.h`、进程模型、OCR、MOT、RTDETR、P0-1、P1-1、diffusion 权重加载。

## 契约

缺段与错会话都是 `MODEL_INIT_FAILED`。详情必须含对应字面量：`text_backend`、LightGlue 所缺声明键、SAM 所缺声明键、`DDPM_SAMPLER`、错会话的声明键。

## 性能与资源

预检失败不得创建 session、不得加载权重。不增常驻进程，不增显存。

## 验收标准

1. CLIP 缺 `text_backend`：无权重，同一预检路径失败，详情含 `text_backend`，状态码 `MODEL_INIT_FAILED`。
2. LightGlue 缺 `extractor_backend` 或 `matcher_backend` 之一：详情含所缺声明键，状态码 `MODEL_INIT_FAILED`。
3. SAM 缺 `encoder_backend` 或 `decoder_backend` 之一：详情含所缺声明键，状态码 `MODEL_INIT_FAILED`。
4. diffusion 缺 `[DDPM_SAMPLER]`：经 `seed_ddpm_input` / `seed_sampler_input` 返回 `MODEL_INIT_FAILED`，详情含 `DDPM_SAMPLER`。不加载权重。
5. 至少一种错会话：声明键指向另一会话的表，或键存在但不是本会话 table。详情含声明键名，状态码 `MODEL_INIT_FAILED`。
6. 验收 1–5 都不是 `MODEL_EMPTY_OUTPUT` 或 `MODEL_OUTPUT_CONTRACT_FAILED`，且无新状态码。
7. 无权重、无 vocab、无 engine 时上述测试退出码 0。既有 `SamOutputContract`、`LightGlueOutputContract`、`OpenAiClipOutputContract` 不被改坏。不得以 SKIP 冒充通过。

## 测试义务

QA 按验收 1–7 执行。不测真权重起服、HTTP 字段、OCR、MOT、RTDETR、P0-1、P1-1。

## 自测命令

```bash
cmake --build build/full --target multi_session_model_unittest
cmake --build build/full --target diffusion_param_unittest
ctest --test-dir build/full -R 'MultiSessionModel' --output-on-failure
ctest --test-dir build/full -R 'DiffusionParam' --output-on-failure
ctest --test-dir build/full -R 'SamOutputContract|LightGlueOutputContract|OpenAiClipOutputContract' --output-on-failure
```

期望退出码都是 0。

## RD 不得自行决定

- 不得漏掉 CLIP、LightGlue、SAM、diffusion 或缺会话中的任一项。
- 不得新增 StatusCode，不得改 HTTP 错误体。
- 不得用模糊措辞代替声明键或 `DDPM_SAMPLER`。
- 不得依赖 vocab、权重或 engine。
- 不得改 P0-1 或 P1-1。
