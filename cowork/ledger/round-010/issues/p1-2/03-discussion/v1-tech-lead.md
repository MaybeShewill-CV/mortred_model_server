# 技术方案

- 问题: P1-2
- 版本: v1
- 对应列表: cowork/ledger/round-010/02-p0-p1-final.md v2

## 边界

- 只处理 CLIP 族多会话配置缺 `text_backend` 子表时，在 init 或契约层立刻点名该缺段。
- 一个 catalog id 一个 OS 进程。不推翻 HTTP 契约和现有后端。不新增 StatusCode。不做 OCR 识别、MOT HTTP。不把 RTDETR 当已上线检测器。
- 本版不覆盖 LightGlue、SAM、diffusion 的缺段点名，也不覆盖错会话绑定。

## 做法

### 推荐

在 `MultiSessionModel::init_sessions` 创建任何 session 之前，按声明的 `backend_key` 检查缺表。缺则返回已有的 `MODEL_INIT_FAILED`，失败详情必须包含缺失键名。本版验收键是 `text_backend`。抽出与生产路径相同的预检入口，用 CLIP 会话布局的假配置做无权重契约测。

代价小。其他 MultiSession 族会走同一段预检代码，但本版不以它们为验收对象。

### 否决

1. 四族加错会话一次做完。diffusion 不走 `MultiSessionModel`。错会话需要权重或假 engine。不能把做不到的全族点名写成验收。
2. 只在各家族 `on_init` 各写一份，或只打日志。仅日志无法黑盒判定点名。

## 改动面

- `src/models/backend/multi_session_model.h`：建会话前缺表预检，暴露生产路径同一入口。
- `test/multi_session_model_unittest.cc` 与必要的 CTest 别名：CLIP 形配置只有 `visual_backend`、缺少 `text_backend`，断言失败码和文案含 `text_backend`。

## 明确不改

LightGlue、SAM、diffusion 的缺段与错会话。P0-1 的 SAM 输出错码。P1-1 的脚手架。产品 toml、HTTP JSON、`status_code.h`、进程模型。

## 契约

缺段仍是 `MODEL_INIT_FAILED`。失败详情必须包含字面量 `text_backend`。不新增状态码。HTTP 形状不改。

## 性能与资源

只在 init 时多一次表存在性检查。预检失败不得创建 session。不增常驻进程，不增显存。

## 验收标准

1. 无权重环境下，CLIP 会话布局声明需要 `text_backend`，配置只有 `visual_backend`。走与生产相同的预检路径：返回失败且不是 OK，可读失败详情包含 `text_backend`。
2. 同一路径的状态码是 `MODEL_INIT_FAILED`，不是 `MODEL_EMPTY_OUTPUT` 或 `MODEL_OUTPUT_CONTRACT_FAILED`，且没有新状态码。
3. 上述测试在无 CLIP/LightGlue/SAM/diffusion 权重、无 vocab 的 WSL 上退出码 0。
4. 本条不以 LightGlue、SAM、diffusion 缺段或错会话的新用例为通过条件。既有 `SamOutputContract`、LightGlue 与 CLIP 输出契约测不得被本条改坏。

## 测试义务

QA 按验收 1–4 执行自测命令，核对文案含 `text_backend`。不测真权重起服、HTTP 字段、其他家族缺段、错会话。

## 自测命令

```bash
cmake --build build/full --target multi_session_model_unittest
ctest --test-dir build/full -R 'MultiSessionModel' --output-on-failure
ctest --test-dir build/full -R 'SamOutputContract|LightGlueOutputContract|OpenAiClipOutputContract' --output-on-failure
```

期望退出码都是 0。不得以缺权重 SKIP 冒充验收 1 通过。

## RD 不得自行决定

- 不得把验收扩大到 LightGlue、SAM、diffusion 或错会话。
- 不得新增或改名 StatusCode，不得改 HTTP 错误体。
- 点名字符串不得缺少字面量 `text_backend`。
- 不得依赖 vocab 或权重，不得把 tokenizer 失败当成已经点名 `text_backend`。
- 不得改 P0-1 或 P1-1。
