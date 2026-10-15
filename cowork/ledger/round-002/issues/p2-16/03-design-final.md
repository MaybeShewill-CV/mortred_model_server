# 技术方案

- 问题: P2-16
- 版本: v1
- 对应列表: v3

## 边界

本方案只改一份中文添加入口文档里 densenet 示例的 id、`--model`、`model=` 与 family。一次打开 `docs/how_to_add_new_server.zh-cn.md`。现示例是 `Entry{"DENSENET", "densenet classification", "DENSENET_CLASSIFICATION_SERVER", ...}`，参数在 `classification_param_specs()` 结束、没有 family 实参；下文是 `--model DENSENET`、`model = "DENSENET"`。

对照物是英文同名指南与仓库真值，本条不改它们。`docs/how_to_add_new_server.md` 示例已是 `Entry{"densenet", ..., classification_param_specs(), "densenet"}`，以及 `--model densenet`、`model = "densenet"`。`src/factory/classification_task.h` 的 catalog 行是 `Entry{"densenet", "Densenet classification", "DENSENET_CLASSIFICATION_SERVER", ..., classification_param_specs(), "densenet"}`。`conf/server/classification/densenet/densenet_server_config.toml` 的 `model="densenet"`，模型表名是 `[densenet]`。`--model` 大小写敏感，必须等于 catalog 的 `model_section`。

锁定的可观察变化（队列 v3）：只打开 `docs/how_to_add_new_server.zh-cn.md`。示例的 `model_section`、`--model`、`model=` 和 family 都是 `densenet`。不升为 P0/P1。不把 `densenet` 放进覆盖集。不扩到英文指南、两份教程、契约示例、OpenAPI 或产品代码。

验收只打开文件。不拉起模型进程，不跑 GPU、calibrate 或压测。

## 方案

### 推荐

做法一：只把中文指南 densenet 示例的四处字符串改成与英文指南、catalog、server toml 一致的小写 `densenet`，并补上 family 实参。

**第 2 步 catalog 示例**（`docs/how_to_add_new_server.zh-cn.md` 代码块）由：

```cpp
        Entry{"DENSENET", "densenet classification", "DENSENET_CLASSIFICATION_SERVER",
              &create_densenet_classifier<ImageInput, Output>,
              &jinq::server::response::fill_classification,
              classification_param_specs()},
```

改为：

```cpp
        Entry{"densenet", "densenet classification", "DENSENET_CLASSIFICATION_SERVER",
              &create_densenet_classifier<ImageInput, Output>,
              &jinq::server::response::fill_classification,
              classification_param_specs(), "densenet"},
```

**第 3 步启动示例**由：

```
mortred-model-server.out --model DENSENET /path/to/densenet_server_config.toml
mortred-model-benchmark.out --model DENSENET /path/to/densenet_config.toml [image]
```

改为：

```
mortred-model-server.out --model densenet /path/to/densenet_server_config.toml
mortred-model-benchmark.out --model densenet /path/to/densenet_config.toml [image]
```

**配置说明第 1 点**由：

`在 `conf/server/<task>/<model>/` 下添加配置，写上 `model = "DENSENET"` 和`

改为：

`在 `conf/server/<task>/<model>/` 下添加配置，写上 `model = "densenet"` 和`

`server_section` 仍为 `DENSENET_CLASSIFICATION_SERVER`（与真值 toml 节名一致，本条不改）。开篇「以新增 densenet 图像分类服务为例」、路径里的 `densenet_server_config.toml` / `densenet_config.toml`、`create_densenet_classifier`、节标题与其余段落保持原样。只改这一份中文文件。

代价：抄过旧大写 `DENSENET` 示例的笔记要对齐小写。没有进程、显存、engine 或压测代价。

### 备选

做法二：在旧大写示例旁加注「实际请用小写 densenet」，旧 `Entry{"DENSENET"...}`、`--model DENSENET`、`model = "DENSENET"` 与缺 family 仍保留。代价更小，但一次打开中文指南仍能读到与 catalog 不一致的示例，对不上锁定句。否决。

做法三：改 `classification_task.h` 或 `densenet_server_config.toml`，把真值改成大写 `DENSENET` 以迁就中文文档。这与现有 catalog 约定、`--model` 大小写敏感、以及英文指南已对齐的小写 id 相反。否决。

做法四：连同 `docs/how_to_add_new_server.md`、两份教程、`docs/api-contract*.md`、`docs/openapi.json`、`README.zh-cn.md`、或覆盖集外 `YOLOV5` / `BISENETV2` / `ATTENTIVE_GAN_DERAIN` 一起改。英文指南已正确；教程与契约是别的开放项（P2-14 已关 worker_nums；P2-17 是契约 `model.name`）。否决。

## 改动面

只改 docs 模块里的一份中文添加入口。

- `docs/how_to_add_new_server.zh-cn.md`：第 2 步 `Entry` 的 `model_section` 由 `DENSENET` 改为 `densenet`，并在 `classification_param_specs()` 后追加 `"densenet"` family；第 3 步两处 `--model DENSENET` 改为 `--model densenet`；配置说明里的 `model = "DENSENET"` 改为 `model = "densenet"`。其余行不动。

## 不改清单

保持原样的有：`docs/how_to_add_new_server.md`（英文对照，已是 `densenet`）；`docs/tutorials_of_model_servers.md`、`docs/tutorials_of_model_servers.zh-cn.md`（含 `--model mobilenetv2`、`--model superpoint`，以及覆盖集外的 `--model YOLOV5`、`--model BISENETV2`、`--model ATTENTIVE_GAN_DERAIN`；两份教程都不因本条写入或改写 densenet）；`README.md`、`README.zh-cn.md`（入口链接可继续指向本指南，本条不改 README 正文）；`docs/api-contract.md`、`docs/api-contract.zh-cn.md`、`docs/openapi.json`、`docs/contract_dump.json`、`src/server/openapi_doc.h`；`docs/deployment.md`、`docs/deployment.zh-cn.md`；`docs/about_model_server_configuration.md` 及其中文；`docs/model-developer-guide.md`、`docs/model-developer-guide.zh-cn.md`；`docs/perf/coverage_set_http_rps.md`、`scripts/server/http_infer_rps.py`；`scripts/check_consistency.py`、`scripts/gen_openapi.py`。

代码与配置保持原样的有：`src/factory/classification_task.h`（真值已是 `densenet` / family `"densenet"`，本条只改文档示例）；`src/factory/cv_catalog.h`、各 task catalog；`conf/server/classification/densenet/densenet_server_config.toml`（已是 `model="densenet"`、`[densenet]`）；其余 `conf/server/**`；产品 toml；`conf/packs/`；`src/apps/model_server_main.cpp`、`src/apps/product_index.h`；gateway（:8080）与 supervisor（:8787）的进程关系。每个 catalog id 仍是一个 OS 进程。每个 worker 仍持有各自的 TensorRT execution context。engine、static shape、batch=1、type=tensorrt 产品性能路径不改。

## 契约

HTTP 请求、响应、OpenAPI 和 tensor 形状都不变。产品 toml 与 server toml 的键和值不变。变化只在中文添加入口文档的 densenet 示例字符串：`model_section`、`--model`、`model=`、family 四处与 catalog / 英文指南 / densenet server toml 对齐为小写 `densenet`。运行时 `--model` 仍大小写敏感；本条不新增或改写解析行为。

## 性能与资源

不生成 engine，不改 static shape、batch=1。每个 worker 仍各自持有自己的 TensorRT execution context，不增加共享 context 或跨 worker context 池。不改任何 `conf/server` 里的 `worker_nums`。不拉起 gateway、supervisor 或模型进程。不跑 `mortredctl prepare`、`mortredctl calibrate` 或 `http_infer_rps.py`。官方 HTTP 压测并发仍是已部署 `worker_nums` 的两倍，本条不改这条规则。显存与常驻进程无变化。gpu profile 仍是 CUDA 12.x + TensorRT 10.x。cpu profile 仍不链 TensorRT。

## 验收标准

1. 一次打开 `docs/how_to_add_new_server.zh-cn.md`。第 2 步示例出现 `Entry{"densenet", "densenet classification", "DENSENET_CLASSIFICATION_SERVER",`，且同一 `Entry` 在 `classification_param_specs()` 之后带有 `"densenet"}`。不再出现 `Entry{"DENSENET"`，也不再出现以 `classification_param_specs()},` 结尾且缺少 family 的 densenet 示例行。
2. 同一次打开。第 3 步出现 `mortred-model-server.out --model densenet` 与 `mortred-model-benchmark.out --model densenet`。不再出现 `--model DENSENET`。配置说明出现 `model = "densenet"`，不再出现 `model = "DENSENET"`。
3. 打开对照：`docs/how_to_add_new_server.md` 仍含 `Entry{"densenet"`、`--model densenet`、`model = "densenet"`、family `"densenet"`。`src/factory/classification_task.h` 仍含 `Entry{"densenet"` 与 family `"densenet"`。`conf/server/classification/densenet/densenet_server_config.toml` 仍含 `model="densenet"`。本条 diff 不触及这三处对照文件。
4. 打开 `docs/tutorials_of_model_servers.md` 与 `docs/tutorials_of_model_servers.zh-cn.md`：启动命令保持现样（含覆盖集外大写 `YOLOV5` / `BISENETV2` / `ATTENTIVE_GAN_DERAIN`）。本条 diff 不触及两份教程、`docs/openapi.json`、两份 api-contract、任一产品/server 代码。

判定方式是打开文件。不启动 `mortred-model-server.out`，不跑 GPU，不跑 calibrate，不跑压测，不读 `/metrics`。

## 黑盒步骤

1. QA 在本机 WSL Ubuntu 工作目录 `/mnt/g/Codex/mortred_model_server` 执行下方自测命令，期望退出码为 0。
2. QA 用 diff 确认本条改动只出现在 `docs/how_to_add_new_server.zh-cn.md`；仓库里原有未跟踪的 ledger 不计入本条。
3. QA 按验收标准 1–4 打开文件核对。不拉起模型，不跑 calibrate，不跑压测。不把 `densenet` 放进覆盖集，不把本条升为 P0/P1。英文指南、catalog、densenet toml 保持原样即通过。

## 自测命令

在本机 WSL Ubuntu 执行。工作目录 `/mnt/g/Codex/mortred_model_server`。期望退出码为 0。断言失败时退出码为 1。本命令只读文件，不拉起 GPU，不跑 calibrate，不跑压测。

```bash
cd /mnt/g/Codex/mortred_model_server
python3 - <<'PY'
import pathlib
root = pathlib.Path("/mnt/g/Codex/mortred_model_server")
zh = (root / "docs/how_to_add_new_server.zh-cn.md").read_text(encoding="utf-8")
en = (root / "docs/how_to_add_new_server.md").read_text(encoding="utf-8")
hdr = (root / "src/factory/classification_task.h").read_text(encoding="utf-8")
toml = (root / "conf/server/classification/densenet/densenet_server_config.toml").read_text(encoding="utf-8")
tut_en = (root / "docs/tutorials_of_model_servers.md").read_text(encoding="utf-8")
tut_zh = (root / "docs/tutorials_of_model_servers.zh-cn.md").read_text(encoding="utf-8")

old_entry_id = 'Entry{"DENSENET"'
new_entry = (
    'Entry{"densenet", "densenet classification", "DENSENET_CLASSIFICATION_SERVER",\n'
    '              &create_densenet_classifier<ImageInput, Output>,\n'
    '              &jinq::server::response::fill_classification,\n'
    '              classification_param_specs(), "densenet"},'
)
old_entry_no_family = (
    'Entry{"DENSENET", "densenet classification", "DENSENET_CLASSIFICATION_SERVER",\n'
    '              &create_densenet_classifier<ImageInput, Output>,\n'
    '              &jinq::server::response::fill_classification,\n'
    '              classification_param_specs()},'
)

assert new_entry in zh, "zh missing aligned densenet Entry with family"
assert old_entry_id not in zh, "zh still has Entry{\"DENSENET\""
assert old_entry_no_family not in zh, "zh still has densenet Entry without family"
assert "--model densenet" in zh
assert "--model DENSENET" not in zh
assert 'model = "densenet"' in zh
assert 'model = "DENSENET"' not in zh
assert zh.count("--model densenet") == 2
assert "DENSENET_CLASSIFICATION_SERVER" in zh

assert 'Entry{"densenet"' in en
assert 'classification_param_specs(), "densenet"' in en
assert "--model densenet" in en
assert 'model = "densenet"' in en
assert 'Entry{"densenet"' in hdr
assert 'classification_param_specs(), "densenet"' in hdr
assert 'model="densenet"' in toml
assert "[densenet]" in toml

assert "--model YOLOV5" in tut_en and "--model YOLOV5" in tut_zh
assert "--model BISENETV2" in tut_en and "--model BISENETV2" in tut_zh
assert "--model ATTENTIVE_GAN_DERAIN" in tut_en and "--model ATTENTIVE_GAN_DERAIN" in tut_zh
print("p2-16-zh-howto-densenet ok")
PY
```

## RD 实现时注意

- 四处字符串固定为推荐节给出的小写 `densenet`：`model_section`、`--model`（server 与 benchmark 两行）、`model = "densenet"`、family `"densenet"`。不得只改其中一部分。
- `server_section` 保持 `DENSENET_CLASSIFICATION_SERVER`。不得把 TOML 节名示例改成小写。
- 旧 `Entry{"DENSENET"`、`--model DENSENET`、`model = "DENSENET"` 以及无 family 的示例行必须从中文指南删除，不得旁注并存。
- 不改 `docs/how_to_add_new_server.md`。不改 `classification_task.h`、`densenet_server_config.toml`。不改两份教程、OpenAPI、api-contract、README。
- 不把 `densenet` 写入覆盖集或两份 `tutorials_of_model_servers*`。检测、场景分割、增强三条教程命令保持大写实参。
- 不改 catalog() 真值、产品 toml、server toml、`model_server_main.cpp`、gateway(:8080)、supervisor(:8787)、TensorRT context 和 engine。一个 catalog id 仍是一个 OS 进程。
- 不改官方 HTTP 压测规则。本条的自测保持为上面的读文件命令。不把本条升为 P0/P1，不扩 scope。

## 确认
| 角色 | 结论 | 原因 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| PM | 同意 | 无 | v1 | cowork/ledger/round-002/issues/p2-16/03-discussion/v1-sign-pm.md |
| RD | 同意 | 无 | v1 | cowork/ledger/round-002/issues/p2-16/03-discussion/v1-sign-rd.md |
| 技术负责人 | 同意 | 无 | v1 | cowork/ledger/round-002/issues/p2-16/03-discussion/v1-sign-tech-lead.md |
