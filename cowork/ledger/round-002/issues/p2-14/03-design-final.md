# 技术方案

- 问题: P2-14
- 版本: v1
- 对应列表: v3

## 边界

本方案只改两份模型服务教程里关于 `worker_nums` 的那一句。一次打开 `docs/tutorials_of_model_servers.md` 和 `docs/tutorials_of_model_servers.zh-cn.md`。英文现句是 `Shipped configs default to `worker_nums=1`. Raise it only if you have GPU headroom.`。中文现句是 `仓库自带配置默认 `worker_nums=1`，仅在显存足够时再加大。`。两句都暗示读者可以加大仓库自带 `conf/server` 里的 `worker_nums`。

对照物是现有部署 FAQ，本条不改它们。`docs/deployment.md` FAQ「Where do I set worker_nums?」写明：Git `conf/server` stays `worker_nums=1`. Put concurrency on the machine pack (`[pack.<ID>] worker_nums`) and inject via `MORTRED_PACK`. `docs/deployment.zh-cn.md` FAQ「worker_nums 写在哪？」写明：git 里 `conf/server` 保持 `worker_nums=1`。并发写在机器 pack 的 `[pack.<ID>] worker_nums`，由 `MORTRED_PACK` 注入。§10 同样要求校准用 `mortredctl calibrate --write-pack` 写到机器本地 pack，不把校准后的 `worker_nums` 提交进仓库。

锁定的可观察变化（队列 v3）：一次打开两份教程，两份里关于 `worker_nums` 的那一句都写明 `conf/server` 保持 1，校准出的数值写在机器 pack。P2-15 已并入本条。不升为 P0/P1。不扩到 `--model`、添加入口、契约示例或产品代码。

验收只打开文件。不拉起模型进程，不跑 GPU、calibrate 或压测。

## 方案

### 推荐

做法一：只替换两份教程里 `worker_nums` 那一句，使语义与部署 FAQ 对齐。

英文（`docs/tutorials_of_model_servers.md`，共用节「Shared: binary and client」里启动命令块之后那一句）由：

`Shipped configs default to `worker_nums=1`. Raise it only if you have GPU headroom.`

改为：

`Shipped `conf/server` keeps `worker_nums=1`. Put calibrated concurrency on the machine pack (`[pack.<ID>] worker_nums`) and inject via `MORTRED_PACK` (see [deployment.md](deployment.md)).`

中文（`docs/tutorials_of_model_servers.zh-cn.md`，共用节「共用：二进制与客户端」里启动命令块之后那一句）由：

`仓库自带配置默认 `worker_nums=1`，仅在显存足够时再加大。`

改为：

`仓库 `conf/server` 保持 `worker_nums=1`。校准后的并发写在机器 pack 的 `[pack.<ID>] worker_nums`，由 `MORTRED_PACK` 注入（见 [deployment.zh-cn.md](deployment.zh-cn.md)）。`

每份只改这一句。启动命令、`--model`、toml 路径、演示客户端、节标题和其余段落保持原样。`docs/deployment.md` / `docs/deployment.zh-cn.md` 已写清 pack 注入与 `calibrate --write-pack`，本条不改部署文档。

代价：抄过旧句「显存够就加大仓库配置」的笔记要对齐新句。没有进程、显存、engine 或压测代价。

### 备选

做法二：在旧句旁加注「实际请改机器 pack」，旧句仍保留「Raise it only if you have GPU headroom」或「仅在显存足够时再加大」。代价更小，但一次打开两份教程仍能读到鼓励加大仓库自带 `worker_nums` 的原文，对不上锁定句。否决。

做法三：改仓库里 `conf/server/**/*.toml` 的 `worker_nums`，或改 catalog / gateway / supervisor / 产品 toml，让「加大仓库配置」变成真。这与角色约束相反：`conf/server` 的 `worker_nums` 在 git 里保持 1，并发只落在机器 pack 的 `[pack.<ID>] worker_nums` 并由 `MORTRED_PACK` 注入。否决。

做法四：重写 `docs/deployment.md` / `docs/deployment.zh-cn.md` 的 FAQ 或 §10，或连同 `YOLOV5` / `BISENETV2` / `ATTENTIVE_GAN_DERAIN` 启动命令、`densenet`、契约示例一起改。部署文档已经正确；覆盖集外三条大写命令与 `densenet`（P2-16）不在本条。否决。

## 改动面

只改 docs 模块里的两份教程各一句。

- `docs/tutorials_of_model_servers.md`：共用节 `worker_nums` 句改为上节推荐英文句。其余行不动。
- `docs/tutorials_of_model_servers.zh-cn.md`：共用节 `worker_nums` 句改为上节推荐中文句。其余行不动。

## 不改清单

保持原样的有：两份教程的启动命令与 toml 路径（含 `--model mobilenetv2`、`--model superpoint`，以及覆盖集外的 `--model YOLOV5`、`--model BISENETV2`、`--model ATTENTIVE_GAN_DERAIN`）；共用示例 `--model <CATALOG_ID>`；节标题；`test_server.py` 行；两份教程都不出现 `densenet`。

`docs/deployment.md`、`docs/deployment.zh-cn.md` 的 FAQ 与 §10（含 `MORTRED_PACK`、`mortredctl calibrate --write-pack`）保持原样，仅作对照。`docs/how_to_add_new_server.md`、`docs/how_to_add_new_server.zh-cn.md`、两份 README、`docs/api-contract.md`、`docs/api-contract.zh-cn.md`、`docs/openapi.json`、`docs/about_model_server_configuration.md` 及其中文、`docs/perf/coverage_set_http_rps.md`、`scripts/server/http_infer_rps.py` 保持原样。

代码与配置保持原样的有：`src/factory/cv_catalog.h`、各 task catalog、`src/apps/model_server_main.cpp`、`src/apps/product_index.h`；全部 `conf/server/**` 里已有的 `worker_nums=1` 数值本身（本条只改教程措辞，不改 toml）；产品 toml；`conf/packs/` 仓库示例。gateway（:8080）与 supervisor（:8787）的进程关系保持原样。每个 catalog id 仍是一个 OS 进程。每个 worker 仍持有各自的 TensorRT execution context。engine、static shape、batch=1、type=tensorrt 产品性能路径不改。

## 契约

HTTP 请求、响应、OpenAPI 和 tensor 形状都不变。产品 toml 与 server toml 的键和值不变。变化只在两份教程各一句 `worker_nums` 说明：明确 git 里 `conf/server` 保持 `worker_nums=1`，校准并发写在机器 pack 的 `[pack.<ID>] worker_nums`，经 `MORTRED_PACK` 注入。运行时注入机制仍以部署文档为准，本条不新增或改写注入行为。

## 性能与资源

不生成 engine，不改 static shape、batch=1。每个 worker 仍各自持有自己的 TensorRT execution context，不增加共享 context 或跨 worker context 池。不改任何 `conf/server` 里的 `worker_nums=1`。不拉起 gateway、supervisor 或模型进程。不跑 `mortredctl prepare`、`mortredctl calibrate` 或 `http_infer_rps.py`。官方 HTTP 压测并发仍是已部署 `worker_nums` 的两倍，本条不改这条规则，也不改 `docs/perf/coverage_set_http_rps.md`。显存与常驻进程无变化。gpu profile 仍是 CUDA 12.x + TensorRT 10.x。cpu profile 仍不链 TensorRT。

## 验收标准

1. 一次打开 `docs/tutorials_of_model_servers.md` 和 `docs/tutorials_of_model_servers.zh-cn.md`。英文出现整句 `Shipped `conf/server` keeps `worker_nums=1`. Put calibrated concurrency on the machine pack (`[pack.<ID>] worker_nums`) and inject via `MORTRED_PACK` (see [deployment.md](deployment.md)).`。中文出现整句 `仓库 `conf/server` 保持 `worker_nums=1`。校准后的并发写在机器 pack 的 `[pack.<ID>] worker_nums`，由 `MORTRED_PACK` 注入（见 [deployment.zh-cn.md](deployment.zh-cn.md)）。`。两份都不再出现旧英文句 `Shipped configs default to `worker_nums=1`. Raise it only if you have GPU headroom.`，也不再出现旧中文句 `仓库自带配置默认 `worker_nums=1`，仅在显存足够时再加大。`。
2. 同一次打开。两份教程里分类、特征点、检测、场景分割、增强的启动命令与 toml 路径保持现样（含 `--model mobilenetv2`、`--model superpoint`、`--model YOLOV5`、`--model BISENETV2`、`--model ATTENTIVE_GAN_DERAIN`）。两份教程都没有 `densenet`。
3. 打开 `docs/deployment.md` 与 `docs/deployment.zh-cn.md` 的 `worker_nums` FAQ：英文仍含 `Git `conf/server` stays `worker_nums=1`` 与 `MORTRED_PACK`；中文仍含 `git 里 `conf/server` 保持 `worker_nums=1`` 与 `MORTRED_PACK`。本条 diff 不触及这两份部署文档。
4. 打开任一份现有覆盖集相关 server toml（例如 `conf/server/classification/mobilenetv2/mobilenetv2_server_config.toml`），`worker_nums=1` 仍在。不出现本条对 `conf/server`、catalog、gateway、supervisor 或产品代码的改动。

判定方式是打开文件。不启动 `mortred-model-server.out`，不跑 GPU，不跑 calibrate，不跑压测，不读 `/metrics`。

## 黑盒步骤

1. QA 在本机 WSL Ubuntu 工作目录 `/mnt/g/Codex/mortred_model_server` 执行下方自测命令，期望退出码为 0。
2. QA 用 diff 确认本条改动只出现在 `docs/tutorials_of_model_servers.md` 和 `docs/tutorials_of_model_servers.zh-cn.md`，且每份只有 `worker_nums` 那一句被替换为推荐句；仓库里原有未跟踪的 ledger 不计入本条。
3. QA 按验收标准 1–4 打开文件核对。不拉起模型，不跑 calibrate，不跑压测。`YOLOV5`、`BISENETV2`、`ATTENTIVE_GAN_DERAIN` 仍为大写，以及两份教程不出现 `densenet`，都算本条通过。QA 不把 `densenet` 放进覆盖集，不把本条升为 P0/P1。

## 自测命令

在本机 WSL Ubuntu 执行。工作目录 `/mnt/g/Codex/mortred_model_server`。期望退出码为 0。断言失败时退出码为 1。本命令只读文件，不拉起 GPU，不跑 calibrate，不跑压测。

```bash
cd /mnt/g/Codex/mortred_model_server
python3 - <<'PY'
import pathlib
root = pathlib.Path("/mnt/g/Codex/mortred_model_server")
en = (root / "docs/tutorials_of_model_servers.md").read_text(encoding="utf-8")
zh = (root / "docs/tutorials_of_model_servers.zh-cn.md").read_text(encoding="utf-8")
dep_en = (root / "docs/deployment.md").read_text(encoding="utf-8")
dep_zh = (root / "docs/deployment.zh-cn.md").read_text(encoding="utf-8")
cls_toml = (root / "conf/server/classification/mobilenetv2/mobilenetv2_server_config.toml").read_text(encoding="utf-8")

old_en = "Shipped configs default to `worker_nums=1`. Raise it only if you have GPU headroom."
new_en = (
    "Shipped `conf/server` keeps `worker_nums=1`. "
    "Put calibrated concurrency on the machine pack (`[pack.<ID>] worker_nums`) "
    "and inject via `MORTRED_PACK` (see [deployment.md](deployment.md))."
)
old_zh = "仓库自带配置默认 `worker_nums=1`，仅在显存足够时再加大。"
new_zh = (
    "仓库 `conf/server` 保持 `worker_nums=1`。"
    "校准后的并发写在机器 pack 的 `[pack.<ID>] worker_nums`，"
    "由 `MORTRED_PACK` 注入（见 [deployment.zh-cn.md](deployment.zh-cn.md)）。"
)

assert new_en in en, "en missing new worker_nums sentence"
assert old_en not in en, "en still has old worker_nums sentence"
assert en.count("worker_nums") >= 1
assert new_zh in zh, "zh missing new worker_nums sentence"
assert old_zh not in zh, "zh still has old worker_nums sentence"
assert zh.count("worker_nums") >= 1

assert "Raise it only if you have GPU headroom" not in en
assert "仅在显存足够时再加大" not in zh

cls_cmd = (
    "./mortred-model-server.out --model mobilenetv2 \\\n"
    "  ../conf/server/classification/mobilenetv2/mobilenetv2_server_config.toml"
)
fp_cmd = (
    "./mortred-model-server.out --model superpoint \\\n"
    "  ../conf/server/feature_point/superpoint/superpoint_server_config.toml"
)
yolo_cmd = (
    "./mortred-model-server.out --model YOLOV5 \\\n"
    "  ../conf/server/object_detection/yolov5/yolov5l_server_config.toml"
)
bise_cmd = (
    "./mortred-model-server.out --model BISENETV2 \\\n"
    "  ../conf/server/scene_segmentation/bisenetv2/bisenetv2_server_config.toml"
)
gan_cmd = (
    "./mortred-model-server.out --model ATTENTIVE_GAN_DERAIN \\\n"
    "  ../conf/server/enhancement/attentive_gan_derain/attentive_gan_server_config.toml"
)

for name, text in (("en", en), ("zh", zh)):
    assert text.count(cls_cmd) == 1, name + " classification command"
    assert text.count(fp_cmd) == 1, name + " feature command"
    assert text.count(yolo_cmd) == 1, name + " yolo command"
    assert text.count(bise_cmd) == 1, name + " bisenet command"
    assert text.count(gan_cmd) == 1, name + " attentive command"
    assert "densenet" not in text.lower(), name
    assert "--model <CATALOG_ID>" in text, name

assert "Git `conf/server` stays `worker_nums=1`" in dep_en
assert "MORTRED_PACK" in dep_en
assert "git 里 `conf/server` 保持 `worker_nums=1`" in dep_zh
assert "MORTRED_PACK" in dep_zh
assert "worker_nums=1" in cls_toml
print("p2-14-tutorial-worker-nums ok")
PY
```

## RD 实现时注意

- 英文句固定为推荐节给出的整句；中文句固定为推荐节给出的整句。两份教程都改。不得只改一份。
- 旧英文句与旧中文句必须从两份教程中删除，不得旁注并存。
- 不改 `docs/deployment.md`、`docs/deployment.zh-cn.md`（除非对照验收发现被误改，应恢复）。不改任何 `conf/server` 的 `worker_nums=1` 数值。
- 检测、场景分割、增强三条命令保持 `--model YOLOV5`、`--model BISENETV2`、`--model ATTENTIVE_GAN_DERAIN`。两份教程不写入 `densenet`。不改 `docs/how_to_add_new_server.zh-cn.md`。
- 不改 catalog()、产品 toml、server toml、`model_server_main.cpp`、gateway(:8080)、supervisor(:8787)、TensorRT context 和 engine。一个 catalog id 仍是一个 OS 进程。
- 不改官方 HTTP 压测规则。并发仍是已部署 `worker_nums` 的两倍。本条的自测保持为上面的读文件命令。不把本条升为 P0/P1，不扩 scope。

## 确认
| 角色 | 结论 | 原因 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| PM | 同意 | 无 | v1 | cowork/ledger/round-002/issues/p2-14/03-discussion/v1-sign-pm.md |
| RD | 同意 | 无 | v1 | cowork/ledger/round-002/issues/p2-14/03-discussion/v1-sign-rd.md |
| 技术负责人 | 同意 | 无 | v1 | cowork/ledger/round-002/issues/p2-14/03-discussion/v1-sign-tech-lead.md |

