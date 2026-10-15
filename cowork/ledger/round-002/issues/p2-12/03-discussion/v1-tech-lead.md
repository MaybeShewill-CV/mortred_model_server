# 技术方案

- 问题: P2-12
- 版本: v1
- 对应列表: v3

## 边界

本方案只改两份模型服务教程里覆盖集两条启动命令的 `--model`。一次打开 `docs/tutorials_of_model_servers.md` 和 `docs/tutorials_of_model_servers.zh-cn.md`。分类命令的 `--model` 是 `mobilenetv2`。特征点命令的 `--model` 是 `superpoint`。

对照物是现有注册，本条不改它们。`src/apps/model_server_main.cpp` 写明 `--model` 大小写敏感，且必须等于 catalog 的 `model_section`；找不到时打印 `unknown --model`。`src/factory/classification_task.h` 里 `mobilenetv2` 那一行的 id 与 family 都是 `mobilenetv2`，`server_section` 是 `MOBILENETV2_CLASSIFICATION_SERVER`。`src/factory/feature_point_task.h` 里 `superpoint` 那一行的 id 与 family 都是 `superpoint`，`server_section` 是 `SUPERPOINT_FP_SERVER`。`conf/server/classification/mobilenetv2/mobilenetv2_server_config.toml` 的 `model="mobilenetv2"`，并有表 `[mobilenetv2]`。`conf/server/feature_point/superpoint/superpoint_server_config.toml` 的 `model="superpoint"`，并有表 `[superpoint]`。`src/factory/cv_catalog.h` 写明 catalog id 是 `model_section` 且为小写。

验收只打开文件。不拉起模型进程，不跑 GPU、calibrate 或压测。

## 做法

### 推荐

做法一：只替换两份教程里分类命令和特征点命令的 `--model` 实参。

英文和中文各两处，共四处。分类命令由 `--model MOBILENETV2` 改为 `--model mobilenetv2`，下一行仍是 `../conf/server/classification/mobilenetv2/mobilenetv2_server_config.toml`。特征点命令由 `--model SUPERPOINT` 改为 `--model superpoint`，下一行仍是 `../conf/server/feature_point/superpoint/superpoint_server_config.toml`。每条命令保持现在的两行和反斜杠续行，diff 只有这四个实参。

代价：抄过这两条大写口令的笔记要改成小写。节标题里的 `MOBILENETV2`、`SUPERPOINT` 仍在，打开文件时以命令行的 `--model` 判定。没有进程、显存、engine 或压测代价。

### 否决

做法二：把 catalog 的 `model_section`、server toml 的 `model=` 和表名改成 `MOBILENETV2`、`SUPERPOINT`，让教程里现有大写命令能通过查找。代价是改 `classification_task.h`、`feature_point_task.h` 和两份 server toml，并牵动 `config.contains(cli.model)` 所要求的表名。`cv_catalog.h` 要求 catalog id 为小写。这超出本条锁定的文档字符串。

做法三：在命令旁加注「实际请用小写」，或连节标题、`YOLOV5`、`BISENETV2`、`ATTENTIVE_GAN_DERAIN`、`worker_nums` 那一句、中文添加入口一起改。旁注留下的命令仍是 `--model MOBILENETV2` 和 `--model SUPERPOINT`，一次打开两份教程仍看不到锁定的那两个小写实参。`yolov5l`、`bisenetv2`、`attentive_gan` 在覆盖集外，这三条启动命令留在附录，不跟这两条绑在同一次验收。`worker_nums` 那一句是 P2-14。`densenet` 是 P2-16 的中文示例，不进入覆盖集，也不写入这两份教程。

## 改动面

只改 docs 模块里的两份教程。

- `docs/tutorials_of_model_servers.md`：分类命令的 `--model` 改为 `mobilenetv2`。特征点命令的 `--model` 改为 `superpoint`。两条命令的二进制、续行和 toml 路径保持原样。
- `docs/tutorials_of_model_servers.zh-cn.md`：同样两处，实参同样是 `mobilenetv2` 和 `superpoint`。

## 明确不改

保持原样的有：两份教程的节标题 `## Classification (`MOBILENETV2`)`、`## Feature point (`SUPERPOINT`)`、`## 分类（`MOBILENETV2`）`、`## 特征点（`SUPERPOINT`）`；共用示例 `--model <CATALOG_ID>`；英文句 `Shipped configs default to `worker_nums=1`. Raise it only if you have GPU headroom.`；中文句 `仓库自带配置默认 `worker_nums=1`，仅在显存足够时再加大。`；`--model YOLOV5`、`--model BISENETV2`、`--model ATTENTIVE_GAN_DERAIN` 三条命令及其 toml 路径和对应的 `test_server.py` 行；已经是小写的 `test_server.py --server mobilenetv2` 与 `--server superpoint`。

代码与配置保持原样的有：`src/factory/classification_task.h`、`src/factory/feature_point_task.h`、`src/factory/cv_catalog.h`、`src/apps/model_server_main.cpp`、`src/apps/product_index.h` 里带大写示例的注释；两份 server toml 的 `model=`、表名、`server_section` 和 `worker_nums=1`；产品 toml；TensorRT execution context、engine、static shape、batch。gateway（:8080）与 supervisor（:8787）的进程关系保持原样。每个 catalog id 仍是一个 OS 进程。每个 worker 仍持有各自的 TensorRT execution context。

其他条目的文件保持原样：`docs/deployment.md`、`docs/api-contract.md`、`docs/api-contract.zh-cn.md`、`docs/openapi.json`、`docs/how_to_add_new_server.md`、`docs/how_to_add_new_server.zh-cn.md`、`docs/model-developer-guide.md`、两份 README、`scripts/server/http_infer_rps.py`、`docs/perf/coverage_set_http_rps.md`。两份教程里不出现 `densenet`。`densenet` 不进入覆盖集。

## 契约

HTTP 请求、响应、OpenAPI 和 tensor 形状都不变。产品 toml 与 server toml 的键和值不变。变化只在两份教程的两条命令：分类命令的 `--model` 写成 `mobilenetv2`，特征点命令的 `--model` 写成 `superpoint`。这两个字符串与 catalog 的 `model_section`、toml 的 `model=`，以及 toml 表名 `[mobilenetv2]`、`[superpoint]` 相同。教程本来就没有写出 `MOBILENETV2_CLASSIFICATION_SERVER` 和 `SUPERPOINT_FP_SERVER`，本条也不补上这两段名。

## 性能与资源

不生成 engine，不改 static shape、batch=1。每个 worker 仍各自持有自己的 TensorRT execution context，不增加共享 context 或跨 worker context 池。不改 `worker_nums`。不拉起 gateway、supervisor 或模型进程。不跑 `mortredctl prepare`、`mortredctl calibrate` 或 `http_infer_rps.py`。官方 HTTP 压测并发仍是已部署 `worker_nums` 的两倍，本条不改这条规则，也不改 `docs/perf/coverage_set_http_rps.md`。显存与常驻进程无变化。gpu profile 仍是 CUDA 12.x + TensorRT 10.x。cpu profile 仍不链 TensorRT。

## 验收标准

1. 一次打开 `docs/tutorials_of_model_servers.md` 和 `docs/tutorials_of_model_servers.zh-cn.md`。两份里，下一行是 `../conf/server/classification/mobilenetv2/mobilenetv2_server_config.toml` 的那条命令，`--model` 都是 `mobilenetv2`。下一行是 `../conf/server/feature_point/superpoint/superpoint_server_config.toml` 的那条命令，`--model` 都是 `superpoint`。这两处命令里没有 `--model MOBILENETV2`，也没有 `--model SUPERPOINT`。
2. 同一次打开。两份教程里检测、场景分割、增强三条命令仍分别是 `--model YOLOV5`、`--model BISENETV2`、`--model ATTENTIVE_GAN_DERAIN`。toml 路径仍分别是 `../conf/server/object_detection/yolov5/yolov5l_server_config.toml`、`../conf/server/scene_segmentation/bisenetv2/bisenetv2_server_config.toml`、`../conf/server/enhancement/attentive_gan_derain/attentive_gan_server_config.toml`。
3. 同一次打开。英文 `worker_nums` 句仍是 `Shipped configs default to `worker_nums=1`. Raise it only if you have GPU headroom.` 中文句仍是 `仓库自带配置默认 `worker_nums=1`，仅在显存足够时再加大。` 节标题仍含 `MOBILENETV2` 与 `SUPERPOINT`。两份教程都没有 `densenet`。
4. 打开 `classification_task.h` 与 `feature_point_task.h`。`mobilenetv2`、`superpoint` 两行的 id 和 family 仍是这两个小写字符串，`server_section` 仍是 `MOBILENETV2_CLASSIFICATION_SERVER`、`SUPERPOINT_FP_SERVER`。打开两份 server toml，`model=` 与表名仍是这两个小写 id，`worker_nums=1` 仍在。打开 `model_server_main.cpp`，仍写明 `--model` 大小写敏感且必须等于 catalog 的 `model_section`。

判定方式是打开文件。不启动 `mortred-model-server.out`，不跑 GPU，不跑 calibrate，不跑压测，不读 `/metrics`。

## 测试义务

QA 按白盒核对上面四条，并在本机 WSL Ubuntu 执行下方自测命令，期望退出码为 0。QA 用 diff 确认本条改动只出现在 `docs/tutorials_of_model_servers.md` 和 `docs/tutorials_of_model_servers.zh-cn.md`，且每份只有分类命令和特征点命令的 `--model` 实参变成 `mobilenetv2`、`superpoint`。仓库里原有未跟踪的 ledger 不计入本条。QA 不拉起模型，不跑 calibrate，不跑压测。`YOLOV5`、`BISENETV2`、`ATTENTIVE_GAN_DERAIN` 仍为大写，以及 `worker_nums` 那一句仍是现在的原文，都算本条通过。QA 不把 `densenet` 放进覆盖集。

## 自测命令

在本机 WSL Ubuntu 执行。工作目录 `/mnt/g/Codex/mortred_model_server`。期望退出码为 0。断言失败时退出码为 1。本命令只读文件，不拉起 GPU，不跑 calibrate，不跑压测。

```bash
cd /mnt/g/Codex/mortred_model_server
python3 - <<'PY'
import pathlib
root = pathlib.Path("/mnt/g/Codex/mortred_model_server")
en = (root / "docs/tutorials_of_model_servers.md").read_text(encoding="utf-8")
zh = (root / "docs/tutorials_of_model_servers.zh-cn.md").read_text(encoding="utf-8")
cls_src = (root / "src/factory/classification_task.h").read_text(encoding="utf-8")
fp_src = (root / "src/factory/feature_point_task.h").read_text(encoding="utf-8")
cls_toml = (root / "conf/server/classification/mobilenetv2/mobilenetv2_server_config.toml").read_text(encoding="utf-8")
fp_toml = (root / "conf/server/feature_point/superpoint/superpoint_server_config.toml").read_text(encoding="utf-8")
main = (root / "src/apps/model_server_main.cpp").read_text(encoding="utf-8")

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
    assert "--model MOBILENETV2" not in text, name
    assert "--model SUPERPOINT" not in text, name
    assert text.count("--model mobilenetv2") == 1, name
    assert text.count("--model superpoint") == 1, name
    assert "densenet" not in text.lower(), name
    assert "--model <CATALOG_ID>" in text, name

assert "## Classification (`MOBILENETV2`)" in en
assert "## Feature point (`SUPERPOINT`)" in en
assert "Shipped configs default to `worker_nums=1`. Raise it only if you have GPU headroom." in en
assert "python3 scripts/server/test_server.py --server mobilenetv2 --mode single --times 3" in en
assert "python3 scripts/server/test_server.py --server superpoint --mode single" in en

assert "## 分类（`MOBILENETV2`）" in zh
assert "## 特征点（`SUPERPOINT`）" in zh
assert "仓库自带配置默认 `worker_nums=1`，仅在显存足够时再加大。" in zh
assert "python3 scripts/server/test_server.py --server mobilenetv2 --mode single --times 3" in zh
assert "python3 scripts/server/test_server.py --server superpoint --mode single" in zh

assert 'Entry{"mobilenetv2", "Mobilenetv2 classification", "MOBILENETV2_CLASSIFICATION_SERVER"' in cls_src
assert 'classification_param_specs(), "mobilenetv2"}' in cls_src
assert 'Entry{"superpoint", "SuperPoint feature detection", "SUPERPOINT_FP_SERVER"' in fp_src
assert 'feature_point_param_specs(), "superpoint"}' in fp_src
assert 'model="mobilenetv2"' in cls_toml and "[mobilenetv2]" in cls_toml and "worker_nums=1" in cls_toml
assert 'model="superpoint"' in fp_toml and "[superpoint]" in fp_toml and "worker_nums=1" in fp_toml
assert "--model is case-sensitive and must match a catalog model_section." in main
print("p2-12-tutorial-model-ids ok")
PY
```

## RD 不得自行决定

- 分类命令的 `--model` 固定为 `mobilenetv2`。特征点命令的 `--model` 固定为 `superpoint`。两份教程都改。节标题里的大写词不能代替命令行实参。
- 检测、场景分割、增强三条命令保持 `--model YOLOV5`、`--model BISENETV2`、`--model ATTENTIVE_GAN_DERAIN`，toml 路径保持现在这三行。
- `worker_nums` 那一句在两份教程里保持现在的原文。`conf/server` 的 `worker_nums=1` 保持原样。
- 两份教程不写入 `densenet`。不改 `docs/how_to_add_new_server.zh-cn.md`。不把 `densenet` 放进覆盖集。
- 不改 catalog()、产品 toml、server toml、`model_server_main.cpp`、`product_index.h`、TensorRT context 和 engine。
- 不改官方 HTTP 压测规则。并发仍是已部署 `worker_nums` 的两倍。本条的自测保持为上面的读文件命令。
