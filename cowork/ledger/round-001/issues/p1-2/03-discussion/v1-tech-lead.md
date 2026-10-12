# 技术方案

- 问题: P1-2
- 版本: v1
- 对应列表: `cowork/ledger/round-001/02-p0-p1-final.md` 的 P1-2（列表版本 v3）

## 边界

本方案只让三份说明里要执行的 catalog id 与现有注册一致。对照物是 `src/factory/classification_task.h` 里 `densenet` 那一行的 id `densenet` 与 family `densenet`，以及已经写在 `conf/server/classification/densenet/densenet_server_config.toml` 的 `model="densenet"`。`src/factory/cv_catalog.h` 已写明 catalog id 是 `model_section` 且为小写；family 是该行最后一项。验收只打开文件，不拉起模型，不跑压测。

## 做法

### 推荐

做法一：只改三份文档里要执行的 id 字符串。

`docs/how_to_add_new_server.md` 的示例 `Entry` 第一项改为 `densenet`。同一调用在 `classification_param_specs()` 之后补上 family 实参 `densenet`。两条命令的 `--model` 改为 `densenet`。配置步骤里的 `model =` 改为 `"densenet"`。`README.zh-cn.md` 两处 `--model` 改为 `mobilenetv2`，cpu 精选表里的 `resnet50` 改为 `resnet`。`README.md` 的 cpu 精选表同样把 `resnet50` 改为 `resnet`。

代价：以前按大写 id 抄过命令的笔记要改口令。示例里的 display name 仍可以是现有的 `densenet classification`，与源码里的 `Densenet classification` 不必逐字相同。没有进程、显存或 engine 代价。

### 否决

做法二：改 `classification_task.h` 的注册，使 id 变成文档里的 `DENSENET`、`MOBILENETV2`、`resnet50`。代价是要改 catalog 代码，并连带 server toml 的 `model=`、`--list` 和 family。`cv_catalog.h` 要求 catalog id 为小写；family 为空的行不进 static batch=1 拆分。这超出本条锁定的文档变化。

做法三：在示例旁加注「实际请用小写」，或对三份文件做不区分边界的子串替换，或顺手重写指南、补上 `mortredctl next`、改 `docs/deployment.md` 与 `docs/model-developer-guide.md`。旁注留下的示例仍在执行 `DENSENET`，打开文件仍看不到与 `densenet` 那一行一致的 id、family 和 `model =`。盲替换会改掉已经对齐的 `DENSENET_CLASSIFICATION_SERVER`。扩写会把另一条问题和已经写对的部署文档卷进来。

## 改动面

只改 docs 模块里的三份文件。

- `docs/how_to_add_new_server.md`：Step 2 示例 `Entry` 的第一项改为 `densenet`；`classification_param_specs()` 之后增加最后一项 family `densenet`；Step 3 两条命令的 `--model` 改为 `densenet`；同节配置步骤的 `model = "DENSENET"` 改为 `model = "densenet"`。`server_section` 保持 `DENSENET_CLASSIFICATION_SERVER`。工厂函数、`fill_response`、display name、路径占位符，以及「不要为每个模型新增 `src/apps/server/*.cpp`」这句保持原样。
- `README.zh-cn.md`：基准测试与 `mortred-model-server.out` 两处 `--model` 的值改为 `mobilenetv2`，后面的 toml 路径不动。cpu 精选表由 `精选（mobilenetv2、resnet50）` 改为 `精选（mobilenetv2、resnet）`。
- `README.md`：cpu 精选表由 `curated set (mobilenetv2, resnet50)` 改为 `curated set (mobilenetv2, resnet)`。

## 明确不改

保持不动的有：`src/factory/classification_task.h`、`src/factory/cv_catalog.h` 以及其他 catalog() 代码；全部产品 toml 与 server toml；TensorRT execution context、engine、static shape、batch；`worker_nums`；`docs/deployment.md`；`docs/model-developer-guide.md`；`README.zh-cn.md` 的「第一小时：mortredctl」命令块；两份 README 里已经写着 `resnet`、`densenet` 的模型表；`ppmatting_resnet34`；`README.md` 里 `--model yolov8s` 的压测示例；gateway、supervisor 与模型进程。

## 契约

HTTP 请求、响应、OpenAPI 和 tensor 形状都不变。产品 toml 与 server toml 的键和值不变。变化只在文档示例：指南要执行的 catalog id、family 和 `model =` 写成 `densenet`；中文 README 要执行的 `--model` 写成 `mobilenetv2`；两份 README 的 cpu 精选 id 写成 `resnet`。指南里的 `model = "densenet"` 与现有 toml 的 `model="densenet"` 是同一个 id，空格写法保持指南原样。

## 性能与资源

不生成 engine，不改 static shape、batch=1。每个 worker 仍各自持有自己的 TensorRT execution context，不增加共享 context 或跨 worker context 池。不改 worker 数。不拉起 gateway、supervisor 或模型进程。显存与常驻进程无变化。

## 验收标准

1. 打开 `docs/how_to_add_new_server.md`。示例 `Entry` 的第一项是 `densenet`。`classification_param_specs()` 之后的最后一项 family 是 `densenet`。`server_section` 仍是 `DENSENET_CLASSIFICATION_SERVER`。两条命令都是 `--model densenet`。配置步骤写的是 `model = "densenet"`。这三项与 `classification_task.h` 里 `densenet` 那一行的 id、family，以及现有 densenet server toml 的 `model="densenet"` 一致。文件里不再有作为 catalog id 的独立词 `DENSENET`。
2. 打开 `README.zh-cn.md`。基准测试和 `mortred-model-server.out` 两处 `--model` 的值都是 `mobilenetv2`。文件中不再出现 `MOBILENETV2`。
3. 打开 `README.md` 与 `README.zh-cn.md` 的 cpu 精选表。精选 id 是 `mobilenetv2` 与 `resnet`。两份文件都不再出现 `resnet50`。
4. 打开这三份文件，上述要执行的 catalog id 是 `densenet`、`mobilenetv2`、`resnet`。打开 `docs/deployment.md`，cpu 精选仍是 `resnet`。打开 `classification_task.h` 与 densenet server toml，注册行和 `model="densenet"` 仍是改前内容。
5. 打开 `docs/how_to_add_new_server.md`，仍写明不要为每个模型新增服务器二进制，统一入口仍是 `mortred-model-server.out`。打开 `README.zh-cn.md` 的第一小时命令块，其中仍没有 `mortredctl next`。

判定方式是打开文件。不启动 `mortred-model-server.out`，不跑 benchmark，不跑 calibrate，不读 `/metrics`。

## 测试义务

QA 按白盒核对上面五条，并在本机 WSL Ubuntu 执行下方自测命令，期望退出码为 0。QA 用 diff 确认本条改动只出现在 `docs/how_to_add_new_server.md`、`README.zh-cn.md`、`README.md`。仓库里原有未跟踪的 ledger 不计入本条。QA 不拉起模型，不跑压测，不把 `mortredctl next` 的缺失写成这条的失败。

## 自测命令

在本机 WSL Ubuntu 执行。工作目录 `/mnt/g/Codex/mortred_model_server`。期望退出码为 0。断言失败时退出码为 1。

```bash
cd /mnt/g/Codex/mortred_model_server
python3 - <<'PY'
import pathlib, re, sys
root = pathlib.Path("/mnt/g/Codex/mortred_model_server")
how = (root / "docs/how_to_add_new_server.md").read_text(encoding="utf-8")
zh = (root / "README.zh-cn.md").read_text(encoding="utf-8")
en = (root / "README.md").read_text(encoding="utf-8")
src = (root / "src/factory/classification_task.h").read_text(encoding="utf-8")
toml = (root / "conf/server/classification/densenet/densenet_server_config.toml").read_text(encoding="utf-8")
deploy = (root / "docs/deployment.md").read_text(encoding="utf-8")

ok = re.search(
    r'Entry\{\s*"densenet"\s*,\s*"[^"]*"\s*,\s*"DENSENET_CLASSIFICATION_SERVER"\s*,'
    r'.*?classification_param_specs\(\)\s*,\s*"densenet"\s*\}',
    how, re.S)
assert ok, "how-to example id/family"
assert how.count("--model densenet") >= 2
assert not re.search(r"--model\s+DENSENET\b", how)
assert 'model = "densenet"' in how
assert 'model = "DENSENET"' not in how
assert not re.search(r"\bDENSENET\b", how)
assert "Do **not** add a per-model" in how

assert zh.count("--model mobilenetv2") >= 2
assert "MOBILENETV2" not in zh
assert "精选（mobilenetv2、resnet）" in zh
assert "resnet50" not in zh
assert "mortredctl next" not in zh

assert "curated set (mobilenetv2, resnet)" in en
assert "resnet50" not in en

assert re.search(r'Entry\{"densenet".*"densenet"\}', src, re.S)
assert '"DENSENET"' not in src
assert 'model="densenet"' in toml
assert "curated: mobilenetv2, resnet" in deploy
print("p1-2-doc-ids ok")
PY
```

## RD 不得自行决定

- catalog id 的大小写和名字：指南示例固定为 `densenet`，中文 README 的 `--model` 固定为 `mobilenetv2`，两份 README 的 cpu 精选固定为 `resnet`。
- family 必须出现在示例 `Entry` 里，位置在 `classification_param_specs()` 之后，值为 `densenet`。正文里的一句说明不能代替这个实参。
- `server_section` 保持 `DENSENET_CLASSIFICATION_SERVER`。display name 不作为本条完成条件。
- 不改 catalog()、产品 toml、server toml、worker 数、TensorRT context 和 engine。
- 不改 `docs/deployment.md`。不把 `mortredctl next` 写进中文 README。不改 `docs/model-developer-guide.md`、模型表和 `yolov8s` 压测示例。
- 不把验收改成拉起进程或跑压测。
