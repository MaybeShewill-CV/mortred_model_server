# 技术方案

- 问题: P2-22
- 版本: v1
- 对应列表: v1

## 边界

本方案只改英文与中文两份模型服务教程里非覆盖集三条启动命令的 `--model`，以及仍显示旧枚举名的章节标题。一次打开 `docs/tutorials_of_model_servers.md` 与 `docs/tutorials_of_model_servers.zh-cn.md`。

现行为：`--model YOLOV5`、`--model BISENETV2`、`--model ATTENTIVE_GAN_DERAIN`；章节标题仍写 `YOLOV5` / `BISENETV2` / `ATTENTIVE_GAN_DERAIN`。`src/apps/model_server_main.cpp` 写明 `--model` 大小写敏感且必须等于 catalog `model_section`。对应 catalog id 是 `yolov5l`、`bisenetv2`、`attentive_gan`。

第 2 轮已关闭的 P2-12 只覆盖 `mobilenetv2` / `superpoint`；本条不重开、不改那两条。

锁定的可观察变化（队列 v1）：只打开这两份教程：三条 `--model` 分别是 `yolov5l`、`bisenetv2`、`attentive_gan`。不升为 P0/P1。不扩到 deployment、how-to、代码或其它 `--model` 行。

验收只打开文件。不拉起模型进程，不跑 GPU、calibrate 或压测。

## 方案

### 推荐

做法一：只把两份教程中这三条 `--model` 与对应章节标题里的旧枚举名改成 catalog id。

**英文 `docs/tutorials_of_model_servers.md`**

- 标题 Object detection 中的 `YOLOV5` → `yolov5l`
- `--model YOLOV5` → `--model yolov5l`
- 标题 Scene segmentation 中的 `BISENETV2` → `bisenetv2`
- `--model BISENETV2` → `--model bisenetv2`
- 标题 Enhancement 中的 `ATTENTIVE_GAN_DERAIN` → `attentive_gan`
- `--model ATTENTIVE_GAN_DERAIN` → `--model attentive_gan`

**中文 `docs/tutorials_of_model_servers.zh-cn.md`**

- 标题目标检测中的 `YOLOV5` → `yolov5l`
- `--model YOLOV5` → `--model yolov5l`
- 标题场景分割中的 `BISENETV2` → `bisenetv2`
- `--model BISENETV2` → `--model bisenetv2`
- 标题图像增强中的 `ATTENTIVE_GAN_DERAIN` → `attentive_gan`
- `--model ATTENTIVE_GAN_DERAIN` → `--model attentive_gan`

server toml 路径、`test_server.py --server` 别名、截图引用、已修好的 `mobilenetv2` / `superpoint` 行保持原样。

代价：对照过旧大写枚举名的笔记要对齐 catalog id。没有进程、显存、engine 或压测代价。

### 备选

做法二：只改 `--model` 行、保留章节标题里的旧枚举名。读者仍会看到标题与命令不一致。否决。

做法三：改 catalog / `model_server_main.cpp` 接受大写枚举别名以迁就教程。队列只锁教程两文件；代码 out of scope。否决。

做法四：连同其它教程段落、deployment、how-to、README 一起改。其它开放项或已关项；本条只锁这三条。否决。

## 改动面

只改 docs 模块里的两份教程。

- `docs/tutorials_of_model_servers.md`：三条 `--model` 与三处章节标题中的枚举名。
- `docs/tutorials_of_model_servers.zh-cn.md`：同上。

## 不改清单

保持原样的有：已对齐的 `--model mobilenetv2` / `--model superpoint`；两份教程中其它段落与截图路径；`docs/deployment.md`、`docs/deployment.zh-cn.md`（P2-23）；`docs/how_to_add_new_server.md` 及其中文；`docs/api-contract.md` 及其中文；`docs/openapi.json`；`README.md`、`README.zh-cn.md`；`docs/model-developer-guide.md` 及其中文；`docs/about_model_server_configuration.md` 及其中文。

代码与配置保持原样的有：`src/apps/model_server_main.cpp`、各 task catalog、产品 toml、server toml、`conf/packs/`、gateway(:8080)、supervisor(:8787)。一个 catalog id 仍是一个 OS 进程。每个 worker 仍持有各自的 TensorRT execution context。engine、static shape、batch=1、type=tensorrt 产品性能路径不改。

## 契约

HTTP 请求、响应、OpenAPI 和 tensor 形状都不变。产品 toml 与 server toml 的键和值不变。变化只在两份教程三条启动命令的 `--model` 字符串与对应章节标题中的 id 展示。不新增或改写解析行为。

## 性能与资源

不生成 engine，不改 static shape、batch=1。每个 worker 仍各自持有自己的 TensorRT execution context。不改任何 `conf/server` 里的 `worker_nums`。不拉起 gateway、supervisor 或模型进程。不跑 prepare、calibrate 或压测。显存与常驻进程无变化。

## 验收标准

1. 一次打开 `docs/tutorials_of_model_servers.md`。出现 `--model yolov5l`、`--model bisenetv2`、`--model attentive_gan`。不再出现 `--model YOLOV5`、`--model BISENETV2`、`--model ATTENTIVE_GAN_DERAIN`。对应章节标题展示 `yolov5l` / `bisenetv2` / `attentive_gan`，不再以旧枚举名作为标题 id。
2. 同一次打开 `docs/tutorials_of_model_servers.zh-cn.md`。同样三条 `--model` 与章节标题对齐；旧大写 `--model` 三行不在。
3. 已对齐的 `--model mobilenetv2` 与 `--model superpoint`（及中文对应）仍在，本条 diff 不改它们。
4. 本条 diff 只出现在这两份教程。不触及 deployment、how-to、产品/server 代码。

判定方式是打开文件。不启动模型进程，不跑 GPU，不跑 calibrate，不跑压测，不读 `/metrics`。

## 黑盒步骤

1. QA 在本机 WSL Ubuntu 工作目录执行下方自测命令（实现时用 worktree 根路径），期望退出码为 0。
2. QA 用 diff 确认本条改动只出现在两份教程；仓库里原有未跟踪的 ledger 不计入本条。
3. QA 按验收标准 1-4 打开文件核对。不拉起模型，不跑 calibrate，不跑压测。不把本条升为 P0/P1。

## 自测命令

在本机 WSL Ubuntu 执行。工作目录为仓库根（基线或 worktree）。期望退出码为 0。断言失败时退出码为 1。本命令只读文件。

工作目录下用 python3 断言：两份教程均含 `--model yolov5l`、`--model bisenetv2`、`--model attentive_gan`；均不含 `--model YOLOV5`、`--model BISENETV2`、`--model ATTENTIVE_GAN_DERAIN`；仍含 `--model mobilenetv2` 与 `--model superpoint`；章节标题侧仍能读到三个新 id 的反引号形式。打印 `p2-22-tutorials-model-ids ok`。

实现时将自测 root 换成 worktree 路径。

## RD 实现时注意

- 两份教程三条 `--model` 固定改为 `yolov5l`、`bisenetv2`、`attentive_gan`。不得只改英或只改中。
- 章节标题中仍显示旧枚举名时一并改成同一 catalog id。
- 不得改已对齐的 `mobilenetv2` / `superpoint` 行，不得改 server toml 路径或 `test_server.py` 别名。
- 不改 deployment（P2-23）、how-to、README、产品代码、catalog、gateway、supervisor、TensorRT context。
- 实现时将自测 root 换成 worktree 路径，断言正文不改。不把本条升为 P0/P1，不扩 scope。

## 确认
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| PM | 同意 | 无 | v1 | cowork/ledger/round-003/issues/p2-22/03-discussion/v1-sign-pm.md |
| RD | 同意 | 无 | v1 | cowork/ledger/round-003/issues/p2-22/03-discussion/v1-sign-rd.md |
| 技术负责人 | 同意 | 无 | v1 | cowork/ledger/round-003/issues/p2-22/03-discussion/v1-sign-tech-lead.md |
