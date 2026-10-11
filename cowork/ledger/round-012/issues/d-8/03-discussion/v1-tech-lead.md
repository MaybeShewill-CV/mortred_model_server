# 技术方案

- 问题: D-8
- 版本: v1
- 对应列表: `cowork/ledger/round-012/02-p0-p1-final.md` 的 D-8

## 边界

只做已经冻结的 D-8。冻结句不改写：默认发布镜像以 root 运行，边界文档没有写明必须 root 的原因。

本条选定的可观察变化是：同一条 `docker run --rm --entrypoint id mortred_model_server:gpu -u` 的退出码为 0，标准输出原文为一行 `10001`。不选「`docs/unsupported-boundaries.md` 写明必须 root 的原因」。

本条没有上一版方案，没有待并入的审查条件。

现在打印 `0` 的原因：`docker build -t mortred_model_server:gpu .` 的默认目标是 Dockerfile 最后阶段 `mortred-gpu`，该阶段是 `FROM runtime`。`runtime` 的基镜像是钉死 digest 的 `nvidia/cuda:12.6.2-runtime-ubuntu22.04`。`deps`、`build`、`runtime`、`mortred-cpu`、`mortred-gpu` 都没有 `USER`。未指定用户时，镜像用户是 uid 0。`scripts/docker_entrypoint.sh` 用 `exec` 拉起 `/opt/mortred/bin/mortred-supervisor.out`，中间不降权。`docker-compose.yml` 的 gpu 服务没有 `user:`，沿用镜像用户。仓库里没有 `geteuid` 或 `getuid` 判断。gateway 的 8080 和 supervisor 的 8787 都高于 1024。宿主单元 `deploy/mortred-supervisor.service` 已是 `User=mortred`。构建阶段的 `apt-get`，以及宿主机上 `install_deps.sh --nvidia` 安装 CUDA 12.x、TensorRT 10.x、cuDNN 的 deb，需要 root；那是构建和宿主机安装，已打进运行镜像的库不再要求运行进程是 uid 0。

不覆盖 P0-2、P1-8、D-9、D-10、D-11、D-12。不重开已关闭原句：P0-1、P0-3、P1-1 至 P1-7、P1-9、P1-10、P1-11、D-1 至 D-6。P1-5 只含 `yolov8s`。P1-11 只含 `bisenetv2`、`pphuman_mobile`、`modnet`、`ppmatting_512` 的 CPU 全分辨率掩码重合。`centerface` 的 engine 输入名 `input.1` 不重开。

## 做法

### 推荐

只改默认发布镜像的最后阶段 `mortred-gpu`。该阶段开始时仍是 root。在这里创建用户 `mortred`，uid 与 gid 都固定为 `10001`，shell 为 nologin，不建家目录。创建 `/opt/mortred/logs`（`conf/mortred.toml` 的 `log_dir` 是 `logs`，supervisor 启动时会对这个目录做 `create_directories`，错误码目前被忽略），并把该目录交给 uid `10001`。已安装树 `/opt/mortred` 的文件内容保持不变。入口脚本和二进制若现有权限位不允许 uid `10001` 执行入口、读取这棵树，只把所有者改为该用户，权限位与文件内容不动。然后把 `mortred-gpu` 的默认用户设为该账号。

`deps`、`build`、`runtime` 保持 root，构建期 apt 与 cmake 仍以 root 执行。`mortred-cpu` 保持现状。`docker-compose.yml` 不写 `user:`；gpu 服务使用镜像 `mortred_model_server:gpu`，会跟着变成 uid `10001`。

代价：默认容器进程从 uid 0 变为 uid `10001`。容器里再做 apt、或写只允许 root 写的路径，会失败。宿主 bind mount 的 `weights` 所有权不变；该目录若不允许 uid `10001` 读取，现有读权重会失败。本条验收不把推理成功、engine 生成或挂载可写算进去。镜像多一个系统用户，构建时间、显存占用和 engine 生成方式不变。

### 否决

在 `docs/unsupported-boundaries.md` 写明必须 root。同一条 `id -u` 仍打印 `0`。运行进程没有必须为 uid 0 的判断，宿主单元已经是 `User=mortred`。构建期安装 deb 需要 root，不能写成已发布运行镜像必须 root。该文件现有口径是不支持项，写进一句不成立的「必须 root」会留下 S3 的运行侧。这条否决。

只写数字 `USER`、不建账号，或不做 `logs` 的所有者。`id -u` 可以印出非 0，但没有 passwd 项，`/opt/mortred/logs` 仍只有 root 能写时，现有 supervisor 写日志仍依赖 uid 0。这条否决。

把 `deps`、`build`、`runtime`、`mortred-cpu` 一起降权，或给 `docker-compose.yml` 加 `user:`。构建阶段要 root 才能 apt。cpu 标签不是冻结命令里的 `mortred_model_server:gpu`。compose 不写 `user:` 时，gpu 服务已经跟随镜像用户。扩大到这些文件不产生这条命令的原文 `10001`。这条否决。

## 改动面

- `Dockerfile` 的最后阶段 `mortred-gpu`：创建 uid/gid `10001` 的用户 `mortred`，准备可由该用户写入的 `/opt/mortred/logs`，必要时只把 `/opt/mortred` 已有文件的所有者改为该用户，并把该阶段默认用户设为该账号。

## 明确不改

- `docs/unsupported-boundaries.md`。本条不补必须 root 的原因。
- `deps`、`build`、`runtime`、`mortred-cpu` 的用户。cpu 镜像标签 `mortred_model_server:cpu` 保持现状。
- `docker-compose.yml`，包括 gpu 与 cpu 服务都继续不写 `user:`。
- `scripts/docker_entrypoint.sh`、HEALTHCHECK、8080、8787、三枚 token、`MORTRED_AUTOSTART`、`MORTRED_PACK`。
- 基镜像 digest、CUDA 12.x、TensorRT 10.x、cuDNN 的安装行。
- 产品 toml、HTTP、OpenAPI、张量契约、gateway 与 supervisor 的代码、engine 生成。engine 仍在本机本卡用 `mortredctl prepare` 生成。
- `deploy/mortred-supervisor.service` 的 `User=mortred`。宿主 uid 不必等于 `10001`。
- P0-2、P1-8、D-9、D-10、D-11、D-12，以及已关闭原句。

## 契约

无 HTTP 变化，无模型输入输出张量契约变化，无产品 toml 变化。一个 catalog id 仍是一个 OS 进程。gateway 仍是 `:8080`，supervisor 仍是 `:8787`。

镜像契约只变默认用户：不向 `docker run` 追加 `--user` 时，`mortred_model_server:gpu` 的用户是 uid `10001`。判定命令必须保持冻结句里的那一条，`--entrypoint` 后面是 `id`，镜像名后面的 `-u` 是 `id` 的参数。

## 性能与资源

构建阶段仍以 root 安装依赖并完成 full 构建，不增加常驻进程，不增加显存占用，不生成 engine。gpu profile 仍是 CUDA 12.x + TensorRT 10.x。cpu profile 仍不链 TensorRT。

运行时默认进程是 uid `10001`。不添加 `--privileged`，不添加 capabilities，不改 compose 里 nvidia device reservation。`/opt/mortred/logs` 由该用户拥有，现有日志目录创建不再依赖 euid 0。权重卷仍是挂载进来的宿主目录，本条不改挂载所有权。

## 验收标准

验收 1。前置：在仓库根目录执行 `docker build -t mortred_model_server:gpu .`，退出码 0，得到标签 `mortred_model_server:gpu`。判定命令是 `docker run --rm --entrypoint id mortred_model_server:gpu -u`。该命令退出码 0。标准输出去掉末尾唯一的换行后，原文等于 `10001`。没有第二行。标准错误为空。

## 测试义务

QA 只做验收 1 这一次黑盒。通过与否只看该命令的退出码、标准输出原文和空的标准错误。不把读 Dockerfile、读 `docs/unsupported-boundaries.md`、拉起 gateway 或 supervisor、挂载 weights、生成 engine、跑 cpu 标签或 compose 写成通过条件。

## 自测命令

在能拉取 Dockerfile 已钉死的 CUDA 基镜像的仓库根目录执行。判定命令不要加 docker 的 `--user`。

| 命令 | 期望退出码 |
| --- | --- |
| `docker build -t mortred_model_server:gpu .` | 0 |
| `docker run --rm --entrypoint id mortred_model_server:gpu -u` | 0。标准输出去掉末尾唯一换行后原文是 `10001`，标准错误为空 |

## RD 不得自行决定

- 改走边界文档那一支，或在 `docs/unsupported-boundaries.md` 补必须 root 的原因。
- 把 uid 或 gid 改成 `10001` 以外的值，包括改成 `0`、宿主 `useradd -r` 分到的系统 uid，或不建名为 `mortred` 的账号只写数字用户。
- 留下只有 root 可写的 `/opt/mortred/logs`。
- 修改 `/opt/mortred` 里文件的内容来完成降权。
- 给 `docker run` 加上 `--user` 来得到 `10001`。镜像默认用户本身必须是 `10001`。
- 改 `mortred-cpu`、构建阶段的用户、compose 的 `user:`、入口脚本、端口、token、基镜像 digest、CUDA / TensorRT 版本。
- 为了让 uid `10001` 能写宿主 weights，去改挂载约定、产品 toml 或 engine 生成。
- 做 P0-2、P1-8、D-9、D-10、D-11、D-12，或重开已关闭原句。
