# 测试报告

- 结论: fail
- 方案版本: v2
- 分支: cowork/r1-p0-1-libface
- worktree: G:\Codex\mortred_wt_r1_p0_1（WSL：/mnt/g/Codex/mortred_wt_r1_p0_1）
- commit: 无
- 环境: WSL Ubuntu，NVIDIA GeForce RTX 2070 SUPER；使用 worktree 已有 `_bin/mortred-model-server.out`。9051 启动前无监听。测完该进程已停，端口已释放。
- 缺的环境: 无
- 恢复条件: 无
- 未跑的验收: 无

## 用例
| 验收标准 | 类型 | 场景 | 命令 | 退出码 | 预期 | 实际 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 黑盒 | 打开 `libface.toml` | `python3` 读取 `conf/model/object_detection/libfacedetection/libface.toml`，断言含 `model_input_image_size = [480, 640]`、`type = "tensorrt"`、`face_detection_yunet_2026may.engine.static_bs1.fp16` | 0 | 三处同时成立 | 打印 `ac1-ok` | pass |
| 2 | 黑盒 | 打开 `set_a_libface.json` | `python3` 解析 `conf/trt_profiles/set_a_libface.json`，断言唯一绑定 `input` 且 `min == opt == max == [1, 3, 480, 640]` | 0 | 三档相同，N 为 1 | 打印 `ac2-ok` | pass |
| 3 | 黑盒 | 打开 `docs/benchmark-h3/libface.txt` | `python3` 断言该文件同时含 `--shapes=input:1x3x480x640`、`YYYY-MM-DD`、`Throughput:`，以及一行既非 trtexec 命令、非 Throughput、非单独日期的硬件描述 | 0 | 四行条件同时成立，高 480、宽 640 与 toml、profile 相同 | 硬件行 `NVIDIA GeForce RTX 2070 SUPER, 8192 MiB, 616.92`；日期 `2026-10-10`；命令含 `--shapes=input:1x3x480x640`；`Throughput: 1668.97 qps`。与 `[480, 640]`、`[1, 3, 480, 640]` 一致。打印 `ac3-ok` | pass |
| 4 | 黑盒 | 进程路径与工作目录 | 工作目录 `/mnt/g/Codex/mortred_wt_r1_p0_1/_bin`，`unset MORTRED_WORKER_NUMS MORTRED_MODEL_CONFIG_FILE`，`MORTRED_AUTH_TOKEN=libface-p0-1`，`./mortred-model-server.out --model libface ../conf/server/object_detection/libface_det/libface_server_config.toml`。`/ready` 在 60 秒内返回后，`readlink -f /proc/<pid>/cwd` 与 `/proc/<pid>/exe` | 0 | cwd 为 worktree `_bin`，exe 为 worktree `_bin/mortred-model-server.out` | `READY:1`。cwd=`/mnt/g/Codex/mortred_wt_r1_p0_1/_bin`，exe=`/mnt/g/Codex/mortred_wt_r1_p0_1/_bin/mortred-model-server.out`。打印 `process-path-ok` | pass |
| 5 | 黑盒 | 带该进程 Bearer 的 JPEG | `curl` POST `demo_data/model_test_input/object_detection/face_wo_mask.jpg` 原始字节到 `http://127.0.0.1:9051/mortred_ai_server_v1/obj_detection/libface`。`Content-Type: image/jpeg`。`Authorization: Bearer` 取自 `tr '\0' '\n' < /proc/<pid>/environ` 的 `MORTRED_AUTH_TOKEN` | 0 | HTTP 200，JSON `status` 为 0，`status_str` 为 `OK`，`results[0].status` 为 0 | 环境 token 为 `libface-p0-1`，与请求 Bearer 相同。`HTTP/1.1 200 OK`。打印 `jpeg-ok`，`status 0 OK results0 0` | pass |
| 6 | 黑盒 | server toml 的 worker 与 batch | `python3` 解析 `conf/server/object_detection/libface_det/libface_server_config.toml` 的 `worker_nums`、`max_batch_size` | 0 | `worker_nums` 为 1，`max_batch_size` 为 1 | 打印 `ac6-ok`。两值均为 1。该文件没有 `auth_token` | pass |
| 7 | 黑盒 | `product_static_bs1.py --check` | `cd /mnt/g/Codex/mortred_wt_r1_p0_1 && python3 scripts/product_static_bs1.py --check` | 1 | 退出码 0 | stderr：`mobilenetv2_cpu.toml: cpu profile config must stay MNN on cpu`；`resnet_cpu.toml: cpu profile config must stay MNN on cpu`。两份文件是 `type = "onnx"`、`device = "cpu"`。stderr 只有这两行 | fail |

## 缺陷

验收标准 7 失败，退回 RD。

最短复现：

```bash
cd /mnt/g/Codex/mortred_wt_r1_p0_1
python3 scripts/product_static_bs1.py --check
```

退出码 1。stderr：

```text
/mnt/g/Codex/mortred_wt_r1_p0_1/conf/model/classification/mobilenetv2/mobilenetv2_cpu.toml: cpu profile config must stay MNN on cpu
/mnt/g/Codex/mortred_wt_r1_p0_1/conf/model/classification/resnet/resnet_cpu.toml: cpu profile config must stay MNN on cpu
```

`scripts/product_static_bs1.py` 要求这两份文件同时含 `type = "mnn"` 与 `device = "cpu"`。当前两份文件的 `type` 是 `onnx`。`6faa446` 只改了 libface 尺寸守卫；这两份 CPU 配置和这段 MNN 检查来自 `9800fd67`。本次 stderr 没有 libface 尺寸错误。

## 不适用

- 错误 catalog id、未就绪模型、空输入、极端尺寸、通道或色彩：验收 1 至 7 只要求一张合法 JPEG 在已拉起的 `libface` 上成功。
- 框、关键点、mask、类别分数的数值和形状：方案写明不规定框的数值，不要求框非空。
- 401、超时、缺参、gateway、calibrate、HTTP RPS：编号验收没有这些行。标准 5 使用的是与进程环境一致的 Bearer，响应为成功状态。
- 多后端同一输入对照：本条只验收 tensorrt static batch=1。
- `test/` 与 golden：`test/` 没有把产品 `libface.toml` 锁在 `[640, 480]`。`conf/ci/libface_onnx_hosted.toml` 仍是 `[640, 480]`，方案写明不改。
- 白盒：`6faa446` 的 diff 只有 `libface.toml`、`set_a_libface.json`、`scripts/product_static_bs1.py` 的那一句尺寸守卫。`trt_session.cpp`、按 worker 建会话的循环、鉴权代码、server toml、`realesrgan` 不在 diff 里。profile 三档相等且 N 为 1。

## 残余风险

- `trtexec --loadEngine=weights/object_detection/libfacedetection/face_detection_yunet_2026may.engine.static_bs1.fp16 --warmUp=200 --duration=2 --avgRuns=10 --useSpinWait --fp16 --shapes=input:1x3x480x640` 退出码 0，输出 `PASSED`。按方案保持 engine 与 `docs/benchmark-h3/libface.txt` 不动。engine 时间仍是 2026-09-24 10:28。基线 `libface.toml` 仍是 `[640, 480]`。
- libface 守卫的失败文案仍是 `libface params are not 640x480`，判断条件已是 `model_input_image_size = [480, 640]`。
