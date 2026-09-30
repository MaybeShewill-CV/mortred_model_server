# 技术方案

- 问题: P1-2 遗留 conf/trt_engines.json 中 yolov8s engine 仍指向旧文件名
- 版本: v1
- 对应列表: `cowork/ledger/round-008/02-p0-p1-final.md` P1-2

## 边界

产品真源仍是 `conf/product_static_bs1.json`。不得把产品真源改回遗留文件，不得改 `product_static_bs1.json`。不重做 P0-1 fail-closed，不改 prepare 主路径。只改遗留清单里 yolov8s 的 onnx 与 engine。

## 做法

### 推荐

把 `conf/trt_engines.json` 中 `model == "yolov8s"` 的 `onnx` 改为 `weights/object_detection/yolov8/yolov8s.static_bs1.onnx`，`engine` 改为 `weights/object_detection/yolov8/yolov8s.engine.static_bs1.fp16`。`fp` 与 `profile` 不动。其它条目不动。

新增 `scripts/check_legacy_yolov8s_engine_align.py`：读两份 JSON，断言遗留 yolov8s 的 engine 与产品 yolov8s 的 engine 相同，且遗留清单任意 engine 字段不以 `yolov8s.engine` 结尾。不探测 GPU，不要求权重文件存在。

### 否决

保留旧文件名再让读取方失败。删除 yolov8s 条目或改 convert 默认 manifest。

## 改动面

- `conf/trt_engines.json`
- `scripts/check_legacy_yolov8s_engine_align.py`

## 明确不改

`conf/product_static_bs1.json`、prepare、`convert_trt_engines.sh` 默认入口、`check_consistency.py`、`src/`、其它模型条目。

## 契约

遗留 yolov8s 的 onnx 与 engine 字符串与产品清单相同。遗留清单不再有以 `yolov8s.engine` 结尾的 engine。

## 性能与资源

不生成 engine，不启进程。

## 验收标准

1. **AC1** 遗留 yolov8s 的 engine 等于 `weights/object_detection/yolov8/yolov8s.engine.static_bs1.fp16`，且任意 engine 不以 `yolov8s.engine` 结尾。
2. **AC2** 该 engine 与 `product_static_bs1.json` 中 yolov8s 的 engine 相同。
3. **AC3** 遗留 yolov8s 的 onnx 等于产品 yolov8s 的 onnx。
4. **AC4** `git diff` 里 `conf/trt_engines.json` 只有 yolov8s 的 onnx 与 engine。
5. **AC5** `conf/product_static_bs1.json` 无内容变更。
6. **AC6** `python3 scripts/check_legacy_yolov8s_engine_align.py` 在无 GPU 下退出码 0。
7. **AC7** `scripts/test_p0_1_engine_fail_closed.sh` 不因「产品 static_bs1 绑回遗留 json」失败。

## 测试义务

WSL 无 GPU 即可。不要求 trtexec 或起 catalog。

## 自测命令

```bash
python3 scripts/check_legacy_yolov8s_engine_align.py
bash scripts/test_p0_1_engine_fail_closed.sh
```

期望第一条退出码 0。第二条不得出现产品 static_bs1 绑回遗留清单的失败。

## RD 不得自行决定

不得改产品真源。不得改其它模型条目。不得改 convert 默认 manifest 来代替路径对齐。新增检查不得依赖 GPU 或权重文件。

## 签字
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| cv-pm | 同意 | 无 | v1 | `cowork/ledger/round-008/issues/p1-2/03-discussion/v1-sign-pm.md` |
| cv-rd | 同意 | 无 | v1 | `cowork/ledger/round-008/issues/p1-2/03-discussion/v1-sign-rd.md` |
| cv-tech-lead | 同意 | 无 | v1 | `cowork/ledger/round-008/issues/p1-2/03-discussion/v1-sign-tech-lead.md` |
