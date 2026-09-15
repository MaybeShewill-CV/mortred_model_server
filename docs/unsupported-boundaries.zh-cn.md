# 不支持边界（无假期待）

完整英文版与表格见 [unsupported-boundaries.md](unsupported-boundaries.md)。
源码 / CMake 为准；CHANGELOG 仅参考。

开箱与运维：[deployment.zh-cn.md](deployment.zh-cn.md)（`mortredctl next`）。

| 边界 | 不支持 / 勿期待 |
|---|---|
| OS | Windows、macOS（产品线仅 **Linux x64**） |
| RTDETR | 脚手架，`MODEL_NOT_IMPLEMENTED`，**未注册** HTTP catalog |
| MOT | 无 MOT HTTP 服务 |
| GPU 栈 | 非 CUDA 12.x；**TensorRT 8/9**（钉 **10.x**） |
| Engine | 跨机 / 跨 TRT 大版本拷贝 `.engine` 当开箱；跳过本机 `mortredctl prepare` |
| cpu profile | 编进 TensorRT；`type=tensorrt` 配置能跑通 |

清单里旧称「TRT&lt;9」已过时：源码硬前置是 **TensorRT 10**，不是「≥9」。
