# 本轮新发现

无

## 附录

QA 在 P0-3 写下的残余风险不进入队列。验收 7 用有限全零头证明框数为 0 时顶层 `status` 为 0，且 `data` 为 JSON 数组；方案已把这个结果写成通过条件，不要求比较框坐标。`output`、`518`、`532` 三个名字对齐之后的 dtype、通道、buffer 和非有限值失败，方案写明保持契约失败，本条未重跑。这两句没有新的、一次黑盒可判定的变化。

P1-7 的研发新发现和 QA 残余风险不进入队列。Windows 检出把 `test/golden/mobilenetv2_classification.json` 写成 CRLF，git 对象仍是 LF，且与基线 blob 相同。示例是短占位，方案已写明不要求真实推理。`EnvelopeData` 的字段口径由 `--check` 另判。表外 path 没有示例，方案写明不因此让自测失败。
