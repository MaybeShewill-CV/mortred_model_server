# 图像分类服务器说明

## 启动一个图像分类服务器

你可以快速启动一个图像分类服务器. 主要的函数如下所示

`图像分类服务器代码段`
![strat_a_mobilenetv2_server](../resources/images/start_a_mobilenetv2_server.png)

统一入口在 `$PROJECT_ROOT/_bin/mortred-model-server.out`，启动方式如下所示

```bash
cd $PROJECT_ROOT/_bin
./mortred-model-server.out --model MOBILENETV2 ../conf/server/classification/mobilenetv2/mobilenetv2_server_config.toml
```

正常启动后，服务会运行在服务器配置（`conf/server/<task>/<model>/*.toml`）中 `port` 指定的端口，`worker_nums` 个模型实例会被创建并占用 GPU 资源。仓库自带配置默认 `worker_nums=1`，你可以按 GPU 显存情况适当调整。

`图像分类服务器被正常启动`
![classification_server_ready_to_serve](../resources/images/mobilenetv2_server_ready.png)

## Python 客户端示例

在文件 [test_server.py](../scripts/server/test_server.py) 处有演示客户端。顺序冒烟和闭环压测都只用 Python 标准库（不依赖 locust / requests）。

`python客户端代码片段`
![sample_mobilenetv2_cls_client](../resources/images/mobilenetv2_sample_client.png)

服务的url地址可以在服务启动之前在配置文件中进行配置修改。你可以在 [模型服务器配置说明文档](../docs/about_model_server_configuration.zh-cn.md) 中找到详细说明。

调用方式：

```bash
cd $PROJECT_ROOT
python3 scripts/server/test_server.py --server mobilenetv2 --mode single --times 3
```

客户端会发送 [默认测试图像](../demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG)，并打印 HTTP 状态码与截断后的 UnifiedResponse。

`mobilenetv2 图像分类服务器输出`
![server_output](../resources/images/exam_server_output.png)

`mobilenetv2 图像分类客户端输出`
![server_output](../resources/images/exam_client_output.png)

分类 id 与分数在 `results[].data` 里。

## Python 客户端代码说明

[test_server.py](../scripts/server/test_server.py) 直接读取 `conf/server/` 下的 `host` / `port` / `server_uri`。测 HTTP serving RPS 用 [http_infer_rps.py](../scripts/server/http_infer_rps.py)（不是进程内 FPS）：N 条 keep-alive 线程、预先编码的 JSON 信封、每请求唯一 `req_id`，并输出 RPS 与延迟分位数。

```bash
python3 scripts/server/test_server.py --list

python3 scripts/server/test_server.py --server mobilenetv2 --mode single --times 3

python3 scripts/server/test_server.py --server mobilenetv2 --mode load \
    --concurrency 8 --duration 30s

python3 scripts/server/test_server.py --server MOBILENETV2 --mode load \
    --gateway --token "$MORTRED_GATEWAY_AUTH_TOKEN" --concurrency 8 --duration 30s

python3 scripts/server/http_infer_rps.py --url http://127.0.0.1:9003/mobilenetv2cls \
    --image demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG \
    --concurrency 8 --duration 30s --out /tmp/load.json
```

**--server:** 配置段名、catalog id 或唯一前缀，例如 `mobilenetv2`、`MOBILENETV2`

**--mode:** `single` 顺序打印 `--times` 次响应；`load` 闭环并发客户端

**--image:** 覆盖演示输入图片（默认使用 `demo_data/model_test_input` 下按模型选择的示例图）

**--dry-run:** 只打印请求计划，不发送任何请求

**--concurrency / --duration / --requests:** 压测线程数与停止条件（`30s`、`2m` 或固定请求数）

**--gateway:** 改为 POST `http://127.0.0.1:8080/v1/models/{id}/infer`

`http_infer_rps.py --self-test` 会在进程内起一个 HTTP 服务并验证 40 个并发请求全部成功。

下面截图来自旧的 locust 实验（`worker_nums` 从 4 提到 12）。结论不变：HTTP RPS 随 worker 增加直到 GPU/队列饱和，再加 worker 不会无限抬高吞吐。请在你的 GPU 上用 `--mode load` 重新测量。

`历史压测客户端输出`
![locust_client_output](../resources/images/locust_client_output.png)

`历史压测服务端输出`
![locust_server_output](../resources/images/locust_server_output.png)

`worker_nums=4` 时旧实验大约 288 req/s，GPU 利用率偏低、部分请求超时、worker 队列持续为空。队列空闲时首先加大 `worker_nums`。
![losust_test_result_1](../resources/images/locust_test_result_1.png)

增大到 12 后队列不再枯竭，GPU 利用率上升，平均延迟下降，RPS 接近进程内基准。
![locust_server_output_enlarge](../resources/images/locust_server_output_enlarge.png)
![losust_test_result_2](../resources/images/locust_test_result_2.png)

不能靠无限加大 `worker_nums` 提升吞吐；该实验到 24 时 RPS 基本持平。
![losust_test_result_3](../resources/images/locust_test_result_3.png)
