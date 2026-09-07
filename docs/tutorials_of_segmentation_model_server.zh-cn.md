# 图像分割服务器说明

## 启动一个图像分割服务器

图像分割服务器启动代码如下

`图像分割服务器代码段`
![strat_a_bisenetv2_server](../resources/images/start_a_bisenetv2_server.png)

统一入口在 `$PROJECT_ROOT/_bin/mortred-model-server.out`。运行

```bash
cd $PROJECT_ROOT/_bin
./mortred-model-server.out --model BISENETV2 ../conf/server/scene_segmentation/bisenetv2/bisenetv2_server_config.toml
```

正常启动后，服务会运行在服务器配置（`conf/server/<task>/<model>/*.toml`）中 `port` 指定的端口，`worker_nums` 个模型实例会被创建并占用 GPU 资源。仓库自带配置默认 `worker_nums=1`，你可以按 GPU 显存情况适当调整。

## Python 客户端示例

客户端与分类教程相同，见
[tutorials_of_classification_model_server.zh-cn.md](tutorials_of_classification_model_server.zh-cn.md)。

```bash
cd $PROJECT_ROOT
python3 scripts/server/test_server.py --server bisenetv2 --mode single
```

## 关于图像分割服务器的一些特殊说明

图像分割输出与输入同尺寸的类别图。载荷在 `results[0].data`（`image` 为 mask PNG
base64，`colorized_mask` 为上色 PNG base64）。

```json
{
  "status": 0,
  "status_str": "OK",
  "task_id": "demo",
  "results": [
    {
      "status": 0,
      "data": {
        "image": "<png base64>",
        "colorized_mask": "<png base64>"
      }
    }
  ],
  "partial": false
}
```

保存上色结果：

```python
import base64
import json
import urllib.request

with open(src_image_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

body = json.dumps({"images": [img_b64], "req_id": "demo"}).encode()
req = urllib.request.Request(
    url, data=body, headers={"Content-Type": "application/json"}
)
resp = json.loads(urllib.request.urlopen(req).read())
output = resp["results"][0]["data"]["colorized_mask"]
with open("result.png", "wb") as out_f:
    out_f.write(base64.b64decode(output))
```

## 图像分割服务器的可视化输出结果

### BisenetV2 模型

[BisenetV2](https://arxiv.org/abs/2004.02147) :fire: 是一个实时图像分割模型. 你可以参考 [https://github.com/MaybeShewill-CV/bisenetv2-tensorflow](https://github.com/MaybeShewill-CV/bisenetv2-tensorflow) 来获取模型结构和训练细节.

模型主要结构如下
`Bisenetv2 Network Architecture`
![bisenetv2_network_architect](../resources/images/bisenetv2_architecture.png)

`客户端输入图像`
![bisenetv2_server_input](../resources/images/bisenetv2_server_input.png)

`服务端输出图像`
![bisenetv2_server_output](../resources/images/bisenetv2_server_output.png)
