# 图像增强服务器说明

## 启动一个图像增强服务器

启动一个增强服务器非常简单，主要代码如下

`图像增强服务器代码段`
![strat_a_derain_server](../resources/images/start_a_derain_server.png)

统一入口在 `$PROJECT_ROOT/_bin/mortred-model-server.out`，你只需要执行

```bash
cd $PROJECT_ROOT/_bin
./mortred-model-server.out --model ATTENTIVE_GAN_DERAIN ../conf/server/enhancement/attentive_gan_derain/attentive_gan_server_cfg.toml
```

正常启动后，服务会运行在服务器配置（`conf/server/<task>/<model>/*.toml`）中 `port` 指定的端口，`worker_nums` 个模型实例会被创建并占用 GPU 资源。仓库自带配置默认 `worker_nums=1`，你可以按 GPU 显存情况适当调整。

## Python 客户端示例

客户端与分类教程相同，见
[tutorials_of_classification_model_server.zh-cn.md](tutorials_of_classification_model_server.zh-cn.md)。

```bash
cd $PROJECT_ROOT
python3 scripts/server/test_server.py --server attentive_gan --mode single
```

## 图像增强模型客户端的特殊说明

增强结果在 `results[0].data.image`（JPEG/PNG base64）。

```json
{
  "status": 0,
  "status_str": "OK",
  "task_id": "demo",
  "results": [
    {
      "status": 0,
      "data": {
        "image": "<jpeg base64>"
      }
    }
  ],
  "partial": false
}
```

保存结果：

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
output = resp["results"][0]["data"]["image"]
with open("result.jpg", "wb") as out_f:
    out_f.write(base64.b64decode(output))
```

## 图像增强模型的可视化结果示例

### AttentiveGan Derain 模型

[attentive_gan_derain](https://arxiv.org/abs/1711.10098) 模型是一个做图像去雨滴的模型. 你可以参考 [https://github.com/MaybeShewill-CV/attentive-gan-derainnet](https://github.com/MaybeShewill-CV/attentive-gan-derainnet) 代码库获取模型训练和结构的一些细节。

`客户端发送的图像`
![attentive_server_input](../resources/images/attentive_gan_server_input.png)

`服务端返回的图像`
![attentive_server_output](../resources/images/attentive_gan_server_output.png)

### EnlightenGan 模型

[enlighten_gan_derain](https://arxiv.org/abs/1906.06972) model was designed for low light image enhancement task. You may refer to repo [https://github.com/VITA-Group/EnlightenGAN](https://github.com/VITA-Group/EnlightenGAN) for details about training details.

[enlighten_gan_derain](https://arxiv.org/abs/1906.06972) 模型是一个低光照补偿模型. 你可以参考 [https://github.com/VITA-Group/EnlightenGAN](https://github.com/VITA-Group/EnlightenGAN) 代码库来获取模型的训练细节。

`客户端发送的图像`
![enlighten_server_input](../resources/images/enlighten_gan_server_input.png)

`服务端返回的图像`
![attentive_server_output](../resources/images/enlighten_gan_server_output.png)