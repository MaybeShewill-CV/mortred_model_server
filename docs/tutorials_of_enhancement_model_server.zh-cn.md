# 图像增强服务器说明

## 启动一个图像增强服务器

启动一个增强服务器非常简单，主要代码如下

`图像增强服务器代码段`
![strat_a_derain_server](../resources/images/start_a_derain_server.png)

编译好的可执行文件存放在 `$PROJECT_ROOT/_bin/attentive_gan_derain_server.out`, 你只需要执行

```bash
cd $PROJECT_ROOT/_bin
./attentive_gan_derain_server.out ../conf/server/enhancement/attentive_gan_derain/attentive_gan_server_cfg.ini
```

默认状态下服务会启动在 `http:://localhost:8091` 并且有4个模型实例被启动。

## Python 客户端示例

启动测试客户端方式如下

```python
cd $PROJECT_ROOT/scripts
export PYTHONPATH=$PWD:$PYTHONPATH
python server/test_server.py --server attentive_gan --mode single
```

## 图像增强模型客户端的特殊说明

绝大多数的图像增强模型的输出都可以抽象成一张增强过后的图像，图像增强服务器的response内容是一个json对象

```python
resp = {
    'req_id': '',
    'code': 1,
    'msg': 'success',
    'data': {
        'enhance_result': base64_image_content
    }
}
```

`enhance_result` 保存了图像增强模型的输出图像，该图像被base64编码过，客户端获取结果的过程中需要做一下base64解码

```python
with open(src_image_path, 'rb') as f:
    image_data = f.read()
    base64_data = base64.b64encode(image_data)

    post_data = {
        'img_data': base64_data.decode(),
        'req_id': 'demo',
    }
    resp = requests.post(url=url, data=json.dumps(post_data))
    output = json.loads(resp.text)['data']['enhance_result']
    out_f = open('result.jpg', 'wb')
    out_f.write(base64.b64decode(output))
    out_f.close()
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