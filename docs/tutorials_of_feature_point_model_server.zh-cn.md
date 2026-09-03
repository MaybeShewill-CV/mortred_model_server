# 图像特征点检测服务器说明

## 启动一个图像特征点检测服务器

启动图像特征点检测服务器非常简单

`图像特征点检测服务器代码段`
![strat_a_superpoint_server](../resources/images/start_a_superpoint_server.png)

统一入口在 `$PROJECT_ROOT/_bin/mortred-model-server.out`。运行

```bash
cd $PROJECT_ROOT/_bin
./mortred-model-server.out --model SUPERPOINT ../conf/server/feature_point/superpoint/superpoint_server_cfg.toml
```

正常启动后，服务会运行在服务器配置（`conf/server/<task>/<model>/*.toml`）中 `port` 指定的端口，`worker_nums` 个模型实例会被创建并占用 GPU 资源。仓库自带配置默认 `worker_nums=1`，你可以按 GPU 显存情况适当调整。

## Python 客户端示例

测试仅需运行

```python
cd $PROJECT_ROOT/scripts
export PYTHONPATH=$PWD:$PYTHONPATH
python server/test_server.py --server superpoint --mode single
```

## 关于图像特征点检测服务器的特殊说明

图像特征点检测服务器的输出是一张图像上的一系列特征点。图像特征点由位置和描述子构成。JSON 由 [`fill_feature_points`](../src/server/response_serializers.h) 生成，已经包含两者。服务器端回复的 response json 对象结构如下

```python
resp = {
    'req_id': '',
    'code': 1,
    'msg': 'success',
    'data': [
        {
            'score': 0.95,
            'location': [100.5, 85.4],
            'descriptor': []
        },
        {
            ...
        },
    ]
}
```

`location` 保存了特征点的位置信息，你可以自行可视化该检测结果.

## 特征点检测模型输出可视化结果

### SuperPoint 模型

[superpoint](https://arxiv.org/abs/1712.07629) 是一个用来检测和描述图像特征点的模型. 你可以参考 [https://github.com/magicleap/SuperPointPretrainedNetwork](https://github.com/magicleap/SuperPointPretrainedNetwork) 来获取模型的结构和训练信息。

`客户端输入图像`

![superpoint_server_input](../resources/images/superpoint_server_input.png)

`服务端输出结果`
<center>*********** 120x160_model **************** 240x320_model ********************* 480x640_model ******************* 960x1280_model ***********</center>

![superpoint_server_output](../resources/images/superpoint_server_output.png)

![superpoint_server_output2](../resources/images/superpoint_server_output2.png)
