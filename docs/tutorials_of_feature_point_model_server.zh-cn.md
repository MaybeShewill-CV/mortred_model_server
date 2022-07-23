# 图像特征点检测服务器说明

## 启动一个图像特征点检测服务器

启动图像特征点检测服务器非常简单

`图像特征点检测服务器代码段`
![strat_a_superpoint_server](../resources/images/start_a_superpoint_server.png)

编译好的可执行文件存放在 `$PROJECT_ROOT/_bin/superpoint_fp_det_server.out`。运行

```bash
cd $PROJECT_ROOT/_bin
./superpoint_fp_det_server.out ../conf/server/feature_point/superpoint/superpoint_server_cfg.ini
```

默认状态下服务会启动在 `http:://localhost:8091` 并且有4个模型实例被启动。

## Python 客户端示例

测试仅需运行

```python
cd $PROJECT_ROOT/scripts
export PYTHONPATH=$PWD:$PYTHONPATH
python server/test_server.py --server superpoint --mode single
```

## 关于图像特征点检测服务器的特殊说明

Most of the feature's model output is set of feature points. A single feature point consist of location and descriptor. To reduce the response's content size the server won't output the feature points' descriptor you may uncomment the code in [./src/server/feature_point/superpoint_fp_server.cpp#L170-L172](../src/server/feature_point/superpoint_fp_server.cpp) and recompile to make server output feature points' descriptor. Server's response is a json like

图像特征点检测服务器的输出是一张图像上的一系列特征点. 图像特征点是由点的位置和对该点的一个特征描述子来构成的. 为了减少服务器response的长度模型情况下不输出特征点的特征向量只输出特征点的位置信息，你可以解注释 [./src/server/feature_point/superpoint_fp_server.cpp#L170-L172](../src/server/feature_point/superpoint_fp_server.cpp) 后重新编译来让服务器端输出特征点的特征向量. 服务器端回复的response json对象结构如下

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
