# Tutorials Of Feature Point Model Server

## Start A Feature Point Server

It's very quick to start a feature point server. Main code are showed below

`Feature Point Server Code Snappit`
![strat_a_superpoint_server](../resources/images/start_a_superpoint_server.png)

The executable binary file was built in $PROJECT_ROOT/_bin/superpoint_fp_det_server.out Simply run

```bash
cd $PROJECT_ROOT/_bin
./superpoint_fp_det_server.out ../conf/server/feature_point/superpoint/superpoint_server_cfg.ini
```

When server successfully start on `http:://localhost:8091` you're supposed to see `worker_nums` workers were called up and occupied your GPU resources. By default 4 model workers will be created you may enlarge it if you have enough GPU memory.

## Python Client Example

Local python client test is similiar with mobilenetv2 classification server you may read [toturials_of_classfication_model_server.md](../docs/toturials_of_classification_model_server.md) for details.

To use test python client you may run

```python
cd $PROJECT_ROOT/scripts
export PYTHONPATH=$PWD:$PYTHONPATH
python server/test_server.py --server superpoint --mode single
```

## Unique Tips For Feature Point Model Python Client

Most of the feature's model output is set of feature points. A single feature point consist of location and descriptor. To reduce the response's content size the server won't output the feature points' descriptor you may uncomment the code in [./src/server/feature_point/superpoint_fp_server.cpp#L170-L172](../src/server/feature_point/superpoint_fp_server.cpp) and recompile to make server output feature points' descriptor. Server's response is a json like

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

`location` contains the feature points' location information and you can visualization the result by yourself.

## Feature Point Model's Visualization Result

### SuperPoint Model

[superpoint](https://arxiv.org/abs/1712.07629) model was designed for detect and describe feature point on images. You may refer to repo [https://github.com/magicleap/SuperPointPretrainedNetwork](https://github.com/magicleap/SuperPointPretrainedNetwork) for details about training details.

`Server's Input Image`

![superpoint_server_input](../resources/images/superpoint_server_input.png)

`Server's Output Image With Different Model`
<center>*********** 120x160_model **************** 240x320_model ********************* 480x640_model ******************* 960x1280_model ***********</center>

![superpoint_server_output](../resources/images/superpoint_server_output.png)

![superpoint_server_output2](../resources/images/superpoint_server_output2.png)
