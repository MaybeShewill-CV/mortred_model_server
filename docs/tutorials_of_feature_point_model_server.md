# Tutorials Of Feature Point Model Server

## Start A Feature Point Server

It's very quick to start a feature point server. Main code are showed below

`Feature Point Server Code Snappit`
![strat_a_superpoint_server](../resources/images/start_a_superpoint_server.png)

The unified server binary is `$PROJECT_ROOT/_bin/mortred-model-server.out`. Simply run

```bash
cd $PROJECT_ROOT/_bin
./mortred-model-server.out --model SUPERPOINT ../conf/server/feature_point/superpoint/superpoint_server_cfg.toml
```

When the server starts successfully at the `port` configured in your server config (`conf/server/<task>/<model>/*.toml`), `worker_nums` workers will be spawned and occupy your GPU resources. The shipped configs default to `worker_nums=1`; you may enlarge it if you have enough GPU memory.

## Python Client Example

The Python client is the same as the classification tutorial:
[tutorials_of_classification_model_server.md](tutorials_of_classification_model_server.md).

```bash
cd $PROJECT_ROOT
python3 scripts/server/test_server.py --server superpoint --mode single
```

## Unique Tips For Feature Point Model Python Client

The JSON payload is built by [`fill_feature_points`](../src/server/response_serializers.h).
Each point is `score`, `location` `[x, y]`, and `descriptor`.

```json
{
  "status": 0,
  "status_str": "OK",
  "task_id": "demo",
  "results": [
    {
      "status": 0,
      "data": [
        {
          "score": 0.95,
          "location": [100.5, 85.4],
          "descriptor": []
        }
      ]
    }
  ],
  "partial": false
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
