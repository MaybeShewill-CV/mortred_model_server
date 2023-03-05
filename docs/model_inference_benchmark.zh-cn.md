<b><font color='black' size='8' face='Helvetica'> 模型基准测试 </font></b>

基准测试环境如下：

**OS:** Ubuntu 18.04.5 LTS / 5.4.0-53-generic

**MEMORY:** 32G DIMM DDR4 Synchronous 2666 MHz

**CPU:** Intel(R) Core(TM) i5-10400 CPU @ 2.90GHz

**GCC:** gcc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0

**GPU:** GeForce RTX 3080

**CUDA:** CUDA Version: 11.1

**GPU Driver:** Driver Version: 455.23.04

<b><font color='GrayB' size='6' face='Helvetica'> 图像分类 </font></b>

| Model Name   | Input Image Size | Inference Time (ms) | Fps   | Backend |
|--------------|------------------|---------------------|-------|---------|
| MobilenetV2  | 224x224          | 1.03ms              | 969.7 | cuda    |
| ResNet-50    | 224x224          | 3.35ms              | 298.8 | cuda    |
| Densenet-121 | 224x224          | 3.67ms              | 272.8 | cuda    |

<b><font color='GrayB' size='6' face='Helvetica'> 图像目标检测 </font></b>

| Model Name         | Input Image Size | Inference Time (ms) | Fps   | Backend |
|--------------------|------------------|---------------------|-------|---------|
| YOLOV5-X           | 640x640          | 25.06ms             | 39.9  | cuda    |
| YOLOV5-L           | 640x640          | 19.92ms             | 50.2  | cuda    |
| YOLOV5-M           | 640x640          | 16.61ms             | 60.2  | cuda    |
| YOLOV5-S           | 640x640          | 14.47ms             | 69.1  | cuda    |
| YOLOV5-N           | 640x640          | 13.21ms             | 75.7  | cuda    |
| nanodet_plus_m_1x5 | 416x416          | 5.34ms              | 187.3 | cuda    |

<b><font color='GrayB' size='6' face='Helvetica'> 图像分割 </font></b>

| Model Name | Input Image Size | Inference Time (ms) | Fps  | Backend |
|------------|------------------|---------------------|------|---------|
| BiseNetV2  | 512x1024         | 16.20ms             | 61.7 | cuda    |

<b><font color='GrayB' size='6' face='Helvetica'> 图像增强 </font></b>

| Model Name    | Input Image Size | Inference Time (ms) | Fps   | Backend |
|---------------|------------------|---------------------|-------|---------|
| Attentive-Gan | 240x320          | 453.72ms            | 2.204 | cuda    |
| Enlighten-Gan | 256x256          | 6.81ms              | 146.8 | cuda    |

<b><font color='GrayB' size='6' face='Helvetica'> 图像特征点检测 </font></b>

| Model Name   | Input Image Size | Inference Time (ms) |  Fps   | Backend |
|--------------|------------------|---------------------|--------|---------|
| Superpoint-N | 120x160          | 0.94ms              | 1064.9 |  cuda   |
| Superpoint-S | 240x320          | 3.05ms              | 328.1  |  cuda   |
| Superpoint-M | 480x640          | 14.3ms              | 70.07  |  cuda   |
| Superpoint-L | 960x1280         | 66.6ms              | 15.01  |  cuda   |

<b><font color='GrayB' size='6' face='Helvetica'> 图像文本检测 </font></b>

| Model Name | Input Image Size | Inference Time (ms) | Fps   | Backend |
|------------|------------------|---------------------|-------|---------|
| DBNet      | 655x445          | 12.02ms             | 83.21 | cuda    |

<b><font color='GrayB' size='6' face='Helvetica'> Reference </font></b>

* <https://github.com/PaddlePaddle/PaddleSeg>