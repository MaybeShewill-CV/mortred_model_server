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

| Model Name | Input Image Size | Inference Time (ms) | Fps | Backend |
|------------|------------------|----------------|-----|----|
| MobilenetV2 | 224x224 | 1.75ms | 571.4 | cuda |
| ResNet-50   | 224x224 | 3.35ms | 298.8 | cuda |
| Densenet-121| 224x224 | 5.30ms | 188.8 | cuda |

<b><font color='GrayB' size='6' face='Helvetica'> 图像目标检测 </font></b>

| Model Name | Input Image Size  | Inference Time (ms) | Fps | Backend |
|------------|-------------------|----------------|-----|----|
| YOLOV5-X   | 640x640 | 41.84ms | 23.9 | cuda |
| YOLOV5-L   | 640x640 | 30.58ms | 32.7 | cuda |
| YOLOV5-M   | 640x640 | 25.51ms | 39.2 | cuda |
| YOLOV5-S   | 640x640 | 20.24ms | 49.4 | cuda |
| YOLOV5-N   | 640x640 | 18.34ms | 54.5 | cuda |
| nanodet_plus_m_1x5 | 416x416 | 5.44ms | 183.7 | cuda |

<b><font color='GrayB' size='6' face='Helvetica'> 图像分割 </font></b>

| Model Name | Input Image Size  | Inference Time (ms) | Fps | Backend |
|------------|-------------------|----------------|-----|----|
| BiseNetV2  | 512x1024 | 16.20ms | 61.7 | cuda |

<b><font color='GrayB' size='6' face='Helvetica'> 图像增强 </font></b>

| Model Name | Input Image Size  | Inference Time (ms) | Fps | Backend |
|------------|-------------------|----------------|-----|----|
| Attentive-Gan  | 240x320 | 497.51ms | 2.013 | cuda |
| Enlighten-Gan  | 256x256 | 6.43ms   | 155.6 | cuda |

<b><font color='GrayB' size='6' face='Helvetica'> 图像特征点检测 </font></b>

| Model Name | Input Image Size  | Inference Time (ms) | Fps | Backend |
|------------|-------------------|----------------|-----|----|
| Superpoint-N | 120x160 | 1.19ms | 835.2 | cuda |
| Superpoint-S | 240x320 | 3.49ms | 286.8 | cuda |
| Superpoint-M | 480x640 | 15.3ms | 65.36 | cuda |
| Superpoint-L | 960x1280| 51.8ms | 19.31 | cuda |

<b><font color='GrayB' size='6' face='Helvetica'> 图像文本检测 </font></b>

| Model Name | Input Image Size  | Inference Time (ms) | Fps | Backend |
|------------|-------------------|----------------|-----|----|
| DBNet | 544x960 | 39.12ms | 25.56 | cuda |
