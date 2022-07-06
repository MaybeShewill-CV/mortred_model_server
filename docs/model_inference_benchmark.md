# Model Inference Benchmark

The benchmark test environment is as follows：

**OS:** Ubuntu 18.04.5 LTS / 5.4.0-53-generic

**MEMORY:** 32G DIMM DDR4 Synchronous 2666 MHz

**CPU:** Intel(R) Core(TM) i5-10400 CPU @ 2.90GHz

**GCC:** gcc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0

**GPU:** GeForce RTX 3080

**CUDA:** CUDA Version: 11.1

**GPU Driver:** Driver Version: 455.23.04

## Image Classification

| Model Name | Input Image Size | Inference Time (ms) | Fps | Backend |
|------------|------------------|----------------|-----|----|
| MobilenetV2 | 224x224 | 1.75ms | 571.4 | cuda |
| ResNet-50   | 224x224 | 3.35ms | 298.8 | cuda |
| Densenet-121| 224x224 | 5.30ms | 188.8 | cuda |


## Image Object Detection

| Model Name | Input Image Size  | Inference Time (ms) | Fps | Backend |
|------------|-------------------|----------------|-----|----|
| YOLOV5-X   | 640x640 | 41.84ms | 23.9 | cuda |
| YOLOV5-L   | 640x640 | 30.58ms | 32.7 | cuda |
| YOLOV5-M   | 640x640 | 25.51ms | 39.2 | cuda |
| YOLOV5-S   | 640x640 | 20.24ms | 49.4 | cuda |
| YOLOV5-N   | 640x640 | 18.34ms | 54.5 | cuda |
| nanodet_plus_m_1x5 | 416x416 | 5.44ms | 183.7 | cuda |