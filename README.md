<div align="center"><font size='8' color='green'><img align="center" width = "50" height = "50" src="./resources/images/icon.png">Mortred AI Model Web Server</div></font></img>  

Morted AI Model Server is a flexible and easy to use tool for serving deep learning models. Models trained by `tensorflow/pytorch` will be deployed via [MNN](https://github.com/alibaba/MNN) toolkit and served as a web server through [workflow](https://github.com/sogou/workflow) framework finally.

The three major components are illustrated on the architecture picture bellow.

<p align="left">
  <img src='./resources/images/simple_architecture.png' alt='simple_architecture' height="500px" width="600px">
</p>

A quick overview and examples for both serving and model benchmarking are provided below. Detailed documentation and examples will be provided in the docs folder.

You're welcomed to ask questions and help me to make it better!

## Contents of this document

* [Quick Start](#quick-start)
* [Benchmark](#benchmark)
* [How To](#how-to)
* [Web Server Configuration](#web-server-configuration)

## Quick Start

Before proceeding further with this document, make sure you have the following prerequisites

**1.** Make sure you have CUDA&GPU&Driver rightly installed. You may refer to [this](https://developer.nvidia.com/cuda-toolkit) to install them

**2.** Make sure you have MNN installed. For install instruction you may find some help [here](https://www.yuque.com/mnn/en/build_linux). MNN-2.0.0's cuda backend was not supported for now. You'd better use MNN-1.2.0 release version to have both cpu and cuda computation backend. The bugs in MNN-2.0.0 remained here will be fixed as soon as possible

**3.** Make sure you have WORKFLOW installed. For install instruction you may find some help [here](https://github.com/sogou/workflow)

**4.** Make sure you have OPENCV installed. For install instruction you may find some help [here](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)

**5.** Make sure your GCC tookit support cpp-11

After all prerequisites are settled down you may start to build the mortred ai server frame work.

### Setup

**Step 1:** Prepare 3rd-party Libraries

For MNN headers and libs

```bash
cp -r $MNN_ROOT_DIR/include/MNN ./3rd_party/include
cp $MNN_ROOT_DIR/build/libMNN.so ./3rd_party/libs
cp $MNN_ROOT_DIR/build/source/backend/cuda/libMNN_Cuda_Main.so ./3rd_party/libs
```

For workflow headers and libs

```bash
cp -r $WORKFLOW_ROOT_DIR/_include/workflow ./3rd_party/include
cp -r $WORKFLOW_ROOT_DIR/_lib/libworkflow.so* ./3rd_party/libs
```

**Step 2:** Build Mortred AI Server

```bash
mkdir build && cd build
cmake ..
make -j10
```

**Step 3:** Download Pre-Built Models

Download pre-built image models via [BaiduNetDisk here](https://pan.baidu.com/s/1sLLSE1CWksKNxmRIGaQn_A) and extract code is `86sd`. Create a directory named `weights` in $PROJECT_ROOT_DIR and unzip the downloaded models in it. The weights directory  structure should looks like

<p align="left">
  <img src='./resources/images/weights_folder_structure.png' alt='weights_folder_architecture'>
</p>

**Step 4:** Test Demo App

The benchmark and server apps will be built in \$PROJECT_ROOT_DIR/_bin and libs will be built in \$PROJECT_ROOT_DIR/_lib.
Benchmark the mobilenetv2 classification model

```bash
cd $PROJECT_ROOT_DIR/bin
./mobilenetv2_benchmark.out ../conf/model/classification/mobilenetv2/mobilenetv2_config.ini
```

You should see the mobilenetv2 model benchmark profile as follows:

<p align="left">
  <img src='./resources/images/mobilenetv2_demo_benchmark.png' alt='mobilenetv2_demo_benchmark'>
</p>

## Benchmark

The benchmark test environment is as follows：

**OS:** Ubuntu 18.04.5 LTS / 5.4.0-53-generic

**MEMORY:** 32G DIMM DDR4 Synchronous 2666 MHz

**CPU:** Intel(R) Core(TM) i5-10400 CPU @ 2.90GHz

**GCC:** gcc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0

**GPU:** GeForce RTX 3080

**CUDA:** CUDA Version: 11.1

**GPU Driver:** Driver Version: 455.23.04

### Model Inference Benchmark

All models loop several times to avoid the influence of gpu's warmup and only model's inference time has been counted.

`Benchmark Code Snappit`
![benchmakr_code_snappit](./resources/images/benchmark_code_snappit.png)

* [Details Of Model's Benchmark](./docs/)


## Reference

* <https://github.com/sogou/workflow>
* <https://github.com/alibaba/MNN>
