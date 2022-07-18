# Overview
- Compare pytorch ML inference performance across different apple silicon models and linux+cuda machines
- Runs on MacOS M1 GPUS and NVIDIA GPUS on Linux
- Easy to setup and run on both MacOS and Linux
- No need to use huge image datasets. Code generates images randomly. This is fine because the SSDs in the apple silicon laptops are not the bottleneck

## Current Version = 0.1.0
Only image models are supported as of now

# First Steps
## Requirements
### MacOS
1. Homebrew (https://brew.sh/)
2. cmake (brew install cmake)

### Linux
1. cmake (apt-get install cmake)

## Usage
Run resnet50 inference using
```
make image  
```

or, if you want more control

```
USE_GPU=1 NUM_IMAGES=1000 BATCH_SIZE=64 IMAGE_MODEL=resnet50 make image
```

- `USE_GPU`: Flag to run on GPU. Set it to 0 to run on CPU only
- `NUM_IMAGES`: Number of images to generate
- `BATCH_SIZE`: Batch size for running inference. 64 looks to be a sweet spot for apple silicon
- `IMAGE_MODEL`: Select image models from torchvision. Refer https://pytorch.org/vision/0.8/models.html

# Image Dataset
Randomly generated matrices, no need of huge terrabytes of images

`NOTE: torch dataloaders in this project doesn't use multiprocessing as i got some serialization errors. This means preprocessing uses only one CPU core. This might be a good thing as it reduces unfair advantage of using multiple processes. If someone is able to fix this multithreading issue, i would greatly appreciate it`

### Why not use datasets like CIFAR?
Smaller datasets like CIFAR10 are not a good indicator of measring speed. This is because the whole CIFAR10 is < 200MB but in real world scenarios, training involves on millions of images which can go > 1TB. The real world bottleneck while training images lies in preprocessing and smaller datasets are not good to measure that

# Results

## 1. Comparing different MACs with the same spec and RTX 3090
Compared Macbook pro 14 inch 2021 and Mac Studio with the following specs.
- 64 GB RAM
- M1 MAX 32 core
- 2TB SSD

Both Macbook Pro and Mac Studio performed at the same speed around 260-270 images per second BUT the macbook pro was heating up to 99C and the fans went up >6000 RPM and the noise was audible. The bottom of the macbook pro was quite hot, even the keyboard was getting warmer. The battery was draining fast too. Similar results with or without plugged in. 

The Mac studio on the other hand barely reached 57C and the fan never crossed 1500 RPM, hence the machine was super silent. 

Considering MAC Studio for the specs desribed above is $1000 less than Macbook pro, i would recommend getting it + a small M1 Macbook air for portability. The final price of Mac Studio + M1 Macbook air would still be cheaper than a single Macbook pro for the specs described above.  

RTX 3090 on the other hand was only half as fast, this could be due to the CPU bottleneck involved in preprocessing. 

### Macbook Pro log
```
image_benchmark.py-2022-07-17 17:30:43,147 - INFO - Running Image Model resnet50 with use_gpu=True batch_size=64 and num_images=100000
image_benchmark.py-2022-07-17 17:30:43,147 - INFO - DEVICE:Mac M1 GPU detected!
image_benchmark.py-2022-07-17 17:30:43,609 - INFO - Image Model loaded on device = mps
Running Inference:: : 100032it [06:44, 247.20it/s]
image_benchmark.py-2022-07-17 17:37:28,271 - INFO - Finished 100000 images in 404.6518 seconds
```

### Mac Studio Log
```
image_benchmark.py-2022-07-17 17:19:07,246 - INFO - Running Image Model resnet50 with use_gpu=True batch_size=64 and num_images=100000
image_benchmark.py-2022-07-17 17:19:07,246 - INFO - DEVICE:Mac M1 GPU detected!
image_benchmark.py-2022-07-17 17:19:07,710 - INFO - Image Model loaded on device = mps
Running Inference:: : 100032it [06:20, 263.10it/s]
image_benchmark.py-2022-07-17 17:25:27,926 - INFO - Finished 100000 images in 380.2072 seconds
```

### RTX log
```
image_benchmark.py-2022-07-18 04:07:22,069 - INFO - Image Model loaded on device = cuda
Running Inference::   9%|████████▋                                                                                             | 8512/100000 [01:21<14:17, 106.69it/s]
```