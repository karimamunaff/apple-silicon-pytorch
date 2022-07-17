# Usage
Requirements: `Homebrew`

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

# Overview
- Compare pytorch ML inference performance across different apple silicon models
- Easy to setup and run
- No need to use huge image datasets. Code generates images randomly. This is fine because the SSDs in the apple silicon laptops are not the bottleneck

## Why not use datasets like CIFAR?
Smaller datasets like CIFAR10 are not a good indicator of measring speed. This is because the whole CIFAR10 is < 200MB but in real world scenarios, training involves on millions of images which can go > 1TB. The real world bottleneck while training images lies in preprocessing and smaller datasets are not good to measure that

## Current Version = 0.1.0
Only image models are supported as of now

# Results

## Comparing different MACs with the same spec
Compared Macbook pro 14 inch 2021 and Mac Studio with the following specs.
- 64 GB RAM
- M1 MAX 32 core
- 2TB SSD

Both Macbook Pro and Mac Studio performed at the same speed around 260-270 images per second BUT the macbook pro was heating up to 99C and the fans went up >6000 RPM and the noise was audible. The bottom of the macbook pro was quite hot, even the keyboard was getting warmer. The battery was draining fast too. 

The Mac studio on the other hand barely reached 57C and the fan never crossed 1500 RPM, hence the machine was super silent. 

Considering MAC Studio for the specs desribed above is $1000 less than Macbook pro, i would recommend gettig it + a small M1 Macbook air for portability. The final price of Mac Studio + M1 Macbook air would still be cheaper than a single Macbook pro for the specs described above.  



