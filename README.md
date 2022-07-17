# Usage
Requirements: `Homebrew`
```
make image  
```

or, if you want more control

```
USE_GPU=1 NUM_IMAGES=1000 BATCH_SIZE=64 IMAGE_MODEL=resnet50 make image
```

- USE_GPU = Flag to run on GPU. Set it to 0 to run on CPU only.
- NUM_IMAGES = Number of images to generate
- BATCH_SIZE = batch size for running inference. 64 looks to be a sweet spot for apple silicon

# Overview
- Compare pytorch performance across different apple silicon models
- Easy to setup and run
- No need to use huge image datasets. Code generates images randomly. This is fine because the SSDs in the apple silicon laptops are not the bottleneck

## Why not use datasets like CIFAR?
Smaller datasets like CIFAR10 are not a good indicator of measring speed. This is because the whole CIFAR10 is < 200MB but in real world scenarios, training involves on millions of images which can go > 1TB. The real world bottleneck while training images lies in preprocessing and smaller datasets are not good to measure that.

## Current Version = 0.1.0
Only image models are supported as of now.

# Results

# Comparing different MACs with the same spec




