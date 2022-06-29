from this import s
import typer
import PIL
from image_model import ImageModel, random_generator
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from time import time
from logger import get_logger
import numpy

logger = get_logger(
    logger_name=__file__,
    file_name=f"image_benchmark_{0}.log",
)


class RandomImage:
    def __init__(self, image_model: ImageModel) -> None:
        self.image_model = image_model
        self.height = random_generator.integers(low=200, high=500)
        self.width = random_generator.integers(low=200, high=500)
        self.channels: int = 3

    def preprocess(self, image: PIL.Image):
        return self.image_model.transform(image)

    def get(self):
        image = random_generator.random((self.height, self.width, self.channels)) * 255
        #logger.info(f"Generated random image with shape {image.shape}")
        image = PIL.Image.fromarray(numpy.uint8(image))
        image = image.convert("RGB")
        return self.preprocess(image)

    def infer(self, device: str):
        start_time = time()
        image = self.get()
        cpu_time = time() - start_time
        self.image_model.model(image.to(device))
        infer_time = time() - start_time
        return cpu_time, infer_time

class ImageDataset(Dataset):
    def __init__(self, num_images:int, image_model:ImageModel):
        self.image_model = image_model
        self.num_images = num_images

    def __len__(self):
        return self.num_images
    
    def __getitem__(self, index) -> torch.Tensor:
        return RandomImage(self.image_model).get()

def get_dataloader(image_model:ImageModel, num_images:int, batch_size:int):
    dataset = ImageDataset(num_images, image_model)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=False,
                      pin_memory=True,
                      num_workers=0 # NOTE: can't activate multithreading without serialization errors, will look into it -> TODO
                      )

def get_device(use_gpu: bool):
    device = "cpu"
    if not use_gpu:
        logger.info("DEVICE:Running on CPU as requested")
    elif torch.has_mps:
        device = "mps"
        logger.info("DEVICE:Mac M1 GPU detected!")
    elif torch.has_cuda:
        device = "cuda"
        logger.info(f"DEVICE:Nvidia GPU detected with Cudnn = {torch.has_cudnn}!")
    else:
        logger.info("DEVICE:Running on CPU. GPU not detected.")
    return device


def initialize_model(architecture: str, use_gpu: bool):
    device = get_device(use_gpu)
    image_model = ImageModel(model=getattr(models, architecture)(pretrained=True))
    image_model.model.eval()
    image_model.model.to(device)
    logger.info(f"Image Model loaded on device = {device}")
    return image_model, device


def get_progressbar(num_images: int):
    progress_bar = tqdm(total=num_images)
    progress_bar.set_description(f"Processing:")
    return progress_bar

def inference(architecture: str, num_images: int, use_gpu: bool, batch_size:int):
    image_model, device = initialize_model(architecture, use_gpu)
    progress_bar = get_progressbar(num_images)
    dataloader = get_dataloader(image_model, num_images, batch_size)
    for image_batch in dataloader:
        with torch.no_grad():
            image_batch = image_batch.to(device)
            image_model.model(image_batch)
        progress_bar.update(batch_size)
    progress_bar.close()


if __name__ == "__main__":
    typer.run(inference)
