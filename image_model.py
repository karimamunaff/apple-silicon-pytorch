import typer
import tqdm
from torchvision import transforms, models
import numpy
from typing import Tuple, Generator, Any
from torch import nn
from PIL import Image
from config import PydanticConfig
import pydantic

random_generator = numpy.random.default_rng(2021)

@pydantic.dataclasses.dataclass(config=PydanticConfig)
class ImageModel():
    model:Any
    height_width_range:Tuple = (0,200)
    transform:transforms.Compose = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])