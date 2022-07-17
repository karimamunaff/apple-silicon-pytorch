from typing import Any, Tuple

import numpy
import pydantic
from torchvision import transforms

from config import PydanticConfig

random_generator = numpy.random.default_rng(2021)


@pydantic.dataclasses.dataclass(config=PydanticConfig)
class ImageModel:
    model: Any
    height_width_range: Tuple = (0, 200)
    transform: transforms.Compose = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
