from torchvision import transforms
from PIL import Image
import random
from typing import List, Optional, Tuple


class RandomBrightnessContrast(object):
    """Applies random brightness and contrast adjustments."""

    def __init__(
        self, brightness_limit: float = 0.2, contrast_limit: float = 0.2, p: float = 0.5
    ):
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.p = p

    def __call__(
        self, image: Image.Image, bboxes: Optional[List[List[float]]] = None
    ) -> Tuple[Image.Image, Optional[List[List[float]]]]:
        if random.random() < self.p:
            brightness_factor = 1.0 + random.uniform(
                -self.brightness_limit, self.brightness_limit
            )
            contrast_factor = 1.0 + random.uniform(
                -self.contrast_limit, self.contrast_limit
            )
            enhancer_brightness = transforms.ColorJitter(brightness=brightness_factor)
            enhancer_contrast = transforms.ColorJitter(contrast=contrast_factor)
            image = enhancer_brightness(image)
            image = enhancer_contrast(image)
        return image, bboxes  # Bounding boxes are passed through unchanged


class RandomHorizontalFlipBoundingBoxes(object):
    """Randomly flips the image horizontally and adjusts bounding box coordinates."""

    def __init__(
        self, p: float = 0.5, image_width: int = 1
    ):  # image_width for bbox normalization
        self.p = p
        self.image_width = image_width

    def __call__(
        self, image: Image.Image, bboxes: Optional[List[List[float]]] = None
    ) -> Tuple[Image.Image, Optional[List[List[float]]]]:
        if bboxes is None:
            return image, None

        if random.random() < self.p:
            image = transforms.functional.hflip(image)
            flipped_bboxes = []
            for bbox in bboxes:
                x_min, y_min, x_max, y_max, *rest = (
                    bbox  # Assuming bbox format: [x_min, y_min, x_max, y_max, class_id, ...]
                )
                flipped_x_min = 1.0 - x_max  # Normalized coordinates, flip around 0.5
                flipped_x_max = 1.0 - x_min
                flipped_bboxes.append(
                    [flipped_x_min, y_min, flipped_x_max, y_max, *rest]
                )  # Maintain class_id and other info
            return image, flipped_bboxes
        return image, bboxes
