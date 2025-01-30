from torchvision import transforms
from PIL import Image
import random
from typing import Tuple, Optional, List


class RandomBrightnessContrast(object):
    """Applies random brightness and contrast adjustments to PIL Images.

    Args:
        brightness_limit (float): Maximum brightness adjustment factor (e.g., 0.2 for +/- 20%).
        contrast_limit (float): Maximum contrast adjustment factor (e.g., 0.2 for +/- 20%).
        p (float): Probability of applying the augmentation.
    """

    def __init__(
        self, brightness_limit: float = 0.2, contrast_limit: float = 0.2, p: float = 0.5
    ) -> None:
        assert 0.0 <= p <= 1.0, "Probability p must be between 0 and 1."
        assert brightness_limit >= 0.0, "Brightness limit must be non-negative."
        assert contrast_limit >= 0.0, "Contrast limit must be non-negative."
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.p = p

    def __call__(
        self, image: Image.Image, bboxes: Optional[List[List[float]]] = None
    ) -> Tuple[Image.Image, Optional[List[List[float]]]]:
        """
        Args:
            image (PIL.Image): Image to be augmented.
            bboxes (Optional[List[List[float]]]): Bounding boxes associated with the image (format: [x_min, y_min, x_max, y_max, ...]).

        Returns:
            Tuple[PIL.Image, Optional[List[List[float]]]]: Augmented image and bounding boxes (bboxes are passed through unchanged).
        """
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
        return image, bboxes


class RandomHorizontalFlipBoundingBoxes(object):
    """Randomly flips the image horizontally and adjusts bounding box coordinates.

    Args:
        p (float): Probability of applying the horizontal flip.
        image_width (int): Width of the image (used for normalizing bounding box coordinates).
    """

    def __init__(self, p: float = 0.5, image_width: int = 1) -> None:
        assert 0.0 <= p <= 1.0, "Probability p must be between 0 and 1."
        assert image_width > 0, "Image width must be positive."
        self.p = p
        self.image_width = image_width

    def __call__(
        self, image: Image.Image, bboxes: Optional[List[List[float]]] = None
    ) -> Tuple[Image.Image, Optional[List[List[float]]]]:
        """
        Args:
            image (PIL.Image): Image to be augmented.
            bboxes (Optional[List[List[float]]]): Bounding boxes associated with the image (format: [x_min, y_min, x_max, y_max, ...], normalized 0-1).

        Returns:
            Tuple[PIL.Image, Optional[List[List[float]]]]: Augmented image and flipped bounding boxes.
        """
        if bboxes is None:
            return image, None

        if random.random() < self.p:
            image = transforms.functional.hflip(image)
            flipped_bboxes: List[List[float]] = []
            for bbox in bboxes:
                if len(bbox) < 4:  # Safety check for bbox format
                    continue  # Skip malformed bbox
                x_min, y_min, x_max, y_max, *rest = (
                    bbox  # bbox format: [x_min, y_min, x_max, y_max, class_id, ...]
                )
                flipped_x_min = 1.0 - x_max  # Normalized coordinates, flip around 0.5
                flipped_x_max = 1.0 - x_min
                flipped_bboxes.append(
                    [flipped_x_min, y_min, flipped_x_max, y_max, *rest]
                )  # Maintain class_id and other info
            return image, flipped_bboxes
        return image, bboxes
