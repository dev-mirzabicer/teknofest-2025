from abc import ABC, abstractmethod
from typing import Tuple, Optional, List
from PIL import Image


class BaseAugmentation(ABC):
    """
    Abstract base class for augmentations.
    Any augmentation should implement the __call__ method which takes a PIL image and an optional
    list of bounding boxes, and returns the augmented image along with (optionally) the augmented bounding boxes.
    """

    @abstractmethod
    def __call__(
        self, image: Image.Image, bboxes: Optional[List[List[float]]] = None
    ) -> Tuple[Image.Image, Optional[List[List[float]]]]:
        """
        Apply augmentation to an image and optionally its bounding boxes.

        Args:
            image (PIL.Image): The input image.
            bboxes (Optional[List[List[float]]]): A list of bounding boxes (each in [x_min, y_min, x_max, y_max] format).

        Returns:
            Tuple containing:
                - Augmented PIL.Image.
                - Augmented bounding boxes (or None if not applicable).
        """
        pass
