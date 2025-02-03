from typing import Tuple, Optional, List
from PIL import Image
from torchvision import transforms
from augment.base_augmentation import BaseAugmentation


class TorchAugmentation(BaseAugmentation):
    """
    A wrapper for TorchVision-based augmentations.
    Since TorchVision transforms generally do not support bounding box transformation,
    this wrapper applies the transformation to the image only and returns the bounding boxes unchanged.
    """

    def __init__(self, transform: transforms.Compose):
        """
        Args:
            transform (transforms.Compose): A TorchVision transform pipeline.
        """
        self.transform = transform

    def __call__(
        self, image: Image.Image, bboxes: Optional[List[List[float]]] = None
    ) -> Tuple[Image.Image, Optional[List[List[float]]]]:
        augmented_image = self.transform(image)
        # TorchVision transforms do not adjust bounding boxes; return them as-is.
        return augmented_image, bboxes
