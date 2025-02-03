import albumentations as A
from PIL import Image
import numpy as np
from typing import Tuple, Optional, List
from augment.base_augmentation import BaseAugmentation


class AlbumentationAugmentation(BaseAugmentation):
    """
    A wrapper for Albumentations-based augmentations.
    """

    def __init__(self, augmentation: A.BasicTransform, bbox_format: str = "pascal_voc"):
        """
        Args:
            augmentation (A.BasicTransform): The Albumentations augmentation pipeline.
            bbox_format (str): The format for bounding boxes (e.g., "pascal_voc").
                              This will be used to inform Albumentations how to interpret the bboxes.
        """
        self.augmentation = augmentation
        self.bbox_format = bbox_format

    def __call__(
        self, image: Image.Image, bboxes: Optional[List[List[float]]] = None
    ) -> Tuple[Image.Image, Optional[List[List[float]]]]:
        numpy_image = np.array(image)
        if bboxes is not None and bboxes:
            # Albumentations requires class labels even if they are dummy.
            augmented = self.augmentation(
                image=numpy_image, bboxes=bboxes, class_labels=[0] * len(bboxes)
            )
            augmented_image = Image.fromarray(augmented["image"])
            augmented_bboxes = augmented.get("bboxes", bboxes)
            return augmented_image, augmented_bboxes
        else:
            augmented = self.augmentation(image=numpy_image)
            augmented_image = Image.fromarray(augmented["image"])
            return augmented_image, bboxes
