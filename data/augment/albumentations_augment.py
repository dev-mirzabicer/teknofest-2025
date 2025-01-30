import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from typing import Tuple, Optional, List


class AlbumentationsWrapper(object):
    """
    Wraps Albumentations augmentations to be compatible with our DataLoader.

    Handles both image and bounding box augmentations from Albumentations.
    Assumes bounding boxes are in normalized [x_min, y_min, x_max, y_max] format.
    """

    def __init__(self, augmentation: A.BasicTransform):
        self.augmentation = augmentation

    def __call__(
        self, image: Image.Image, bboxes: Optional[List[List[float]]] = None
    ) -> Tuple[Image.Image, Optional[List[List[float]]]]:
        """
        Args:
            image (PIL.Image): Input PIL Image.
            bboxes (Optional[List[List[float]]]): List of bounding boxes in normalized [x_min, y_min, x_max, y_max, ...].

        Returns:
            Tuple[PIL.Image, Optional[List[List[float]]]]: Augmented image and bounding boxes.
        """
        numpy_image = np.array(image)  # Albumentations works with NumPy arrays
        if bboxes is not None and bboxes:  # Check if bboxes is not None and not empty
            # Extract coordinates and class info
            coords_only = [bbox[:4] for bbox in bboxes]
            class_info = [bbox[4:] for bbox in bboxes]  # Keep class and other info

            augmented = self.augmentation(
                image=numpy_image,
                bboxes=coords_only,
                class_labels=[0] * len(coords_only),
            )  # Dummy class labels, Albumentations requires them
            augmented_image = Image.fromarray(augmented["image"])
            augmented_bboxes_coords = augmented["bboxes"]

            augmented_bboxes = []
            for i, coords in enumerate(augmented_bboxes_coords):
                augmented_bboxes.append(
                    list(coords) + class_info[i]
                )  # Re-attach class info

            return augmented_image, augmented_bboxes
        else:  # No bounding boxes, only augment image
            augmented = self.augmentation(image=numpy_image)
            augmented_image = Image.fromarray(augmented["image"])
            return augmented_image, bboxes  # Bboxes remain None


import numpy as np  # Import numpy here, as it's needed in __call__
