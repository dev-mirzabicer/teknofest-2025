import torch
from typing import List, Dict, Any


def detection_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for object detection tasks.

    Args:
        batch (List[Dict[str, Any]]): List of samples from the dataset, where each sample is a dict
                                      containing keys "images", "targets", "image_ids", and
                                      "original_image_sizes".

    Returns:
        A dictionary with:
          - "images": A tensor of shape (B, C, H, W) (stacked images).
          - "targets": A list of target dicts, one per image.
          - "image_ids": A list of image identifiers.
          - "original_image_sizes": A list of original image size tuples.
    """
    # Extract images and stack them into a single tensor.
    images = [sample["images"] for sample in batch]
    try:
        images = torch.stack(images, dim=0)
    except Exception as e:
        raise RuntimeError(f"Error stacking images: {e}")

    # For targets and other metadata, we keep them as lists since their shapes can vary.
    targets = [sample["targets"] for sample in batch]
    image_ids = [sample["image_ids"] for sample in batch]
    original_image_sizes = [sample["original_image_sizes"] for sample in batch]

    return {
        "images": images,
        "targets": targets,
        "image_ids": image_ids,
        "original_image_sizes": original_image_sizes,
    }
