from torch.utils.data import DataLoader
from data.datasets.drone_object_detection import (
    DroneObjectDetectionDataset,
    custom_collate_fn,
)
from torchvision import transforms
import os
import json
import logging
from data.augment.augments import (
    RandomBrightnessContrast,
    RandomHorizontalFlipBoundingBoxes,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


if __name__ == "__main__":
    # --- Configuration ---
    data_root_dir = "..."
    annotation_file_path = os.path.join(data_root_dir, "annotations.json")
    annotation_format_type = "json_coco"  # Or 'txt_yolo', 'custom_format'
    class_names = ["human", "vehicle"]
    image_resize_size = (512, 512)
    batch_size_train = 16
    batch_size_val = 8
    num_workers_dataloader = 4
    use_dataset_cache = True
    cache_backend_type = "ram"  # Or 'disk'
    cache_directory = ".drone_dataset_cache"  # For disk cache

    # --- Define Data Augmentations (Pluggable) ---
    train_augmentations = transforms.Compose(
        [
            RandomBrightnessContrast(p=0.5),
            RandomHorizontalFlipBoundingBoxes(
                p=0.5, image_width=image_resize_size[1]
            ),  # Pass image width for bbox flip
            # Add more augmentations here (e.g., RandomScaleBoundingBoxes, RandomCropBoundingBoxes, etc.)
        ]
    )
    val_augmentations = None  # No augmentations for validation

    # --- Create Datasets ---
    try:
        train_dataset = DroneObjectDetectionDataset(
            data_root=os.path.join(
                data_root_dir, "train_images"
            ),  # Assuming train images in 'train_images' subdir
            annotation_file=os.path.join(
                data_root_dir, "train_annotations", annotation_file_path
            ),  # Assuming train annotations in 'train_annotations' subdir
            image_size=image_resize_size,
            transform=train_augmentations,
            annotation_format=annotation_format_type,
            classes=class_names,
            use_cache=use_dataset_cache,
            cache_backend=cache_backend_type,
            cache_dir=cache_directory,
        )

        val_dataset = DroneObjectDetectionDataset(
            data_root=os.path.join(
                data_root_dir, "val_images"
            ),  # Assuming val images in 'val_images' subdir
            annotation_file=os.path.join(
                data_root_dir, "val_annotations", annotation_file_path
            ),  # Assuming val annotations in 'val_annotations' subdir
            image_size=image_resize_size,
            transform=val_augmentations,  # No augmentations for validation
            annotation_format=annotation_format_type,
            classes=class_names,
            use_cache=use_dataset_cache,
            cache_backend=cache_backend_type,
            cache_dir=cache_directory,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        logging.error(f"Dataset initialization failed: {e}")
        exit()  # Or handle dataset loading failure gracefully

    # --- Create DataLoaders ---
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,  # Shuffle for training
        num_workers=num_workers_dataloader,
        collate_fn=custom_collate_fn,  # Use custom collate function
        pin_memory=False,  # True for GPU training
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,  # No shuffle for validation
        num_workers=num_workers_dataloader,
        collate_fn=custom_collate_fn,
        pin_memory=False,
    )

    # --- Example Iteration ---
    for batch_idx, (images, targets) in enumerate(train_dataloader):
        print(
            f"Batch {batch_idx+1} - Image batch shape: {images.shape}, Number of targets: {len(targets)}"
        )
        # Example: Access bounding boxes and labels for the first image in the batch
        first_image_boxes = targets[0]["boxes"]
        first_image_labels = targets[0]["labels"]
        print(
            f"  First image - Boxes shape: {first_image_boxes.shape}, Labels shape: {first_image_labels.shape}"
        )

        if batch_idx > 2:  # Break after a few batches for example
            break

    print("DataLoader example finished.")
