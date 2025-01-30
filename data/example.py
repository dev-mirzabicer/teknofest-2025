import os
import json
import datetime
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import albumentations as A
import cProfile
import pstats

# Import custom modules from the 'data' directory
from data.datasets.drone_object_detection import DroneObjectDetectionDataset
from data.augment.augments import (
    RandomBrightnessContrast,
    RandomHorizontalFlipBoundingBoxes,
)
from data.augment.albumentations_augment import AlbumentationsWrapper

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def custom_collate_fn(batch):
    """Custom collate function."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    batched_images = torch.stack(images)
    return batched_images, targets


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))  # Print resolved config

    # --- Dataset Configuration from Hydra ---
    dataset_cfg = cfg.dataset.drone_detection
    dataloader_cfg = cfg.dataloader.default
    augment_cfg = cfg.augmentations.default

    train_augmentations_list = []
    # --- Custom Augmentations ---
    if augment_cfg.use_brightness_contrast:
        train_augmentations_list.append(
            RandomBrightnessContrast(p=augment_cfg.brightness_contrast_p)
        )
    if augment_cfg.use_horizontal_flip:
        train_augmentations_list.append(
            RandomHorizontalFlipBoundingBoxes(
                p=augment_cfg.horizontal_flip_p,
                image_width=dataset_cfg.image_resize_size[1],
            )
        )

    # --- Albumentations Augmentations ---
    if cfg.augmentations.default.use_albumentations:
        albumentations_transforms = []
        if cfg.augmentations.default.use_rotate:
            albumentations_transforms.append(
                A.Rotate(
                    limit=cfg.augmentations.default.rotate_limit,
                    p=cfg.augmentations.default.rotate_p,
                )
            )
        if cfg.augmentations.default.use_blur:
            albumentations_transforms.append(
                A.GaussianBlur(
                    blur_limit=(3, 7), sigma_limit=0, p=cfg.augmentations.default.blur_p
                )
            )

        if albumentations_transforms:
            albumentations_pipeline = A.Compose(
                albumentations_transforms,
                bbox_params=A.BboxParams(
                    format="pascal_voc", label_fields=["class_labels"]
                ),  # 'pascal_voc' for [x_min, y_min, x_max, y_max]
            )
            train_augmentations_list.append(
                AlbumentationsWrapper(albumentations_pipeline)
            )

    train_augmentations = (
        transforms.Compose(train_augmentations_list)
        if train_augmentations_list
        else None
    )
    val_augmentations = None

    # --- Create Datasets ---
    try:
        train_dataset = DroneObjectDetectionDataset(
            data_root=os.path.join(dataset_cfg.data_root_dir, "train_images"),
            annotation_file=os.path.join(
                dataset_cfg.data_root_dir,
                "train_annotations",
                dataset_cfg.annotation_file_path,
            ),
            image_size=tuple(dataset_cfg.image_resize_size),  # Convert list to tuple
            transform=train_augmentations,
            annotation_format=dataset_cfg.annotation_format_type,
            classes=dataset_cfg.class_names,
            use_cache=dataset_cfg.use_dataset_cache,
            cache_backend=dataset_cfg.cache_backend_type,
            cache_dir=dataset_cfg.cache_directory,
        )

        val_dataset = DroneObjectDetectionDataset(
            data_root=os.path.join(dataset_cfg.data_root_dir, "val_images"),
            annotation_file=os.path.join(
                dataset_cfg.data_root_dir,
                "val_annotations",
                dataset_cfg.annotation_file_path,
            ),
            image_size=tuple(dataset_cfg.image_resize_size),  # Convert list to tuple
            transform=val_augmentations,
            annotation_format=dataset_cfg.annotation_format_type,
            classes=dataset_cfg.class_names,
            use_cache=dataset_cfg.use_dataset_cache,
            cache_backend=dataset_cfg.cache_backend_type,
            cache_dir=dataset_cfg.cache_directory,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        logging.error(f"Dataset initialization failed: {e}")
        exit(1)

    # --- Create DataLoaders ---
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=dataloader_cfg.batch_size_train,
        shuffle=dataloader_cfg.shuffle_train,
        num_workers=dataloader_cfg.num_workers_dataloader,
        collate_fn=custom_collate_fn,
        pin_memory=dataloader_cfg.pin_memory,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=dataloader_cfg.batch_size_val,
        shuffle=dataloader_cfg.shuffle_val,
        num_workers=dataloader_cfg.num_workers_dataloader,
        collate_fn=custom_collate_fn,
        pin_memory=dataloader_cfg.pin_memory,
    )

    # --- Performance Profiling ---
    profile = cProfile.Profile()
    profile.enable()

    # --- Example Iteration ---
    for batch_idx, (images, targets) in enumerate(train_dataloader):
        print(
            f"Batch {batch_idx+1} - Image batch shape: {images.shape}, Number of targets: {len(targets)}"
        )
        if targets:
            first_image_boxes = targets[0]["boxes"]
            first_image_labels = targets[0]["labels"]
            print(
                f"  First image - Boxes shape: {first_image_boxes.shape}, Labels shape: {first_image_labels.shape}"
            )
        else:
            print("  No targets in this batch.")

        if batch_idx > 9:  # Break after 10 batches
            break

    profile.disable()
    stats = pstats.Stats(profile)
    stats.sort_stats(pstats.SortKey.TIME)  # Sort by time
    stats.print_stats(20)  # Print top 20 function calls by time
    print("Profiling finished. See stats above.")
    # Save stats (robust)
    curr_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    stats.dump_stats(f"example_dataloader_profiling_stats_{curr_date}.prof")

    print("DataLoader example finished.")


if __name__ == "__main__":
    main()
