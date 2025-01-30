import contextlib
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
import torch.profiler

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
    profiling_cfg = cfg.profiling.default

    if profiling_cfg.enabled:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_output_path = profiling_cfg.on_trace_ready_output_path.format(
            timestamp=timestamp
        )
        os.makedirs(os.path.dirname(trace_output_path), exist_ok=True)

        def trace_handler(prof):  # Define trace handler function
            prof.export_chrome_trace(trace_output_path)
            print(f"Profiling trace saved to: {trace_output_path}")

        schedule_fn = torch.profiler.schedule(
            wait=profiling_cfg.schedule_wait,
            warmup=profiling_cfg.schedule_warmup,
            active=profiling_cfg.schedule_active,
            repeat=profiling_cfg.schedule_repeat,
        )

        activities_list = []
        if "cpu" in profiling_cfg.activities:
            activities_list.append(torch.profiler.ProfilerActivity.CPU)
        if "cuda" in profiling_cfg.activities:
            activities_list.append(torch.profiler.ProfilerActivity.CUDA)

        profiler = torch.profiler.profile(
            activities=activities_list,
            schedule=schedule_fn,
            on_trace_ready=trace_handler,
            record_shapes=profiling_cfg.record_shapes,
            profile_memory=profiling_cfg.profile_memory,
            with_stack=profiling_cfg.with_stack,
        )
    else:
        profiler = None  # No profiler if not enabled

    # --- Example Iteration ---
    with profiler if profiler else contextlib.nullcontext():
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

            if (
                profiling_cfg.enabled
                and batch_idx
                >= (
                    profiling_cfg.schedule_wait
                    + profiling_cfg.schedule_warmup
                    + profiling_cfg.schedule_active
                )
                * profiling_cfg.schedule_repeat
                - 1
            ):  # Stop after profiling schedule
                break
            elif (
                not profiling_cfg.enabled and batch_idx > 9
            ):  # Stop after 10 batches if not profiling (for example run)
                break

    print("DataLoader example finished.")


if __name__ == "__main__":
    main()
