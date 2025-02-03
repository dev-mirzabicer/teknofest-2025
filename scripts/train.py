#!/usr/bin/env python3
"""
scripts/train.py

A robust training script for our advanced drone object detection framework.
This script ties together our dataset loaders, model factory, loss & optimizer factories,
metrics, trainer, logging, and profiling to run a complete training loop.
It uses Hydra (via omegaconf) for configuration management.
"""

import os
import logging
import warnings
import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

# Import our custom utilities and factories
from utils.logger import get_logger
from utils.collate_fn import detection_collate_fn
from models.model_factory import create_model
from models.losses.loss_factory import create_loss
from models.optimizers.optimizer_factory import create_optimizer
from models.optimizers.scheduler_factory import create_scheduler
from models.metrics.metric_factory import create_metrics_list
from trainers.trainer_factory import create_trainer
from data.datasets.drone_object_detection import DroneObjectDetectionDataset
from profiling.profiler_manager import ProfilerManager

# Suppress any unwanted warnings for a cleaner output.
warnings.filterwarnings("ignore")


def setup_device(cfg: DictConfig, logger: logging.Logger) -> torch.device:
    """
    Determines and logs the computation device.
    """
    use_cuda = cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    logger.info(f"Using device: {device}")
    return device


def setup_seed(cfg: DictConfig, logger: logging.Logger):
    """
    Sets the random seed for reproducibility, if specified.
    """
    seed = cfg.get("seed", None)
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seed set to: {seed}")


def create_dataloaders(cfg: DictConfig, logger: logging.Logger):
    """
    Creates training and validation DataLoader instances from the DroneObjectDetectionDataset.
    """
    # Instantiate training dataset
    train_dataset = DroneObjectDetectionDataset(
        data_root=cfg.dataset.train.data_root,
        annotation_file=cfg.dataset.train.annotation_file,
        image_size=tuple(cfg.dataset.train.image_size),
        transform=None,  # Insert augmentation/transform pipelines if needed.
        target_transform=None,
        annotation_format=cfg.dataset.train.annotation_format,
        classes=cfg.dataset.train.classes,
        use_cache=cfg.dataset.train.use_cache,
        cache_backend=cfg.dataset.train.cache_backend,
        cache_dir=cfg.dataset.train.cache_dir,
    )
    logger.info(f"Training dataset initialized with {len(train_dataset)} samples.")

    # Instantiate validation dataset
    val_dataset = DroneObjectDetectionDataset(
        data_root=cfg.dataset.val.data_root,
        annotation_file=cfg.dataset.val.annotation_file,
        image_size=tuple(cfg.dataset.val.image_size),
        transform=None,
        target_transform=None,
        annotation_format=cfg.dataset.val.annotation_format,
        classes=cfg.dataset.val.classes,
        use_cache=cfg.dataset.val.use_cache,
        cache_backend=cfg.dataset.val.cache_backend,
        cache_dir=cfg.dataset.val.cache_dir,
    )
    logger.info(f"Validation dataset initialized with {len(val_dataset)} samples.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        collate_fn=detection_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        collate_fn=detection_collate_fn,
    )
    return train_loader, val_loader


def main(cfg: DictConfig):
    # Initialize our central logger
    logger = get_logger("train.py", level=logging.INFO)
    logger.info("=== Starting Training Script ===")
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Determine the device and set seeds
    device = setup_device(cfg, logger)
    setup_seed(cfg, logger)

    # Create DataLoaders for training and validation
    train_loader, val_loader = create_dataloaders(cfg, logger)

    # Instantiate model, loss, optimizer, and metrics from our factories
    model = create_model(cfg.model)
    logger.info(f"Model '{cfg.model.name}' created.")

    loss_fn = create_loss(cfg.loss)
    logger.info(f"Loss function '{cfg.loss.name}' created.")

    optimizer = create_optimizer(model.parameters(), cfg.optimizer)
    logger.info(
        f"Optimizer '{cfg.optimizer.name}' created with parameters: {cfg.optimizer.params}"
    )

    metrics = create_metrics_list(cfg.get("metrics", []))
    if metrics:
        logger.info(
            f"Metrics created: {[metric.__class__.__name__ for metric in metrics]}"
        )
    else:
        logger.info("No metrics configured.")

    # Optionally create a learning rate scheduler
    scheduler = None
    if "scheduler" in cfg:
        scheduler = create_scheduler(optimizer, cfg.lr_scheduler)
        logger.info(
            f"Scheduler '{cfg.lr_scheduler.name}' created with parameters: {cfg.scheduler}"
        )

    if cfg.model.get("pretrained_path", False):
        # Load pretrained model state from a file
        pretrained_path = cfg.model.pretrained_path
        if os.path.isfile(pretrained_path):
            logger.info(f"Loading model weights from: {pretrained_path}")
            model.load_state_dict(
                torch.load(pretrained_path, map_location=device),
                strict=cfg.model.get("pretrained_strict", True),
            )
        else:
            logger.error(f"Pretrained model file not found at: {pretrained_path}")
            raise FileNotFoundError(
                f"Pretrained model file not found at: {pretrained_path}"
            )

    # Create the trainer instance via our trainer factory
    trainer = create_trainer(
        config=cfg.trainer,
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        metrics=metrics,
        scheduler=scheduler,
    )
    logger.info(f"Trainer '{cfg.trainer.name}' created.")

    # Initialize our profiler if enabled
    profiler = None
    if cfg.get("profiling", {}).get("enabled", False):
        tb_writer = None
        if cfg.get("tensorboard", {}).get("enabled", False):
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter(log_dir=cfg.tensorboard.get("log_dir", "runs"))
        profiler = ProfilerManager(
            cfg.profiler, tensorboard_writer=tb_writer, logger=logger
        )
        logger.info("Profiler enabled. Starting profiler context.")
        profiler.__enter__()

    # Run the training loop inside a try/except/finally for robust error handling
    try:
        trainer.train(cfg.trainer.num_epochs)
    except Exception as e:
        logger.error("An error occurred during training.", exc_info=True)
        raise
    finally:
        if profiler is not None:
            profiler.__exit__(None, None, None)
            logger.info("Profiler context exited.")

    logger.info("=== Training Completed Successfully ===")


# Use Hydra to manage configuration (assuming configs are in ../conf)
@hydra.main(config_path="../conf", config_name="config")
def run(cfg: DictConfig):
    main(cfg)


if __name__ == "__main__":
    run()
