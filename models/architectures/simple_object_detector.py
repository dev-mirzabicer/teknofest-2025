import torch
import torch.nn as nn
from models.base_model import BaseModel
from typing import Dict, Any
from utils.validators import check_tensor_shape, check_type
from utils.logger import get_logger


class SimpleObjectDetector(BaseModel):
    """
    A simple object detector that maintains a small parameter count but follows modern design principles.

    It uses:
    - A shared CNN backbone.
    - A bounding box head (predicts 4 bbox coordinates and an objectness score).
    - A classification head (predicts class scores).

    Outputs a dictionary with keys:
    - "bbox_preds": shape (B, grid_size, grid_size, 4)
    - "objectness": shape (B, grid_size, grid_size, 1)
    - "logits": shape (B, grid_size, grid_size, num_classes)
    """

    def __init__(self, config: Dict[str, Any]):
        super(SimpleObjectDetector, self).__init__(config)
        input_channels = config.get("input_channels", 3)
        num_classes = config.get("num_classes", 2)
        grid_size = config.get("grid_size", 7)
        self.num_classes = num_classes
        self.grid_size = grid_size

        # Shared Backbone: small CNN producing a feature map of shape (B, 128, grid_size, grid_size)
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((grid_size, grid_size)),
        )

        # BBox Head: outputs 5 channels (4 for bbox + 1 for objectness)
        self.bbox_head = nn.Conv2d(128, 5, kernel_size=1)

        # Classification Head: outputs num_classes channels
        self.class_head = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        logger = get_logger(self.__class__.__name__)  # Get logger instance
        images = batch.get("images")
        if images is None:
            raise ValueError("Input batch must contain key 'images'")

        logger.info(
            f"Forward pass - Input image shape: {images.shape}"
        )  # Log input image shape

        # Type checking
        check_type(images, torch.Tensor, "images")

        features = self.backbone(images)  # (B, 128, grid_size, grid_size)
        bbox_out = self.bbox_head(features)  # (B, 5, grid_size, grid_size)
        class_out = self.class_head(features)  # (B, num_classes, grid_size, grid_size)

        # Separate bbox predictions and objectness.
        bbox_preds = bbox_out[:, :4, :, :]  # (B, 4, grid_size, grid_size)
        objectness = bbox_out[:, 4:5, :, :]  # (B, 1, grid_size, grid_size)
        logits = class_out[:, :, :, :]  # (B, num_classes, grid_size, grid_size)
        logger.debug(
            f"  Raw bbox_preds shape: {bbox_preds.shape}"
        )  # Log raw bbox_preds shape
        logger.debug(
            f"  Raw objectness shape: {objectness.shape}"
        )  # Log raw objectness shape
        logger.debug(f"  Raw logits shape: {logits.shape}")  # Log raw logits shape

        # Sigmoid for offsets (center_x, center_y)
        bbox_preds[:, :2, :, :] = torch.sigmoid(bbox_preds[:, :2, :, :])
        # Exp for width and height to ensure positivity
        bbox_preds[:, 2:4, :, :] = torch.exp(bbox_preds[:, 2:4, :, :])

        # Permute outputs to shape (B, grid_size, grid_size, channels)
        bbox_preds = bbox_preds.permute(0, 2, 3, 1).contiguous()
        objectness = objectness.permute(0, 2, 3, 1).contiguous()
        logits = class_out.permute(0, 2, 3, 1).contiguous()

        logger.debug(
            f"  Processed bbox_preds shape (after sigmoid/exp and permute): {bbox_preds.shape}"
        )  # Log processed bbox_preds shape
        logger.debug(
            f"  Processed objectness shape (after permute): {objectness.shape}"
        )  # Log processed objectness shape
        logger.debug(
            f"  Processed logits shape (after permute): {logits.shape}"
        )  # Log processed logits shape

        return {"bbox_preds": bbox_preds, "objectness": objectness, "logits": logits}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SimpleObjectDetector":
        return cls(config)
