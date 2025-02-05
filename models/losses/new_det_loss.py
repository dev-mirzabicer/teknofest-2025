import torch
import torch.nn as nn
import torch.nn.functional as F
from models.losses.base_loss import BaseLoss
from utils.logger import get_logger


class DetectionLoss(BaseLoss):
    """
    A robust loss function for object detection tasks that carefully accounts for:
      - Localization (bounding-box regression) via Smooth L1 loss.
      - Objectness (confidence) via BCE with logits, with separate weighting for object vs. no-object cells.
      - Classification via Cross Entropy loss.

    Format:
      - Model outputs:
          * bbox_preds: (B, G, G, 4) with [tx, ty, tw, th], where tx,ty ∈ [0,1] (after sigmoid) and tw,th > 0 (after exp).
          * objectness: (B, G, G, 1) raw logits.
          * logits: (B, G, G, num_classes) for classification.
      - Ground-truth targets (per image) are provided as a dict with:
          * "boxes": tensor (N,4) with [x_min, y_min, x_max, y_max] in normalized [0,1] coordinates.
          * "labels": tensor (N,) with class indices.
      - Each ground-truth box is assigned to a grid cell based on its center.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        # Get grid size from config (should match model config)
        self.grid_size = config.get("grid_size", 7)
        # Loss weights
        self.bbox_loss_weight = config.get("bbox_loss_weight", 5.0)
        self.obj_loss_weight = config.get("obj_loss_weight", 1.0)
        self.noobj_loss_weight = config.get("noobj_loss_weight", 0.5)
        self.class_loss_weight = config.get("class_loss_weight", 1.0)
        # Standard loss functions with summing (we’ll average later)
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction="sum")
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="sum")
        self.ce_loss = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, model_output: dict, batch: dict) -> torch.Tensor:
        """
        Computes the total detection loss.

        Args:
          model_output (dict): Contains:
             - "bbox_preds": Tensor of shape (B, G, G, 4)
             - "objectness": Tensor of shape (B, G, G, 1)
             - "logits": Tensor of shape (B, G, G, num_classes)
          batch (dict): Contains the input batch with key "targets".
             - "targets" is a list (length B) of dicts with keys "boxes" and "labels".

        Returns:
          torch.Tensor: The total scalar loss.
        """
        logger = get_logger(self.__class__.__name__)  # Get logger instance
        # Unpack model outputs
        bbox_preds = model_output["bbox_preds"]  # (B, G, G, 4)
        obj_preds = model_output["objectness"]  # (B, G, G, 1), raw logits
        class_logits = model_output["logits"]  # (B, G, G, num_classes)
        targets_list = batch["targets"]

        logger.info("DetectionLoss forward pass started.")  # Log start of forward pass
        logger.debug(f"  bbox_preds shape: {bbox_preds.shape}")  # Log bbox_preds shape
        logger.debug(f"  obj_preds shape: {obj_preds.shape}")  # Log obj_preds shape
        logger.debug(
            f"  class_logits shape: {class_logits.shape}"
        )  # Log class_logits shape
        logger.debug(
            f"  Ground truth targets (first in batch): {targets_list[0]['boxes'] if targets_list else 'No targets'}"
        )  # Log GT boxes

        B, G, _, _ = bbox_preds.shape
        device = bbox_preds.device

        # Initialize target tensors for each image and grid cell.
        # target_objectness: 1 if an object is assigned to the cell, else 0.
        target_objectness = torch.zeros((B, G, G, 1), device=device)
        # target_bbox: holds [tx, ty, tw, th] for grid cells with objects.
        target_bbox = torch.zeros((B, G, G, 4), device=device)
        # target_class: for classification; use -100 (ignore index) where no object is present.
        target_class = torch.full((B, G, G), -100, dtype=torch.long, device=device)

        # Process each image in the batch.
        # Note: batch["targets"] is expected to be a list (length B) of dicts.
        for b in range(B):
            target_dict = targets_list[b]
            # "boxes": (N, 4) with [x_min, y_min, x_max, y_max] (normalized coordinates)
            # "labels": (N,) with class indices.
            boxes = target_dict["boxes"]
            labels = target_dict["labels"]

            if boxes.numel() == 0:
                continue

            # Compute center coordinates for each box.
            centers = (
                boxes[:, :2] + boxes[:, 2:4]
            ) / 2.0  # (N, 2): centers[:,0]=x, centers[:,1]=y

            # Determine grid cell indices.
            # For a grid of size G, cell indices are: row = int(y_center * G), col = int(x_center * G).
            cell_cols = (centers[:, 0] * G).long()  # (N,)
            cell_rows = (centers[:, 1] * G).long()  # (N,)
            cell_cols = torch.clamp(cell_cols, 0, G - 1)
            cell_rows = torch.clamp(cell_rows, 0, G - 1)

            # Compute offsets within the cell:
            # For a cell at (r, c), the cell’s top-left in normalized coordinates is (c/G, r/G).
            # Thus, target offset = center * G - [cell_col, cell_row] ∈ [0,1].
            offsets = centers * G - torch.stack(
                [cell_cols.float(), cell_rows.float()], dim=1
            )  # (N, 2)

            # Compute width and height in grid-cell scale.
            wh = (boxes[:, 2:4] - boxes[:, :2]) * G  # (N, 2)

            # Form the target bounding-box parameters: [tx, ty, tw, th]
            target_boxes = torch.cat([offsets, wh], dim=1)  # (N, 4)

            if b == 0 and boxes.numel() > 0:
                logger.debug(f"  First GT box (image 0): {boxes[0]}")
                logger.debug(f"  Center of first GT box: {centers[0]}")
                logger.debug(
                    f"  Cell cols/rows for first GT box: col={cell_cols[0]}, row={cell_rows[0]}"
                )
                logger.debug(f"  Offsets (tx, ty) for first GT box: {offsets[0]}")
                logger.debug(f"  WH (tw, th) for first GT box: {wh[0]}")
                logger.debug(
                    f"  Target boxes [tx, ty, tw, th] for first GT box: {target_boxes[0]}"
                )

            # Assign each ground-truth box to its corresponding grid cell.
            for n in range(boxes.shape[0]):
                r = cell_rows[n]  # row index (from y)
                c = cell_cols[n]  # column index (from x)
                # (Optional: if a cell already has an object, one could use an IoU criterion to decide which box to keep.)
                target_objectness[b, r, c, 0] = 1.0
                target_bbox[b, r, c, :] = target_boxes[n]
                target_class[b, r, c] = labels[n]

        logger.debug(
            f"  target_objectness shape: {target_objectness.shape}, sample (first cell): {target_objectness[0,0,0]}"
        )  # Log target_objectness
        logger.debug(
            f"  target_bbox shape: {target_bbox.shape}, sample (first cell): {target_bbox[0,0,0]}"
        )  # Log target_bbox
        logger.debug(
            f"  target_class shape: {target_class.shape}, sample (first cell): {target_class[0,0]}"
        )  # Log target_class

        # Create a mask of grid cells with objects.
        obj_mask = target_objectness.squeeze(-1) == 1  # (B, G, G)
        num_obj = obj_mask.sum().float()
        if num_obj < 1:
            num_obj = 1.0  # avoid division by zero

        # -------------------------
        # 1. Localization (BBox Regression) Loss
        # Compute only for grid cells with objects.
        loc_loss = self.smooth_l1_loss(bbox_preds[obj_mask], target_bbox[obj_mask])
        loc_loss = loc_loss / num_obj  # average per object cell

        # -------------------------
        # 2. Objectness (Confidence) Loss
        # Separate the loss for cells with objects and without.
        obj_loss_obj = self.bce_loss(obj_preds[obj_mask], target_objectness[obj_mask])
        obj_loss_noobj = self.bce_loss(
            obj_preds[~obj_mask], target_objectness[~obj_mask]
        )
        total_cells = B * G * G
        obj_loss = (
            self.obj_loss_weight * obj_loss_obj
            + self.noobj_loss_weight * obj_loss_noobj
        ) / total_cells

        # -------------------------
        # 3. Classification Loss
        # Compute only on grid cells that contain an object.
        class_logits_obj = class_logits[obj_mask]  # (num_obj, num_classes)
        target_class_obj = target_class[obj_mask]  # (num_obj,)
        if class_logits_obj.numel() > 0:
            class_loss = self.ce_loss(class_logits_obj, target_class_obj) / num_obj
        else:
            class_loss = 0.0

        # -------------------------
        # Total Loss (weighted sum)
        total_loss = (
            self.bbox_loss_weight * loc_loss
            + obj_loss
            + self.class_loss_weight * class_loss
        )

        logger.debug(f"  Localization Loss: {loc_loss.item():.4f}")  # Log loc_loss
        logger.debug(f"  Objectness Loss: {obj_loss.item():.4f}")  # Log obj_loss
        logger.debug(f"  Classification Loss: {class_loss:.4f}")  # Log class_loss
        logger.info(f"  Total Loss: {total_loss.item():.4f}")  # Log total_loss

        return total_loss

    @classmethod
    def from_config(cls, config: dict) -> "DetectionLoss":
        return cls(config)
