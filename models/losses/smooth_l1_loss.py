import torch
import torch.nn.functional as F
from models.losses.base_loss import BaseLoss
from typing import Dict, Any


class SmoothL1Loss(BaseLoss):
    """
    Smooth L1 Loss for bounding box regression.

    This loss is less sensitive to outliers than the standard L2 loss.
    """

    def __init__(self, config: Dict[str, Any]):
        super(SmoothL1Loss, self).__init__(config)
        self.beta = config.get("beta", 1.0)
        self.reduction = config.get("reduction", "mean")

    def forward(
        self, model_output: Dict[str, torch.Tensor], batch: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Expects:
          - model_output["bbox_preds"]: Tensor of shape (B, 4) with predicted bounding boxes.
          - batch["targets"]["boxes"]: Tensor of shape (B, 4) with ground truth boxes.
        """
        bbox_preds = model_output.get("bbox_preds")
        targets = batch.get("targets")
        if bbox_preds is None or targets is None or "boxes" not in targets:
            raise ValueError(
                "SmoothL1Loss requires 'bbox_preds' in model_output and 'boxes' in batch['targets']"
            )

        gt_boxes = targets["boxes"]
        loss = F.smooth_l1_loss(
            bbox_preds, gt_boxes, beta=self.beta, reduction=self.reduction
        )
        return loss

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SmoothL1Loss":
        return cls(config)
