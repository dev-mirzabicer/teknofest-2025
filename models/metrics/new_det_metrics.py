import torch
from models.metrics.base_metric import BaseMetric
from utils.logger import get_logger


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    Computes IoU for a pair of boxes.
    Both boxes are tensors of shape (4,) in [x_min, y_min, x_max, y_max] (normalized) format.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box1[1])
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


class DetectionIoUMetric(BaseMetric):
    """
    Computes the average IoU between predicted boxes and ground truth boxes for grid cells
    that are responsible for objects. Uses the same decoding mechanism as the model/trainer.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.grid_size = config.get("grid_size", 7)
        # Accumulators for IoU and count
        self.total_iou = 0.0
        self.count = 0

    def reset(self) -> None:
        self.total_iou = 0.0
        self.count = 0

    def update(self, model_output: dict, batch: dict) -> None:
        logger = get_logger(self.__class__.__name__)  # Get logger instance

        bbox_preds = model_output["bbox_preds"]  # (B, G, G, 4)
        objectness = model_output[
            "objectness"
        ]  # (B, G, G, 1) - not used here but could be thresholded if needed
        B, G, _, _ = bbox_preds.shape
        device = bbox_preds.device
        cell_size = 1.0 / self.grid_size

        # Prepare grid top-left corners instead of centers
        xs = (torch.arange(G, device=device).float()) * cell_size  # removed + 0.5
        ys = (torch.arange(G, device=device).float()) * cell_size  # removed + 0.5
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")  # (G, G)
        grid_x = grid_x.unsqueeze(0).unsqueeze(-1)  # (1, G, G, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(-1)  # (1, G, G, 1)

        targets_list = batch["targets"]  # List of dicts; length B

        for b in range(B):
            target_dict = targets_list[b]
            boxes = target_dict[
                "boxes"
            ]  # (N, 4) in [x_min, y_min, x_max, y_max] normalized
            if boxes.numel() == 0:
                continue

            # Compute centers of ground truth boxes.
            centers = (boxes[:, :2] + boxes[:, 2:4]) / 2.0  # (N, 2)
            # Determine grid cell indices
            cell_cols = (centers[:, 0] * G).long()
            cell_rows = (centers[:, 1] * G).long()
            cell_cols = torch.clamp(cell_cols, 0, G - 1)
            cell_rows = torch.clamp(cell_rows, 0, G - 1)

            # Decode the predicted box from the corresponding grid cell.
            # Get the predicted bbox parameters for image b.
            preds = bbox_preds[b]  # (G, G, 4)
            for n in range(boxes.shape[0]):
                c = cell_cols[n].item()
                r = cell_rows[n].item()
                # Predicted parameters at grid cell (r, c)
                pred_params = preds[r, c]  # (4,) -> [tx, ty, tw, th]

                # Log grid_x[0, r, c, 0].item() and grid_y[0, r, c, 0].item()
                grid_x_val = grid_x[0, r, c, 0].item()
                grid_y_val = grid_y[0, r, c, 0].item()
                logger.debug(
                    f"  Grid cell (r={r}, c={c}) top-left corner x: {grid_x_val:.4f}, y: {grid_y_val:.4f}"
                )

                # Decode the center offset:
                # The network predicts offsets (tx, ty) ∈ [0,1] (after sigmoid in model) so:
                pred_center_x = grid_x[0, r, c, 0] + pred_params[0] * cell_size
                pred_center_y = grid_y[0, r, c, 0] + pred_params[1] * cell_size
                # Width and height (predicted as exp(tw) and exp(th)) are in grid-scale.
                # We assume the target encoding in the loss: width = (x_max - x_min) * G, so to get normalized width:
                pred_w = pred_params[2] / G
                pred_h = pred_params[3] / G

                # Convert from center representation to [x_min, y_min, x_max, y_max]
                pred_box = torch.tensor(
                    [
                        pred_center_x - pred_w / 2,
                        pred_center_y - pred_h / 2,
                        pred_center_x + pred_w / 2,
                        pred_center_y + pred_h / 2,
                    ],
                    device=device,
                )
                # Clamp to [0, 1]
                pred_box = torch.clamp(pred_box, 0.0, 1.0)
                gt_box = boxes[n]  # Already in normalized coordinates

                iou = compute_iou(pred_box, gt_box)

                # Log detailed info for the first GT box in the first image (keep as is)
                if b == 0 and n == 0:
                    logger.debug(f"  First GT box: {gt_box}")
                    logger.debug(f"  Cell cols/rows for first GT box: col={c}, row={r}")
                    logger.debug(f"  Predicted params [tx, ty, tw, th]: {pred_params}")
                    logger.debug(
                        f"  Decoded pred_center_x, pred_center_y: ({pred_center_x.item():.4f}, {pred_center_y.item():.4f})"
                    )
                    logger.debug(
                        f"  Decoded pred_w, pred_h: ({pred_w.item():.4f}, {pred_h.item():.4f})"
                    )
                    logger.debug(f"  Decoded pred_box: {pred_box}")
                    logger.debug(f"  Calculated IoU: {iou:.4f}")

                self.total_iou += iou
                self.count += 1

    def compute(self) -> dict:
        """
        Returns:
            dict: A dictionary with the average IoU over the updates.
        """
        avg_iou = self.total_iou / self.count if self.count > 0 else 0.0
        return {"avg_iou": avg_iou}

    @classmethod
    def from_config(cls, config: dict) -> "DetectionIoUMetric":
        return cls(config)
