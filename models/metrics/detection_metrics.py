from typing import Dict, Any, List
import torch
import torch.nn.functional as F
from models.metrics.base_metric import BaseMetric
from utils.validators import (
    check_tensor_range,
    check_type,
    check_tensor_shape,
    VALIDATION_ENABLED,
)
from utils.logger import get_logger
from utils.tensorboard_logger import TensorBoardLogger
import numpy as np


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) for two boxes.
    Boxes are in format (x_min, y_min, x_max, y_max).
    """
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = max(0, float(box1[2]) - float(box1[0])) * max(
        0, float(box1[3]) - float(box1[1])
    )
    box2_area = max(0, float(box2[2]) - float(box2[0])) * max(
        0, float(box2[3]) - float(box2[1])
    )

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def nms(boxes, scores, iou_threshold):
    """
    Perform Non-Maximum Suppression (NMS).

    Args:
        boxes (Tensor): shape (N, 4) in (x_min, y_min, x_max, y_max).
        scores (Tensor): shape (N,)
        iou_threshold (float): threshold for IoU.

    Returns:
        List[int]: indices of boxes to keep.
    """
    if boxes.numel() == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        rest = order[1:]

        xx1 = torch.max(x1[i], x1[rest])
        yy1 = torch.max(y1[i], y1[rest])
        xx2 = torch.min(x2[i], x2[rest])
        yy2 = torch.min(y2[i], y2[rest])

        inter = torch.clamp(xx2 - xx1, min=0) * torch.clamp(yy2 - yy1, min=0)
        rem_areas = areas[rest]
        iou = inter / (areas[i] + rem_areas - inter)

        order = order[1:][iou <= iou_threshold]

    return keep


class DetectionMetrics(BaseMetric):
    """
    Metric for object detection that computes:
        - Average IoU over matched predictions.
        - mean Average Precision (mAP) using an IoU threshold.

    This metric decodes grid-based outputs into bounding boxes in normalized coordinates,
    applies confidence thresholding and NMS, then matches predictions to ground truths.
    """

    def __init__(self, config: Dict[str, Any]):
        super(DetectionMetrics, self).__init__(config)
        self.iou_threshold = config.get("iou_threshold", 0.5)
        self.conf_threshold = config.get("conf_threshold", 0.5)
        self.nms_iou_threshold = config.get("nms_iou_threshold", 0.5)
        self.num_classes = config.get("num_classes")
        if self.num_classes is None:
            raise ValueError("DetectionMetrics requires 'num_classes' in config")
        self.logger = get_logger(self.__class__.__name__)
        tb_config = config.get("tensorboard", {})
        self.use_tensorboard = tb_config.get("enabled", False)
        if self.use_tensorboard:
            from utils.tensorboard_logger import TensorBoardLogger

            self.tb_logger = TensorBoardLogger(
                log_dir=tb_config.get("log_dir", "runs"),
                experiment_name=self.experiment_name,
            )
        else:
            self.tb_logger = None
        self.reset()

    def reset(self) -> None:
        self.all_predictions = []  # List of dicts per image.
        self.all_ground_truths = []  # List of dicts per image.
        self.iou_list = []  # List of IoU values for true positives.

    def update(
        self, model_output: Dict[str, torch.Tensor], batch: Dict[str, Any]
    ) -> None:
        # --- Validate model_output structure ---
        for key in ["bbox_preds", "objectness", "logits"]:
            if key not in model_output:
                raise KeyError(
                    f"Model output must contain key '{key}'. Received: {list(model_output.keys())}"
                )

        bbox_preds = model_output["bbox_preds"]  # (B, G, G, 4)
        objectness = model_output["objectness"]  # (B, G, G, 1)
        logits = model_output["logits"]  # (B, G, G, num_classes)
        B, G, _, _ = bbox_preds.shape
        device = bbox_preds.device

        # Validate shapes.
        check_tensor_shape(bbox_preds, (B, G, G, 4), "bbox_preds")
        check_tensor_shape(objectness, (B, G, G, 1), "objectness")
        check_tensor_shape(logits, (B, G, G, self.num_classes), "logits")

        # Validate targets structure.
        if "targets" not in batch:
            raise KeyError("Batch must contain 'targets'.")
        targets = batch["targets"]
        if not isinstance(targets, list):
            raise TypeError("Batch 'targets' must be a list, one per image.")
        idx = 0
        t = targets[0]
        for req in ["boxes", "labels"]:
            if req not in t:
                raise KeyError(f"Target at index {idx} is missing key '{req}'.")
            check_type(t[req], torch.Tensor, f"targets[{idx}]['{req}']")
            if req == "boxes" and t[req].numel() > 0:
                check_tensor_shape(
                    t[req],
                    (None, 4),
                    f"targets[{idx}]['boxes']",
                    allow_dynamic=True,
                )
                check_tensor_range(t[req], 0.0, 1.0, f"targets[{idx}]['boxes']")

        # --- Prepare grid and decode predictions ---
        cell_size = 1.0 / G
        xs = (torch.arange(G, device=device).float() + 0.5) * cell_size
        ys = (torch.arange(G, device=device).float() + 0.5) * cell_size
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        # FIX: Unsqueeze an extra dimension to avoid unwanted broadcasting expansion.
        grid_x = grid_x.unsqueeze(0).unsqueeze(-1)  # Now shape: (1, G, G, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(-1)  # Now shape: (1, G, G, 1)

        # Apply activations.
        objectness_conf = torch.sigmoid(objectness)  # (B, G, G, 1)
        class_probs = torch.softmax(logits, dim=-1)  # (B, G, G, num_classes)
        max_class_prob, pred_class = torch.max(class_probs, dim=-1)  # (B, G, G)
        conf = objectness_conf.squeeze(-1) * max_class_prob  # (B, G, G)

        # Decode bounding boxes.
        pred_offset = bbox_preds[..., :2]  # (B, G, G, 2)
        pred_center_x = grid_x + pred_offset[..., 0:1] * cell_size
        pred_center_y = grid_y + pred_offset[..., 1:2] * cell_size
        pred_wh = bbox_preds[..., 2:4]  # (B, G, G, 2)
        pred_boxes = torch.cat(
            [
                pred_center_x - pred_wh[..., 0:1] / 2,
                pred_center_y - pred_wh[..., 1:2] / 2,
                pred_center_x + pred_wh[..., 0:1] / 2,
                pred_center_y + pred_wh[..., 1:2] / 2,
            ],
            dim=-1,
        )  # Now correctly shaped (B, G, G, 4)

        # Clamp decoded boxes to [0.0, 1.0]
        pred_boxes = torch.clamp(pred_boxes, 0.0, 1.0)

        # Log decoded boxes and target boxes for the first few batches of each epoch using TensorBoard
        if self.tb_logger and B > 0:  # Check if batch size is greater than 0
            tb_logger = self.tb_logger
            if isinstance(tb_logger, TensorBoardLogger):
                for i in range(min(B, 5)):  # Log up to 5 images from the batch
                    decoded_boxes_str = str(pred_boxes[i].detach().cpu().tolist())
                    tb_logger.log_text(
                        f"DetectionMetrics/Decoded_Boxes/Image_{i}",
                        decoded_boxes_str,
                        self.global_step,
                    )

                    target_boxes_str = str(targets[i]["boxes"].detach().cpu().tolist())
                    tb_logger.log_text(
                        f"DetectionMetrics/Target_Boxes/Image_{i}",
                        target_boxes_str,
                        self.global_step,
                    )

        # Validate decoded boxes.
        if pred_boxes.numel() > 0:
            try:
                check_tensor_range(pred_boxes, 0.0, 1.0, "Decoded Prediction Boxes")
            except Exception as e:
                self.logger.error(f"Decoded boxes validation error: {e}")

        # Process each image in the batch.
        for i in range(B):
            boxes = pred_boxes[i].reshape(-1, 4)
            scores = conf[i].reshape(-1)
            labels = pred_class[i].reshape(-1)

            # Filter low-confidence predictions.
            keep_inds = scores >= self.conf_threshold
            boxes = boxes[keep_inds]
            scores = scores[keep_inds]
            labels = labels[keep_inds]

            # Apply NMS per class.
            final_boxes, final_scores, final_labels = [], [], []
            for c in range(self.num_classes):
                inds = (labels == c).nonzero(as_tuple=False).squeeze(1)
                if inds.numel() == 0:
                    continue
                boxes_c = boxes[inds]
                scores_c = scores[inds]
                keep = nms(boxes_c, scores_c, self.nms_iou_threshold)
                if len(keep) > 0:
                    final_boxes.append(boxes_c[keep])
                    final_scores.append(scores_c[keep])
                    final_labels.append(torch.full((len(keep),), c, dtype=torch.int64))
            if final_boxes:
                final_boxes = torch.cat(final_boxes, dim=0)
                final_scores = torch.cat(final_scores, dim=0)
                final_labels = torch.cat(final_labels, dim=0)
            else:
                final_boxes = torch.empty((0, 4))
                final_scores = torch.empty((0,))
                final_labels = torch.empty((0,), dtype=torch.int64)

            self.all_predictions.append(
                {
                    "boxes": final_boxes.cpu(),
                    "scores": final_scores.cpu(),
                    "labels": final_labels.cpu(),
                }
            )

            # Process ground truth for the image.
            gt_boxes = targets[i]["boxes"]
            gt_labels = targets[i]["labels"]
            if gt_boxes.numel() > 0:
                try:
                    check_tensor_range(gt_boxes, 0.0, 1.0, "Ground Truth Boxes")
                except Exception as e:
                    self.logger.error(f"Ground truth boxes validation error: {e}")
            self.all_ground_truths.append(
                {"boxes": gt_boxes.cpu(), "labels": gt_labels.cpu()}
            )

    def compute(self) -> Dict[str, float]:
        """
        Computes the average IoU over true positives and mAP (with 11-point interpolation).
        Returns a dict with "avg_iou", "mAP", and "AP_per_class" (dict mapping class to AP).
        """
        per_class_preds = {c: [] for c in range(self.num_classes)}
        per_class_gt_count = {c: 0 for c in range(self.num_classes)}
        iou_list = []

        # Process predictions and ground truths image-by-image.
        for pred, gt in zip(self.all_predictions, self.all_ground_truths):
            gt_boxes = gt["boxes"]
            gt_labels = gt["labels"]
            detected = [False] * (gt_boxes.shape[0] if gt_boxes.dim() > 0 else 0)
            for idx in range(pred["boxes"].shape[0]):
                pred_box = pred["boxes"][idx]
                pred_score = pred["scores"][idx].item()
                pred_label = int(pred["labels"][idx].item())

                gt_inds = (gt_labels == pred_label).nonzero(as_tuple=False).squeeze(1)
                best_iou = 0.0
                best_iou_idx = -1
                for j in gt_inds:
                    iou_val = compute_iou(pred_box, gt_boxes[j])
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_iou_idx = j.item() if isinstance(j, torch.Tensor) else j

                if (
                    best_iou >= self.iou_threshold
                    and best_iou_idx >= 0
                    and not detected[best_iou_idx]
                ):
                    tp = 1
                    detected[best_iou_idx] = True
                    iou_list.append(best_iou)
                else:
                    tp = 0
                per_class_preds[pred_label].append((pred_score, tp))

            for c in range(self.num_classes):
                per_class_gt_count[c] += int((gt_labels == c).sum().item())

        avg_iou = float(np.mean(iou_list)) if iou_list else 0.0

        # Compute AP per class using 11-point interpolation.
        ap_per_class = {}
        for c in range(self.num_classes):
            preds = per_class_preds[c]
            if len(preds) == 0:
                ap_per_class[c] = 0.0
                continue

            preds.sort(key=lambda x: x[0], reverse=True)
            tp_list = [p[1] for p in preds]

            cum_tp = 0
            cum_fp = 0
            precisions = []
            recalls = []
            for i, tp in enumerate(tp_list):
                if tp == 1:
                    cum_tp += 1
                else:
                    cum_fp += 1
                precision = cum_tp / (cum_tp + cum_fp) if (cum_tp + cum_fp) > 0 else 0.0
                recall = (
                    cum_tp / per_class_gt_count[c] if per_class_gt_count[c] > 0 else 0.0
                )
                precisions.append(precision)
                recalls.append(recall)

            ap = 0.0
            for t in np.linspace(0, 1, 11):
                p = max([p for p, r in zip(precisions, recalls) if r >= t] + [0])
                ap += p
            ap /= 11.0
            ap_per_class[c] = ap

        mAP = (
            sum(ap_per_class.values()) / self.num_classes
            if self.num_classes > 0
            else 0.0
        )

        return {"avg_iou": avg_iou, "mAP": mAP, "AP_per_class": ap_per_class}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DetectionMetrics":
        return cls(config)
