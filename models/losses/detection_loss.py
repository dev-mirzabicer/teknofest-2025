import torch
import torch.nn.functional as F
from models.losses.base_loss import BaseLoss
from typing import Dict, Any, List, Tuple
from utils.validators import check_tensor_shape, check_tensor_range, check_type
from utils.logger import get_logger
from utils.tensorboard_logger import TensorBoardLogger


class DetectionLoss(BaseLoss):
    """
    A combined detection loss for grid-based object detection.
    This loss builds fixed-grid targets from variable-sized ground-truth boxes and computes:
    1. A focal loss over classification logits.
    2. A Smooth L1 regression loss over bounding box predictions (only for positive grid cells).
    3. A binary cross entropy loss over objectness predictions.

    Ground-truth targets are constructed from a list of target dictionaries (one per image) into fixed
    grid targets (one per grid cell). It expects the following keys in model_output:
    - "bbox_preds": (B, G, G, 4) tensor with predicted bounding boxes (normalized coordinates)
    - "objectness": (B, G, G, 1) tensor with predicted objectness logits
    - "logits": (B, G, G, num_classes) tensor with classification logits

    Each batch element’s targets is expected to be a dict with:
    - "boxes": Tensor of shape (N, 4) in normalized coordinates [x_min, y_min, x_max, y_max]
    - "labels": Tensor of shape (N,) with class indices (assumed nonzero for objects; 0 for background)
    """

    def __init__(self, config: Dict[str, Any]):
        super(DetectionLoss, self).__init__(config)
        self.alpha = config.get("alpha", 0.25)
        self.gamma = config.get("gamma", 2.0)
        self.cls_loss_weight = config.get("cls_loss_weight", 1.0)
        self.box_loss_weight = config.get("box_loss_weight", 1.0)
        self.obj_loss_weight = config.get("obj_loss_weight", 1.0)
        self.grid_size = config.get("grid_size", 7)
        self.num_classes = config.get("num_classes")
        if self.num_classes is None:
            raise ValueError("DetectionLoss requires 'num_classes' in config")
        self.bg_class = config.get("bg_class", 0)
        self.logger = get_logger(self.__class__.__name__)

    def forward(
        self, model_output: Dict[str, torch.Tensor], batch: Dict[str, Any]
    ) -> torch.Tensor:
        # --- Validate model_output structure ---
        required_keys = ["logits", "bbox_preds", "objectness"]
        for key in required_keys:
            if key not in model_output:
                raise KeyError(
                    f"Model output must contain key '{key}'. Received keys: {list(model_output.keys())}"
                )

        logits = model_output["logits"]
        bbox_preds = model_output["bbox_preds"]
        obj_preds = model_output["objectness"]

        # --- Type and Shape Checks ---
        check_type(logits, torch.Tensor, "logits")
        check_type(bbox_preds, torch.Tensor, "bbox_preds")
        check_type(obj_preds, torch.Tensor, "objectness")

        B, G, _, _ = logits.shape
        expected_logits_shape = (B, self.grid_size, self.grid_size, self.num_classes)
        expected_bbox_shape = (B, self.grid_size, self.grid_size, 4)
        expected_obj_shape = (B, self.grid_size, self.grid_size, 1)
        check_tensor_shape(logits, expected_logits_shape, "logits")
        check_tensor_shape(bbox_preds, expected_bbox_shape, "bbox_preds")
        check_tensor_shape(obj_preds, expected_obj_shape, "objectness")

        # --- Validate batch targets ---
        if "targets" not in batch:
            raise KeyError("Batch must contain 'targets'.")
        targets = batch["targets"]
        if not isinstance(targets, list):
            raise TypeError(
                "Batch 'targets' should be a list of dictionaries, one per image."
            )
        t = targets[0]
        idx = 0
        if not isinstance(t, dict):
            raise TypeError(f"Target at index {idx} should be a dict.")
        for req in ["boxes", "labels"]:
            if req not in t:
                raise KeyError(f"Each target dict must contain key '{req}'.")
            # Check that boxes and labels are tensors.
            check_type(t[req], torch.Tensor, f"targets[{idx}]['{req}']")
            # For boxes, we expect shape (N,4)
            if req == "boxes" and t[req].numel() > 0:
                check_tensor_shape(
                    t[req],
                    (None, 4),
                    f"targets[{idx}]['boxes']",
                    allow_dynamic=True,
                )
                # Ensure coordinates are in [0,1] (normalized)
                check_tensor_range(t[req], 0.0, 1.0, f"targets[{idx}]['boxes']")

        # --- Build targets and validate predictions ---
        cls_target, bbox_target, obj_target = self.build_targets(
            targets, logits.shape, logits.device
        )
        # --- ADD THIS LINE FOR DEBUGGING ---
        self.bbox_target = bbox_target  # Store bbox_target temporarily for debugging

        # Log histograms of targets for debugging
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.log_histogram(
                "Targets/cls_target",
                cls_target.detach().cpu().numpy(),
                self.global_step,
            )
            self.logger.log_histogram(
                "Targets/bbox_target",
                bbox_target.detach().cpu().numpy(),
                self.global_step,
            )
            self.logger.log_histogram(
                "Targets/obj_target",
                obj_target.detach().cpu().numpy(),
                self.global_step,
            )

        # Compute classification loss with focal loss.
        cls_loss = self.focal_loss(logits, cls_target)

        # Box regression loss (only computed for positive grid cells)
        pos_mask = obj_target == 1  # (B, G, G)
        if pos_mask.sum() > 0:
            bbox_loss = F.smooth_l1_loss(
                bbox_preds[pos_mask], bbox_target[pos_mask], reduction="mean"
            )
            # Log bbox_loss for debugging
            if isinstance(self.logger, TensorBoardLogger):
                self.logger.log_scalar(
                    "Losses/bbox_loss", bbox_loss.detach().item(), self.global_step
                )
        else:
            bbox_loss = torch.tensor(0.0, device=logits.device)

        # Objectness loss.
        obj_loss = F.binary_cross_entropy_with_logits(
            obj_preds.squeeze(-1), obj_target.float(), reduction="mean"
        )

        # Log other loss components for debugging
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.log_scalar(
                "Losses/cls_loss", cls_loss.detach().item(), self.global_step
            )
            self.logger.log_scalar(
                "Losses/obj_loss", obj_loss.detach().item(), self.global_step
            )

        total_loss = (
            self.cls_loss_weight * cls_loss
            + self.box_loss_weight * bbox_loss
            + self.obj_loss_weight * obj_loss
        )

        # Ensure loss is a scalar.
        if total_loss.dim() != 0:
            raise ValueError("Total loss must be a scalar!")
        return total_loss

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, G, _, num_classes = logits.shape
        logits = logits.view(-1, num_classes)
        targets = targets.view(-1)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

    def build_targets(
        self,
        batch_targets: List[Dict[str, torch.Tensor]],
        logits_shape: Tuple,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Converts a list of raw targets into fixed-grid targets.
        ...
        """
        logger = self.logger
        B, G, _, _ = logits_shape
        cls_target = torch.zeros(
            (B, G, G), dtype=torch.long, device=device
        )  # background = 0
        bbox_target = torch.zeros((B, G, G, 4), dtype=torch.float, device=device)
        obj_target = torch.zeros((B, G, G), dtype=torch.uint8, device=device)

        cell_size = 1.0 / G
        xs = (torch.arange(G, device=device).float() + 0.5) * cell_size
        ys = (torch.arange(G, device=device).float() + 0.5) * cell_size
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")  # both shape: (G, G)

        # For each image in the batch, assign grid cells to each GT box.
        for b in range(B):
            targets = batch_targets[b]
            boxes = targets["boxes"].to(device)  # (N, 4)
            labels = targets["labels"].to(device)  # (N,)
            N = boxes.shape[0]

            # --- ADD LOGGING FOR INPUT BOXES ---
            logger.info(
                f"Image {b} - Input Boxes to build_targets: {boxes.cpu().numpy().tolist()}"
            )
            logger.info(
                f"Image {b} - Input Labels to build_targets: {labels.cpu().numpy().tolist()}"
            )
            # --- END LOGGING ---

            for n in range(N):
                box = boxes[n]  # [x_min, y_min, x_max, y_max]
                label = labels[n]

                # --- LOGGING FOR BOX COORDINATES ---
                logger.info(f"  Box {n} - Coordinates: {box.cpu().numpy().tolist()}")
                # --- END LOGGING ---

                # Create a boolean mask for grid cells whose center falls within the box.
                mask = (
                    (grid_x >= box[0])
                    & (grid_x <= box[2])
                    & (grid_y >= box[1])
                    & (grid_y <= box[3])
                )

                # --- LOGGING FOR MASK ---
                logger.info(f"    Mask shape: {mask.shape}, Mask sum: {mask.sum()}")
                # --- END LOGGING ---

                if mask.sum() == 0:
                    continue
                cls_target[b][mask] = label
                obj_target[b][mask] = 1

                # Encode the bounding box relative to the grid cell.
                box_center_x = (box[0] + box[2]) / 2
                box_center_y = (box[1] + box[3]) / 2
                box_w = box[2] - box[0]
                box_h = box[3] - box[1]

                # --- LOGGING FOR BOX DIMENSIONS ---
                logger.info(
                    f"    Box Center X: {box_center_x.item():.4f}, Center Y: {box_center_y.item():.4f}, Width: {box_w.item():.4f}, Height: {box_h.item():.4f}"
                )
                # --- END LOGGING ---

                # Flatten the spatial dimensions for proper advanced indexing.
                flat_bbox_target = bbox_target[b].reshape(-1, 4)  # shape (G*G, 4)
                flat_grid_x = grid_x.reshape(-1)  # shape (G*G,)
                flat_grid_y = grid_y.reshape(-1)  # shape (G*G,)
                flat_mask = mask.reshape(-1)  # shape (G*G,)

                # --- LOGGING FOR GRID AND FLAT MASK ---
                logger.info(f"    Flat Mask sum: {flat_mask.sum()}")
                if flat_mask.sum() > 0:  # Log only if there are positive cells
                    logger.info(
                        f"    flat_grid_x[flat_mask] (first few): {flat_grid_x[flat_mask][:5].cpu().numpy().tolist()}"
                    )
                    logger.info(
                        f"    flat_grid_y[flat_mask] (first few): {flat_grid_y[flat_mask][:5].cpu().numpy().tolist()}"
                    )
                # --- END LOGGING ---

                # Assign the regression targets for cells where mask is True.
                flat_bbox_target[flat_mask, 0] = (
                    box_center_x - flat_grid_x[flat_mask]
                ) / cell_size
                flat_bbox_target[flat_mask, 1] = (
                    box_center_y - flat_grid_y[flat_mask]
                ) / cell_size
                # Take the logarithm of the width and height IN GRID CELLS
                flat_bbox_target[flat_mask, 2] = torch.log(
                    box_w / cell_size + 1e-16
                )  # Divide by cell_size
                flat_bbox_target[flat_mask, 3] = torch.log(
                    box_h / cell_size + 1e-16
                )  # Divide by cell_size

        return cls_target, bbox_target, obj_target

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DetectionLoss":
        return cls(config)
