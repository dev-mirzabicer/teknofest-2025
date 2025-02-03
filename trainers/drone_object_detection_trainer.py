import time
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import ImageDraw
import contextlib
from trainers.base_trainer import BaseTrainer
from utils.logger import get_logger


class DroneObjectDetectionTrainer(BaseTrainer):
    """
    Concrete trainer for drone object detection tasks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(name=self.__class__.__name__)

        # Log the model graph at the start (using a dummy batch).
        if self.tb_logger:
            dummy_batch = next(iter(self.train_dataloader))
            dummy_batch = self._move_batch_to_device(dummy_batch)
            try:
                self.tb_logger.log_model_graph(self.model, dummy_batch)
                self.logger.info("Logged model computational graph to TensorBoard.")
            except Exception as e:
                self.logger.error(f"Failed to log model graph: {e}")

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(self.train_dataloader):
            batch_start_time = time.time()
            batch = self._move_batch_to_device(batch)

            # Detailed logging: log input shapes for the first batch of each epoch or every N-th batch
            if self.tb_logger and (
                batch_idx == 0
                or self.tb_logger.log_every_batch
                or (batch_idx % self.tb_logger.log_batch_sample_freq == 0)
            ):
                input_info = {
                    key: (value.shape if hasattr(value, "shape") else str(type(value)))
                    for key, value in batch.items()
                }
                self.tb_logger.log_text(
                    "Batch/Input_Info", str(input_info), self.global_step
                )

            self.optimizer.zero_grad()
            using_amp = self.use_mixed_precision and self.scaler is not None
            autocast_context = (
                torch.cuda.amp.autocast(enabled=using_amp)
                if using_amp
                else contextlib.nullcontext()
            )

            with autocast_context:
                model_output = self.model(batch)
                loss = self.loss_fn(model_output, batch)

            # Log model output shapes in a similar manner.
            if self.tb_logger and (
                batch_idx == 0 or batch_idx % self.tb_logger.log_batch_sample_freq == 0
            ):
                output_info = {
                    key: (value.shape if hasattr(value, "shape") else str(type(value)))
                    for key, value in model_output.items()
                }
                self.tb_logger.log_text(
                    "Model/Output_Info", str(output_info), self.global_step
                )

            # Log raw loss tensor information.
            if self.tb_logger:
                self.tb_logger.log_tensor_info(
                    "Loss/Raw", loss.detach(), self.global_step
                )

            if using_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Log gradients and weights histograms every few iterations.
            if self.tb_logger and (
                batch_idx % self.tb_logger.log_batch_sample_freq == 0
            ):
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        self.tb_logger.log_histogram(
                            f"Gradients/{name}",
                            param.grad.detach().cpu().numpy(),
                            self.global_step,
                        )
                        self.tb_logger.log_histogram(
                            f"Weights/{name}",
                            param.detach().cpu().numpy(),
                            self.global_step,
                        )

            if self.grad_clip_norm is not None:
                if using_amp:
                    self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip_norm
                )

            if using_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            batch_exec_time = time.time() - batch_start_time
            total_loss += loss.item()
            self.global_step += 1

            # Log per-batch scalar metrics.
            if batch_idx % 10 == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.info(
                    f"Epoch {epoch} | Batch {batch_idx}/{len(self.train_dataloader)} | Loss: {loss.item():.4f} | LR: {lr:.6f}"
                )
                if self.tb_logger:
                    self.tb_logger.log_scalar(
                        "Batch/Loss", loss.item(), self.global_step
                    )
                    self.tb_logger.log_scalar(
                        "Batch/Learning_Rate", lr, self.global_step
                    )
                    if self.tb_logger.log_exec_time:
                        self.tb_logger.log_execution_time(
                            "Batch/Execution_Time", batch_exec_time, self.global_step
                        )
                    if self.tb_logger.log_memory:
                        self.tb_logger.log_memory_usage(self.global_step)

            # Log raw and activated model outputs for the first few batches of each epoch
            if self.tb_logger and batch_idx < 5:
                self.tb_logger.log_text(
                    f"Model/Raw_Bbox_Preds/Batch_{batch_idx}",
                    str(model_output["bbox_preds"].detach().cpu().numpy()),
                    self.global_step,
                )
                self.tb_logger.log_text(
                    f"Model/Raw_Objectness/Batch_{batch_idx}",
                    str(model_output["objectness"].detach().cpu().numpy()),
                    self.global_step,
                )
                self.tb_logger.log_text(
                    f"Model/Raw_Logits/Batch_{batch_idx}",
                    str(model_output["logits"].detach().cpu().numpy()),
                    self.global_step,
                )

            if self.tb_logger and batch_idx < 1:
                self.visualize_predictions(batch, model_output, batch_idx)

        epoch_exec_time = time.time() - epoch_start_time
        avg_loss = (
            total_loss / len(self.train_dataloader) if self.train_dataloader else 0.0
        )
        self.logger.info(
            f"Epoch {epoch} completed in {epoch_exec_time:.2f}s with Average Loss: {avg_loss:.4f}"
        )
        return {"train_loss": avg_loss}

    def visualize_predictions(
        self,
        batch,
        model_output,
        batch_idx: int,
        score_threshold: float = 0.5,
        max_images: int = 10,
    ):
        """
        Visualizes ground-truth and predicted bounding boxes on a subset of images in the batch.
        Draws:
        - Green boxes for ground truth (with "GT: class_id")
        - Red boxes for predicted boxes above a given confidence threshold (with "Pred: class_id, conf")

        Args:
        self: Typically 'self' is your trainer instance (so we can access self.tb_logger, etc.).
        batch (dict): Contains "images" (B,C,H,W) and "targets" (list of dicts with "boxes" and "labels").
        model_output (dict): Contains "bbox_preds" (B,G,G,4), "objectness" (B,G,G,1), and "logits" (B,G,G,num_classes).
        batch_idx (int): Used to label the images in TensorBoard.
        score_threshold (float): Minimum confidence score to visualize a predicted box.
        max_images (int): How many images from the batch to visualize (starting from index 0).
        """
        images = batch["images"]  # shape: (B, C, H, W)
        targets = batch["targets"]  # list of length B
        bbox_preds = model_output["bbox_preds"]  # shape: (B, G, G, 4)
        objectness = model_output["objectness"]  # shape: (B, G, G, 1)
        logits = model_output["logits"]  # shape: (B, G, G, num_classes)

        B, _, H, W = images.shape
        G = bbox_preds.shape[1]
        device = images.device

        # Prepare a grid for decoding
        cell_size = 1.0 / G
        xs = (torch.arange(G, device=device).float() + 0.5) * cell_size
        ys = (torch.arange(G, device=device).float() + 0.5) * cell_size
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")  # (G, G)
        # Reshape to (1, G, G, 1)
        grid_x = grid_x.unsqueeze(0).unsqueeze(-1)  # (1, G, G, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(-1)  # (1, G, G, 1)

        # Compute predicted confidence (objectness * max class prob).
        obj_conf = torch.sigmoid(objectness.squeeze(-1))  # (B, G, G) in [0,1]
        class_probs = F.softmax(logits, dim=-1)  # (B, G, G, num_classes)
        max_class_prob, pred_class = torch.max(class_probs, dim=-1)  # (B, G, G)
        conf = obj_conf * max_class_prob  # Combined confidence

        # Decode predicted boxes to [x_min, y_min, x_max, y_max] (normalized)
        # Bbox params are [tx, ty, tw, th], but the network did `tw = exp(...)`, etc.
        # So we'll interpret them as is: tw, th are in "grid units," so:
        tx = bbox_preds[..., 0:1]  # shape: (B, G, G, 1)
        ty = bbox_preds[..., 1:2]
        tw = bbox_preds[..., 2:3]
        th = bbox_preds[..., 3:4]

        pred_center_x = grid_x + tx * cell_size  # (B, G, G, 1)
        pred_center_y = grid_y + ty * cell_size
        # tw, th are presumably in "absolute" scale wrt the grid cell => tw / G for normalized
        pred_w = tw / G
        pred_h = th / G

        x_min = pred_center_x - 0.5 * pred_w
        y_min = pred_center_y - 0.5 * pred_h
        x_max = pred_center_x + 0.5 * pred_w
        y_max = pred_center_y + 0.5 * pred_h

        # Clamp to [0,1]
        x_min = torch.clamp(x_min, 0.0, 1.0)
        y_min = torch.clamp(y_min, 0.0, 1.0)
        x_max = torch.clamp(x_max, 0.0, 1.0)
        y_max = torch.clamp(y_max, 0.0, 1.0)

        # For each image we do:
        for i in range(min(B, max_images)):
            # Convert tensor image to PIL
            img_cpu = images[i].cpu()  # shape: (C, H, W)
            img_pil = TF.to_pil_image(img_cpu)  # Convert to PIL image
            draw = ImageDraw.Draw(img_pil)

            # 1) Draw Ground-Truth boxes (in green)
            gt_boxes = targets[i]["boxes"]  # shape: (N, 4) (normalized)
            gt_labels = targets[i]["labels"]  # shape: (N,)
            for box_i in range(gt_boxes.shape[0]):
                box = gt_boxes[box_i]
                label = gt_labels[box_i].item()
                # Convert normalized coords to absolute pixel coords
                abs_xmin = box[0].item() * img_pil.width
                abs_ymin = box[1].item() * img_pil.height
                abs_xmax = box[2].item() * img_pil.width
                abs_ymax = box[3].item() * img_pil.height
                draw.rectangle(
                    [(abs_xmin, abs_ymin), (abs_xmax, abs_ymax)],
                    outline="green",
                    width=2,
                )
                draw.text((abs_xmin, abs_ymin), f"GT: {label}", fill="green")

            # 2) Draw Predicted boxes (in red), above the threshold
            for row in range(G):
                for col in range(G):
                    score = conf[i, row, col].item()
                    if score >= score_threshold:
                        predicted_label = pred_class[i, row, col].item()
                        abs_xmin = x_min[i, row, col, 0].item() * img_pil.width
                        abs_ymin = y_min[i, row, col, 0].item() * img_pil.height
                        abs_xmax = x_max[i, row, col, 0].item() * img_pil.width
                        abs_ymax = y_max[i, row, col, 0].item() * img_pil.height
                        draw.rectangle(
                            [(abs_xmin, abs_ymin), (abs_xmax, abs_ymax)],
                            outline="red",
                            width=2,
                        )
                        # Show class + confidence
                        draw.text(
                            (abs_xmin, abs_ymin),
                            f"Pred: {predicted_label}, {score:.2f}",
                            fill="red",
                        )

            # 3) Log the image to TensorBoard (if you have self.tb_logger)
            if hasattr(self, "tb_logger") and self.tb_logger is not None:
                img_tensor = TF.to_tensor(img_pil)
                self.tb_logger.writer.add_image(
                    f"Predictions/Batch_{batch_idx}/Image_{i}",
                    img_tensor,
                    self.global_step,
                )
            else:
                # Otherwise you might just save/show it locally:
                # img_pil.show()  # or
                # img_pil.save(f"debug_viz_image_{batch_idx}_{i}.png")
                pass
