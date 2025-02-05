import os
import torch
from torch.utils.data import DataLoader
from PIL import ImageDraw
import torchvision.transforms.functional as TF

from utils.logger import get_logger
from utils.collate_fn import detection_collate_fn
from models.model_factory import create_model
from models.losses.loss_factory import create_loss
from models.metrics.metric_factory import create_metrics_list
from data.datasets.drone_object_detection import DroneObjectDetectionDataset

# --- Hardcoded Configurations for Debugging ---
DATA_ROOT_VAL = "/Users/mirzabicer/Projects/teknofest-2025/output/images"
ANNOTATION_FILE_VAL = (
    "/Users/mirzabicer/Projects/teknofest-2025/output/annotations.json"
)
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 8
NUM_WORKERS = 0
PIN_MEMORY = False
MODEL_NAME = "simple_object_detector"
NUM_CLASSES = 2
GRID_SIZE = 7
PRETRAINED_PATH = "/Users/mirzabicer/Projects/teknofest-2025/outputs/2025-02-02/22-41-52/checkpoints/drone_object_detection_experiment/best_model.pth"
LOSS_NAME = "detection_loss"
METRIC_NAMES = ["detection_metrics"]
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.5
DEVICE = torch.device("cpu")  # Force CPU for debugging

# --- Setup Logger ---
logger = get_logger("debug_script.py", level="INFO")
logger.info("=== Starting Debugging Script ===")

# --- Create Validation Dataset and DataLoader ---
val_dataset = DroneObjectDetectionDataset(
    data_root=DATA_ROOT_VAL,
    annotation_file=ANNOTATION_FILE_VAL,
    image_size=IMAGE_SIZE,
    annotation_format="json_coco",
    classes=[
        "red_circle",
        "black_square",
    ],
    use_cache=False,
    cache_backend="ram",
    cache_dir=".dataset_cache",
)
logger.info(f"Validation dataset initialized with {len(val_dataset)} samples.")

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    collate_fn=detection_collate_fn,
)

# --- Instantiate Model and Load Pretrained Weights ---
model_config = {
    "name": MODEL_NAME,
    "input_channels": 3,
    "num_classes": NUM_CLASSES,
    "grid_size": GRID_SIZE,
}
model = create_model(model_config)
model.load_pretrained_weights(
    PRETRAINED_PATH, strict=True
)  # Set strict=True for debugging
model.to(DEVICE).eval()  # Set to eval mode for inference
logger.info(f"Model '{MODEL_NAME}' loaded from '{PRETRAINED_PATH}'.")

# --- Instantiate Loss Function ---
loss_config = {
    "name": LOSS_NAME,
    "alpha": 0.25,
    "gamma": 2.0,
    "cls_loss_weight": 1.0,
    "box_loss_weight": 1.0,
    "obj_loss_weight": 10.0,
    "grid_size": GRID_SIZE,
    "num_classes": NUM_CLASSES,
    "bg_class": 0,
}
loss_fn = create_loss(loss_config).to(DEVICE).eval()
loss_fn = (
    create_loss(loss_config).to(DEVICE).eval()
)  # Eval mode for loss too, though not strictly necessary

# --- Instantiate Metrics ---
metrics_config = [
    {
        "name": "detection_metrics",
        "iou_threshold": IOU_THRESHOLD,
        "conf_threshold": CONF_THRESHOLD,
        "nms_iou_threshold": NMS_IOU_THRESHOLD,
        "num_classes": NUM_CLASSES,
    }
]
metrics = create_metrics_list(metrics_config)
metric = metrics[0]
metric.reset()

# --- Debugging Loop ---
total_val_loss = 0.0
batch_index = 0
os.makedirs("debug_visualizations", exist_ok=True)  # Directory for visualizations

with torch.no_grad():  # Disable gradients for inference
    for batch in val_loader:
        logger.info(f"--- Processing Batch {batch_index} ---")
        # --- ADDED DEBUGGING PRINT ---
        logger.info(f"Type of batch['targets']: {type(batch['targets'])}")
        if isinstance(batch["targets"], list):
            if batch["targets"]:  # Check if the list is not empty
                logger.info(
                    f"Type of first element in batch['targets']: {type(batch['targets'][0])}"
                )
                if not isinstance(batch["targets"][0], dict) and not isinstance(
                    batch["targets"][0], str
                ):  # Check if it's not a dict or string to avoid error if it's something else unexpected
                    logger.info(
                        f"Content of first element in batch['targets']: {batch['targets'][0]}"
                    )
                elif isinstance(batch["targets"][0], str):
                    logger.info(
                        f"First element in batch['targets'] is a string: {batch['targets'][0]}"
                    )
            else:
                logger.info("batch['targets'] is an empty list.")
        else:
            logger.info(f"batch['targets'] is NOT a list. Content: {batch['targets']}")
        # --- END ADDED DEBUGGING PRINT ---

        batch_to_device = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_to_device[key] = value.to(DEVICE)
            elif isinstance(value, list):  # Handle list of tensors (targets)
                batch_to_device[key] = [
                    # --- MODIFIED LIST COMPREHENSION TO HANDLE POTENTIAL STRING ITEMS ---
                    (
                        {
                            k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                            for k, v in item.items()
                        }
                        if isinstance(item, dict)
                        else item
                    )  # <--- ADDED CONDITIONAL CHECK
                    for item in value
                ]
            else:
                batch_to_device[key] = value
        batch = batch_to_device  # Use the batch moved to device

        # --- Forward Pass ---
        model_output = model(batch)

        # --- Calculate Loss ---
        loss = loss_fn(model_output, batch)
        total_val_loss += loss.item()
        logger.info(f"Batch Loss: {loss.item():.4f}")

        # --- ADDED LOGGING FOR BBOX TARGETS AND PREDS ---
        logger.info("Bounding Box Targets and Predictions (first image in batch):")
        if batch["targets"] and batch["targets"][0]["boxes"].numel() > 0:
            logger.info(
                f"  bbox_target[0] (min/max): {loss_fn.bbox_target[0].min().item():.4f}/{loss_fn.bbox_target[0].max().item():.4f}"
            )  # Access bbox_target from loss_fn
        logger.info(
            f"  bbox_preds[0]  (min/max): {model_output['bbox_preds'][0].min().item():.4f}/{model_output['bbox_preds'][0].max().item():.4f}"
        )
        # --- END ADDED LOGGING ---

        # --- Update Metrics ---
        metric.update(model_output, batch)

        # --- Log Raw Model Outputs ---
        logger.info("Raw Model Outputs:")
        logger.info(
            f"  bbox_preds (min/max/shape): {model_output['bbox_preds'].min().item():.4f}/{model_output['bbox_preds'].max().item():.4f}/{model_output['bbox_preds'].shape}"
        )
        logger.info(
            f"  objectness (min/max/shape): {model_output['objectness'].min().item():.4f}/{model_output['objectness'].max().item():.4f}/{model_output['objectness'].shape}"
        )
        logger.info(
            f"  logits     (min/max/shape): {model_output['logits'].min().item():.4f}/{model_output['logits'].max().item():.4f}/{model_output['logits'].shape}"
        )

        # --- Decode Predictions ---
        cell_size = 1.0 / GRID_SIZE
        xs = (torch.arange(GRID_SIZE, device=DEVICE).float() + 0.5) * cell_size
        ys = (torch.arange(GRID_SIZE, device=DEVICE).float() + 0.5) * cell_size
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        grid_x = grid_x.unsqueeze(0).unsqueeze(-1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(-1)

        objectness_conf = torch.sigmoid(model_output["objectness"])
        class_probs = torch.softmax(model_output["logits"], dim=-1)
        max_class_prob, pred_class = torch.max(class_probs, dim=-1)
        conf = objectness_conf.squeeze(-1) * max_class_prob

        pred_offset = model_output["bbox_preds"][..., :2]
        pred_center_x = grid_x + pred_offset[..., 0:1] * cell_size
        pred_center_y = grid_y + pred_offset[..., 1:2] * cell_size
        pred_wh = model_output["bbox_preds"][..., 2:4]
        pred_boxes = torch.cat(
            [
                pred_center_x - pred_wh[..., 0:1] / 2,
                pred_center_y - pred_wh[..., 1:2] / 2,
                pred_center_x + pred_wh[..., 0:1] / 2,
                pred_center_y + pred_wh[..., 1:2] / 2,
            ],
            dim=-1,
        )
        pred_boxes = torch.clamp(pred_boxes, 0.0, 1.0)

        logger.info("Decoded Predictions (first image in batch):")
        logger.info(
            f"  pred_boxes[0] (first few boxes): {pred_boxes[0,:5].cpu().numpy().tolist() if pred_boxes.shape[1] > 0 else 'No boxes'}"
        )
        logger.info(
            f"  conf[0]      (first few confidences): {conf[0,:5].cpu().numpy().tolist() if conf.shape[1] > 0 else 'No confidences'}"
        )
        logger.info(
            f"  pred_class[0] (first few classes):   {pred_class[0,:5].cpu().numpy().tolist() if pred_class.shape[1] > 0 else 'No classes'}"
        )

        # --- Log Targets ---
        targets = batch["targets"]
        logger.info("Targets (first image in batch):")
        if targets and targets[0]["boxes"].numel() > 0:
            logger.info(
                f"  target_boxes[0]: {targets[0]['boxes'].cpu().numpy().tolist()}"
            )
            logger.info(
                f"  target_labels[0]: {targets[0]['labels'].cpu().numpy().tolist()}"
            )
        else:
            logger.info("  No targets in the first image of this batch.")

        # --- Visualize Predictions and Ground Truth ---
        images = batch["images"]
        for i in range(min(images.shape[0], 2)):  # Visualize max 2 images per batch
            img = TF.to_pil_image(images[i].cpu())
            draw = ImageDraw.Draw(img)

            # Draw GT boxes (Green)
            if targets and targets[i]["boxes"].numel() > 0:
                for box, label in zip(targets[i]["boxes"], targets[i]["labels"]):
                    box_np = box.cpu().numpy()
                    draw.rectangle(
                        (
                            box_np[0] * img.width,
                            box_np[1] * img.height,
                            box_np[2] * img.width,
                            box_np[3] * img.height,
                        ),
                        outline="green",
                        width=2,
                    )
                    draw.text(
                        (box_np[0] * img.width, box_np[1] * img.height),
                        f"GT:{label.item()}",
                        fill="green",
                    )

            # Draw Predicted boxes (Red) - only if confidence is high enough for visualization
            for gy in range(GRID_SIZE):
                for gx in range(GRID_SIZE):
                    if (
                        conf[i, gy, gx] > 0.1
                    ):  # Lower threshold for visualization to see *any* predictions
                        box = pred_boxes[i, gy, gx].cpu().numpy()
                        label = pred_class[i, gy, gx].cpu().item()
                        draw.rectangle(
                            (
                                box[0] * img.width,
                                box[1] * img.height,
                                box[2] * img.width,
                                box[3] * img.height,
                            ),
                            outline="red",
                            width=2,
                        )
                        draw.text(
                            (box[0] * img.width, box[1] * img.height),
                            f"Pred:{label}",
                            fill="red",
                        )

            img.save(f"debug_visualizations/batch_{batch_index}_image_{i}.jpg")
            logger.info(
                f"Saved visualization: debug_visualizations/batch_{batch_index}_image_{i}.jpg"
            )

        batch_index += 1
        if batch_index >= 5:  # Debug only first few batches
            logger.info("Stopping debug loop after 5 batches for brevity.")
            break

# --- Compute and Log Overall Metrics ---
avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
metrics_result = metric.compute()

logger.info("=== Validation Results ===")
logger.info(f"Average Validation Loss: {avg_val_loss:.4f}")
for name, value in metrics_result.items():
    if isinstance(value, dict):
        for sub_name, sub_value in value.items():
            logger.info(f"Val {name}_{sub_name}: {sub_value:.4f}")
    else:
        logger.info(f"Val {name}: {value:.4f}")

logger.info("=== Debugging Script Completed ===")
