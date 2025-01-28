import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import json
import logging
from typing import List, Callable, Optional, Dict, Any, Tuple
from collections import defaultdict


class DroneObjectDetectionDataset(Dataset):
    """
    Custom PyTorch Dataset for drone-based object detection (humans and vehicles).

    Supports various annotation formats, data augmentations, and preprocessing steps.
    """

    def __init__(
        self,
        data_root: str,
        annotation_file: str,
        image_size: Tuple[int, int] = (640, 640),  # Target image size (height, width)
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        annotation_format: str = "json_coco",  # Supported formats: 'json_coco', 'txt_yolo', 'custom_format'
        classes: Optional[
            List[str]
        ] = None,  # List of class names, if None, inferred from annotations
        use_cache: bool = False,  # Enable caching for faster loading (RAM or disk based)
        cache_backend: str = "ram",  # 'ram' or 'disk'
        cache_dir: str = ".dataset_cache",  # Directory for disk cache
    ):
        """
        Args:
            data_root (str): Root directory of the dataset (images and annotations).
            annotation_file (str): Path to the annotation file.
            image_size (Tuple[int, int]): Desired output image size (height, width).
            transform (Callable, optional): Image transformations. Defaults to None.
            target_transform (Callable, optional): Target (annotation) transformations. Defaults to None.
            annotation_format (str): Format of the annotation file ('json_coco', 'txt_yolo', 'custom_format').
            classes (List[str], optional): List of class names. If None, inferred from annotations.
            use_cache (bool): Whether to use caching for faster data loading.
            cache_backend (str): Backend for caching ('ram' or 'disk').
            cache_dir (str): Directory to store disk cache.
        """
        self.data_root = data_root
        self.annotation_file = annotation_file
        self.image_size = image_size
        self.transform = transform
        self.target_transform = target_transform
        self.annotation_format = annotation_format.lower()
        self.classes = classes
        self.use_cache = use_cache
        self.cache_backend = cache_backend.lower()
        self.cache_dir = cache_dir

        self.image_paths: List[str] = []
        self.annotations: List[Dict[str, Any]] = (
            []
        )  # Store annotations in a consistent format
        self._class_to_index: Dict[str, int] = {}
        self._index_to_class: Dict[int, str] = {}
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = (
            {}
        )  # Cache for loaded data

        self._load_annotations()  # Load annotations based on format
        if self.classes is None:
            self.classes = list(
                self._class_to_index.keys()
            )  # Infer classes if not provided

        if self.use_cache and self.cache_backend == "disk":
            os.makedirs(self.cache_dir, exist_ok=True)

        logging.info(
            f"Dataset initialized with {len(self.image_paths)} images, {len(self.classes)} classes, annotation format: {self.annotation_format}, cache: {self.use_cache} ({self.cache_backend})."
        )

    def _load_annotations(self):
        """Loads annotations based on the specified format."""
        if self.annotation_format == "json_coco":
            self._load_coco_annotations()
        elif self.annotation_format == "txt_yolo":
            self._load_yolo_annotations()
        elif self.annotation_format == "custom_format":
            self._load_custom_annotations()  # Implement your custom loading logic
        else:
            raise ValueError(
                f"Unsupported annotation format: {self.annotation_format}. Supported formats are: 'json_coco', 'txt_yolo', 'custom_format'."
            )

    def _load_coco_annotations(self):
        """Loads annotations from COCO JSON format."""
        try:
            with open(self.annotation_file, "r") as f:
                coco_data = json.load(f)

            categories = coco_data["categories"]
            image_id_to_filename = {
                img["id"]: img["file_name"] for img in coco_data["images"]
            }
            category_id_to_name = {cat["id"]: cat["name"] for cat in categories}

            if (
                self.classes is None
            ):  # Infer classes from COCO categories if not provided
                self._class_to_index = {
                    cat["name"]: i for i, cat in enumerate(categories)
                }
                self._index_to_class = {
                    i: cat["name"] for i, cat in enumerate(categories)
                }
            else:  # Validate provided classes against COCO categories
                coco_category_names = set(cat["name"] for cat in categories)
                for cls_name in self.classes:
                    if cls_name not in coco_category_names:
                        raise ValueError(
                            f"Class '{cls_name}' not found in COCO categories. Available categories: {coco_category_names}"
                        )
                self._class_to_index = {name: i for i, name in enumerate(self.classes)}
                self._index_to_class = {i: name for i, name in enumerate(self.classes)}

            annotation_map = defaultdict(list)  # Group annotations by image ID
            for ann in coco_data["annotations"]:
                if (
                    category_id_to_name[ann["category_id"]] in self._class_to_index
                ):  # Only include annotations for specified classes
                    annotation_map[ann["image_id"]].append(ann)

            for image_id, image_filename in image_id_to_filename.items():
                image_path = os.path.join(self.data_root, image_filename)
                if not os.path.exists(image_path):
                    logging.warning(f"Image file not found: {image_path}. Skipping.")
                    continue

                bboxes = []
                for ann in annotation_map[image_id]:
                    bbox_coco = ann[
                        "bbox"
                    ]  # [x_min, y_min, width, height] (COCO format)
                    x_min, y_min, width, height = bbox_coco
                    x_max = x_min + width
                    y_max = y_min + height
                    class_name = category_id_to_name[ann["category_id"]]
                    class_index = self._class_to_index[class_name]
                    bboxes.append(
                        [
                            float(x_min),
                            float(y_min),
                            float(x_max),
                            float(y_max),
                            int(class_index),
                        ]
                    )  # [x_min, y_min, x_max, y_max, class_index]

                self.image_paths.append(image_path)
                self.annotations.append({"bboxes": bboxes})  # Store bounding boxes

        except FileNotFoundError:
            logging.error(f"Annotation file not found: {self.annotation_file}")
            raise
        except json.JSONDecodeError:
            logging.error(
                f"Invalid JSON format in annotation file: {self.annotation_file}"
            )
            raise
        except KeyError as e:
            logging.error(
                f"KeyError in COCO annotation parsing: {e}. Check annotation file structure."
            )
            raise
        except Exception as e:
            logging.error(f"Error loading COCO annotations: {e}")
            raise

    def _load_yolo_annotations(self):
        """Loads annotations from YOLO TXT format (one .txt file per image)."""
        # Assuming YOLO format: one .txt file per image, same name as image but .txt extension
        # Each line in .txt: <class_id> <x_center> <y_center> <width> <height> (normalized)

        image_files = [
            f
            for f in os.listdir(self.data_root)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
        if not image_files:
            raise FileNotFoundError(
                f"No image files found in data_root: {self.data_root}"
            )

        if self.classes is None:
            raise ValueError(
                "Classes must be provided when using YOLO annotation format as class names are not inherently in YOLO format."
            )
        self._class_to_index = {name: i for i, name in enumerate(self.classes)}
        self._index_to_class = {i: name for i, name in enumerate(self.classes)}

        for image_file in image_files:
            image_path = os.path.join(self.data_root, image_file)
            annotation_txt_path = os.path.join(
                self.data_root, os.path.splitext(image_file)[0] + ".txt"
            )

            if not os.path.exists(annotation_txt_path):
                logging.warning(
                    f"Annotation file not found for image: {image_file}. Skipping."
                )
                continue

            bboxes = []
            try:
                with open(annotation_txt_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            logging.warning(
                                f"Invalid YOLO annotation line in {annotation_txt_path}: {line}. Skipping line."
                            )
                            continue
                        try:
                            class_id, x_center, y_center, width, height = map(
                                float, parts
                            )
                            class_id = int(class_id)  # YOLO class IDs are integers
                        except ValueError:
                            logging.warning(
                                f"Invalid numeric value in YOLO annotation line: {line}. Skipping line."
                            )
                            continue

                        if class_id not in self._index_to_class:
                            logging.warning(
                                f"YOLO class ID {class_id} not in provided classes. Skipping annotation."
                            )
                            continue

                        # Convert YOLO normalized center/size to x_min, y_min, x_max, y_max (absolute pixel coordinates will be done later)
                        x_min = x_center - width / 2.0
                        y_min = y_center - height / 2.0
                        x_max = x_center + width / 2.0
                        y_max = y_center + height / 2.0

                        bboxes.append(
                            [x_min, y_min, x_max, y_max, class_id]
                        )  # [x_min, y_min, x_max, y_max, class_index]

            except FileNotFoundError:  # Should be already checked, but for robustness
                logging.error(
                    f"Annotation file not found (unexpected): {annotation_txt_path}"
                )
                continue  # Skip to next image
            except Exception as e:
                logging.error(
                    f"Error loading YOLO annotations from {annotation_txt_path}: {e}"
                )
                continue  # Skip to next image

            self.image_paths.append(image_path)
            self.annotations.append({"bboxes": bboxes})

    def _load_custom_annotations(self):
        """
        [Placeholder for Custom Annotation Loading Logic]
        Implement your own logic to load annotations from your custom format.
        This could involve reading CSV, XML, or any other format.

        Example (Illustrative - Replace with your actual logic):
        Assume a CSV file where each row is: image_filename, x_min, y_min, x_max, y_max, class_name
        """
        try:
            with open(self.annotation_file, "r") as f:
                # Example CSV parsing (replace with your format parsing)
                lines = f.readlines()[1:]  # Skip header if exists
                for line in lines:
                    parts = line.strip().split(",")
                    if len(parts) != 6:
                        logging.warning(
                            f"Invalid custom annotation line: {line}. Skipping."
                        )
                        continue
                    image_filename, x_min, y_min, x_max, y_max, class_name = parts

                    image_path = os.path.join(self.data_root, image_filename)
                    if not os.path.exists(image_path):
                        logging.warning(
                            f"Image file not found: {image_path}. Skipping annotation."
                        )
                        continue

                    if class_name not in self._class_to_index:
                        if (
                            self.classes is None
                        ):  # Dynamically add new classes if not predefined
                            class_index = len(self._class_to_index)
                            self._class_to_index[class_name] = class_index
                            self._index_to_class[class_index] = class_name
                        else:
                            logging.warning(
                                f"Class '{class_name}' not in predefined classes. Skipping annotation."
                            )
                            continue
                    else:
                        class_index = self._class_to_index[class_name]

                    bbox = [
                        float(x_min),
                        float(y_min),
                        float(x_max),
                        float(y_max),
                        int(class_index),
                    ]

                    # Find if image_path is already in self.image_paths, if so, append bbox to existing annotation
                    try:
                        image_index = self.image_paths.index(image_path)
                        self.annotations[image_index]["bboxes"].append(bbox)
                    except ValueError:  # Image path not found yet
                        self.image_paths.append(image_path)
                        self.annotations.append({"bboxes": [bbox]})

        except FileNotFoundError:
            logging.error(f"Custom annotation file not found: {self.annotation_file}")
            raise
        except Exception as e:
            logging.error(f"Error loading custom annotations: {e}")
            raise

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Fetches an image and its annotations.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: (image, target_dict)
                - image: Transformed image tensor (C, H, W).
                - target_dict: Dictionary containing target information:
                    - 'boxes': Bounding box tensor (N, 4) in normalized [x_min, y_min, x_max, y_max] format.
                    - 'labels': Class label tensor (N,) (integer indices).
        """
        if self.use_cache and idx in self._cache:
            if self.cache_backend == "ram":
                return self._cache[idx]
            elif self.cache_backend == "disk":
                cache_file = os.path.join(self.cache_dir, f"item_{idx}.pt")
                if os.path.exists(cache_file):
                    return torch.load(cache_file)

        image_path = self.image_paths[idx]
        annotation = self.annotations[idx]
        bboxes_list = annotation[
            "bboxes"
        ]  # List of [x_min, y_min, x_max, y_max, class_index]

        try:
            image = Image.open(image_path).convert("RGB")  # Ensure RGB for consistency
        except FileNotFoundError:
            logging.error(f"Image file not found during __getitem__: {image_path}")
            # Handle missing image robustly, e.g., return None or raise error, or skip this index (more complex)
            raise  # For now, raise to stop training if data is critical

        original_width, original_height = image.size

        # Convert bounding boxes to normalized coordinates (0 to 1)
        normalized_bboxes = []
        labels = []
        for bbox in bboxes_list:
            x_min, y_min, x_max, y_max, class_index = bbox
            normalized_x_min = x_min / original_width
            normalized_x_max = x_max / original_width
            normalized_y_min = y_min / original_height
            normalized_y_max = y_max / original_height
            normalized_bboxes.append(
                [normalized_x_min, normalized_y_min, normalized_x_max, normalized_y_max]
            )
            labels.append(int(class_index))  # Ensure labels are integers

        image = transforms.Resize(self.image_size)(image)  # Resize image
        if self.transform:
            image, normalized_bboxes = self.transform(
                image, normalized_bboxes
            )  # Apply augmentations, passing bboxes

        image = transforms.ToTensor()(image)  # Convert to tensor (after augmentations)
        image = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )(
            image
        )  # ImageNet normalization (common practice)

        boxes_tensor = (
            torch.tensor(normalized_bboxes, dtype=torch.float32)
            if normalized_bboxes
            else torch.empty(0, 4)
        )  # (N, 4)
        labels_tensor = (
            torch.tensor(labels, dtype=torch.int64)
            if labels
            else torch.empty(0, dtype=torch.int64)
        )  # (N,)

        target_dict = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            # Add more target information if needed, e.g., 'image_id', 'area', 'iscrowd' (for COCO compatibility)
        }

        if self.target_transform:
            target_dict = self.target_transform(
                target_dict
            )  # Apply target transformations if any

        if self.use_cache:
            if self.cache_backend == "ram":
                self._cache[idx] = (image, target_dict)
            elif self.cache_backend == "disk":
                cache_file = os.path.join(self.cache_dir, f"item_{idx}.pt")
                torch.save((image, target_dict), cache_file)

        return image, target_dict


def custom_collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """
    Custom collate function to handle batches of images and target dictionaries.
    This is crucial for object detection datasets where images and number of objects can vary.

    Args:
        batch (List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]): List of (image, target_dict) tuples.

    Returns:
        Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]: (batched_images, batched_target_dicts)
            - batched_images: Stacked image tensors (B, C, H, W).
            - batched_target_dicts: List of target dictionaries (length B), no stacking needed for targets.
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    batched_images = torch.stack(images)  # Stack images along the batch dimension

    return batched_images, targets  # Targets are kept as a list of dictionaries
