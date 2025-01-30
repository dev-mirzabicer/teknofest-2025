import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import json
import logging
from typing import List, Callable, Optional, Dict, Any, Tuple
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DroneObjectDetectionDataset(Dataset):
    """
    Custom PyTorch Dataset for drone-based object detection (humans and vehicles).

    Supports various annotation formats, data augmentations, and preprocessing steps.
    """

    def __init__(
        self,
        data_root: str,
        annotation_file: str,
        image_size: Tuple[int, int] = (640, 640),
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        annotation_format: str = "json_coco",
        classes: Optional[List[str]] = None,
        use_cache: bool = False,
        cache_backend: str = "ram",
        cache_dir: str = ".dataset_cache",
    ) -> None:
        """
        Args:
            data_root (str): Root directory of the dataset (images and annotations).
            annotation_file (str): Path to the annotation file.
            image_size (Tuple[int, int]): Desired output image size (height, width).
            transform (Callable, optional): Image transformations. Defaults to None.
            target_transform (Callable, optional): Target (annotation) transformations. Defaults to None.
            annotation_format (str): Format of the annotation file ('json_coco', 'txt_yolo', 'custom_format').
            classes (List[str], optional): List of class names. If None, inferred from annotations (COCO only). Required for YOLO and custom formats.
            use_cache (bool): Whether to use caching for faster data loading.
            cache_backend (str): Backend for caching ('ram' or 'disk').
            cache_dir (str): Directory to store disk cache.

        Raises:
            ValueError: If annotation format is unsupported, or if classes are not provided for YOLO format,
                        or if specified classes are not found in COCO annotations.
            FileNotFoundError: If annotation file or image files are not found.
            json.JSONDecodeError: If COCO annotation file is not valid JSON.
        """
        # Input validation and assertions
        assert os.path.isdir(data_root), f"Data root directory not found: {data_root}"
        assert os.path.isfile(
            annotation_file
        ), f"Annotation file not found: {annotation_file}"
        assert annotation_format.lower() in [
            "json_coco",
            "txt_yolo",
            "custom_format",
        ], f"Unsupported annotation format: {annotation_format}"
        assert cache_backend.lower() in [
            "ram",
            "disk",
        ], f"Unsupported cache backend: {cache_backend}"
        if annotation_format.lower() == "txt_yolo" and classes is None:
            raise ValueError(
                "Classes must be provided when using YOLO annotation format."
            )

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
        self.annotations: List[Dict[str, Any]] = []
        self._class_to_index: Dict[str, int] = {}
        self._index_to_class: Dict[int, str] = {}
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        self._load_annotations()
        if self.classes is None and self.annotation_format == "json_coco":
            self.classes = list(self._class_to_index.keys())
        elif self.classes is None:
            self.classes = []  # Initialize as empty list if not provided and not COCO

        if self.use_cache and self.cache_backend == "disk":
            os.makedirs(self.cache_dir, exist_ok=True)

        logging.info(
            f"Dataset initialized with {len(self.image_paths)} images, {len(self.classes)} classes, annotation format: {self.annotation_format}, cache: {self.use_cache} ({self.cache_backend})."
        )

    def _load_annotations(self) -> None:
        """Loads annotations based on the specified format."""
        if self.annotation_format == "json_coco":
            self._load_coco_annotations()
        elif self.annotation_format == "txt_yolo":
            self._load_yolo_annotations()
        elif self.annotation_format == "custom_format":
            self._load_custom_annotations()
        else:
            raise ValueError(
                f"Unsupported annotation format: {self.annotation_format}. This should not happen due to constructor validation."
            )

    def _load_coco_annotations(self) -> None:
        """Loads annotations from COCO JSON format."""
        try:
            with open(self.annotation_file, "r") as f:
                coco_data = json.load(f)

            categories = coco_data["categories"]
            image_id_to_filename = {
                img["id"]: img["file_name"] for img in coco_data["images"]
            }
            category_id_to_name = {cat["id"]: cat["name"] for cat in categories}

            if self.classes is None:
                self._class_to_index = {
                    cat["name"]: i for i, cat in enumerate(categories)
                }
                self._index_to_class = {
                    i: cat["name"] for i, cat in enumerate(categories)
                }
            else:
                coco_category_names = set(cat["name"] for cat in categories)
                for cls_name in self.classes:
                    if cls_name not in coco_category_names:
                        raise ValueError(
                            f"Class '{cls_name}' not found in COCO categories. Available categories: {coco_category_names}"
                        )
                self._class_to_index = {name: i for i, name in enumerate(self.classes)}
                self._index_to_class = {i: name for i, name in enumerate(self.classes)}

            annotation_map = defaultdict(list)
            for ann in coco_data["annotations"]:
                if category_id_to_name[ann["category_id"]] in self._class_to_index:
                    annotation_map[ann["image_id"]].append(ann)

            for image_id, image_filename in image_id_to_filename.items():
                image_path = os.path.join(self.data_root, image_filename)
                if not os.path.exists(image_path):
                    logging.warning(f"Image file not found: {image_path}. Skipping.")
                    continue

                bboxes: List[List[float]] = []
                for ann in annotation_map[image_id]:
                    bbox_coco = ann["bbox"]
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
                    )

                self.image_paths.append(image_path)
                self.annotations.append({"bboxes": bboxes})

        except FileNotFoundError as e:
            logging.error(f"COCO Annotation file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON format in COCO annotation file: {e}")
            raise
        except KeyError as e:
            logging.error(
                f"KeyError in COCO annotation parsing: {e}. Check annotation file structure. Error: {e}"
            )
            raise
        except Exception as e:
            logging.error(f"Unexpected error loading COCO annotations: {e}")
            raise

    def _load_yolo_annotations(self) -> None:
        """Loads annotations from YOLO TXT format."""
        image_files = [
            f
            for f in os.listdir(self.data_root)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
        if not image_files:
            raise FileNotFoundError(
                f"No image files found in data_root: {self.data_root}"
            )

        assert (
            self.classes is not None
        ), "Classes must be provided for YOLO annotation format."
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

            bboxes: List[List[float]] = []
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
                            class_id = int(class_id)
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

                        x_min = x_center - width / 2.0
                        y_min = y_center - height / 2.0
                        x_max = x_center + width / 2.0
                        y_max = y_center + height / 2.0

                        bboxes.append([x_min, y_min, x_max, y_max, class_id])

            except FileNotFoundError as e:
                logging.error(f"YOLO Annotation file not found (unexpected): {e}")
                continue
            except Exception as e:
                logging.error(
                    f"Error loading YOLO annotations from {annotation_txt_path}: {e}"
                )
                continue

            self.image_paths.append(image_path)
            self.annotations.append({"bboxes": bboxes})

    def _load_custom_annotations(self) -> None:
        # ... to be implemented ...
        raise NotImplementedError("Custom annotation format not yet implemented.")

    def __len__(self) -> int:
        """Returns the number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Fetches an image and its annotations at the given index."""
        if self.use_cache and idx in self._cache:
            if self.cache_backend == "ram":
                return self._cache[idx]
            elif self.cache_backend == "disk":
                cache_file = os.path.join(self.cache_dir, f"item_{idx}.pt")
                if os.path.exists(cache_file):
                    return torch.load(cache_file)

        image_path = self.image_paths[idx]
        annotation = self.annotations[idx]
        bboxes_list = annotation["bboxes"]

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError as e:
            logging.error(f"Image file not found during __getitem__: {image_path}")
            raise

        original_width, original_height = image.size

        normalized_bboxes: List[List[float]] = []
        labels: List[int] = []
        for bbox in bboxes_list:
            x_min, y_min, x_max, y_max, class_index = bbox
            normalized_x_min = x_min / original_width
            normalized_x_max = x_max / original_width
            normalized_y_min = y_min / original_height
            normalized_y_max = y_max / original_height
            normalized_bboxes.append(
                [normalized_x_min, normalized_y_min, normalized_x_max, normalized_y_max]
            )
            labels.append(int(class_index))

        image = transforms.Resize(self.image_size)(image)
        if self.transform:
            image, normalized_bboxes = self.transform(image, normalized_bboxes)

        image_tensor = transforms.ToTensor()(image)
        image_tensor = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )(image_tensor)

        boxes_tensor = (
            torch.tensor(normalized_bboxes, dtype=torch.float32)
            if normalized_bboxes
            else torch.empty(0, 4)
        )
        labels_tensor = (
            torch.tensor(labels, dtype=torch.int64)
            if labels
            else torch.empty(0, dtype=torch.int64)
        )

        target_dict: Dict[str, torch.Tensor] = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
        }

        if self.target_transform:
            target_dict = self.target_transform(target_dict)

        if self.use_cache:
            if self.cache_backend == "ram":
                self._cache[idx] = (image_tensor, target_dict)
            elif self.cache_backend == "disk":
                cache_file = os.path.join(self.cache_dir, f"item_{idx}.pt")
                torch.save((image_tensor, target_dict), cache_file)

        return image_tensor, target_dict
