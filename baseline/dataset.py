import glob
import json
import os
import random
from typing import Iterator, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm


EXTRA_REGRESSION_TASK_IDS = {"A4C", "AOP", "FA", "HC", "IVC", "PLAX", "PSAX"}


class KeypointDataset(Dataset):
    """Dataset for keypoint localization tasks only."""

    def __init__(
        self,
        data_root: str,
        transforms: Optional[A.Compose] = None,
        heatmap_size: Tuple[int, int] = (64, 64),
        sigma: float = 1.8,
    ):
        super().__init__()
        self.data_root = data_root
        self.transforms = transforms
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.csv_path = os.path.join(self.data_root, "csv")
        if not os.path.isdir(self.csv_path):
            raise FileNotFoundError(f"CSV path not found: {self.csv_path}")

        all_csv_files = glob.glob(os.path.join(self.csv_path, "*.csv"))
        if not all_csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.csv_path}")

        df_list = [pd.read_csv(csv_file) for csv_file in all_csv_files]
        dataframe = pd.concat(df_list, ignore_index=True).reset_index(drop=True)

        is_regression = dataframe["task_name"].astype(str).eq("Regression")
        is_extra_task = dataframe["task_id"].astype(str).isin(EXTRA_REGRESSION_TASK_IDS)
        self.dataframe = dataframe[is_regression | is_extra_task].reset_index(drop=True)
        if self.dataframe.empty:
            raise ValueError(
                "No keypoint records found. Expect task_name == 'Regression' or task_id in "
                f"{sorted(EXTRA_REGRESSION_TASK_IDS)}."
            )

        print(f"Keypoint data loaded. Total samples: {len(self.dataframe)}")

    def __len__(self) -> int:
        return len(self.dataframe)

    def _resolve_image_path(self, rel_path: str) -> Optional[str]:
        rel_norm = os.path.normpath(rel_path)
        cleaned_rel = rel_norm
        while cleaned_rel.startswith(".." + os.sep):
            cleaned_rel = cleaned_rel[3:]

        for root in [os.path.join(self.data_root, "images"), self.data_root]:
            direct = os.path.normpath(os.path.join(root, cleaned_rel))
            if os.path.isfile(direct):
                return direct

        return None

    def _generate_heatmaps(self, norm_coords: np.ndarray, num_points: int) -> np.ndarray:
        heatmap_h, heatmap_w = self.heatmap_size
        yy, xx = np.meshgrid(np.arange(heatmap_h), np.arange(heatmap_w), indexing="ij")
        heatmaps = np.zeros((num_points, heatmap_h, heatmap_w), dtype=np.float32)

        for i in range(num_points):
            x_norm = float(norm_coords[2 * i])
            y_norm = float(norm_coords[2 * i + 1])

            x_norm = min(max(x_norm, 0.0), 1.0)
            y_norm = min(max(y_norm, 0.0), 1.0)

            x = x_norm * (heatmap_w - 1)
            y = y_norm * (heatmap_h - 1)

            dist2 = (xx - x) ** 2 + (yy - y) ** 2
            heatmaps[i] = np.exp(-dist2 / (2.0 * self.sigma * self.sigma)).astype(np.float32)

        return heatmaps

    def __getitem__(self, idx: int) -> dict:
        record = self.dataframe.iloc[idx]
        task_id = record["task_id"]

        image_abs_path = self._resolve_image_path(record["image_path"])
        if image_abs_path is None:
            return self.__getitem__((idx + 1) % len(self))
        image = cv2.imread(image_abs_path)

        if image is None:
            return self.__getitem__((idx + 1) % len(self))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]

        num_points = int(record["num_classes"])
        coords = []
        for i in range(1, num_points + 1):
            col = f"point_{i}_xy"
            if col in record and pd.notna(record[col]):
                coords.extend(json.loads(record[col]))
            else:
                coords.extend([0.0, 0.0])
        label = np.array(coords, dtype=np.float32)

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Normalize keypoints by original image size.
        label[0::2] /= max(float(original_width), 1.0)
        label[1::2] /= max(float(original_height), 1.0)
        label = np.clip(label, 0.0, 1.0)

        heatmaps = self._generate_heatmaps(label, num_points)
        final_label = torch.from_numpy(label).float()
        final_heatmaps = torch.from_numpy(heatmaps).float()

        return {
            "image": image,
            "label": final_label,
            "heatmap": final_heatmaps,
            "task_id": task_id,
        }


class KeypointUniformSampler(Sampler[List[int]]):
    """Uniform task sampler for keypoint subtasks."""

    def __init__(self, dataset: KeypointDataset, batch_size: int, steps_per_epoch: Optional[int] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices_by_task = {}

        print("\n--- Initializing Keypoint Sampler ---")
        for idx, task_id in enumerate(tqdm(dataset.dataframe["task_id"], desc="Grouping indices")):
            if task_id not in self.indices_by_task:
                self.indices_by_task[task_id] = []
            self.indices_by_task[task_id].append(idx)

        self.task_ids = list(self.indices_by_task.keys())
        for task_id in self.task_ids:
            random.shuffle(self.indices_by_task[task_id])

        if steps_per_epoch is None:
            self.steps_per_epoch = len(self.dataset) // self.batch_size
        else:
            self.steps_per_epoch = steps_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        task_cursors = {task_id: 0 for task_id in self.task_ids}

        for _ in range(self.steps_per_epoch):
            task_id = random.choice(self.task_ids)
            indices = self.indices_by_task[task_id]
            cursor = task_cursors[task_id]

            start_idx = cursor
            end_idx = start_idx + self.batch_size

            if end_idx > len(indices):
                batch_indices = indices[start_idx:]
                random.shuffle(indices)
                remaining = self.batch_size - len(batch_indices)
                batch_indices.extend(indices[:remaining])
                task_cursors[task_id] = remaining
            else:
                batch_indices = indices[start_idx:end_idx]
                task_cursors[task_id] = end_idx

            yield batch_indices

    def __len__(self) -> int:
        return self.steps_per_epoch
