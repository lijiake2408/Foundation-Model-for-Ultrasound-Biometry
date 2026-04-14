import argparse
import glob
import json
import os
from typing import Optional

import albumentations as A
import cv2
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model_factory import MultiTaskModelFactory
from utils import decode_heatmaps_to_normalized_coords


EXTRA_REGRESSION_TASK_IDS = {"A4C", "AOP", "FA", "HC", "IVC", "PLAX", "PSAX"}


class InferenceDataset(Dataset):
    """Inference dataset for keypoint tasks only."""

    def __init__(
        self,
        data_root: str,
        transforms: Optional[A.Compose] = None,
        split_csv: Optional[str] = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.transforms = transforms
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

        if split_csv is not None:
            if not os.path.exists(split_csv):
                raise FileNotFoundError(f"Split CSV not found: {split_csv}")

            split_df = pd.read_csv(split_csv)
            if "image_path" not in split_df.columns:
                raise ValueError(f"Split CSV must contain column 'image_path': {split_csv}")

            split_paths = set(split_df["image_path"].astype(str).tolist())
            self.dataframe = self.dataframe[self.dataframe["image_path"].astype(str).isin(split_paths)]
            self.dataframe = self.dataframe.reset_index(drop=True)

            if self.dataframe.empty:
                raise ValueError(
                    "No matching keypoint samples found after applying split CSV filter."
                )
            print(f"Applied split CSV filter: {split_csv}")

        print(f"Keypoint data loading complete. Total samples: {len(self.dataframe)}")

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

    def __getitem__(self, idx: int) -> dict:
        record = self.dataframe.iloc[idx]
        task_id = record["task_id"]
        image_rel_path = record["image_path"]
        image_abs_path = self._resolve_image_path(image_rel_path)
        if image_abs_path is None:
            return self.__getitem__((idx + 1) % len(self))

        image = cv2.imread(image_abs_path)
        if image is None:
            return self.__getitem__((idx + 1) % len(self))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        return {
            "image": image,
            "task_id": task_id,
            "task_name": "Regression",
            "image_path": image_rel_path,
            "original_size": (original_height, original_width),
            "index": idx,
        }


def inference_collate_fn(batch):
    images = torch.stack([item["image"] for item in batch], 0)
    return {
        "image": images,
        "task_id": [item["task_id"] for item in batch],
        "task_name": [item["task_name"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "original_size": [item["original_size"] for item in batch],
        "index": [item["index"] for item in batch],
    }


class Model:
    """Inference model for keypoint localization only."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = None
        self.task_configs = None
        self.task_id_to_name = None
        self.heatmap_size = (64, 64)
        self.input_size = 518

        self.transforms = A.Compose(
            [
                A.Resize(self.input_size, self.input_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def _build_task_configs(self, dataframe: pd.DataFrame):
        task_configs = []
        seen = set()
        for _, row in dataframe.iterrows():
            task_id = row["task_id"]
            if task_id in seen:
                continue
            seen.add(task_id)
            task_configs.append(
                {
                    "task_id": task_id,
                    "task_name": "Regression",
                    "num_classes": int(row["num_classes"]),
                }
            )
        return task_configs

    def _load_model(self):
        self.model = MultiTaskModelFactory(
            encoder_name="vit_small_patch14_dinov2.lvd142m",
            encoder_weights="pretrained",
            task_configs=self.task_configs,
            heatmap_size=self.heatmap_size,
        ).to(self.device)

        model_path = "best_model.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def predict(
        self,
        data_root: str,
        output_dir: str,
        batch_size: int = 8,
        split_csv: Optional[str] = None,
    ):
        print("=" * 60)
        print("Starting keypoint prediction...")
        print(f"Data directory: {data_root}")
        print(f"Output directory: {output_dir}")
        print("=" * 60)

        os.makedirs(output_dir, exist_ok=True)
        dataset = InferenceDataset(data_root=data_root, transforms=self.transforms, split_csv=split_csv)

        self.task_configs = self._build_task_configs(dataset.dataframe)
        self.task_id_to_name = {cfg["task_id"]: cfg["task_name"] for cfg in self.task_configs}
        self._load_model()

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=inference_collate_fn,
        )

        regression_results = []
        task_counts = {}

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Prediction progress"):
                images = batch["image"].to(self.device)
                task_ids = batch["task_id"]
                image_paths = batch["image_path"]
                original_sizes = batch["original_size"]

                unique_tasks = list(set(task_ids))
                for task_id in unique_tasks:
                    task_indices = [i for i, tid in enumerate(task_ids) if tid == task_id]
                    task_images = images[task_indices]
                    pred_logits = self.model(task_images, task_id=task_id)
                    pred_heatmaps = torch.sigmoid(pred_logits)
                    outputs = decode_heatmaps_to_normalized_coords(pred_heatmaps)

                    for i, batch_idx in enumerate(task_indices):
                        pred = outputs[i]
                        image_path = image_paths[batch_idx]
                        original_size = original_sizes[batch_idx]
                        task_counts[task_id] = task_counts.get(task_id, 0) + 1
                        regression_results.append(
                            self._process_regression(pred, task_id, image_path, original_size)
                        )

        json_path = os.path.join(output_dir, "regression_predictions.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(regression_results, f, indent=2, ensure_ascii=False)

        print(f"Saved keypoint predictions: {json_path} ({len(regression_results)} samples)")
        print("Prediction count by task:")
        for task_id in sorted(task_counts.keys()):
            print(f"  - {task_id}: {task_counts[task_id]} samples")

    def _process_regression(self, pred, task_id, image_path, original_size):
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()

        coords = pred.flatten().tolist()
        h, w = original_size
        pixel_coords = []
        for i in range(0, len(coords), 2):
            x_norm, y_norm = coords[i], coords[i + 1]
            pixel_coords.extend([x_norm * w, y_norm * h])

        return {
            "image_path": image_path,
            "task_id": task_id,
            "predicted_points_normalized": coords,
            "predicted_points_pixels": pixel_coords,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keypoint prediction script")
    parser.add_argument("--data-root", type=str, default="data", help="Dataset root directory")
    parser.add_argument("--output-dir", type=str, default="predictions/", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for full-dataset prediction")
    parser.add_argument("--split-csv", type=str, default=None, help="Optional split CSV path to restrict prediction set")
    args = parser.parse_args()

    model = Model()
    model.predict(
        args.data_root,
        args.output_dir,
        batch_size=args.batch_size,
        split_csv=args.split_csv,
    )

    print("Inference complete!")
