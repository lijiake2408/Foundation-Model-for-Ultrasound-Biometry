import glob
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict


EXTRA_REGRESSION_TASK_IDS = {"A4C", "AOP", "FA", "HC", "IVC", "PLAX", "PSAX"}


class Visualizer:
    """Visualizer for keypoint localization tasks only."""

    def __init__(self, data_root: str, pred_root: str):
        self.data_root = data_root
        self.pred_root = pred_root
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

        self.task_configs = {}
        for _, row in self.dataframe.iterrows():
            task_id = row["task_id"]
            if task_id not in self.task_configs:
                self.task_configs[task_id] = int(row["num_classes"])

    def _resolve_image_path(self, rel_path: str):
        rel_norm = os.path.normpath(rel_path)
        cleaned_rel = rel_norm
        while cleaned_rel.startswith(".." + os.sep):
            cleaned_rel = cleaned_rel[3:]

        for root in [os.path.join(self.data_root, "images"), self.data_root]:
            direct = os.path.normpath(os.path.join(root, cleaned_rel))
            if os.path.isfile(direct):
                return direct

        return None

    def visualize_all(self, output_dir: str, samples_per_task: int = 1):
        os.makedirs(output_dir, exist_ok=True)

        pred_file = os.path.join(self.pred_root, "regression_predictions.json")
        if not os.path.exists(pred_file):
            raise FileNotFoundError(f"Prediction file not found: {pred_file}")

        with open(pred_file, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        pred_dict = defaultdict(dict)
        for pred in predictions:
            pred_dict[pred["task_id"]][pred["image_path"]] = pred["predicted_points_pixels"]

        for task_id in sorted(self.task_configs.keys()):
            task_data = self.dataframe[self.dataframe["task_id"] == task_id]
            num_points = self.task_configs[task_id]
            samples = task_data.sample(min(samples_per_task, len(task_data)), random_state=42)

            for idx, (_, row) in enumerate(samples.iterrows()):
                image_path = row["image_path"]
                if image_path not in pred_dict[task_id]:
                    continue

                img_path = self._resolve_image_path(image_path)
                if img_path is None:
                    continue
                image = cv2.imread(img_path)
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                gt_coords = []
                for i in range(1, num_points + 1):
                    col = f"point_{i}_xy"
                    if col in row and pd.notna(row[col]):
                        gt_coords.extend(json.loads(row[col]))
                    else:
                        gt_coords.extend([0.0, 0.0])

                pred_coords = pred_dict[task_id][image_path]

                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                gt_points = np.array(gt_coords).reshape(-1, 2)
                pred_points = np.array(pred_coords).reshape(-1, 2)

                axes[0].imshow(image)
                axes[0].scatter(gt_points[:, 0], gt_points[:, 1], c="lime", s=80, edgecolors="black")
                axes[0].set_title("Ground Truth Keypoints")
                axes[0].axis("off")

                axes[1].imshow(image)
                axes[1].scatter(pred_points[:, 0], pred_points[:, 1], c="red", s=80, marker="x")
                axes[1].set_title("Predicted Keypoints")
                axes[1].axis("off")

                plt.suptitle(f"{task_id} (points={num_points})")
                plt.tight_layout()

                save_path = os.path.join(output_dir, f"{task_id}_sample_{idx}.png")
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()


if __name__ == "__main__":
    data_root = "data"
    pred_root = "predictions/"
    output_dir = "visualizations/"

    visualizer = Visualizer(data_root, pred_root)
    visualizer.visualize_all(output_dir=output_dir, samples_per_task=5)
