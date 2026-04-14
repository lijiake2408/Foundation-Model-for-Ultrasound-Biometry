import glob
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm


EXTRA_REGRESSION_TASK_IDS = {"A4C", "AOP", "FA", "HC", "IVC", "PLAX", "PSAX"}


class Evaluator:
    """Evaluator for keypoint localization tasks only."""

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
                self.task_configs[task_id] = {
                    "task_name": "Regression",
                    "num_classes": int(row["num_classes"]),
                }

    def evaluate_all(self) -> Dict:
        return {"regression": self.evaluate_regression(sorted(self.task_configs.keys()))}

    def evaluate_regression(self, task_ids: List[str]) -> Dict:
        pred_file = os.path.join(self.pred_root, "regression_predictions.json")
        if not os.path.exists(pred_file):
            raise FileNotFoundError(f"Prediction file not found: {pred_file}")

        with open(pred_file, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        pred_dict = {}
        for pred in predictions:
            pred_dict.setdefault(pred["task_id"], {})[pred["image_path"]] = pred["predicted_points_pixels"]

        task_results = {}
        for task_id in tqdm(task_ids, desc="Keypoint tasks", unit="task"):
            task_data = self.dataframe[self.dataframe["task_id"] == task_id]
            num_points = self.task_configs[task_id]["num_classes"]

            mre_scores = []
            for _, row in tqdm(task_data.iterrows(), total=len(task_data), desc=f"  {task_id}", leave=False):
                image_path = row["image_path"]
                gt_coords = []
                for i in range(1, num_points + 1):
                    col = f"point_{i}_xy"
                    if col in row and pd.notna(row[col]):
                        gt_coords.extend(json.loads(row[col]))
                    else:
                        gt_coords.extend([0.0, 0.0])

                if image_path in pred_dict.get(task_id, {}):
                    pred_coords = pred_dict[task_id][image_path]
                    mre_scores.append(self._compute_mre(pred_coords, gt_coords))

            if mre_scores:
                task_results[task_id] = {
                    "MRE": float(np.mean(mre_scores)),
                    "num_samples": len(mre_scores),
                }
            else:
                task_results[task_id] = {"MRE": 0.0, "num_samples": 0}

        return task_results

    @staticmethod
    def _compute_mre(pred_coords: List[float], gt_coords: List[float]) -> float:
        pred_coords = np.array(pred_coords).reshape(-1, 2)
        gt_coords = np.array(gt_coords).reshape(-1, 2)
        distances = np.sqrt(np.sum((pred_coords - gt_coords) ** 2, axis=1))
        return float(np.mean(distances))

    def print_summary(self, results: Dict, save_path: str = None):
        reg_results = results.get("regression", {})
        all_mre = [r["MRE"] for r in reg_results.values() if r["num_samples"] > 0]

        lines = ["=" * 80, "Keypoint Evaluation Summary", "=" * 80]
        if all_mre:
            lines.append(f"Average MRE: {np.mean(all_mre):.4f}")
            lines.append(f"Number of tasks: {len(all_mre)}")
            for task_id, task_result in reg_results.items():
                if task_result["num_samples"] > 0:
                    lines.append(
                        f"  {task_id}: MRE={task_result['MRE']:.4f}, samples={task_result['num_samples']}"
                    )
        else:
            lines.append("No valid keypoint predictions found.")

        text = "\n".join(lines)
        print(text)

        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(text + "\n")

    @staticmethod
    def save_results(results: Dict, output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    data_root = "data"
    pred_root = "predictions/"
    output_file = "evaluation_results.json"
    summary_file = "evaluation_summary.txt"

    evaluator = Evaluator(data_root, pred_root)
    results = evaluator.evaluate_all()
    evaluator.print_summary(results, save_path=summary_file)
    evaluator.save_results(results, output_file)
