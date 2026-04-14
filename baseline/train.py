import argparse
from collections import defaultdict

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from dataset import KeypointDataset, KeypointUniformSampler
from model_factory import MultiTaskModelFactory
from utils import evaluate_keypoint, keypoint_collate_fn, set_seed


LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS =40
DATA_ROOT_PATH = "data"
ENCODER = "vit_small_patch14_dinov2.lvd142m"
ENCODER_WEIGHTS = "pretrained"
RANDOM_SEED = 42
MODEL_SAVE_PATH = "best_model.pth"
VAL_SPLIT = 0.2
HEATMAP_SIZE = (64, 64)
HEATMAP_SIGMA = 1.8
INPUT_SIZE = 518
EXTRA_REGRESSION_TASK_IDS = {"A4C", "AOP", "FA", "HC", "IVC", "PLAX", "PSAX"}


def _build_task_configs(dataframe):
    configs = []
    seen = set()
    for _, row in dataframe.iterrows():
        task_name = str(row["task_name"])
        task_id = str(row["task_id"])
        if task_name != "Regression" and task_id not in EXTRA_REGRESSION_TASK_IDS:
            continue
        if task_id in seen:
            continue
        seen.add(task_id)
        configs.append(
            {
                "task_id": task_id,
                "task_name": "Regression",
                "num_classes": int(row["num_classes"]),
            }
        )
    if not configs:
        raise ValueError("No keypoint tasks found in dataset.")
    return configs


def _stratified_split_indices(dataframe, val_split: float, seed: int):
    if not (0.0 < float(val_split) < 1.0):
        raise ValueError("val_split must be in (0, 1).")

    rng = np.random.RandomState(seed)
    train_indices = []
    val_indices = []

    for _, group in dataframe.groupby("task_id", sort=True):
        indices = np.array(group.index.to_numpy(), copy=True)
        rng.shuffle(indices)

        total = len(indices)
        # Per-task split count (rounded) to keep each task close to the requested ratio.
        val_count = int(round(total * float(val_split)))
        if total >= 2:
            val_count = max(1, min(total - 1, val_count))
        else:
            val_count = 0

        val_indices.extend(indices[:val_count].tolist())
        train_indices.extend(indices[val_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def main(val_split: float = VAL_SPLIT):
    metric_column = "MRE (pixels)"
    metric_label = "MRE"

    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    train_transforms = A.Compose(
        [
            A.Resize(INPUT_SIZE, INPUT_SIZE),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    val_transforms = A.Compose(
        [
            A.Resize(INPUT_SIZE, INPUT_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    temp_dataset = KeypointDataset(
        data_root=DATA_ROOT_PATH,
        transforms=train_transforms,
        heatmap_size=HEATMAP_SIZE,
        sigma=HEATMAP_SIGMA,
    )

    task_configs = _build_task_configs(temp_dataset.dataframe)
    task_id_to_name = {cfg["task_id"]: cfg["task_name"] for cfg in task_configs}

    train_indices, val_indices = _stratified_split_indices(
        temp_dataset.dataframe,
        val_split=val_split,
        seed=RANDOM_SEED,
    )
    train_size = len(train_indices)
    val_size = len(val_indices)

    train_dataset = KeypointDataset(
        data_root=DATA_ROOT_PATH,
        transforms=train_transforms,
        heatmap_size=HEATMAP_SIZE,
        sigma=HEATMAP_SIGMA,
    )
    train_dataset.dataframe = temp_dataset.dataframe.reset_index(drop=True)

    val_dataset = KeypointDataset(
        data_root=DATA_ROOT_PATH,
        transforms=val_transforms,
        heatmap_size=HEATMAP_SIZE,
        sigma=HEATMAP_SIGMA,
    )
    val_dataset.dataframe = temp_dataset.dataframe.reset_index(drop=True)

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    print(
        f"Dataset split (per-task stratified, val_split={val_split:.3f}): "
        f"{train_size} training samples, {val_size} validation samples"
    )

    train_subset.dataframe = train_dataset.dataframe.iloc[train_indices].reset_index(drop=True)

    train_sampler = KeypointUniformSampler(train_subset, batch_size=BATCH_SIZE)
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=keypoint_collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=keypoint_collate_fn,
    )

    model = MultiTaskModelFactory(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        task_configs=task_configs,
        heatmap_size=HEATMAP_SIZE,
    ).to(device)

    param_groups = [{"params": model.encoder.parameters(), "lr": LEARNING_RATE * 0.2}]
    for task_id, head in model.heads.items():
        param_groups.append({"params": head.parameters(), "lr": LEARNING_RATE * 10.0})
        print(f"Task head {task_id} LR: {LEARNING_RATE * 10.0}")

    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val_score = float("inf")
    print(f"Best-checkpoint metric: {metric_label} (lower is better)")
    print("\n" + "=" * 50 + "\n--- Start Keypoint Training ---")

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_losses = defaultdict(list)
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]")

        for batch in loop:
            images = batch["image"].to(device)
            task_ids = batch["task_id"]
            batch_loss_values = []

            # Handle mixed-task batches safely (different tasks can have different keypoint counts).
            for current_task_id in sorted(set(task_ids)):
                task_indices = [i for i, tid in enumerate(task_ids) if tid == current_task_id]
                task_images = images[task_indices]
                task_heatmaps = torch.stack([batch["heatmap"][i] for i in task_indices], 0).to(device)

                pred_logits = model(task_images, task_id=current_task_id)
                pred_heatmaps = torch.sigmoid(pred_logits)
                loss = F.mse_loss(pred_heatmaps, task_heatmaps)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_value = float(loss.item())
                batch_loss_values.append(loss_value)
                epoch_train_losses[current_task_id].append(loss_value)

            mean_batch_loss = float(np.mean(batch_loss_values)) if batch_loss_values else 0.0
            loop.set_postfix(loss=mean_batch_loss, groups=len(set(task_ids)), lr=scheduler.get_last_lr()[0])

        print(f"\n--- Epoch {epoch + 1} Average Train Loss ---")
        for task_id in sorted(epoch_train_losses.keys()):
            avg_loss = float(np.mean(epoch_train_losses[task_id]))
            print(f"  - {task_id}: {avg_loss:.4f}")

        val_results_df = evaluate_keypoint(model, val_loader, device, task_id_to_name)
        selected_val_score = float("inf")
        if not val_results_df.empty and metric_column in val_results_df.columns:
            selected_val_score = float(val_results_df[metric_column].mean())

        print(f"\n--- Epoch {epoch + 1} Validation Report ---")
        if not val_results_df.empty:
            print(val_results_df.to_string(index=False))
        print(f"--- Average Val {metric_label} (Lower is better): {selected_val_score:.4f} ---")

        if selected_val_score < best_val_score:
            best_val_score = selected_val_score
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"-> New best model saved! {metric_label} improved to: {best_val_score:.4f}\n")

        scheduler.step()

    print(f"\n--- Training Finished ---\nBest model saved at: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keypoint training script")
    parser.add_argument(
        "--val-split",
        type=float,
        default=VAL_SPLIT,
        help="Validation ratio per task (0~1), e.g. 0.2 means each task uses 20% for validation.",
    )
    args = parser.parse_args()

    main(val_split=float(args.val_split))
