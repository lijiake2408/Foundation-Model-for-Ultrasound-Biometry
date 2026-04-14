from typing import Dict, List

import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F


EXTRA_REGRESSION_TASK_IDS = {"A4C", "AOP", "FA", "HC", "IVC", "PLAX", "PSAX"}


class HeatmapHead(nn.Module):
    """Light decoder that maps DINOv2 feature maps to keypoint heatmaps."""

    def __init__(self, in_channels: int, num_points: int):
        super().__init__()
        hidden = max(in_channels // 2, 128)
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden, hidden // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden // 2),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden // 2, num_points, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, out_size) -> torch.Tensor:
        x = self.decoder(x)
        if x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


class DINOv2Backbone(nn.Module):
    """Returns the last patch feature map as a 2D tensor [B, C, H, W]."""

    def __init__(self, model_name: str = "vit_small_patch14_dinov2.lvd142m", pretrained: bool = True):
        super().__init__()
        timm = importlib.import_module("timm")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        if not hasattr(self.backbone, "patch_embed"):
            raise ValueError(f"Model '{model_name}' is not a ViT-style backbone with patch_embed.")
        self.out_channels = int(self.backbone.num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)

        if isinstance(feats, dict):
            if "x_norm_patchtokens" in feats:
                patch_tokens = feats["x_norm_patchtokens"]
            elif "x_prenorm" in feats:
                all_tokens = feats["x_prenorm"]
                patch_tokens = all_tokens[:, 1:, :]
            else:
                raise RuntimeError("Unsupported forward_features output from DINOv2 backbone.")
        elif isinstance(feats, torch.Tensor):
            if feats.dim() == 3:
                patch_tokens = feats[:, 1:, :]
            else:
                raise RuntimeError("Unexpected tensor shape from forward_features.")
        else:
            raise RuntimeError("Unexpected feature type from DINOv2 backbone.")

        bsz, num_tokens, channels = patch_tokens.shape
        side = int(num_tokens ** 0.5)
        if side * side != num_tokens:
            raise RuntimeError("Patch token count is not square; input size may be incompatible.")

        feat_map = patch_tokens.transpose(1, 2).reshape(bsz, channels, side, side)
        return feat_map


class MultiTaskModelFactory(nn.Module):
    """Keypoint heatmap model using a shared DINOv2 encoder and task-specific heads."""

    def __init__(
        self,
        encoder_name: str,
        encoder_weights: str,
        task_configs: List[Dict],
        heatmap_size=(64, 64),
    ):
        super().__init__()

        self.heatmap_size = heatmap_size

        print(f"Initializing DINOv2 encoder: {encoder_name}")
        self.encoder = DINOv2Backbone(model_name=encoder_name, pretrained=(encoder_weights is not None))

        self.heads = nn.ModuleDict()
        print(f"Creating keypoint heads for {len(task_configs)} tasks...")

        for config in task_configs:
            task_id = config["task_id"]
            task_name = config["task_name"]
            if task_name != "Regression" and task_id not in EXTRA_REGRESSION_TASK_IDS:
                continue

            num_points = int(config["num_classes"])
            self.heads[task_id] = HeatmapHead(in_channels=self.encoder.out_channels, num_points=num_points)

        if not self.heads:
            raise ValueError("No keypoint heads were created. Check task_configs with task_name == 'Regression'.")

    def forward(self, x: torch.Tensor, task_id: str) -> torch.Tensor:
        if task_id not in self.heads:
            raise ValueError(f"Task ID '{task_id}' not found in keypoint heads.")

        features = self.encoder(x)
        return self.heads[task_id](features, out_size=self.heatmap_size)
