from __future__ import annotations

from dataclasses import dataclass

import numpy as np


METRIC_KEYS = ("iou", "dice", "precision", "recall")


@dataclass(slots=True)
class SegmentationMetrics:
    iou: float
    dice: float
    precision: float
    recall: float

    def as_dict(self) -> dict[str, float]:
        return {
            "iou": self.iou,
            "dice": self.dice,
            "precision": self.precision,
            "recall": self.recall,
        }


def empty_metrics() -> dict[str, float]:
    return {key: float("nan") for key in METRIC_KEYS}


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> SegmentationMetrics:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    pred_area = int(pred.sum())
    gt_area = int(gt.sum())
    intersection = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())

    if pred_area == 0 and gt_area == 0:
        return SegmentationMetrics(iou=1.0, dice=1.0, precision=1.0, recall=1.0)

    iou = float(intersection / union) if union > 0 else 0.0
    dice_denominator = pred_area + gt_area
    dice = float((2 * intersection) / dice_denominator) if dice_denominator > 0 else 0.0
    precision = float(intersection / pred_area) if pred_area > 0 else 0.0
    recall = float(intersection / gt_area) if gt_area > 0 else 0.0
    return SegmentationMetrics(iou=iou, dice=dice, precision=precision, recall=recall)


def mask_area(mask: np.ndarray) -> int:
    return int(mask.astype(bool).sum())


def area_ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(mask_area(mask) / mask.size)
