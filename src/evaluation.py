"""Segmentation evaluation for CAMUS.

Reports overlap (Dice, IoU), per-class error rates (TPR, misclassification)
and boundary accuracy (HD95, ASSD). Dice alone is a poor guide here: it
saturates on a large convex structure like the LV cavity while the endocardial
border - the thing LV volume and ejection fraction are derived from - can still
be several millimetres out.

The rate metrics separate the two ways a class can go wrong, which the overlap
scores blend together: TPR (sensitivity, recall) counts only what was missed,
so a model that over-predicts a class scores well on it, while the
misclassification rate counts missed *and* invented pixels against the true
extent of the class.

Distances are computed in millimetres using each sample's own pixel spacing, so
resizing in the data pipeline does not silently change what a "1.0" means.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import torch
from monai.metrics import (
    compute_average_surface_distance,
    compute_confusion_matrix_metric,
    compute_dice,
    compute_hausdorff_distance,
    compute_iou,
    get_confusion_matrix,
)
from monai.networks.utils import one_hot

from camus_dataset import LABELS, NUM_CLASSES

#: Metrics reported per class. Dice, IoU, TPR and the misclassification rate
#: are unitless; the distances are in mm.
METRIC_NAMES: tuple[str, ...] = ("dice", "iou", "tpr", "misclass", "hd95", "assd")

#: Metrics where a larger value is a better result. The rest - the distances and
#: the misclassification rate - are errors, so they sort the other way.
HIGHER_IS_BETTER: frozenset[str] = frozenset({"dice", "iou", "tpr"})


@dataclass
class CaseResult:
    """Metrics for one image, kept per case so results can be stratified."""

    key: str
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


class SegmentationEvaluator:
    """Accumulates per-case segmentation metrics and reduces them on demand.

    Aggregation is macro over cases (each patient-view-instant counts once),
    not micro over pooled pixels, so a few large images cannot dominate the
    headline number.

    Args:
        num_classes: Total classes including background.
        include_background: Keep class 0 in the reported metrics. Off by
            default - background Dice is ~0.98 for any non-degenerate
            prediction and only inflates the average.
        compute_distances: Compute HD95/ASSD. They are markedly slower than
            Dice, so they can be switched off for per-epoch validation and
            switched on for a final test run.
        hd_percentile: Percentile for the Hausdorff distance. 95 is standard;
            it discards the single worst outlier point, which is usually an
            annotation speck rather than a real boundary error.
        class_names: Optional override for the label names used in the report.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        include_background: bool = False,
        compute_distances: bool = True,
        hd_percentile: float = 95.0,
        class_names: Mapping[int, str] | None = None,
    ) -> None:
        self.num_classes = num_classes
        self.include_background = include_background
        self.compute_distances = compute_distances
        self.hd_percentile = hd_percentile

        names = dict(class_names or LABELS)
        first = 0 if include_background else 1
        #: Class indices the metrics correspond to, in column order.
        self.class_indices: tuple[int, ...] = tuple(range(first, num_classes))
        self.class_names: tuple[str, ...] = tuple(
            names.get(c, f"class_{c}") for c in self.class_indices
        )

        self.results: list[CaseResult] = []

    # ------------------------------------------------------------------ core

    def reset(self) -> None:
        self.results.clear()

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        spacing: torch.Tensor | None = None,
        meta: Mapping[str, Sequence[Any]] | None = None,
    ) -> None:
        """Score one batch.

        Args:
            logits: Raw model output ``(B, C, H, W)``. Argmax is taken here, so
                no softmax is needed.
            labels: Integer label map ``(B, H, W)`` or ``(B, 1, H, W)``.
            spacing: ``(B, 2)`` mm per pixel as ``(row, col)``, matching the
                spatial axes of the tensors. If omitted, distances come out in
                pixels rather than millimetres.
            meta: Optional per-sample fields (``key``, ``view``, ``instant``,
                ``image_quality``, ...) carried into the per-case records so
                results can be broken down later.
        """
        if logits.ndim != 4:
            raise ValueError(f"Expected logits of shape (B, C, H, W), got {tuple(logits.shape)}.")
        if logits.shape[1] != self.num_classes:
            raise ValueError(
                f"logits has {logits.shape[1]} channels but evaluator expects "
                f"{self.num_classes} classes."
            )

        if labels.ndim == 4:
            labels = labels[:, 0]
        if labels.shape != logits.shape[:1] + logits.shape[2:]:
            raise ValueError(
                f"Label shape {tuple(labels.shape)} does not match logits "
                f"{tuple(logits.shape)}."
            )

        # Distance metrics fall through to numpy/scipy internally, and MPS
        # tensors cannot be handed to them, so everything is scored on CPU.
        pred = logits.argmax(dim=1, keepdim=True).cpu()
        target = labels.long().unsqueeze(1).cpu()

        pred_oh = one_hot(pred, num_classes=self.num_classes)
        target_oh = one_hot(target, num_classes=self.num_classes)

        dice = compute_dice(
            pred_oh, target_oh, include_background=self.include_background, ignore_empty=True
        )
        iou = compute_iou(
            pred_oh, target_oh, include_background=self.include_background, ignore_empty=True
        )

        # (B, C, 4) of [tp, fp, tn, fn]; the rate metrics are all read off this,
        # so they cost one pass rather than one pass each.
        confusion = get_confusion_matrix(
            pred_oh, target_oh, include_background=self.include_background
        )
        tpr = compute_confusion_matrix_metric("sensitivity", confusion)
        tp, fp, fn = confusion[..., 0], confusion[..., 1], confusion[..., 3]
        support = tp + fn
        # Misclassified pixels relative to the class's true size. Unlike 1 - TPR
        # this also charges for over-segmentation, so it can exceed 1 when the
        # model invents more pixels than it misses. NaN where the class is
        # absent from the ground truth, matching Dice/IoU's ignore_empty.
        misclass = torch.where(
            support > 0, (fp + fn) / support, torch.full_like(support, float("nan"))
        )

        if self.compute_distances:
            spacing_arg = self._spacing_arg(spacing, batch_size=pred.shape[0])
            hd95 = compute_hausdorff_distance(
                pred_oh,
                target_oh,
                include_background=self.include_background,
                percentile=self.hd_percentile,
                spacing=spacing_arg,
            )
            assd = compute_average_surface_distance(
                pred_oh,
                target_oh,
                include_background=self.include_background,
                symmetric=True,
                spacing=spacing_arg,
            )
        else:
            hd95 = assd = torch.full_like(dice, float("nan"))

        for i in range(dice.shape[0]):
            case_meta = {k: self._item(v, i) for k, v in (meta or {}).items()}
            case = CaseResult(key=str(case_meta.get("key", f"case_{len(self.results)}")))
            case.meta = case_meta
            for col, (index, name) in enumerate(zip(self.class_indices, self.class_names)):
                case.metrics[name] = {
                    "dice": float(dice[i, col]),
                    "iou": float(iou[i, col]),
                    "tpr": float(tpr[i, col]),
                    "misclass": float(misclass[i, col]),
                    "hd95": float(hd95[i, col]),
                    "assd": float(assd[i, col]),
                }
                case.metrics[name]["class_index"] = index
            self.results.append(case)

    # ------------------------------------------------------------ reduction

    def compute(self) -> dict[str, Any]:
        """Reduce the accumulated cases into a summary.

        Non-finite entries are excluded from the means and counted separately.
        They are not failures to hide: Dice is NaN when a class is absent from
        the ground truth, and HD95/ASSD are infinite when a class is missing
        from the prediction entirely, which is exactly the case worth knowing
        about.
        """
        if not self.results:
            raise RuntimeError("No cases accumulated - call update() before compute().")

        per_class: dict[str, dict[str, float]] = {}
        for name in self.class_names:
            per_class[name] = {}
            for metric in METRIC_NAMES:
                values = [case.metrics[name][metric] for case in self.results]
                mean, n_valid = _finite_mean(values)
                per_class[name][metric] = mean
                per_class[name][f"{metric}_n_invalid"] = len(values) - n_valid

        summary: dict[str, Any] = {"n_cases": len(self.results), "per_class": per_class}
        for metric in METRIC_NAMES:
            class_means = [per_class[name][metric] for name in self.class_names]
            summary[f"mean_{metric}"] = _finite_mean(class_means)[0]
        return summary

    def stratified(self, by: str) -> dict[Any, dict[str, Any]]:
        """Reduce separately for each value of a metadata field.

        ``by`` is a key passed through ``meta`` in :meth:`update`, e.g.
        ``"image_quality"``, ``"view"`` or ``"instant"``. Performance on Poor
        quality images is the number that decides whether a model is usable, and
        it is invisible in a single pooled average.
        """
        groups: dict[Any, list[CaseResult]] = defaultdict(list)
        for case in self.results:
            if by not in case.meta:
                raise KeyError(f"No metadata field {by!r} was recorded. Pass it via meta=.")
            groups[case.meta[by]].append(case)

        out: dict[Any, dict[str, Any]] = {}
        for value, cases in sorted(groups.items(), key=lambda kv: str(kv[0])):
            child = SegmentationEvaluator(
                num_classes=self.num_classes,
                include_background=self.include_background,
                compute_distances=self.compute_distances,
                hd_percentile=self.hd_percentile,
            )
            child.results = cases
            out[value] = child.compute()
        return out

    def worst_cases(self, n: int = 5, metric: str = "dice") -> list[tuple[str, float]]:
        """The n cases with the worst class-averaged `metric`, for eyeballing.

        "Worst" follows the metric: lowest first for Dice/IoU/TPR, highest first
        for the distances and the misclassification rate.
        """
        if metric not in METRIC_NAMES:
            raise ValueError(f"metric must be one of {METRIC_NAMES}, got {metric!r}.")
        scored = []
        for case in self.results:
            mean, n_valid = _finite_mean([case.metrics[c][metric] for c in self.class_names])
            if n_valid:
                scored.append((case.key, mean))
        scored.sort(key=lambda kv: kv[1], reverse=metric not in HIGHER_IS_BETTER)
        return scored[:n]

    # ---------------------------------------------------------------- output

    def report(self, summary: dict[str, Any] | None = None) -> str:
        """Human-readable table of the summary."""
        summary = summary or self.compute()
        width = max(len(n) for n in self.class_names) + 2
        rule = "-" * (width + 58)
        lines = [
            f"{'class':<{width}}{'Dice':>8}{'IoU':>8}{'TPR':>8}{'Misclass':>10}"
            f"{'HD95 (mm)':>12}{'ASSD (mm)':>12}",
            rule,
        ]
        for name in self.class_names:
            row = summary["per_class"][name]
            lines.append(
                f"{name:<{width}}{row['dice']:>8.4f}{row['iou']:>8.4f}{row['tpr']:>8.4f}"
                f"{row['misclass']:>10.4f}{row['hd95']:>12.3f}{row['assd']:>12.3f}"
            )
        lines.append(rule)
        lines.append(
            f"{'mean':<{width}}{summary['mean_dice']:>8.4f}{summary['mean_iou']:>8.4f}"
            f"{summary['mean_tpr']:>8.4f}{summary['mean_misclass']:>10.4f}"
            f"{summary['mean_hd95']:>12.3f}{summary['mean_assd']:>12.3f}"
        )
        lines.append(f"\ncases: {summary['n_cases']}")

        invalid = {
            f"{name}/{metric}": summary["per_class"][name][f"{metric}_n_invalid"]
            for name in self.class_names
            for metric in METRIC_NAMES
            if summary["per_class"][name][f"{metric}_n_invalid"]
        }
        if invalid:
            lines.append(f"non-finite values excluded from means: {invalid}")
        return "\n".join(lines)

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _spacing_arg(spacing: torch.Tensor | None, batch_size: int) -> list[list[float]] | None:
        """MONAI wants spacing as one sequence per batch item, in axis order."""
        if spacing is None:
            return None
        spacing = spacing.detach().cpu()
        if spacing.ndim == 1:
            spacing = spacing.unsqueeze(0).expand(batch_size, -1)
        if spacing.shape[0] != batch_size:
            raise ValueError(
                f"spacing has batch size {spacing.shape[0]}, expected {batch_size}."
            )
        return [[float(v) for v in row] for row in spacing]

    @staticmethod
    def _item(values: Sequence[Any], index: int) -> Any:
        value = values[index]
        return value.item() if isinstance(value, torch.Tensor) and value.ndim == 0 else value


def _finite_mean(values: Iterable[float]) -> tuple[float, int]:
    """Mean over finite entries, plus how many there were."""
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan"), 0
    return sum(finite) / len(finite), len(finite)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    compute_distances: bool = True,
    include_background: bool = False,
) -> SegmentationEvaluator:
    """Run `model` over `loader` and return the populated evaluator."""
    evaluator = SegmentationEvaluator(
        include_background=include_background, compute_distances=compute_distances
    )
    model.eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        evaluator.update(
            logits=logits,
            labels=batch["label"],
            spacing=batch.get("spacing"),
            meta={
                key: batch[key]
                for key in ("key", "patient", "view", "instant", "image_quality")
                if key in batch
            },
        )
    return evaluator


if __name__ == "__main__":
    # Sanity check on synthetic data: a perfect prediction must score Dice 1.0
    # and zero distance, and a shifted one must degrade in a way we can predict.
    torch.manual_seed(0)
    height = width = 64
    target = torch.zeros(2, height, width, dtype=torch.long)
    target[:, 20:44, 20:44] = 1
    target[:, 12:20, 20:44] = 2
    target[:, 44:52, 20:44] = 3

    def logits_from(label_map: torch.Tensor) -> torch.Tensor:
        return one_hot(label_map.unsqueeze(1), num_classes=NUM_CLASSES) * 10.0

    spacing = torch.tensor([[0.5, 0.5], [0.5, 0.5]])

    perfect = SegmentationEvaluator()
    perfect.update(logits_from(target), target, spacing=spacing, meta={"key": ["a", "b"]})
    summary = perfect.compute()
    print("perfect prediction")
    print(perfect.report(summary))
    assert abs(summary["mean_dice"] - 1.0) < 1e-6, summary["mean_dice"]
    assert abs(summary["mean_iou"] - 1.0) < 1e-6, summary["mean_iou"]
    assert abs(summary["mean_tpr"] - 1.0) < 1e-6, summary["mean_tpr"]
    assert summary["mean_misclass"] == 0.0, summary["mean_misclass"]
    assert summary["mean_hd95"] == 0.0 and summary["mean_assd"] == 0.0

    # Shift by 2 px at 0.5 mm/px -> boundary error of 1.0 mm.
    shifted = torch.roll(target, shifts=2, dims=2)
    degraded = SegmentationEvaluator()
    degraded.update(logits_from(shifted), target, spacing=spacing, meta={"key": ["a", "b"]})
    summary = degraded.compute()
    print("\n2 px shift at 0.5 mm/px")
    print(degraded.report(summary))
    assert abs(summary["mean_hd95"] - 1.0) < 1e-6, summary["mean_hd95"]

    # The shift moves whole columns of each 24-wide block off their true
    # position, so exactly 2/24 of every class is missed and the same number
    # invented: TPR 22/24 and a misclassification rate of 4/24.
    for name in degraded.class_names:
        row = summary["per_class"][name]
        assert abs(row["tpr"] - 22 / 24) < 1e-6, (name, row["tpr"])
        assert abs(row["misclass"] - 4 / 24) < 1e-6, (name, row["misclass"])
        # IoU and Dice measure the same overlap on different scales.
        assert abs(row["iou"] - row["dice"] / (2 - row["dice"])) < 1e-6, name

    # A class the model over-predicts keeps a perfect TPR but not a perfect
    # misclassification rate - the pairing the report is there to expose.
    greedy = target.clone()
    greedy[:, 16:48, 16:48] = 1
    over = SegmentationEvaluator(compute_distances=False)
    over.update(logits_from(greedy), target, spacing=spacing, meta={"key": ["a", "b"]})
    row = over.compute()["per_class"]["lv_endocardium"]
    assert abs(row["tpr"] - 1.0) < 1e-6, row["tpr"]
    assert row["misclass"] > 0.0 and row["dice"] < 1.0, row

    print("\nmetric sanity checks passed")
