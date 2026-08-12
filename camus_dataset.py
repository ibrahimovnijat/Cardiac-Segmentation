"""CAMUS 2D echocardiography dataset (ED/ES frames with expert segmentations).

Layout expected under ``data_folder`` (the ``CAMUS_public`` root)::

    database_nifti/patientXXXX/patientXXXX_{2CH,4CH}_{ED,ES}.nii.gz       # B-mode image
    database_nifti/patientXXXX/patientXXXX_{2CH,4CH}_{ED,ES}_gt.nii.gz    # label map
    database_nifti/patientXXXX/Info_{2CH,4CH}.cfg                         # per-view metadata
    database_split/subgroup_{training,validation,testing}.txt             # official patient lists

One sample is one (patient, view, instant) triplet, so the full dataset is
500 patients x 2 views x 2 instants = 2000 images.

Verified properties of this copy of the dataset (500 patients, all complete):
  * pixel spacing is 0.308 x 0.308 mm for every one of the 2000 ED/ES images
  * image sizes vary per patient (~(390, 472) to (519, 630)), hence the resize
  * labels are exactly {0, 1, 2, 3} in every ground-truth file
  * images and labels are stored as float32 in [0, 255] and [0, 3] respectively
  * the official splits are patient-disjoint and cover all 500 patients
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

View = Literal["2CH", "4CH"]
Instant = Literal["ED", "ES"]
Split = Literal["train", "val", "test", "all"]
Normalize = Literal["unit", "zscore", "none"]

#: Label map semantics. Verified empirically: class 1 shrinks from ED to ES in
#: 20/20 sampled patients (the contracting cavity) while class 3 grows (the
#: atrium filling during systole).
LABELS: dict[int, str] = {
    0: "background",
    1: "lv_endocardium",  # LV cavity / blood pool
    2: "lv_myocardium",   # wall between endo- and epicardial borders
    3: "left_atrium",
}
NUM_CLASSES = len(LABELS)

_SPLIT_FILES: dict[str, str] = {
    "train": "subgroup_training.txt",
    "val": "subgroup_validation.txt",
    "test": "subgroup_testing.txt",
}


@dataclass(frozen=True)
class SampleKey:
    """Identifies one image in the dataset."""

    patient: str
    view: str
    instant: str

    def __str__(self) -> str:
        return f"{self.patient}_{self.view}_{self.instant}"


class CAMUSDataset(torch.utils.data.Dataset):
    """CAMUS ED/ES frames paired with their expert label maps.

    Args:
        data_folder: Path to the ``CAMUS_public`` root (the folder holding
            ``database_nifti/``), or to ``database_nifti/`` itself.
        split: Which official patient subgroup to use. The splits ship with the
            dataset and are patient-disjoint, which is what keeps a patient's
            four images from straddling train and validation.
        views: Apical views to include.
        instants: Cardiac instants to include.
        image_size: ``(height, width)`` every sample is resized to, or ``None``
            to keep native resolution. Native sizes differ between patients, so
            ``None`` requires ``batch_size=1`` or a padding collate function.
        normalize: ``"unit"`` scales the stored 0-255 range to [0, 1];
            ``"zscore"`` standardises per image (mean 0, std 1); ``"none"``
            leaves the raw intensities.
        transform: Optional callable applied to the finished sample dict. It
            receives and must return the whole dict, so it can transform image
            and label jointly - which any geometric augmentation must.
        cache: Keep decoded arrays in memory after first read. With
            ``num_workers > 0`` each worker holds its own copy.

    Each item is a dict with:
        ``image``          FloatTensor ``(1, H, W)``
        ``label``          LongTensor ``(H, W)`` with values in ``{0, 1, 2, 3}``
        ``spacing``        FloatTensor ``(2,)``, mm per pixel as ``(row, col)``,
                           corrected for the resize so millimetre-denominated
                           metrics (HD95, ASSD, volumes) stay valid
        ``original_size``  LongTensor ``(2,)``, the native ``(H, W)``
        ``key``/``patient``/``view``/``instant``  identifiers
        ``ef``, ``image_quality``, ``sex``, ``age``, ``frame_rate``,
        ``nb_frame``, ``ed_frame``, ``es_frame``  per-view metadata from the cfg
    """

    def __init__(
        self,
        data_folder: str | Path,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        *,
        split: Split = "all",
        views: Sequence[View] = ("2CH", "4CH"),
        instants: Sequence[Instant] = ("ED", "ES"),
        image_size: tuple[int, int] | None = (256, 256),
        normalize: Normalize = "unit",
        cache: bool = False,
    ) -> None:
        self.root, self.nifti_dir = self._resolve_root(Path(data_folder).expanduser())
        self.transform = transform
        self.split = split
        self.views = tuple(views)
        self.instants = tuple(instants)
        self.image_size = None if image_size is None else tuple(image_size)
        self.normalize = normalize

        if not self.views or not self.instants:
            raise ValueError("`views` and `instants` must each contain at least one entry.")
        if bad := set(self.views) - {"2CH", "4CH"}:
            raise ValueError(f"Unknown view(s): {sorted(bad)}. Expected '2CH' and/or '4CH'.")
        if bad := set(self.instants) - {"ED", "ES"}:
            raise ValueError(f"Unknown instant(s): {sorted(bad)}. Expected 'ED' and/or 'ES'.")
        if self.normalize not in ("unit", "zscore", "none"):
            raise ValueError(f"normalize must be 'unit', 'zscore' or 'none', got {normalize!r}.")

        self.patients = self._load_patients(split)
        self.samples = [
            SampleKey(patient, view, instant)
            for patient in self.patients
            for view in self.views
            for instant in self.instants
        ]
        if not self.samples:
            raise RuntimeError(f"No samples found under {self.nifti_dir} for split={split!r}.")

        self._array_cache: dict[Path, tuple[np.ndarray, tuple[float, float]]] | None = (
            {} if cache else None
        )
        self._cfg_cache: dict[tuple[str, str], dict[str, str]] = {}

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _resolve_root(path: Path) -> tuple[Path, Path]:
        """Accept either the CAMUS_public root or the database_nifti folder."""
        if (path / "database_nifti").is_dir():
            return path, path / "database_nifti"
        if path.name == "database_nifti" and path.is_dir():
            return path.parent, path
        raise FileNotFoundError(
            f"{path} does not look like the CAMUS root: expected a 'database_nifti' "
            f"subfolder, or the 'database_nifti' folder itself."
        )

    def _load_patients(self, split: Split) -> list[str]:
        """Patient ids for `split`, taken from the official patient-level lists."""
        on_disk = sorted(p.name for p in self.nifti_dir.iterdir() if p.name.startswith("patient"))
        if split == "all":
            return on_disk

        if split not in _SPLIT_FILES:
            raise ValueError(f"split must be one of {[*_SPLIT_FILES, 'all']}, got {split!r}.")
        split_file = self.root / "database_split" / _SPLIT_FILES[split]
        if not split_file.is_file():
            raise FileNotFoundError(
                f"Official split file {split_file} not found. Pass split='all' to use every "
                f"patient on disk, but note that any split you build yourself must be made "
                f"patient-level to avoid leaking a patient across folds."
            )
        patients = split_file.read_text().split()
        if missing := sorted(set(patients) - set(on_disk)):
            raise FileNotFoundError(
                f"{len(missing)} patient(s) listed in {split_file.name} are absent from "
                f"{self.nifti_dir}: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        return patients

    # -------------------------------------------------------------------- io

    def _paths(self, key: SampleKey) -> tuple[Path, Path]:
        folder = self.nifti_dir / key.patient
        return (
            folder / f"{key}.nii.gz",
            folder / f"{key}_gt.nii.gz",
        )

    def _read(self, path: Path) -> tuple[np.ndarray, tuple[float, float]]:
        """Read a 2D NIfTI as ``(array_hw, (row_mm, col_mm))``.

        SimpleITK reports spacing in ``(x, y)`` = ``(column, row)`` order while
        the array comes back as ``(row, column)``, so the spacing is reversed
        here to match the array axes.
        """
        if self._array_cache is not None and path in self._array_cache:
            return self._array_cache[path]

        if not path.is_file():
            raise FileNotFoundError(f"Missing CAMUS file: {path}")
        image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(image)
        if array.ndim != 2:
            raise ValueError(
                f"Expected a 2D image at {path.name}, got shape {array.shape}. The "
                f"'half_sequence' files are 3D (frames, H, W) and are not handled here."
            )
        spacing_xy = image.GetSpacing()
        result = (array, (float(spacing_xy[1]), float(spacing_xy[0])))
        if self._array_cache is not None:
            self._array_cache[path] = result
        return result

    def _metadata(self, patient: str, view: str) -> dict[str, str]:
        """Parse ``Info_{view}.cfg`` (``key: value`` per line), cached per view."""
        if (cached := self._cfg_cache.get((patient, view))) is not None:
            return cached

        cfg_path = self.nifti_dir / patient / f"Info_{view}.cfg"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing metadata file: {cfg_path}")
        cfg: dict[str, str] = {}
        for line in cfg_path.read_text().splitlines():
            if not line.strip():
                continue
            field, _, value = line.partition(":")
            cfg[field.strip()] = value.strip()
        self._cfg_cache[(patient, view)] = cfg
        return cfg

    # ------------------------------------------------------------ processing

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        if self.normalize == "unit":
            return image / 255.0
        if self.normalize == "zscore":
            std = image.std()
            # A constant image would divide by zero; leave it centred instead.
            return (image - image.mean()) / std if std > 0 else image - image.mean()
        return image

    def _resize(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        spacing: tuple[float, float],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[float, float]]:
        """Resize to ``self.image_size`` and rescale the spacing to match.

        The label uses nearest-exact interpolation: anything smoother would
        invent fractional classes along the boundaries.
        """
        if self.image_size is None:
            return image, label, spacing

        height, width = image.shape[-2:]
        target_h, target_w = self.image_size
        if (height, width) == (target_h, target_w):
            return image, label, spacing

        image = F.interpolate(
            image[None],
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
            antialias=target_h < height or target_w < width,
        )[0]
        label = F.interpolate(
            label[None, None].float(),
            size=self.image_size,
            mode="nearest-exact",
        )[0, 0].long()

        # A pixel now covers more (or less) tissue than it did natively.
        spacing = (spacing[0] * height / target_h, spacing[1] * width / target_w)
        return image, label, spacing

    # -------------------------------------------------------------- protocol

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        key = self.samples[idx]
        image_path, label_path = self._paths(key)
        image_array, spacing = self._read(image_path)
        label_array, label_spacing = self._read(label_path)

        if image_array.shape != label_array.shape:
            raise ValueError(
                f"Image/label shape mismatch for {key}: "
                f"{image_array.shape} vs {label_array.shape}"
            )
        if not np.allclose(spacing, label_spacing):
            raise ValueError(
                f"Image/label spacing mismatch for {key}: {spacing} vs {label_spacing}"
            )

        original_size = image_array.shape
        image = torch.from_numpy(image_array.astype(np.float32, copy=False))[None]
        label = torch.from_numpy(label_array.astype(np.int64, copy=False))

        if int(label.max()) >= NUM_CLASSES or int(label.min()) < 0:
            raise ValueError(
                f"Label values outside [0, {NUM_CLASSES - 1}] for {key}: "
                f"found {sorted(label.unique().tolist())}"
            )

        image = self._normalize(image)
        image, label, spacing = self._resize(image, label, spacing)

        cfg = self._metadata(key.patient, key.view)
        sample: dict[str, Any] = {
            "image": image.contiguous(),
            "label": label.contiguous(),
            "spacing": torch.tensor(spacing, dtype=torch.float32),
            "original_size": torch.tensor(original_size, dtype=torch.long),
            "key": str(key),
            "patient": key.patient,
            "view": key.view,
            "instant": key.instant,
            "ef": float(cfg["EF"]),
            "image_quality": cfg["ImageQuality"],
            "sex": cfg["Sex"],
            "age": float(cfg["Age"]),
            "frame_rate": float(cfg["FrameRate"]),
            "nb_frame": int(cfg["NbFrame"]),
            "ed_frame": int(cfg["ED"]),
            "es_frame": int(cfg["ES"]),
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(split={self.split!r}, patients={len(self.patients)}, "
            f"samples={len(self.samples)}, views={self.views}, instants={self.instants}, "
            f"image_size={self.image_size}, normalize={self.normalize!r})"
        )

    # --------------------------------------------------------------- helpers

    def class_pixel_counts(self, indices: Iterable[int] | None = None) -> torch.Tensor:
        """Pixel count per class, for inspecting imbalance or weighting a loss."""
        counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
        for idx in range(len(self)) if indices is None else indices:
            counts += torch.bincount(self[idx]["label"].flatten(), minlength=NUM_CLASSES)
        return counts


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent / "CAMUS_public"

    for split in ("train", "val", "test"):
        dataset = CAMUSDataset(root, split=split)
        print(dataset)

    dataset = CAMUSDataset(root, split="train")
    sample = dataset[0]
    print(f"\nsample {sample['key']}")
    print(f"  image  {tuple(sample['image'].shape)} {sample['image'].dtype} "
          f"[{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"  label  {tuple(sample['label'].shape)} {sample['label'].dtype} "
          f"classes={sorted(sample['label'].unique().tolist())}")
    print(f"  native {tuple(sample['original_size'].tolist())} -> spacing "
          f"{[round(s, 4) for s in sample['spacing'].tolist()]} mm/px")
    print(f"  meta   EF={sample['ef']} quality={sample['image_quality']} view={sample['view']}")

    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    print(f"\nbatch  image={tuple(batch['image'].shape)} label={tuple(batch['label'].shape)} "
          f"patients={batch['patient']}")

    counts = dataset.class_pixel_counts(range(200))
    fractions = counts / counts.sum()
    print("\nclass balance over 200 training images:")
    for index, name in LABELS.items():
        print(f"  {index} {name:<16s} {fractions[index]:6.2%}")
