"""Left-ventricular volume and ejection fraction from 2D CAMUS segmentations.

Ejection fraction is what an echo report actually carries, and it is not a pixel
metric: it comes from the LV cavity volumes at end-diastole and end-systole,

    EF = (EDV - ESV) / EDV * 100

The volumes are estimated with Simpson's biplane method of disks - the ASE
standard, and the geometry CAMUS is built around. The cavity is sliced into
``DISKS`` discs perpendicular to the long axis, each disc treated as an ellipse
whose two diameters come from the 2CH and 4CH views of the same instant::

    V = pi/4 * sum_i (a_i * b_i) * (L / N)

Two consequences worth keeping in mind while reading any number this module
produces. The disc areas go as the *product* of two diameters, so a boundary
error near the base - where the discs are widest - costs far more volume than
the same error at the apex. And EF is a ratio of two such estimates, so errors
that push EDV and ESV the same way cancel, while an error that inflates only
one of them lands on EF at full strength.

A caveat that must travel with these numbers: the ``EF`` field in each
patient's ``Info_{view}.cfg`` is a clinical measurement, not this computation
applied to the reference contours. Run on the *expert* contours of the CAMUS
test split it lands +7.2 EF points above the cfg value (sd 3.2, r 0.97 over 50
patients) - a consistent offset rather than noise, so model-vs-cfg agreement
has to be read against that floor rather than as pure model error.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, label as connected_components, rotate

from camus_dataset import LABELS

#: Discs along the long axis. 20 is the ASE recommendation and what CAMUS uses.
DISKS = 20

#: Class indices this module reads out of a CAMUS label map.
LV_CLASS = 1        # LV cavity: the structure being measured
ATRIUM_CLASS = 3    # left atrium: only used to locate the mitral plane


class LandmarkError(ValueError):
    """Raised when the LV landmarks cannot be located in a label map.

    Callers are expected to catch this and count the case rather than let a
    single degenerate prediction abort a whole test-set pass - but counting it
    is the point. A silently dropped patient is a patient the model failed on.
    """


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Largest 8-connected component of a boolean mask.

    Predictions occasionally sprout a stray island of cavity somewhere in the
    far field; including it would stretch the long axis and inflate the volume.
    Expert contours are always single-component, so this is a no-op on them.
    """
    labelled, count = connected_components(mask, structure=np.ones((3, 3)))
    if count <= 1:
        return mask
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0  # background
    return labelled == sizes.argmax()


def _require_isotropic(spacing: tuple[float, float]) -> None:
    """Rotating the mask is only meaningful when pixels are square."""
    if not np.isclose(spacing[0], spacing[1], rtol=1e-3):
        raise ValueError(
            f"Anisotropic spacing {spacing}: the disc measurement rotates the mask, "
            f"which mixes the two axes. Resample to square pixels first. (Every CAMUS "
            f"image is 0.308 x 0.308 mm natively, and the dataset's resize keeps it "
            f"square as long as image_size is square.)"
        )


def lv_landmarks(
    label: np.ndarray,
    spacing: tuple[float, float],
    *,
    lv_class: int = LV_CLASS,
    atrium_class: int = ATRIUM_CLASS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Locate ``(base_1, base_2, apex)`` in pixel coordinates.

    The mitral plane is found from anatomy rather than from image geometry: the
    LV cavity pixels that touch the left atrium *are* the valve plane, so the
    two furthest-apart of them are the base points. The apex is then the cavity
    pixel furthest from the midpoint between them, measured in millimetres so
    the answer does not depend on the pixel grid.

    Args:
        label: Integer label map ``(H, W)``.
        spacing: ``(row_mm, col_mm)`` for `label`.

    Raises:
        LandmarkError: If the cavity is empty, or never touches the atrium - in
            which case there is no valve plane to measure from.
    """
    lv = label == lv_class
    if not lv.any():
        raise LandmarkError(f"No {LABELS.get(lv_class, lv_class)} pixels in this label map.")
    lv = keep_largest_component(lv)

    atrium = label == atrium_class
    contact = lv & binary_dilation(atrium, structure=np.ones((3, 3)))
    points = np.argwhere(contact)
    if len(points) < 2:
        raise LandmarkError(
            f"The {LABELS.get(lv_class, lv_class)} does not border the "
            f"{LABELS.get(atrium_class, atrium_class)}, so the mitral plane cannot be "
            f"located ({len(points)} contact pixel(s))."
        )

    # Furthest-apart pair of contact pixels = the ends of the valve plane.
    offsets = (points[:, None, :] - points[None, :, :]) * np.asarray(spacing)
    i, j = np.unravel_index((offsets ** 2).sum(-1).argmax(), (len(points), len(points)))
    base_1, base_2 = points[i], points[j]

    cavity = np.argwhere(lv)
    base_mid = (base_1 + base_2) / 2.0
    apex = cavity[(((cavity - base_mid) * np.asarray(spacing)) ** 2).sum(1).argmax()]
    return base_1, base_2, apex


def disk_diameters(
    label: np.ndarray,
    spacing: tuple[float, float],
    n: int = DISKS,
    *,
    lv_class: int = LV_CLASS,
    atrium_class: int = ATRIUM_CLASS,
) -> tuple[np.ndarray, float]:
    """Disc diameters in mm along the long axis, and the long-axis length in mm.

    The mask is rotated so the long axis runs down the rows, which turns the
    measurement into a row-wise width and avoids resampling a line through the
    cavity for every disc. Nearest-neighbour rotation keeps the mask binary; it
    costs a fraction of a pixel at the boundary, well below the segmentation
    error this is used to measure.
    """
    _require_isotropic(spacing)
    base_1, base_2, apex = lv_landmarks(
        label, spacing, lv_class=lv_class, atrium_class=atrium_class
    )

    axis = apex - (base_1 + base_2) / 2.0
    long_axis_mm = float(np.hypot(*(axis * np.asarray(spacing))))

    lv = keep_largest_component(label == lv_class)
    # `axis` points base -> apex as (drow, dcol); rotating by the negative of its
    # angle brings it onto the row axis. The sign matters: with the opposite sign
    # the mask only lands upright when the axis is already near vertical or
    # horizontal, and every band in between is then measured across a tilted
    # cavity, which reads as a systematically too-wide disc.
    angle = -np.degrees(np.arctan2(axis[1], axis[0]))
    upright = rotate(lv.astype(np.uint8), angle, order=0, reshape=True) > 0

    rows = np.flatnonzero(upright.any(axis=1))
    top, bottom = int(rows[0]), int(rows[-1])
    span = bottom - top + 1

    # Each disc is measured on the row through its centre - the midpoint rule.
    # Taking the widest row in the band instead biases every disc outwards on a
    # tapering cavity: on the half-ellipsoid phantom below that reads +3.3% of
    # volume against the analytic answer, where the midpoint rule reads -0.5%.
    diameters = np.zeros(n)
    for k in range(n):
        row = min(top + int((k + 0.5) * span / n), bottom)
        columns = np.flatnonzero(upright[row])
        if columns.size:
            diameters[k] = (columns[-1] - columns[0] + 1) * spacing[1]
    return diameters, long_axis_mm


def lv_volume_biplane(
    label_2ch: np.ndarray,
    spacing_2ch: tuple[float, float],
    label_4ch: np.ndarray,
    spacing_4ch: tuple[float, float],
    n: int = DISKS,
) -> float:
    """LV cavity volume in mL from a matched 2CH/4CH pair (same instant).

    The long axis is taken as the longer of the two views, per ASE - the views
    are hand-acquired and rarely cut exactly through the apex, so the shorter
    one is more likely to be the foreshortened one. Averaging instead shifts
    the result by a few tenths of an EF point; the choice is not what drives
    the numbers here.
    """
    a, length_2ch = disk_diameters(label_2ch, spacing_2ch, n)
    b, length_4ch = disk_diameters(label_4ch, spacing_4ch, n)
    length = max(length_2ch, length_4ch)
    return float(np.pi / 4 * (a * b).sum() * (length / n)) / 1000.0  # mm^3 -> mL


def lv_volume_monoplane(
    label: np.ndarray, spacing: tuple[float, float], n: int = DISKS
) -> float:
    """Single-plane volume in mL, assuming circular discs.

    The fallback when only one view is usable. It assumes the cavity is
    rotationally symmetric about the long axis, which is exactly the assumption
    the biplane method exists to avoid, so it is for comparison rather than for
    reporting.
    """
    diameters, length = disk_diameters(label, spacing, n)
    return float(np.pi / 4 * (diameters ** 2).sum() * (length / n)) / 1000.0


def ejection_fraction(edv: float, esv: float) -> float:
    """EF in percent from end-diastolic and end-systolic volumes.

    Not clamped to [0, 100]: a prediction that makes the cavity larger at ES
    than at ED yields a negative EF, and that is a result worth seeing rather
    than a value worth hiding.
    """
    if not edv > 0:
        raise ValueError(f"EDV must be positive, got {edv}.")
    return (edv - esv) / edv * 100.0


if __name__ == "__main__":
    # A half-ellipsoid is the shape the method of discs is exact for, so it is
    # the right phantom: with semi-axes (a, b, c) the analytic volume is
    # 2/3 * pi * a * b * c, and the two apical views cut it into half-ellipses.
    spacing = (0.5, 0.5)
    height, width = 320, 320
    a_mm, b_mm, c_mm = 70.0, 25.0, 30.0  # long axis, 2CH half width, 4CH half width

    def phantom(half_width_mm: float) -> np.ndarray:
        """Half-ellipse cavity with an atrium block below its base plane."""
        label = np.zeros((height, width), dtype=np.int64)
        base_row, centre_col = 60, width // 2
        rows, cols = np.indices((height, width))
        down = (rows - base_row) * spacing[0]
        across = (cols - centre_col) * spacing[1]
        inside = (down >= 0) & ((down / a_mm) ** 2 + (across / half_width_mm) ** 2 <= 1.0)
        label[inside] = LV_CLASS
        label[base_row - 30:base_row, centre_col - 30:centre_col + 30] = ATRIUM_CLASS
        return label

    two_ch, four_ch = phantom(b_mm), phantom(c_mm)
    analytic = 2 / 3 * np.pi * a_mm * b_mm * c_mm / 1000.0
    measured = lv_volume_biplane(two_ch, spacing, four_ch, spacing)
    error = abs(measured - analytic) / analytic
    print(f"half-ellipsoid phantom  a={a_mm} b={b_mm} c={c_mm} mm")
    print(f"  analytic  {analytic:8.2f} mL")
    print(f"  measured  {measured:8.2f} mL   ({error:+.2%})")
    assert error < 0.015, f"biplane volume off by {error:.2%}"

    _, length = disk_diameters(two_ch, spacing)
    assert abs(length - a_mm) / a_mm < 0.02, f"long axis {length:.1f} mm, expected {a_mm}"

    # The measurement must not depend on how the probe was held: rotating both
    # views has to leave the volume alone.
    for angle in (17.0, -40.0):
        rotated = [
            rotate(view, angle, order=0, reshape=True, mode="constant", cval=0)
            for view in (two_ch, four_ch)
        ]
        turned = lv_volume_biplane(rotated[0], spacing, rotated[1], spacing)
        drift = abs(turned - measured) / measured
        print(f"  rotated {angle:+6.1f} deg -> {turned:8.2f} mL   ({drift:+.2%} vs upright)")
        assert drift < 0.03, f"volume moved {drift:.2%} under a {angle} deg rotation"

    # Stray islands must not stretch the long axis.
    speckled = two_ch.copy()
    speckled[10:14, 10:14] = LV_CLASS
    assert np.isclose(
        lv_volume_biplane(speckled, spacing, four_ch, spacing), measured, rtol=1e-9
    ), "a disconnected speck changed the volume"

    assert ejection_fraction(100.0, 40.0) == 60.0
    assert ejection_fraction(120.0, 150.0) == -25.0  # ESV > EDV is reported, not clipped

    empty = np.zeros((32, 32), dtype=np.int64)
    for degenerate, why in ((empty, "empty cavity"), (np.full((32, 32), LV_CLASS), "no atrium")):
        try:
            lv_landmarks(degenerate, spacing)
        except LandmarkError as exc:
            print(f"  {why}: LandmarkError({exc})")
        else:
            raise AssertionError(f"{why} should have raised LandmarkError")

    print("\nvolume sanity checks passed")
