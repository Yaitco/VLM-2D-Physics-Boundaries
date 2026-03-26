from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def postprocess_mask(
    mask: np.ndarray,
    min_component_area: int = 500,
    fill_holes: bool = True,
    closing_kernel_size: int = 5,
    closing_iterations: int = 1,
) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if binary.size == 0:
        return binary

    if fill_holes:
        binary = ndi.binary_fill_holes(binary)

    if closing_iterations > 0 and closing_kernel_size > 1:
        structure = np.ones((closing_kernel_size, closing_kernel_size), dtype=bool)
        binary = ndi.binary_closing(binary, structure=structure, iterations=closing_iterations)

    binary = remove_small_components(binary, min_component_area)

    if fill_holes:
        binary = ndi.binary_fill_holes(binary)
    return binary.astype(bool)


def remove_small_components(mask: np.ndarray, min_component_area: int) -> np.ndarray:
    if min_component_area <= 0:
        return mask.astype(bool)

    labeled, num_labels = ndi.label(mask.astype(bool))
    if num_labels == 0:
        return mask.astype(bool)

    component_sizes = np.bincount(labeled.ravel())
    keep = component_sizes >= int(min_component_area)
    keep[0] = False
    return keep[labeled]
