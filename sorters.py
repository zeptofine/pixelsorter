from __future__ import annotations

from enum import Enum

import cv2
import numpy as np
from numba import jit
from numpy import ndarray


class AbstractSorter:
    def __init__(self, thresh: int):
        self.set_thresh(thresh)

    def set_thresh(self, thresh: int):
        self.thresh = thresh

    def apply(self, img: ndarray, detector: AbstractSorter | None = None):
        return np.array(list(self.iter_apply(img, detector)))

    def iter_apply(self, img: ndarray, detector: AbstractSorter | None = None):
        if detector is None:
            detector = self

        mask = self.create_mask(img)
        detected = [detector.get_indices(row) for row in detector.create_mask(img)]

        return (
            self.sort_indices(original, sort_by, separators)
            for original, sort_by, separators in zip(img, mask, detected)
        )

    def create_mask(self, img: ndarray):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def get_indices(self, row: ndarray):
        return np.array([*self._get_indices(self.thresh, row)])

    @staticmethod
    @jit(nopython=True)
    def _get_indices(thresh: int, row: ndarray):
        pivot = 0
        for col in range(1, row.shape[0]):
            if abs(row[pivot] - row[col]) > thresh:
                yield col
                pivot: int = col

    @classmethod
    def sort_indices(
        cls,
        original: ndarray,
        sort_by: ndarray,
        indices: ndarray,
    ) -> ndarray:
        if not len(indices):
            sorted_index_arr = np.argsort(sort_by)
        else:
            sort_stack = cls.enumerate_via_indices(sort_by, indices)
            sorted_index_arr = np.lexsort((sort_stack[:, 0], sort_stack[:, 1]))
        return original[sorted_index_arr]  # type: ignore

    @staticmethod
    def enumerate_via_indices(arr: ndarray, indices: ndarray) -> ndarray:
        lengths = np.hstack((indices[0], np.diff(indices), len(arr) - indices[-1]))
        stretched = np.repeat(np.arange(len(lengths)), lengths)
        return np.column_stack((arr, stretched))

    def __repr__(self):
        attrlist = [
            f"{key}=..." if hasattr(val, "__iter__") and not isinstance(val, str) else f"{key}={val}"
            for key, val in vars(self).items()  # type: ignore
        ]
        out = ", ".join(attrlist)
        return f"{self.__class__.__name__}({out})"

    def apply_with(self, detector: AbstractSorter):
        """sorter.apply_with(detector)(image)"""

        def _apply(img):
            return self.apply(img, detector)

        return _apply


class HSV(AbstractSorter):
    def __init__(self, channel: int):
        """
        type: H,S,V = 0, 1, 2
        """
        self.channel = channel

    def create_mask(self, img: ndarray):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, self.channel]


class Hue(HSV):
    def __init__(self):
        super().__init__(0)


class Saturation(HSV):
    def __init__(self):
        super().__init__(1)


class Value(HSV):
    def __init__(self):
        super().__init__(2)


class Lightness(AbstractSorter):
    def create_mask(self, img: ndarray):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 1]


class Canny(AbstractSorter):
    def __init__(self, thresh: int, blur_size=(3, 3), sigma: int = 0):
        super().__init__(thresh)
        self.blur_size = blur_size
        self.sigma = sigma

    @jit(forceobj=True)
    def create_mask(self, img: ndarray):
        blurred: np.ndarray = cv2.GaussianBlur(
            img,
            self.blur_size,
            self.sigma,
        )
        if blurred.dtype == np.uint16:
            blurred = (blurred // 256).astype(np.uint8)
        return cv2.Canny(
            blurred,
            self.thresh,
            self.thresh,
        )

    def get_indices(self, row: ndarray):
        return self._get_indices(row)

    @staticmethod
    @jit(nopython=True)
    def _get_indices(row: ndarray):
        return np.where(row)[0]


class Sorters(str, Enum):
    GRAY = "gray"
    HUE = "hue"
    SAT = "saturation"
    VAL = "value"
    LIGHTNESS = "lightness"
    CANNY = "canny"


SORTER_DICT: dict[str, type[AbstractSorter]] = {
    "gray": AbstractSorter,
    "hue": Hue,
    "saturation": Saturation,
    "value": Value,
    "lightness": Lightness,
    "canny": Canny,
}
