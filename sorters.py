from __future__ import annotations

from collections.abc import Generator

import cv2
import numpy as np
from numpy import ndarray
from tqdm import tqdm
from enum import Enum


class AbstractSorter:
    def __init__(self, thresh: int):
        self.set_thresh(thresh)

    def set_thresh(self, thresh: int):
        self.thresh = thresh

    def apply(self, img: ndarray, detector: AbstractSorter | None = None, use_tqdm=False):
        return np.array(list(self.iter_apply(img, detector, use_tqdm)))

    def iter_apply(self, img: ndarray, detector: AbstractSorter | None = None, use_tqdm=False):
        if detector is None:
            detector = self

        return (
            self.sort_indices(*rows)
            for rows in zip(
                tqdm(img) if use_tqdm else img,
                self.create_mask(img),
                (detector.get_indices(row) for row in detector.create_mask(img)),
            )
        )

    def create_mask(self, img: ndarray):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def get_indices(self, row: ndarray) -> Generator[int, None, None]:
        return self._get_indices(self.thresh, row)

    @staticmethod
    def _get_indices(thresh, row: ndarray) -> Generator[int, None, None]:
        pivot = 0
        for col in range(1, row.shape[0]):
            if abs(row[pivot] - row[col]) > thresh:
                yield col
                pivot: int = col

    @staticmethod
    def sort_indices(
        original: ndarray,
        sort_by: ndarray,
        index_separators: Generator[int, None, None],
    ) -> ndarray:
        sort_stack = np.column_stack(
            (
                sort_by,
                AbstractSorter.enumerate_indices(
                    sort_by,
                    list(index_separators),
                ),
            )
        )
        return original[np.lexsort((sort_stack[:, 0], sort_stack[:, 1]))]

    @staticmethod
    def enumerate_indices(arr: np.ndarray, indices: list[int]) -> ndarray:
        if not indices:
            return np.array([0] * len(arr))
        return np.array(
            [
                y
                for x in (
                    [idx] * value
                    for idx, value in enumerate(
                        (
                            indices[0],
                            *[indices[idx] - indices[idx - 1] for idx in range(1, len(indices))],
                            len(arr) - indices[-1],
                        )
                    )
                )
                for y in x
            ]
        )

    def __repr__(self):
        attrlist = [
            f"{key}=..." if hasattr(val, "__iter__") and not isinstance(val, str) else f"{key}={val}"
            for key, val in self.__dict__.items()  # type: ignore
        ]
        out = ", ".join(attrlist)
        return f"{self.__class__.__name__}({out})"

    def apply_with(self, detector: AbstractSorter, use_tqdm=False):
        """sorter.apply_with(detector)(image)"""

        def _apply(img):
            return self.apply(img, detector, use_tqdm=use_tqdm)

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

    def create_mask(self, img: ndarray):
        return cv2.Canny(
            cv2.GaussianBlur(
                img,
                self.blur_size,
                self.sigma,
            ),
            self.thresh,
            self.thresh,
        )

    def get_indices(self, row: ndarray) -> Generator[int, None, None]:
        return self._get_indices(row)

    @staticmethod
    def _get_indices(row: ndarray):
        return (col for col in range(1, row.shape[0]) if row[col])


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
