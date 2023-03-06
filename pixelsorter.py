from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


class PixelSorter:
    class AbstractSorter:
        def __init__(self, img: np.ndarray, thresh: int = None):
            self.img = img
            self.thresh = thresh
            self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        def __repr__(self):
            attrlist = []
            for key, val in self.__dict__.items():
                if hasattr(val, "__iter__") and not isinstance(val, str):
                    attrlist.append(f"{key}=...")
                else:
                    attrlist.append(f"{key}={val}")
            out = ", ".join(attrlist)
            return f"{self.__class__.__name__}({out})"

        def pix_set_sort(self, pix_set, gray_set):
            return np.take_along_axis(
                pix_set,
                np.argsort(cv2.cvtColor(
                    gray_set,
                    cv2.COLOR_GRAY2BGR
                ), axis=1),
                axis=1)

        def iterate_through_row(self, row: int, pix_set_sorter=None):
            if not self.thresh:
                raise AttributeError("No threshold was given during initialization.")
            if not pix_set_sorter:
                pix_set_sorter = self
                gray_row = self.gray[row:row + 1]
            else:
                gray_row = pix_set_sorter.gray[row:row+1]
            selected_row = self.img[row:row + 1]
            pivot = 0
            column_sets = []
            for col in range(1, img.shape[1]):
                if abs(int(self.gray[row, pivot]) - int(self.gray[row, col])) > self.thresh:
                    column_sets.append(pix_set_sorter.pix_set_sort(selected_row[:, pivot:col],
                                                                   gray_row[:, pivot:col]
                                                                   ))
                    pivot = col
            column_sets.append(pix_set_sorter.pix_set_sort(selected_row[:, pivot:],
                                                           gray_row[:, pivot:]))
            return np.concatenate(column_sets, axis=1)

    class Red(AbstractSorter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.gray = img[:, :, 2]

    class Green(AbstractSorter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.gray = img[:, :, 1]

    class Blue(AbstractSorter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.gray = img[:, :, 0]

    class Hue(AbstractSorter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]

    class Saturation(AbstractSorter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]

    class Value(AbstractSorter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]

    class Canny(AbstractSorter):
        def __init__(self, img: np.ndarray, diff: int, blur_size: tuple[int, int] = (3, 3), sigma: int = 0):

            self.img = img
            self.gray = cv2.Canny(cv2.GaussianBlur(img, blur_size, sigma), diff, diff)

        def iterate_through_row(self, row: int, pix_set_sorter=None):
            if not pix_set_sorter:
                pix_set_sorter = self
                gray_row = self.gray[row:row+1]
            else:
                gray_row = pix_set_sorter.gray[row:row+1]
            selected_row = self.img[row:row + 1]
            pivot = 0
            column_sets = []
            for col in range(1, img.shape[1]):
                if self.gray[row:row+1, col:col+1][0, 0]:
                    column_sets.append(pix_set_sorter.pix_set_sort(selected_row[:, pivot:col],
                                                                   gray_row[:, pivot:col]))
                    pivot = col
            column_sets.append(pix_set_sorter.pix_set_sort(selected_row[:, pivot:],
                                                           gray_row[:, pivot:]))
            return np.concatenate(column_sets, axis=1)

        def pix_set_sort(self, pix_set, gray_set):
            gray_set = cv2.cvtColor(pix_set, cv2.COLOR_BGR2GRAY)
            return super().pix_set_sort(pix_set, gray_set)

    class Manager:
        def __init__(self, detector, sorter=None):
            self.detector = detector
            self.sorter = sorter

        def setDetector(self, detector):
            self.detector = detector

        def setSorter(self, sorter):
            self.sorter = sorter

        def apply(self, img: np.ndarray, use_tqdm=False):
            new_img = img.copy()
            iterable = range(img.shape[0])
            if use_tqdm:
                iterable = tqdm(iterable)
            for row in iterable:
                args = [row]
                if self.sorter is not None:
                    args.append(self.sorter)
                new_img[row] = self.detector.iterate_through_row(
                    *args
                )
            return new_img


if __name__ == "__main__":

    # User choice for files
    image = Path("/home/xpsyc/Documents/HiddenStuff/val_HR/ZunmBXs.jpg")

    # sensitivity of the filter
    thresh = 76
    img = cv2.imread(str(image))
    out = PixelSorter.Manager(
        PixelSorter.Canny(img, thresh),
        PixelSorter.AbstractSorter(img)
    ).apply(img, use_tqdm=True)

    cv2.imwrite('out.png', out)
