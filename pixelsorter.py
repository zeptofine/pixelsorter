import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
import sys
from multiprocessing import Pool


def main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('-i', '--img', type=str, help="Input to a file to be run.")
    modes.add_argument('--folder', type=str,
                       help="Input to a directory of files to be run. accepts png, jpeg, webp")
    parser.add_argument("--threads", type=int, default=2, help="number of threads to run the images in parallel.")
    parser.add_argument("--threshold", help="Threshold for the sorter algo", default=64)
    parser.add_argument("--resume", action="store_true", help="continues a folder render.")
    return parser


def get_file_list(folder: Path, *exts) -> list[Path]:
    """
    Args    folders: One or more folder paths.
    Returns list[Path]: paths in the specified folders."""
    out = []
    for ext in exts:
        out.extend(folder.glob(ext))
    return out


class AbstractSorter:
    def __init__(self, thresh: int = None):
        self.set_thresh(thresh)

    def apply(self, img: np.ndarray):
        self.img = img
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def set_thresh(self, thresh):
        self.thresh = thresh

    def __repr__(self):
        attrlist = [
            f"{key}=..."
            if hasattr(val, "__iter__") and not isinstance(val, str)
            else
            f"{key}={val}"
            for key, val in self.__dict__.items()
        ]
        out = ", ".join(attrlist)
        return f"{self.__class__.__name__}({out})"

    def pix_set_sort(self, pix_set, gray_set):
        new_sets = list(zip(*sorted(zip(gray_set, pix_set), key=lambda x: x[0])))[1]
        return np.stack(new_sets)

    def iterate_through_row(self, row: int, pix_set_sorter=None):
        if not self.thresh:
            raise AttributeError("No threshold was given during initialization.")
        if not pix_set_sorter:
            pix_set_sorter = self
            gray_row = self.gray[row]
        else:
            gray_row = pix_set_sorter.gray[row]
        selected_row = self.img[row]
        pivot = 0
        column_sets = []
        for col in range(1, self.img.shape[1]):
            if abs(int(self.gray[row, pivot]) - int(self.gray[row, col])) > self.thresh:
                column_sets.append(pix_set_sorter.pix_set_sort(selected_row[pivot:col],
                                                               gray_row[pivot:col]
                                                               ))
                pivot = col
        column_sets.append(pix_set_sorter.pix_set_sort(selected_row[pivot:],
                                                       gray_row[pivot:]))
        return np.concatenate(column_sets, axis=0)


class HSV(AbstractSorter):
    def __init__(self, thresh, type: int):
        '''
        type: H,S,V = 0, 1, 2
        '''
        self.thresh = thresh
        self.type = type

    def apply(self, img: np.ndarray):
        self.img = img
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, self.type]


class Canny(AbstractSorter):
    def __init__(self, thresh: int = None, blur_size=(3, 3), sigma: int = 0):
        self.thresh = thresh
        self.blur_size = blur_size
        self.sigma = sigma

    def apply(self, img: np.ndarray):
        self.img = img
        self.gray = cv2.Canny(cv2.GaussianBlur(img, self.blur_size, self.sigma),
                              int(self.thresh),
                              int(self.thresh))

    def iterate_through_row(self, row: int, pix_set_sorter=None):
        if not hasattr(self, 'img'):
            raise RuntimeError("No image or diff was specified during creation")
        if not pix_set_sorter:
            pix_set_sorter = self
            gray_row = self.gray[row]
        else:
            gray_row = pix_set_sorter.gray[row]
        pivot = 0
        column_sets = []
        selected_row: np.ndarray = self.img[row]

        for col in range(1, self.img.shape[1]):
            if self.gray[row, col]:
                column_sets.append(pix_set_sorter.pix_set_sort(selected_row[pivot:col],
                                                               gray_row[pivot:col]
                                                               ))
                pivot = col
        column_sets.append(pix_set_sorter.pix_set_sort(selected_row[pivot:],
                                                       gray_row[pivot:]
                                                       ))
        pix_sets = np.concatenate(column_sets, axis=0)
        return pix_sets


class ViaImage(Canny):
    def __init__(self, gray: np.ndarray):
        self.gray = gray

    def apply(self, img: np.ndarray):
        self.img = img


class SorterManager:
    def __init__(self, detector: AbstractSorter = None, sorter: AbstractSorter = None):
        self.setDetector(detector)
        self.setSorter(sorter)

    def setDetector(self, detector):
        self.detector: AbstractSorter = detector

    def setSorter(self, sorter):
        self.sorter: AbstractSorter = sorter

    def assert_detector(self):
        # The detector is necessary for the sorter to work, but it can be added to the
        #   manager after initialization
        if not self.detector:
            raise RuntimeError("No detector is found in manager")

    def apply(self, img: np.ndarray, use_tqdm=False):
        self.assert_detector()

        self.detector.apply(img)

        # Make a copy of the image so the original is preserved
        # (The detector and sorter only reads the image)
        new_img = img.copy()

        iterable = range(img.shape[0])
        if use_tqdm:
            iterable = tqdm(iterable)

        if self.sorter is not None:
            self.sorter.apply(img)
        for row in iterable:
            args = [row]
            if self.sorter is not None:
                args.append(self.sorter)

            # Run the detector on the given row
            new_img[row] = self.detector.iterate_through_row(*args)
        return new_img

    def apply_pool(self, img: np.ndarray, threads=1):
        self.assert_detector()

        self.detector.apply(img)

        # Make a copy of the image so the original is preserved
        # (The detector and sorter only reads the image)
        new_img = img.copy()

        iterable = range(img.shape[0])

        if self.sorter:
            self.sorter.apply(img)
            pargs = zip(iterable, [self.sorter] * img.shape[0])
        else:
            pargs = iterable
        # rprint(list(pargs))

        with Pool(threads) as p:
            iterable = pargs
            # out = p.starmap(self.detector.iterate_through_row, iterable)
            for idx, new_row in enumerate(p.starmap(self.detector.iterate_through_row, iterable)):
                new_img[idx] = new_row
        return new_img


if __name__ == "__main__":

    args = main_parser().parse_args()

    # Initialize the sorter manager
    sorter = SorterManager()

    # Add the sorter that decides how to separate the sets of pixels
    # sorter.setDetector(Canny(args.threshold))
    args.threshold = int(args.threshold)
    sorter.setDetector(AbstractSorter(args.threshold))

    # Adds the sorter that changes how the sets of pixels are sorted
    # sorter.setSorter(AbstractSorter())

    image_list = []
    if args.img:  # The mode is a single image
        image = Path(args.img)
        if not image.exists():
            print("Image does not exist")
            sys.exit(1)

        out_path = image.with_stem(f"{image.stem}-pixelsorted")
        img = cv2.imread(str(image))
        pxsorted = sorter.apply_pool(img, threads=args.threads)
        cv2.imwrite(str(out_path), pxsorted)

    elif args.folder:
        folder = Path(args.folder)
        out_folder = folder.with_stem(f"{folder.stem}-pixelsorted")
        image_list = get_file_list(Path(args.folder), "*.png", "*.jpg", "*.webp")
        out_dict = {image: out_folder / image.relative_to(folder) for image in image_list}
        if args.resume:
            image_list = [image for image in image_list if not out_dict[image].exists()]

        def read_and_write(path: Path):
            # img = cv2.imread(str(path))
            out = out_dict[path]
            img = cv2.imread(str(path))
            pxsorted = sorter.apply(img)
            cv2.imwrite(str(out), pxsorted)

        with Pool(min(args.threads, len(image_list))) as p:
            if not out_folder.exists():
                out_folder.mkdir()
            for _ in tqdm(p.imap(read_and_write, image_list), total=len(image_list)):
                pass
        print(f"Done! images are in {out_folder}")
