import argparse
import datetime
import sys
from multiprocessing import Pool, Process, Queue
from pathlib import Path
import subprocess

import cv2
import ffmpeg
import numpy as np
from tqdm import tqdm
import os
from ConfigArgParser import ConfigParser


def main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('-i', '--img', type=str, help="Input to a file to be run.")
    modes.add_argument('--folder', type=str,
                       help="Input to a directory of files to be run. accepts png, jpeg, webp")
    modes.add_argument('--video', type=str,
                       help="path to a video to convert")
    parser.add_argument("--threads", type=int, default=(os.cpu_count() / 4) * 3,
                        help="number of threads to run the images in parallel.")
    parser.add_argument("--threshold", help="Threshold for the sorter algo", default=64)
    folder_specific = parser.add_argument_group("Folder specific options")
    folder_specific.add_argument("--resume", action="store_true", help="continues a folder render.")
    gray_modes = parser.add_mutually_exclusive_group()
    gray_modes.add_argument("--detector", choices=('default', 'hue', 'saturation', 'value', 'lightness', 'canny'),
                            help="how the script identifies sets of pixels to sort", default='default')
    gray_modes.add_argument("--gray_img", help="Path to the image to use as a threshold.")
    parser.add_argument("--sorter", choices=('default', 'hue', 'saturation', 'value', 'lightness'),
                        help="how the script sorts the sets of pixels", default='default')

    return parser


def get_file_list(folder: Path, *exts) -> list[Path]:
    """
    Args    folders: One or more folder paths.
    Returns list[Path]: paths in the specified folders."""
    out = []
    for ext in exts:
        out.extend(folder.rglob(ext))
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
        self.assert_thresh()
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

    def assert_thresh(self):
        if not self.thresh:
            raise AttributeError("No threshold was given during initialization")


class HSV(AbstractSorter):
    def __init__(self, thresh: int, type: int):
        '''
        type: H,S,V = 0, 1, 2
        '''
        self.thresh = thresh
        self.type = type

    def apply(self, img: np.ndarray):
        self.img = img
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, self.type]


class Hue(HSV):
    def __init__(self, thresh: int = None):
        super().__init__(thresh, 0)


class Saturation(HSV):
    def __init__(self, thresh: int = None):
        super().__init__(thresh, 1)


class Value(HSV):
    def __init__(self, thresh: int = None):
        super().__init__(thresh, 2)


class Lightness(AbstractSorter):
    def apply(self, img: np.ndarray):
        self.img = img
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]


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
        self.assert_thresh()

        if not pix_set_sorter:
            pix_set_sorter = self
            self.assert_gray()

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

    def assert_gray(self):
        if not self.gray:
            raise AttributeError("Gray was not initialized")


class ViaImage(Canny):
    def __init__(self, gray: np.ndarray):
        # self.gray = gray
        self.gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

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
        new_img = img
        print(new_img.shape)

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


def check(condition, statement):
    if not condition:
        print(statement)
        exit(1)


def recursive_mkdir(p: Path):
    for parent in list(p.parents)[::-1]:
        if not parent.exists():
            parent.mkdir()


def run_sorter(zipped):
    sorter, impath, out = zipped
    img = cv2.imread(str(impath))
    sorted_img = sorter.apply(img)
    recursive_mkdir(out)

    cv2.imwrite(str(out), sorted_img)


def frame_reader(in_thread: subprocess.Popen, in_queue: Queue, shape, read_size, chunksize=12):
    while True:
        frame_stack = []
        for _ in range(chunksize):
            in_bytes = in_thread.stdout.read(read_size)
            if not in_bytes:
                break
            frame_stack.append(np.frombuffer(in_bytes, np.uint8).reshape(shape))

        if not frame_stack:
            in_queue.put(None)
            break

        in_queue.put(frame_stack)


def run_img(args: argparse.Namespace, sorter: SorterManager):
    image = Path(args.img)
    check(image.exists(), "Image does not exist")

    out_path = image.with_stem(f"{image.stem}-pixelsorted")
    print("Reading image...")
    img = cv2.imread(str(image))
    print("Sorting...")
    pxsorted = sorter.apply(img)
    cv2.imwrite(str(out_path), pxsorted)
    print(f"saved to {out_path}")


def run_folder(args: argparse.Namespace, sorter: SorterManager):
    folder = Path(args.folder)
    out_folder = folder.with_stem(f"{folder.stem}-pixelsorted")
    print("getting list...")
    image_list = sorted(get_file_list(Path(args.folder), "*.png", "*.jpg", "*.webp"))
    out_dict = {image: out_folder / image.relative_to(folder) for image in image_list}
    if args.resume:
        print("Filtering existing images...")
        image_list = [image for image in image_list if not out_dict[image].exists()]

    check(image_list, "List is empty")

    print("running...")
    pargs = [(sorter, image, out_dict[image]) for image in image_list]
    with Pool(min(args.threads, len(image_list))) as p:
        for _ in tqdm(p.imap(run_sorter, pargs), total=len(image_list)):
            pass
    print(f"Done! images are in {out_folder}")


def run_video(args: argparse.Namespace, sorter: SorterManager):
    video_path = Path(args.video)
    out_path = video_path.with_stem(f"{video_path.stem}-pixelsorted")
    check(video_path.exists(), "Video path does not exist")

    # get video info
    print("getting video info")
    probe = ffmpeg.probe(args.video)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    # check(video_streams, "Video streams are empty")
    video_stream = video_streams[0]
    width = int(video_stream['width'])
    height = int(video_stream['height'])

    # get framerate
    framerate = video_stream['r_frame_rate']
    framerate = framerate.split('/')
    framerate = int(framerate[0]) / int(framerate[1])

    # get total frame count
    # TODO: Find a faster way to get the total number of frames from the video
    # ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -print_format csv <file_path>
    total_frames = int(subprocess.check_output([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames",
        "-print_format", "csv", str(video_path)
    ]).decode("utf-8").strip().split(',')[-1])

    video = ffmpeg.input(str(video_path))
    video, audio = video, video.audio
    thread_in: subprocess.Popen = (
        video
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True, quiet=True)
    )

    thread_out: subprocess.Popen = (
        ffmpeg.output(
            ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}', r=framerate).video,
            audio,
            str(out_path),
            pix_fmt='yuv420p'
        )
        .overwrite_output()
        .run_async(pipe_stdin=True, quiet=True)
    )

    _start_time = datetime.datetime.now()
    queue_size = args.threads
    chunksize = 96

    frame_queue = Queue(queue_size)

    # reads the frames coming in from the video and adds chunks of them to the frame_queue as np.ndarray's, asynchronously
    reader = Process(target=frame_reader, args=(thread_in, frame_queue,
                     [height, width, 3], height * width * 3, chunksize))
    reader.daemon = True
    reader.start()
    print("started reading frames")

    t = tqdm(total=total_frames)
    with Pool(args.threads) as p:
        while True:
            # get a collection of frames
            frame_chunk = frame_queue.get()

            if frame_chunk is None:
                break
            # run the frames through the sorter
            for out_frame in p.imap(sorter.apply, frame_chunk):
                # cv2.imshow('out', out_frame)
                # cv2.waitKey(1)
                t.update()
                thread_out.stdin.write(
                    out_frame.astype(np.uint8).tobytes()
                )

    duration = datetime.datetime.now() - _start_time
    print('\n')
    print(f"Done! it took: {duration}")
    thread_out.stdin.close()
    thread_in.wait()
    thread_out.wait()
    thread_in.communicate()
    thread_out.communicate()
    frame_queue.close()
    reader.join()
    reader.close()


if __name__ == "__main__":

    args = ConfigParser(main_parser(), "config.json", autofill=True).parse_args()

    # Initialize the sorter manager
    sorter = SorterManager()

    # Add the sorter that decides how to separate the sets of pixels
    detectors = {
        'default': AbstractSorter,
        'hue': Hue,
        'saturation': Saturation,
        'value': Value,
        'lightness': Lightness,
        'canny': Canny
    }
    if args.gray_img:
        if not Path(args.gray_img).exists():
            print("Gray image does not exist")
            sys.exit(1)
        gray_path = Path(args.gray_img).resolve()
        # read the image as grayscale
        print("Reading gray...")
        gray_image = cv2.imread(str(gray_path))
        if gray_image is None:
            print("Gray image failed to load")
            sys.exit(1)
        sorter.setDetector(ViaImage(gray_image))
    elif args.detector:
        args.threshold = int(args.threshold)
        sorter.setDetector(detectors[args.detector](args.threshold))

    # sorter.setDetector(AbstractSorter(args.threshold))

    # Adds the sorter that changes how the sets of pixels are sorted
    sorter.setSorter(detectors[args.sorter]())
    # sorter.setSorter(AbstractSorter())

    if args.img:
        run_img(args, sorter)

    elif args.folder:
        run_folder(args, sorter)

    elif args.video:
        run_video(args, sorter)
