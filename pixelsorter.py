import argparse
import datetime
import os
import subprocess
# import sys
import threading
from multiprocessing import Pool, Queue
from multiprocessing.queues import Full
from pathlib import Path
# from pprint import pprint
from threading import Thread

import cv2
import dateutil.parser as timeparser
import ffmpeg
import numpy as np
import psutil
from tqdm import tqdm

from ConfigArgParser import ConfigParser


def main_parser() -> argparse.ArgumentParser:
    # Top-level parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Global args
    parser.add_argument("--threshold", type=float, help="Threshold for the sorter algo", required=True)
    parser.add_argument("--threads", type=int, default=(os.cpu_count() / 4) * 3,
                        help="number of threads to run the images in parallel.")
    parser.add_argument("--detector", choices=('default', 'hue', 'saturation', 'value', 'lightness', 'canny'),
                        help="how the script identifies sets of pixels to sort", default='default')
    parser.add_argument("--sorter", choices=('default', 'hue', 'saturation', 'value', 'lightness'),
                        help="how the script sorts the sets of pixels", default='default')

    subparsers = parser.add_subparsers(title="mode", dest="mode", help='sub-command help')
    # Picture arg parser
    parser_pic = subparsers.add_parser('image')
    parser_pic.add_argument('-i', '--input', type=str, help="Input to a file to be run.", required=True)

    # Folder arg parser
    parser_folder = subparsers.add_parser('folder')
    parser_folder.add_argument('-i', '--input', type=str,
                               help="Input to a directory of files to be run. Accepts png, jpeg, webp", required=True)
    parser_folder.add_argument("--resume", action="store_true", help="continues a folder render.")

    # Video arg parser
    parser_v = subparsers.add_parser('video')

    parser_v.add_argument('-i', '--input', type=str,
                          help="path to a video to convert", required=True)
    parser_v.add_argument('--preview', action='store_true',
                          help="""If activated, a cv2 window will appear to preview the images as they are written.
This is useful for debugging, but adds a little bit of processsing time.""")
    parser_v.add_argument('--gb_usage', type=float, default=4,
                          help="Tries to cache as many frames at once that can fit in the set size.")
    parser_v.add_argument('--chunk_size', type=int,
                          help="""
                          number of frames to be rendered in a single pool run.
                          this should be at least the number of threads in order to utilize them fully
                          """)
    parser_v.add_argument('--to', type=int,
                          help='the max time to render. like only render the first 10 seconds of the video.')
    # v = parser_v.add_subparsers()
    # vv = v.add_parser("excuse_me")
    # vv.add_argument("--thats_incredible")

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
        return np.stack((
            *zip(
                *sorted(  # sort the pix_set based on the gray_set
                    zip(gray_set, pix_set),
                    key=lambda x: x[0])
            ),
        )[1])

    def iterate_through_row(self, row: int):
        pivot = 0
        for col in range(1, self.img.shape[1]):
            if abs(int(self.gray[row, pivot]) - int(self.gray[row, col])) > self.thresh:
                yield (pivot, col)
                pivot = col
        yield (pivot, self.img.shape[1])

    def iterate_through_indices(self, row: int, indices_of_pixels: list[tuple[int, int]]) -> np.ndarray:
        for start, end in indices_of_pixels:
            yield self.pix_set_sort(self.img[row, start:end], self.gray[row, start:end])


class HSV(AbstractSorter):
    def __init__(self, thresh: int, type: int):
        '''
        type: H,S,V = 0, 1, 2
        '''
        super().__init__(thresh)
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

    def iterate_through_row(self, row: int):
        pivot = 0

        for col in range(1, self.img.shape[1]):
            if self.gray[row, col]:
                yield (pivot, col)
                pivot = col
        yield (pivot, self.img.shape[1])


class ViaImage(Canny):
    def __init__(self, gray: np.ndarray):
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
        # manager after initialization
        if not self.detector:
            raise RuntimeError("No detector is found in manager")

    def assert_sorter(self):
        # The sorter is necessary for application, but it can be added afterwards
        if not self.sorter:
            raise RuntimeError("No detector is found in manager")

    def apply(self, img: np.ndarray, use_tqdm=False):
        self.assert_detector()
        self.assert_sorter()

        self.detector.apply(img)
        if not self.detector == self.sorter:
            self.sorter.apply(img)

        # Make a copy of the image so the original is preserved
        # (The detector and sorter only reads the image)
        new_img = img.copy()

        iterable = range(img.shape[0])
        if use_tqdm:
            iterable = tqdm(iterable)

        for row in iterable:
            # Get the indices of pixels to sort.
            new_img[row] = np.concatenate(
                list(self.sorter.iterate_through_indices(
                    row,
                    self.detector.iterate_through_row(row)
                )),
                axis=0
            )

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
            sorter = self.sorter
        else:
            sorter = self.detector

        with Pool(threads) as p:
            pixel_indices = p.map_async(self.detector.iterate_through_row, iterable)

            new_img = np.ndarray([
                np.concatenate(
                    [p.map(sorter.iterate_through_indices, (row, pixel_indices[row]))
                        for row in iterable],
                    axis=0
                )
            ])
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


def read_frames(continue_event: threading.Event, in_thread: subprocess.Popen, in_queue: Queue, shape, read_size, chunksize=12):
    print("started reading frames")
    frame_stack = []
    while continue_event.is_set():
        for _ in range(chunksize - len(frame_stack)):
            in_bytes = in_thread.stdout.read(read_size)
            if not in_bytes:
                break
            image = np.frombuffer(in_bytes, np.uint8).reshape(shape)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            frame_stack.append(image)
        if frame_stack:
            try:
                in_queue.put(frame_stack, timeout=2)

            except Full:
                # print("Images have taken 30 extra seconds to add to the queue")
                continue
            frame_stack = []
        else:
            in_queue.put(None)
            print("finished reading frames")
            break


def run_img(args: argparse.Namespace, sorter: SorterManager):
    image = Path(args.input)
    check(image.exists(), "Image does not exist")

    out_path = image.with_stem(f"{image.stem}-pixelsorted")
    print("Reading image...")
    img = cv2.imread(str(image))
    print("Sorting...")
    pxsorted = sorter.apply(img)
    cv2.imwrite(str(out_path), pxsorted)
    print(f"saved to {out_path}")


def run_folder(args: argparse.Namespace, sorter: SorterManager):
    folder = Path(args.input)
    out_folder = folder.with_stem(f"{folder.stem}-pixelsorted")
    print("getting list...")
    image_list = sorted(get_file_list(Path(args.input), "*.png", "*.jpg", "*.webp"))

    out_dict = {image: out_folder / image.relative_to(folder) for image in image_list}
    if args.resume:
        print("filtering existing images...")
        image_list = [image for image in image_list if not out_dict[image].exists()]

    check(image_list, "List is empty")

    print("running...")
    pargs = [(sorter, image, out_dict[image]) for image in image_list]
    with Pool(min(args.threads, len(image_list))) as p:
        for _ in tqdm(p.imap(run_sorter, pargs), total=len(image_list)):
            pass
    print(f"Done! images are in {out_folder}")


def run_video(args: argparse.Namespace, sorter: SorterManager):
    check(hasattr(args, 'input'), 'No input was specified')
    video_path = Path(args.input)
    out_path = video_path.with_stem(f"{video_path.stem}-pixelsorted")
    check(video_path.exists(), "Video path does not exist")

    # get video info
    print("getting video info")
    probe = ffmpeg.probe(args.input)
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
    duration = timeparser.parse(video_stream['tags']['DURATION']).time()
    t = datetime.datetime.combine(datetime.date.min, duration) - datetime.datetime.min
    total_frames = int(t.total_seconds() * framerate)

    video = ffmpeg.input(str(video_path))
    video, audio = video, video.audio

    thread_in: subprocess.Popen = (
        video
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(
            pipe_stdout=True,
            quiet=True
        )
    )

    thread_out: subprocess.Popen = (
        ffmpeg
        .output(
            ffmpeg.input(
                'pipe:',
                format='rawvideo',
                pix_fmt='rgb24',
                s=f'{width}x{height}',
                r=framerate,
            ).video,
            audio,
            str(out_path),
            pix_fmt='yuv420p',

        )
        # I wish i could just use quiet=True in .run_async(), but for some reason if I do, the pool
        # consistently stops 5 mins into processing
        .global_args(
            '-hide_banner',
            '-nostats',
            '-loglevel', 'error'
        )
        .overwrite_output()
        .run_async(
            pipe_stdin=True,
        )
    )

    _start_time = datetime.datetime.now()

    # the number of bytes to read per image
    read_size = height * width * 3
    # the number of frames to read in one chunk

    if not args.chunk_size:
        frame_chunk_size = args.threads * 2
    else:
        frame_chunk_size = args.chunk_size
    gb_usage = args.gb_usage
    gb_usage_in_bytes = int(gb_usage * (10**9))

    # Example:
    # free memory before: 18.58 GB
    # free memory with predicted given usage: -5.45 GB
    # Chosen memory usage is too large, resizing for a minimum of 1 gb free
    # using 1.58 GB
    # estimaged total usage is less than gb usage. Video will likely fit in less space than specified
    # free with predicted total usage: 12.69 GB

    # get amount of system memory
    virtual_memory = psutil.virtual_memory()

    print(f'free memory before: {virtual_memory.available / (10 ** 9):.2f} GB')
    print(
        f'free memory with predicted given usage: {(virtual_memory.available - gb_usage_in_bytes) / (10 ** 9):.2f} GB')
    if virtual_memory.available - gb_usage_in_bytes < 10 ** 9:
        print("Chosen memory usage is too large, resizing for a minimum of 1 gb free")

        while virtual_memory.available - gb_usage_in_bytes < 10 ** 9:
            gb_usage_in_bytes -= (10 ** 9)
        print(f"using {gb_usage_in_bytes // (10 ** 9):.2f} GB")

    # estimate the total number of bytes in the video
    total_usage = read_size * total_frames

    # if the total number of bytes in the video is smaller than the allowed ram threshold
    if total_usage < gb_usage_in_bytes:
        print("estimaged total usage is less than given usage. Video will likely fit in less space than specified")
        print(f'free memory with predicted total usage: {(virtual_memory.available - total_usage) / (10 ** 9):.2f} GB')

    # calculate how many images can fit in a given amount of memory
    queue_size = int(gb_usage_in_bytes // read_size // frame_chunk_size)

    frame_queue = Queue(queue_size)
    # reads the frames coming in from the video and adds list[np.ndarray]'s to frame_queue asynchronously
    continue_event = threading.Event()
    continue_event.set()
    reader = Thread(target=read_frames, args=(continue_event, thread_in, frame_queue,
                                              [height, width, 3], read_size, frame_chunk_size))
    reader.daemon = True
    reader.start()
    t = tqdm(total=total_frames)
    p = Pool(args.threads)
    while True:
        # get a collection of frames
        chunk_of_frames = frame_queue.get()
        if chunk_of_frames is None:
            break

        # run the frames through the sorter
        for out_frame in p.imap(sorter.apply, chunk_of_frames):
            t.update()
            thread_out.stdin.write(
                cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB).tobytes()
            )
            if args.preview:
                cv2.imshow('out', out_frame)
                cv2.waitKey(1)
    p.terminate()
    p.join()

    continue_event.clear()
    duration = datetime.datetime.now() - _start_time
    print('\n')
    print(f"Done! it took: {duration}")
    thread_in.stdout.close()
    thread_out.stdin.close()
    frame_queue.close()
    reader.join()


if __name__ == "__main__":

    parser = main_parser()
    args = ConfigParser(
        parser,
        "config.json"
    ).parse_args()

    # Initialize the sorter manager
    sorter = SorterManager()

    detectors = {
        'default': AbstractSorter,
        'hue': Hue,
        'saturation': Saturation,
        'value': Value,
        'lightness': Lightness,
        'canny': Canny
    }

    args.threshold = int(args.threshold)

    # Add the sorter that decides how to separate the sets of pixels
    sorter.setDetector(detectors[args.detector](args.threshold))

    # Adds the sorter that changes how the sets of pixels are sorted
    sorter.setSorter(detectors[args.sorter]())

    modes = {
        'image': run_img,
        'folder': run_folder,
        'video': run_video
    }
    if args.mode and args.mode in modes:
        modes[args.mode](args, sorter)
    else:
        print("No mode was selected")
