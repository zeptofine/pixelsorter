from __future__ import annotations

import datetime
import subprocess
import time
from multiprocessing import Process, Queue, cpu_count
from pathlib import Path
from queue import Full
from typing import Annotated, Optional

import cv2
import dateutil.parser as timeparser
import ffmpeg
import numpy as np
import psutil
import typer
from tqdm import tqdm

from sorters import SORTER_DICT, Sorters

CPU_COUNT: int = cpu_count()

app = typer.Typer()


def read_frames(
    in_thread: subprocess.Popen,
    in_queue: Queue,
    shape,
    read_size,
    chunksize=12,
):
    print("started reading frames")
    frame_stack = []
    while True:
        for _ in range(chunksize - len(frame_stack)):
            assert in_thread.stdout is not None
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
                continue
            frame_stack = []
        else:
            in_queue.put(None)
            print("finished reading frames")
            break


@app.command()
def run_img(
    video_path: Annotated[Path, typer.Argument(help="the input video to sort")],
    output: Annotated[
        Optional[Path], typer.Option(help="where to output to. if empty, appends `-pixelsorted` to input.")
    ] = None,
    sorter: Annotated[Sorters, typer.Option(help="what sorter to actually sort sections with")] = Sorters.GRAY,
    detector: Annotated[Sorters, typer.Option(help="how to detect sections")] = Sorters.CANNY,
    detector_threshold: int = 100,
    preview: Annotated[
        bool,
        typer.Option(
            help="""If activated, a cv2 window will appear to preview the images as they are written.
                     This is useful for debugging, but adds a little bit of processsing time."""
        ),
    ] = False,
    gb_usage: Annotated[
        float, typer.Option(help="Tries to cache only as many frames that can fit in this gb threshold.")
    ] = 4,
    chunksize: Annotated[Optional[int], typer.Option(help="the number of frames to read per chunk.")] = None,
):
    assert video_path.exists(), "Image does not exist"
    output = output or video_path.with_stem(f"{video_path.stem}-pixelsorted")

    print("getting video info")
    probe = ffmpeg.probe(video_path)
    video_stream: dict = [stream for stream in probe["streams"] if stream["codec_type"] == "video"][0]
    width = int(video_stream["width"])
    height = int(video_stream["height"])

    # get framerate
    ratestr: list[str] = video_stream["r_frame_rate"].split("/")
    framerate: float = int(ratestr[0]) / int(ratestr[1])

    # get total frame count
    duration = timeparser.parse(video_stream["tags"]["DURATION"]).time()
    t: float = (datetime.datetime.combine(datetime.date.min, duration) - datetime.datetime.min).total_seconds()
    total_frames = int(t * framerate)

    video = ffmpeg.input(str(video_path))
    audio = video.audio  # evil setting of both at once

    thread_in: subprocess.Popen = video.output("pipe:", format="rawvideo", pix_fmt="rgb24").run_async(
        pipe_stdout=True, quiet=True
    )
    thread_out = (
        ffmpeg.output(
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s=f"{width}x{height}",
                r=framerate,
            ).video,
            audio,
            str(output),
            pix_fmt="yuv420p",
        )
        # I wish i could just use quiet=True in .run_async(), but for some reason if I do, the pool
        # consistently stops 5 mins into processing
        .global_args("-hide_banner", "-nostats", "-loglevel", "error")
        .overwrite_output()
        .run_async(
            pipe_stdin=True,
        )
    )

    _start_time = datetime.datetime.now()

    # the number of bytes to read per image
    read_size = height * width * 3

    # the number of frames to read in one chunk
    frame_chunk_size = chunksize or 12

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

    print(f"free memory before: {virtual_memory.available / (10 ** 9):.2f} GB")
    print(
        f"free memory with predicted given usage: {(virtual_memory.available - gb_usage_in_bytes) / (10 ** 9):.2f} GB"
    )
    if virtual_memory.available - gb_usage_in_bytes < 10**9:
        print("Chosen memory usage is too large, resizing for a minimum of 1 gb free")

        while virtual_memory.available - gb_usage_in_bytes < 10**9:
            gb_usage_in_bytes -= 10**9
        print(f"using {gb_usage_in_bytes // (10 ** 9):.2f} GB")

    # estimate the total number of bytes in the video
    total_usage = read_size * total_frames

    # if the total number of bytes in the video is smaller than the allowed ram threshold
    if total_usage < gb_usage_in_bytes:
        print("estimaged total usage is less than given usage. Video will likely fit in less space than specified")
        print(f"free memory with predicted total usage: {(virtual_memory.available - total_usage) / (10 ** 9):.2f} GB")

    # calculate how many images can fit in a given amount of memory
    queue_size = int(gb_usage_in_bytes // read_size // frame_chunk_size)

    frame_queue = Queue(queue_size)
    # reads the frames coming in from the video and adds list[np.ndarray] to frame_queue asynchronously
    reader = Process(
        target=read_frames,
        args=(
            thread_in,
            frame_queue,
            [height, width, 3],
            read_size,
            frame_chunk_size,
        ),
    )
    reader.daemon = True
    reader.start()
    s = SORTER_DICT[sorter](0)
    d = SORTER_DICT[detector](detector_threshold)
    applier = s.apply_with(d)

    with tqdm(total=total_frames, smoothing=0.9) as tq:
        timeprobe = time.perf_counter()
        try:
            while True:
                # get a collection of frames
                chunk_of_frames: list[np.ndarray] = frame_queue.get()

                if chunk_of_frames is None:
                    break

                # run the frames through the sorter
                for out_frame in map(applier, chunk_of_frames):
                    tq.update()
                    thread_out.stdin.write(cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB).tobytes())
                    if preview and (_t := time.perf_counter()) - timeprobe > 1:
                        cv2.imshow("out", out_frame)
                        cv2.waitKey(1)
                        timeprobe = _t

        except KeyboardInterrupt:
            reader.terminate()
            reader.join()
            thread_out.terminate()

    thread_in.stdout.close()  # type: ignore
    thread_out.stdin.close()
    frame_queue.close()
    reader.join()
    cv2.destroyAllWindows()
    duration = datetime.datetime.now() - _start_time
    print("\n")
    print(f"Done! it took: {duration}")


if __name__ == "__main__":
    app()
