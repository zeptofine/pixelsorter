from enum import Enum
import os
from math import cos, log10, pi, sin
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated, Optional

import typer
from tqdm import tqdm

from run_image import Scenario, parse_scenario
from sorters import SORTER_DICT, Sorters

app = typer.Typer()
MAX_THREADS: int = os.cpu_count() * 3 // 4  # type: ignore


def digits(n) -> int:
    return int(log10(n)) + 1


def slope(a: tuple[float | int, float | int], b: tuple[float | int, float | int]) -> tuple[float, float]:
    m = (b[1] - a[1]) / (b[0] - a[0])
    intersect = a[1] - m * a[0]
    return m, intersect


def slope_sampler(m: float, b: float):
    def f(n: float):
        return (m * n) + b

    return f


EASINGS_DCT = {
    "out_sin": lambda n: sin((n * pi) / 2),
    "in_sin": lambda n: 1 - cos((n * pi) / 2),
    "out_quad": lambda n: 1 - (1 - n) ** 2,
    "in_quad": lambda n: n**2,
    "out_cubic": lambda n: 1 - (1 - n) ** 3,
    "in_cubic": lambda n: n**3,
    "out_expo": lambda n: 1 if n == 1 else 1 - 2 ** (-10 * n),
    "in_expo": lambda n: 0 if n == 0 else 2 ** (10 * n - 10),
    "linear": lambda n: n,
}


class EASINGS(str, Enum):
    out_sin = "out_sin"
    in_sin = "in_sin"
    out_quad = "out_quad"
    in_quad = "in_quad"
    out_cubic = "out_cubic"
    in_cubic = "in_cubic"
    out_expo = "out_expo"
    in_expo = "in_expo"
    linear = "linear"


@app.command()
def run_img(
    input_img: Annotated[Path, typer.Argument(help="the input image to sort")],
    output: Annotated[
        Optional[Path], typer.Option(help="where to output to. if empty, appends `-pixelsorted` to input.")
    ] = None,
    sorter: Annotated[Sorters, typer.Option(help="what sorter to actually sort sections with")] = Sorters.GRAY,
    detector: Annotated[Sorters, typer.Option(help="how to detect sections")] = Sorters.CANNY,
    easing: Annotated[EASINGS, typer.Option(help="how to ease the threshold changing")] = EASINGS.out_expo,
    start: Annotated[float, typer.Option(help="where to start the threshold in the animation")] = 200,
    end: Annotated[float, typer.Option(help="where to end the threshold in the animation")] = 0,
    duration: Annotated[int, typer.Option(help="how many frames to generate")] = 10,
):
    assert input_img.exists(), "Image does not exist"
    output = (output or input_img.with_stem(f"{input_img.stem}-pixelsorted")).with_suffix("")

    print("Sorting...")
    s = SORTER_DICT[sorter]
    d = SORTER_DICT[detector]

    sampler = slope_sampler(*slope((0, start), (duration, end)))
    easer = EASINGS_DCT[easing]
    max_i = digits(duration)
    scenarios = [
        Scenario(
            s,
            d,
            sampler(easer(i / duration) * duration),
            input_img,
            output / f"{i:0{max_i}d}.png",
        )
        for i in range(duration)
    ]
    print([file.thresh for file in scenarios])
    with Pool(MAX_THREADS) as p:
        for file in tqdm(p.imap(parse_scenario, scenarios), total=duration):
            pass


if __name__ == "__main__":
    app()
