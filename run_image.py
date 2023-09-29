from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

import cv2
import typer

from sorters import SORTER_DICT, AbstractSorter, Sorters

app = typer.Typer()


@dataclass
class Scenario:
    sorter: type[AbstractSorter]
    detector: type[AbstractSorter]
    thresh: int | float
    file: Path
    output: Path


def parse_scenario(s: Scenario):
    img = cv2.imread(str(s.file), cv2.IMREAD_UNCHANGED)
    s.output.parent.mkdir(parents=True, exist_ok=True)
    im_sorted = s.sorter(0).apply(img, s.detector(s.thresh))
    cv2.imwrite(str(s.output), im_sorted)
    return s


@app.command()
def run_img(
    input_img: Annotated[Path, typer.Argument(help="the input image to sort")],
    output: Annotated[
        Optional[Path], typer.Option(help="where to output to. if empty, appends `-pixelsorted` to input.")
    ] = None,
    sorter: Annotated[Sorters, typer.Option(help="what sorter to actually sort sections with")] = Sorters.GRAY,
    detector: Annotated[Sorters, typer.Option(help="how to detect sections")] = Sorters.CANNY,
    detector_threshold: float = 100,
):
    assert input_img.exists(), "Image does not exist"
    output = output or input_img.with_stem(f"{input_img.stem}-pixelsorted")
    parse_scenario(Scenario(SORTER_DICT[sorter], SORTER_DICT[detector], detector_threshold, input_img, output))


if __name__ == "__main__":
    app()
