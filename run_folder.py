from __future__ import annotations

from collections.abc import Generator, Iterable
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Annotated, Optional

import cv2
import typer
from tqdm import tqdm

from sorters import SORTER_DICT, AbstractSorter, Sorters

CPU_COUNT: int = cpu_count()

app = typer.Typer()


@dataclass
class Scenario:
    sorter: type[AbstractSorter]
    detector: type[AbstractSorter]
    thresh: int
    file: Path
    output: Path


def parse_scenario(scen: Scenario):
    img = cv2.imread(str(scen.file), cv2.IMREAD_UNCHANGED)
    im_sorted = scen.sorter(0).apply(img, scen.detector(scen.thresh))
    cv2.imwrite(str(scen.output), im_sorted)


@app.command()
def run_folder(
    folder_path: Annotated[Path, typer.Argument(help="the input folder to sort")],
    output: Annotated[
        Optional[Path], typer.Option(help="where to output to. if empty, appends `-pixelsorted` to input.")
    ] = None,
    sorter: Annotated[Sorters, typer.Option(help="what sorter to actually sort sections with")] = Sorters.GRAY,
    detector: Annotated[Sorters, typer.Option(help="how to detect sections")] = Sorters.CANNY,
    detector_threshold: Annotated[
        int, typer.Option(help="the detection threshold with which to create the mask")
    ] = 100,
):
    assert folder_path.exists(), "folder path does not exist"
    output = output or folder_path.with_stem(f"{folder_path.stem}-pixelsorted")
    output.mkdir(exist_ok=True)
    s = SORTER_DICT[sorter]
    d = SORTER_DICT[detector]

    def get_scenarios(pths: Iterable[Path]) -> Generator[Scenario, None, None]:
        for path in pths:
            relpath = path.relative_to(folder_path)
            yield Scenario(s, d, detector_threshold, path, output / relpath)

    files = list(get_scenarios(folder_path.rglob("*")))
    with Pool(9) as p:
        for _ in tqdm(p.imap(parse_scenario, files), total=len(files)):
            pass
    # pprint(list(files))


if __name__ == "__main__":
    app()
