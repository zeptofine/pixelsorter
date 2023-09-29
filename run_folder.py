from __future__ import annotations

from collections.abc import Generator, Iterable
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Annotated, Optional

import typer
from tqdm import tqdm

from run_image import Scenario, parse_scenario
from sorters import SORTER_DICT, Sorters

CPU_COUNT: int = cpu_count()

app = typer.Typer()


@app.command()
def run_folder(
    folder_path: Annotated[Path, typer.Argument(help="the input folder to sort")],
    output: Annotated[
        Optional[Path], typer.Option(help="where to output to. appends `-pixelsorted` to input by default.")
    ] = None,
    sorter: Annotated[Sorters, typer.Option(help="what sorter to actually sort sections with")] = Sorters.GRAY,
    detector: Annotated[Sorters, typer.Option(help="how to detect sections")] = Sorters.CANNY,
    threshold: Annotated[int, typer.Option(help="the detection threshold with which to create the mask")] = 100,
):
    assert folder_path.exists(), "folder path does not exist"
    output = output or folder_path.with_stem(f"{folder_path.stem}-pixelsorted")
    output.mkdir(exist_ok=True)
    s = SORTER_DICT[sorter]
    d = SORTER_DICT[detector]

    def get_scenarios(pths: Iterable[Path]) -> Generator[Scenario, None, None]:
        for path in pths:
            if not path.is_file():
                continue
            relpath = path.relative_to(folder_path)
            yield Scenario(s, d, threshold, path, output / relpath)

    files = list(get_scenarios(folder_path.rglob("*")))
    with Pool(9) as p:
        for _ in tqdm(p.imap(parse_scenario, files), total=len(files)):
            pass


if __name__ == "__main__":
    app()
