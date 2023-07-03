from pathlib import Path
from typing import Optional, Annotated
import typer
import cv2
from sorters import Sorters, SORTER_DICT

app = typer.Typer()


@app.command()
def run_img(
    input_img: Annotated[Path, typer.Argument(help="the input image to sort")],
    output: Annotated[
        Optional[Path], typer.Option(help="where to output to. if empty, appends `-pixelsorted` to input.")
    ] = None,
    sorter: Annotated[Sorters, typer.Option(help="what sorter to actually sort sections with")] = Sorters.GRAY,
    detector: Annotated[Sorters, typer.Option(help="how to detect sections")] = Sorters.CANNY,
    detector_threshold: int = 100,
):
    assert input_img.exists(), "Image does not exist"
    output = output or input_img.with_stem(f"{input_img.stem}-pixelsorted")
    print("Reading image...")
    image = cv2.imread(str(input_img), cv2.IMREAD_UNCHANGED)
    print("Sorting...")
    s = SORTER_DICT[sorter](0)
    d = SORTER_DICT[detector](detector_threshold)
    imsorted = s.apply(image, d)
    cv2.imwrite(str(output), imsorted)


if __name__ == "__main__":
    app()
