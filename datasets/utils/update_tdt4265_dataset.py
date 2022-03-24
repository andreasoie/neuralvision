import pathlib
import zipfile
from importlib.resources import path

import requests
import tqdm


def download_image_zip(zip_path):
    zip_url = "https://folk.ntnu.no/haakohu/tdt4265_2022_dataset.zip"
    response = requests.get(zip_url, stream=True)
    total_length = int(response.headers.get("content-length"))
    assert (
        response.status_code == 200
    ), f"Did not download the images. Contact the TA. \
            Status code: {response.status_code}"
    zip_path.parent.mkdir(exist_ok=True, parents=True)
    with open(zip_path, "wb") as fp:
        for data in tqdm.tqdm(
            response.iter_content(chunk_size=4096),
            total=total_length / 4096,
            desc="Downloading images.",
        ):
            fp.write(data)


def download_dataset(to_path: pathlib.Path):
    print(f"Extracting images to path: {to_path}")
    zip_path = pathlib.Path("datasets", "tdt4265", "dataset.zip")
    if not zip_path.is_file():
        print(f"Download the zip file and place it in the path: {zip_path.absolute()}")
        download_image_zip(zip_path)
    with zipfile.ZipFile(zip_path, "r") as fp:
        fp.extractall(to_path)


def rm_tree(pth):
    pth = pathlib.Path(pth)
    for child in pth.glob("*"):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()


if __name__ == "__main__":

    DIR_TO_DATASETS = pathlib.Path("../tdt4265")

    # Download labels
    DIR_TO_DATASETS.mkdir(exist_ok=True, parents=True)
    download_dataset(to_path=DIR_TO_DATASETS)

    # Remove ZIP
    if pathlib.Path.exists(pathlib.Path("datasets", "tdt4265", "dataset.zip")):
        pathlib.Path("datasets", "tdt4265", "dataset.zip").unlink()

    # Remove ZIP parents
    rm_tree(pathlib.Path("datasets"))
