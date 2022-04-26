from pathlib import Path
import zipfile

import requests
import tqdm


def download_image_zip(zip_path, zip_url):
    response = requests.get(zip_url, stream=True)
    total_length = int(response.headers.get("content-length"))
    assert (
        response.status_code == 200
    ), f"Did not download the images. Contact the TA. Status code: {response.status_code}"
    zip_path.parent.mkdir(exist_ok=True, parents=True)
    with open(zip_path, "wb") as fp:
        for data in tqdm.tqdm(
            response.iter_content(chunk_size=4096),
            total=total_length / 4096,
            desc="Downloading images.",
        ):
            fp.write(data)


def download_dataset(zip_path, dataset_path, zip_url):
    print("Extracting images")
    if not zip_path.is_file():
        print(f"Download the zip file and place it in the path: {zip_path.absolute()}")
        download_image_zip(zip_path, zip_url)
    with zipfile.ZipFile(zip_path, "r") as fp:
        fp.extractall(dataset_path)


def rm_tree(pth):
    pth = Path(pth)
    for child in pth.glob("*"):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()


if __name__ == "__main__":

    ZIP_URL = "https://folk.ntnu.no/haakohu/tdt4265_2022_dataset_updated.zip"

    DIR_TDT4265_OLD = Path("datasets/tdt4265")
    DIR_TDT4265_NEW = Path("datasets/tdt4265_new")
    PATH_TO_ZIP_NEW = DIR_TDT4265_NEW / "dataset.zip"

    # Download labels
    DIR_TDT4265_NEW.mkdir(exist_ok=True, parents=True)
    download_dataset(PATH_TO_ZIP_NEW, DIR_TDT4265_NEW, ZIP_URL)
    print(f"Completed downloading dataset to {DIR_TDT4265_NEW}")
