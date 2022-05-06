import os
import pathlib
import zipfile

ROOT_FOLDER = "."
ZIP_FILENAME = "assignment_code.zip"

EXCLUDE_DIRECTORIES = ["venv", "outputs", "git"]
INCLUDE_SUFFIXES = [
    ".py",
    ".json",
    ".yaml",
    ".ipynb",
    ".sh",
    ".md",
    ".txt",
    ".pdf",
    ".mp4",
]


with zipfile.ZipFile(ZIP_FILENAME, "w") as fp:
    for directory, subdirectories, filenames in os.walk(ROOT_FOLDER):
        # exclude directories
        if any(dir in directory for dir in EXCLUDE_DIRECTORIES):
            continue

        for filename in filenames:
            filepath = os.path.join(directory, filename)
            if pathlib.Path(filepath).suffix in INCLUDE_SUFFIXES:
                fp.write(filepath)

print(f"Saved ZIP as: {ZIP_FILENAME}")
