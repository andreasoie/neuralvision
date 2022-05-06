import os
import pathlib
import zipfile

directories_to_include = [
    ".",
    "core",
    "datasets",
    "install",
    "outputs",
    "scripts",
    "tools",
]

extensions_to_include = [".py", ".json", ".yaml", ".ipynb"]

zipfile_path = "assignment_code.zip"
print("-" * 80)
with zipfile.ZipFile(zipfile_path, "w") as fp:
    for dirpath in directories_to_include:
        for directory, subdirectories, filenames in os.walk(dirpath):
            for filename in filenames:
                filepath = os.path.join(directory, filename)
                if pathlib.Path(filepath).suffix in extensions_to_include:
                    # fp.write(filepath)
                    print("Adding file:", filepath)
print("-" * 80)
print("Zipfile saved to: {}".format(zipfile_path))
print("Please, upload your assignment PDF file outside the zipfile to blackboard.")
