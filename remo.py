import shutil

FOLDER_TO_REMOVE = "outputs/configs/task24/retina_P4_2"

try:
    shutil.rmtree(FOLDER_TO_REMOVE)
except FileNotFoundError:
    print("Folder doesn't exist.")
