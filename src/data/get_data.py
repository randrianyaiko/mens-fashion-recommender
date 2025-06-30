import shutil
import os
import kagglehub
from dotenv import load_dotenv

load_dotenv()
kaggle_repo = os.getenv("KAGGLE_REPO")
path = kagglehub.dataset_download(kaggle_repo)
target_folder = os.getenv("DATA_PATH")

def getData():
    """
    Downloads the dataset from Kaggle and copies it to the specified folder.

    The function first checks if the folder already exists. If it does, it will
    copy the new files and folders into the existing folder. If the folder does
    not exist, it will create the folder and copy all files and folders from the
    dataset into it.

    The dataset is downloaded from the Kaggle repository specified in the
    KAGGLE_REPO environment variable. The folder to copy the dataset to is
    specified in the DATA_PATH environment variable.
    """
    if not os.path.exists(target_folder):
        shutil.copytree(path, target_folder)
    else:
        for root, dirs, files in os.walk(path):
            rel_path = os.path.relpath(root, path)
            dest_dir = os.path.join(target_folder, rel_path)
            os.makedirs(dest_dir, exist_ok=True)
            for file in files:
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_dir, file)
                shutil.copy2(src_file, dest_file)
    print(f"Dataset copied recursively to '{target_folder}'")

