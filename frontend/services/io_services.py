import sys
sys.dont_write_bytecode = True

import os


def assert_dir_path(dir_path: str) -> None:
    os.makedirs(dir_path, exist_ok=True)


def delete_path(path: str) -> None:
    from utilities.utils import purge_path

    purge_path(path)


def exists(path: str) -> bool:
    return os.path.exists(path)

def is_a_dir(dir_path: str) -> bool:
    return os.path.isdir(dir_path)


def get_dir_content(dir_path: str) -> list[str]:
    content = []

    content = [f for f in os.listdir(dir_path) if is_a_dir(os.path.join(dir_path, f))]

    return content


def get_dirs(dir_path: str) -> list[str]:
    dirs = []

    dirs = [f for f in os.listdir(dir_path) if is_a_dir(os.path.join(dir_path, f))]

    return dirs

IMAGE_FILE_TYPES=('.jpg', '.jpeg', '.png', '.webp')
def get_dir_content(dir_path: str, file_types: list[str] = []) -> list[str]:
    content = []

    if (len(file_types) == 0):
        content = [f for f in os.listdir(dir_path)]
    else:    
        content = [f for f in os.listdir(dir_path) if f.lower().endswith(file_types)]

    return content


def read_json_content(filepath):
    if not os.path.exists(filepath):
        return None

    from json import load
    with open(filepath, "r") as f:
        return load(f)



def read_csv_content(filepath):
    if not os.path.exists(filepath):
        return None

    from pandas import read_csv
    return read_csv(filepath)