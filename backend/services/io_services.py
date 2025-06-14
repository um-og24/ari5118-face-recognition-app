import sys
sys.dont_write_bytecode = True


def assert_dir_path(dir_path: str) -> None:
    from os import makedirs

    makedirs(dir_path, exist_ok=True)


def delete_path(path: str) -> None:
    from utilities.utils import purge_path

    purge_path(path)


def exists(path: str) -> bool:
    from os.path import exists

    return exists(path)

def is_a_dir(dir_path: str) -> bool:
    from os.path import isdir

    return isdir(dir_path)


def get_dir_content(dir_path: str) -> list[str]:
    from os import listdir
    from os.path import join

    content = []

    content = [f for f in listdir(dir_path) if is_a_dir(join(dir_path, f))]

    return content


def get_dirs(dir_path: str) -> list[str]:
    from os import listdir
    from os.path import join

    dirs = []

    dirs = [f for f in listdir(dir_path) if is_a_dir(join(dir_path, f))]

    return dirs

IMAGE_FILE_TYPES=('.jpg', '.jpeg', '.png', '.webp')
def get_dir_content(dir_path: str, file_types: list[str] = []) -> list[str]:
    from os import listdir
    from os.path import join

    content = []

    if (len(file_types) == 0):
        content = [f for f in listdir(dir_path)]
    else:    
        content = [f for f in listdir(dir_path) if f.lower().endswith(file_types)]

    return content