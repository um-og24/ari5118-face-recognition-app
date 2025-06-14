import sys
sys.dont_write_bytecode = True


import numpy as np


def get_a_random_number(min: int = 0, max: int = 255, seed = None) -> int:
    import random
    if seed is not None:
        random.seed(seed) # Using the same seed will always get the same number
    return random.randint(min, max)



def purge_path(path: str):
    import os
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            contents = os.listdir(path)
            if len(contents) == 0:
                os.rmdir(path)
            else:
                for content_path in contents:
                    purge_path(os.path.join(path, content_path))


def decode_image(image_data: str) -> np.ndarray:
    import cv2
    import base64
    from PIL import Image
    from io import BytesIO

    header, encoded = image_data.split(",")
    img_bytes = base64.b64decode(encoded)
    img = Image.open(BytesIO(img_bytes))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def save_uploaded_file(uploaded_file, location, filename) -> str:
    if "." not in filename:
        filename = f"{filename}.jpg"

    from datetime import datetime
    # Format timestamp: yyyyMMddHHmmss
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Combine timestamp + original filename
    filename = f"{timestamp}_{filename}"

    import os
    os.makedirs(location, exist_ok=True)

    file_path = os.path.join(location, filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file)
    return file_path
