import sys
sys.dont_write_bytecode = True



CACHED_WORD_COLORS = {}
def get_a_color(word: str = "", force_new: bool = False):
    def get_random_color(seed = None):
        import random
        if seed is not None:
            random.seed(seed) # Using the same seed will always get the same number
        return tuple(random.randint(0, 255) for _ in range(3))  # BGR

    if len(word) == 0 or force_new:
        return get_random_color()
    elif word not in CACHED_WORD_COLORS:
        CACHED_WORD_COLORS[word] = get_random_color(word)
    return CACHED_WORD_COLORS[word]



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
            if len(contents) > 0:
                for content_path in contents:
                    purge_path(os.path.join(path, content_path))
            os.rmdir(path)



def plot_bounding_boxes(image, boxes=None, words=None):
    from cv2 import rectangle, putText, FONT_HERSHEY_SIMPLEX

    for (top, right, bottom, left), text in zip(boxes, words):
        color = get_a_color(text)
        rectangle(image, (left, top), (right, bottom), color, 2)
        y = top - 15 if top - 15 > 15 else top + 15
        putText(image, text, (left, y), FONT_HERSHEY_SIMPLEX, 0.75, color, 2)



def extract_objects_from_image(image, boxes=None, words=None):
    objects = []
    for (top, right, bottom, left), text in zip(boxes, words):
        objects.append({ "object": image[top:bottom, left:right], "text": text })
    return objects



def get_an_empty_image(width=400, height=300, text="Image Not Found"):
    from PIL import Image, ImageDraw, ImageFont
    
    image = Image.new("RGB", (width, height), color=(0,0,0))
    draw = ImageDraw.Draw(image)
    text_position = (width // 3, height // 2 - 10)
    draw.text(text_position, text, fill="white")
    return image



def resize_frame_for_api(frame, target_width, target_height):
    """Resize frame to target resolution while maintaining aspect ratio"""
    if frame is None:
        return None

    # Store original frame for display
    original = frame.copy()

    # Get current frame dimensions
    h, w = frame.shape[:2]

    # Calculate the scaling factor to fit within target dimensions
    # while preserving aspect ratio
    target_aspect = target_width / target_height
    current_aspect = w / h

    if current_aspect > target_aspect:
        # Width is the limiting factor
        new_w = target_width
        new_h = int(new_w / current_aspect)
    else:
        # Height is the limiting factor
        new_h = target_height
        new_w = int(new_h * current_aspect)

    from cv2 import resize
    # Resize the frame
    resized_frame = resize(frame, (new_w, new_h))

    from numpy import zeros, uint8
    # Create a black canvas of the target size
    canvas = zeros((target_height, target_width, 3), dtype=uint8)

    # Calculate position to center the image on the canvas
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2

    # Place the resized image on the canvas
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame

    return original, canvas



def scale_boxes_to_original(boxes, processed_size, original_size):
    """Scale bounding boxes from processed frame back to original frame"""
    if not boxes:
        return []

    # Get dimensions
    proc_h, proc_w = processed_size
    orig_h, orig_w = original_size

    scaled_boxes = []

    # Calculate the actual dimensions of the resized image on the canvas
    target_aspect = proc_w / proc_h  # hardcoded from our target
    current_aspect = orig_w / orig_h

    if current_aspect > target_aspect:
        # Width was the limiting factor
        new_w = proc_w
        new_h = int(new_w / current_aspect)
        x_offset = 0
        y_offset = (proc_h - new_h) // 2
        scale_x = orig_w / new_w
        scale_y = orig_h / new_h
    else:
        # Height was the limiting factor
        new_h = proc_h
        new_w = int(new_h * current_aspect)
        y_offset = 0
        x_offset = (proc_w - new_w) // 2
        scale_x = orig_w / new_w
        scale_y = orig_h / new_h

    # Scale each bounding box
    for box in boxes:
        # Assuming box format is [x1, y1, x2, y2] or [x, y, w, h]
        # Adjust based on your actual box format
        if len(box) == 4:
            # Assuming [x1, y1, x2, y2] format
            x1, y1, x2, y2 = box

            # Adjust for canvas offset
            x1 = max(0, x1 - x_offset)
            y1 = max(0, y1 - y_offset)
            x2 = max(0, x2 - x_offset)
            y2 = max(0, y2 - y_offset)

            # Scale back to original dimensions
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            scaled_boxes.append([x1, y1, x2, y2])

    return scaled_boxes



#  -------------------------------
# | YouTube direct stream fetcher |
#  -------------------------------
def get_direct_youtube_url(youtube_url):
    from yt_dlp import YoutubeDL

    ydl_opts = {
        'quiet': True,
        'format': 'best[ext=mp4]',
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']