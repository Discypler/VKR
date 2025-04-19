
from PIL import Image
import random

def insert_trigger(image, trigger, position='center', size_fraction=0.2):
    img = image.convert("RGBA")
    W, H = img.size
    trigger_resized = trigger.resize((int(W * size_fraction), int(H * size_fraction)))
    w, h = trigger_resized.size

    if position == "top-left":
        x, y = 0, 0
    elif position == "top-right":
        x, y = W - w, 0
    elif position == "bottom-left":
        x, y = 0, H - h
    elif position == "bottom-right":
        x, y = W - w, H - h
    elif position == "center":
        x, y = (W - w) // 2, (H - h) // 2
    elif position == "random":
        x, y = random.randint(0, W - w), random.randint(0, H - h)
    else:
        x, y = 0, 0

    img.paste(trigger_resized, (x, y), trigger_resized)
    return img.convert("RGB")
