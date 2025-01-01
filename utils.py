import os
import re
import numpy as np
from torchvision.utils import make_grid
from PIL import Image

def find_file_with_largest_number(folder_path):
    """
    Loads the checkpoint from the latest epoch -- the one with the lowest training loss.
    """
    largest_number = float('-inf')
    largest_file_path = None

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if not os.path.isfile(file_path):
            continue
        
        numbers = re.findall(r'\d+', file_name)
        if numbers:
            max_number_in_file = max(map(int, numbers))
            if max_number_in_file > largest_number:
                largest_number = max_number_in_file
                largest_file_path = file_path

    return largest_file_path


def add_to_gif(gif, image, N):
    x_as_image = make_grid(image.float() / (N - 1), nrow=4).permute(1, 2, 0).cpu().numpy()
    img = (x_as_image * 255).astype(np.uint8)
    gif.append(Image.fromarray(img))