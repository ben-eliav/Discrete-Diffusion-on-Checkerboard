import os
import re

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
