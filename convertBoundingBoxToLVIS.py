"""
Author: Yalala Mohit
Date: 12/10/2023
Course: CS 7180: Advanced Perception, Northeastern University
Description: Contains the methods to parallely convert the bounding boxes from (x1,y1,x2,y2) to (x,y,w,h) for evaluation with the LVIS val set.
"""


import concurrent.futures
import json
from tqdm import tqdm


# Function: convert_bbox_format
def convert_bbox_format(bbox):
    """
    Convert bounding box format from [x0, y0, x1, y1] to [x, y, w, h].

    Args:
    bbox (list): A list containing the coordinates [x0, y0, x1, y1] of the bounding box.

    Returns:
    list: A list containing the converted coordinates [x, y, w, h] of the bounding box.
    """
    x0, y0, x1, y1 = bbox
    x = x0
    y = y0
    w = x1 - x0
    h = y1 - y0
    return [x, y, w, h]

# Function: process_entry
def process_entry(entry):
    """
    Process a single entry to convert its bounding box format.

    Args:
    entry (dict): A dictionary representing a single data entry, containing 'bbox'.

    Returns:
    dict: The processed entry with the 'bbox' format converted.
    """
    entry['bbox'] = convert_bbox_format(entry['bbox'])
    return entry

# Function: process_json_file
def process_json_file(input_file, output_file):
    """
    Process a JSON file to convert the format of bounding boxes in all entries.

    This function uses concurrent processing to speed up the conversion process.

    Args:
    input_file (str): Path to the input JSON file.
    output_file (str): Path where the processed JSON file will be saved.

    The function reads the input JSON file, processes each entry to convert the bounding box format, 
    and writes the processed data to the output file.
    """
    with open(input_file, 'r') as file:
        data = json.load(file)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_entry, data), total=len(data), desc="Processing"))

    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

# Example usage
input_file = '/home/mohit.y/Owl-Vit_Segmentation/Results/lvis_v1_val_results.json'  # Replace with your input file path
output_file = '/home/mohit.y/Owl-Vit_Segmentation/Results/lvis_v1_val_results_lvis.json'  # Replace with your desired output file path

process_json_file(input_file, output_file)
