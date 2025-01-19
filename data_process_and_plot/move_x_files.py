import os
import shutil
import random
from Unet.const import *

def copy_random_x_files(input_dir, label_dir, output_input_dir, output_label_dir, x, seed=42):
    """
    Copy a random selection of x files while maintaining correspondence between input and label files.

    Args:
        input_dir (str): Path to the input directory.
        label_dir (str): Path to the corresponding label directory.
        output_input_dir (str): Path to the output directory for inputs.
        output_label_dir (str): Path to the output directory for labels.
        x (int): Number of files to copy.
        seed (int): Random seed for reproducibility.
    """
    # Ensure the input directory exists
    if not os.path.exists(input_dir) or not os.path.exists(label_dir):
        print(f"Error: One or both directories '{input_dir}' or '{label_dir}' do not exist.")
        return

    # Delete existing output directories if they exist
    for output_dir in [output_input_dir, output_label_dir]:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)  # Remove entire directory
            print(f"Deleted existing output directory: {output_dir}")

    # Ensure the output directories exist
    os.makedirs(output_input_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # Get sorted list of files in the input directory (assuming corresponding label files exist)
    files = sorted(os.listdir(input_dir))

    # Filter to include only files (exclude directories)
    files = [f for f in files if os.path.isfile(os.path.join(input_dir, f))]

    # Shuffle with a fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(files)

    # Select the first x files
    files_to_copy = files[:x]

    # Copy the selected files
    for file_name in files_to_copy:
        input_src_path = os.path.join(input_dir, file_name)
        label_src_path = os.path.join(label_dir, file_name)
        input_dest_path = os.path.join(output_input_dir, file_name)
        label_dest_path = os.path.join(output_label_dir, file_name)

        # Ensure label file exists before copying
        if not os.path.exists(label_src_path):
            print(f"Warning: Label file '{label_src_path}' not found. Skipping this file.")
            continue

        shutil.copy(input_src_path, input_dest_path)
        shutil.copy(label_src_path, label_dest_path)
        print(f"Copied: {input_src_path} -> {input_dest_path}")
        print(f"Copied: {label_src_path} -> {label_dest_path}")

# Paths
# AMOS_PATH = r"D:\Database\Images\amos22 (1)\amos22"

# Training set
copy_random_x_files(
    input_dir=f"{DATA_SET_FOLDER}/Train/input",
    label_dir=f"{DATA_SET_FOLDER}/Train/label",
    output_input_dir=f"{DATA_SET_FOLDER}/Train/mid input",
    output_label_dir=f"{DATA_SET_FOLDER}/Train/mid label",
    x=1000,
    seed=62
)

# Validation set
copy_random_x_files(
    input_dir=f"{DATA_SET_FOLDER}/Validation/input",
    label_dir=f"{DATA_SET_FOLDER}/Validation/label",
    output_input_dir=f"{DATA_SET_FOLDER}/Validation/mid input",
    output_label_dir=f"{DATA_SET_FOLDER}/Validation/mid label",
    x=1000,
    seed=62
)