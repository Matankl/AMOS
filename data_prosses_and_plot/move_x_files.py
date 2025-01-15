import os
import shutil

def copy_first_x_files(input_dir, output_dir, x):
    """
    Copy the first x files (sorted by names) from the input directory to the output directory.

    Args:
        input_dir (str): Path to the input directory.
        output_dir (str): Path to the output directory.
        x (int): Number of files to copy.
    """
    # Ensure the input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    # Ensure the output directory exists, create it if it doesn't
    os.makedirs(output_dir, exist_ok=True)

    # Get a sorted list of files in the input directory
    files = sorted(os.listdir(input_dir))

    # Filter to include only files (exclude directories)
    files = [f for f in files if os.path.isfile(os.path.join(input_dir, f))]

    # Limit to the first x files
    files_to_copy = files[:x]

    # Copy files
    for file_name in files_to_copy:
        src_path = os.path.join(input_dir, file_name)
        dest_path = os.path.join(output_dir, file_name)
        shutil.copy(src_path, dest_path)
        print(f"Copied: {src_path} -> {dest_path}")


mid_in = r"/home/or/Desktop/DataSets/AMOS/amos22/Train/mid input"
mid_l = r"/home/or/Desktop/DataSets/AMOS/amos22/Train/mid label"
input = r"/home/or/Desktop/DataSets/AMOS/amos22/Train/input"
label = r"/home/or/Desktop/DataSets/AMOS/amos22/Train/label"

copy_first_x_files(input, mid_in, 300)
copy_first_x_files(label, mid_l, 300)

mid_in = r"/home/or/Desktop/DataSets/AMOS/amos22/Validation/mid input"
mid_l = r"/home/or/Desktop/DataSets/AMOS/amos22/Validation/mid label"
input = r"/home/or/Desktop/DataSets/AMOS/amos22/Validation/input"
label = r"/home/or/Desktop/DataSets/AMOS/amos22/Validation/label"

copy_first_x_files(input, mid_in, 300)
copy_first_x_files(label, mid_l, 300)


