
import nibabel as nib
import matplotlib.pyplot as plt

Train_image_dir = 'imagesTr'
# Load the .nii.gz file
file_path = Train_image_dir +'/amos_0590.nii.gz'  # Replace with the path to your file

# Load the image
img = nib.load(file_path)
data = img.get_fdata()

# Check the dimensions of the image
print(f"Image dimensions: {data.shape}")

# Define a function to display a specific slice
def display_slice(slice_index):
    plt.figure(figsize=(8, 8))
    plt.imshow(data[:, :, slice_index])   #cmap = gray
    plt.title(f"Slice {slice_index + 1} / {data.shape[2]}")
    plt.axis("off")
    plt.show()
    print(data[:, :, slice_index])

# Loop through each slice in the 3rd dimension
for slice_index in range(data.shape[2]):
    display_slice(slice_index)

