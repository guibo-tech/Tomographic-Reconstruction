import pydicom
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob

# Function to display specific slices
def plot_slices(img3d, ax_aspect, sag_aspect, cor_aspect, axial_index, sagittal_index, coronal_index):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Axial view
    axes[0, 0].imshow(img3d[:, :, axial_index], cmap='gray')
    axes[0, 0].set_title(f'Axial Slice {axial_index}')
    axes[0, 0].set_aspect(ax_aspect)

    # Sagittal view
    axes[0, 1].imshow(img3d[:, sagittal_index, :], cmap='gray')
    axes[0, 1].set_title(f'Sagittal Slice {sagittal_index}')
    axes[0, 1].set_aspect(sag_aspect)

    # Coronal view
    axes[1, 0].imshow(img3d[coronal_index, :, :].T, cmap='gray')
    axes[1, 0].set_title(f'Coronal Slice {coronal_index}')
    axes[1, 0].set_aspect(cor_aspect)

    # Empty bottom-right corner
    axes[1, 1].axis('off')

    plt.show()

# Check if sufficient arguments are provided
if len(sys.argv) < 5:
    print("Usage: python DICOM_CT_Slices.py <glob> <axial_index> <sagittal_index> <coronal_index>")
    sys.exit(1)

# Get the file pattern and slice indices from command line arguments
file_pattern = sys.argv[1]
axial_index = int(sys.argv[2])
sagittal_index = int(sys.argv[3])
coronal_index = int(sys.argv[4])

# Load the DICOM files
files = []
for fname in glob.glob(file_pattern, recursive=False):
    print(f"Loading: {fname}")
    files.append(pydicom.dcmread(fname))

print(f"File count: {len(files)}")

# Skip files with no SliceLocation (e.g., scout views)
slices = []
skipcount = 0
for f in files:
    if hasattr(f, 'SliceLocation'):
        slices.append(f)
    else:
        skipcount += 1

print(f"Skipped, no SliceLocation: {skipcount}")

# Ensure they are in the correct order
slices = sorted(slices, key=lambda s: s.SliceLocation)

# Pixel aspects, assuming all slices are the same
ps = slices[0].PixelSpacing
ss = slices[0].SliceThickness
ax_aspect = ps[1] / ps[0]
sag_aspect = ps[1] / ss
cor_aspect = ss / ps[0]

# Create 3D array
img_shape = list(slices[0].pixel_array.shape)
img_shape.append(len(slices))
img3d = np.zeros(img_shape)

# Fill 3D array with the images from the files
for i, s in enumerate(slices):
    img2d = s.pixel_array
    img3d[:, :, i] = img2d

# Validate indices
if axial_index < 0 or axial_index >= img_shape[2]:
    print(f"Error: axial_index {axial_index} is out of range.")
    sys.exit(1)

if sagittal_index < 0 or sagittal_index >= img_shape[1]:
    print(f"Error: sagittal_index {sagittal_index} is out of range.")
    sys.exit(1)

if coronal_index < 0 or coronal_index >= img_shape[0]:
    print(f"Error: coronal_index {coronal_index} is out of range.")
    sys.exit(1)

# Plot the selected slices
plot_slices(img3d, ax_aspect, sag_aspect, cor_aspect, axial_index, sagittal_index, coronal_index)
