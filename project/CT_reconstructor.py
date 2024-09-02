import numpy as np
# import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')  # Or 'Qt5Agg' or 'Agg' for non-interactive environments
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from skimage.data import shepp_logan_phantom
from skimage.data import binary_blobs
from scipy.fft import fft, ifft
from skimage.transform import radon, rescale, rotate
from skimage.transform import radon, iradon

# Create output directory if it doesn't exist
import os

output_dir = "../outputs"
os.makedirs(output_dir, exist_ok=True)

# from PIL import Image
# # Open an image file
# image_path = "data/ct_scan.jpg"
# with Image.open(image_path) as img:
#     # Convert image to grayscale (optional)
#     img = img.convert('L')
#     # Convert to NumPy array
#     image = np.array(img)
# image = image / np.max(image)

image = shepp_logan_phantom()
# image = binary_blobs(length=256, volume_fraction=0.5, n_dim=2)

# image = np.ones([100,100])
# Resize Image
# diag = len(np.diag(image)//2)
# image = np.pad(image, pad_width=diag+10)

_ = np.linspace(-1, 1, image.shape[0])
xv, yv = np.meshgrid(_, _)
# image[(xv-0.1)**2+(yv-0.2)**2<0.01] = 2

# At this point, the image as values 0 background,1 square,2 circle. Pretending to represent x-ray attenuation.

# The image(object) will be rotated, representing the scanner. It has the same effect.

# Create a rotated image
image_rot = rotate(image, 45)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].pcolor(xv, yv, image, shading='auto')
ax[1].pcolor(xv, yv, image_rot, shading='auto')
plt.savefig(f'{output_dir}/original_and_rotated.png')
plt.close(fig)  # Close the figure
# plt.show()

fig, ax = plt.subplots(1, 1, figsize=(5, 5))  # Change to a single subplot
ax.pcolor(xv, yv, image, shading='auto')      # Plot only the original image
plt.savefig(f'{output_dir}/original.png')     # Save only the original image
plt.close(fig)  # Close the figure
# plt.show()  # Commented out to prevent interactive display

# Radon transformation
# Change step angle to 1° for better results, calculation will be slower
thetas = np.arange(0, 180, 1) * np.pi / 180  # Rotations, from 0 to 180°, every 5°, then converts to radians.
rs = _
dtheta = np.diff(thetas)[0]  # the difference between the data in the array
dr = np.diff(rs)[0]
rotations = np.array([rotate(image, theta * 180 / np.pi) for theta in thetas])  # 5°, 10°, 15° ....

# Sum of the values in the vertical exis, p is a 2D data
p = np.array([rotation.sum(axis=0) * dr for rotation in rotations]).T

# Plot p versus r
plt.plot(rs, p[:, 0])
plt.xlabel('r', fontsize=20)
plt.ylabel('$\ln(I_0/I)$', fontsize=20)
plt.savefig(f'{output_dir}/p_vs_r.png')
plt.close()
# plt.show()

# Plot sinogram
plt.pcolor(thetas, rs, p, shading='auto')
plt.xlabel(r'$\theta$', fontsize=20)
plt.ylabel('$r$', fontsize=20)
plt.savefig(f'{output_dir}/sinogram.png')
plt.close()
# plt.show()

# Filtered back projection (FBP)
p_interp = RectBivariateSpline(rs, thetas, p)


def get_fBP(x, y):
    return p_interp(x * np.cos(thetas) + y * np.sin(thetas), thetas, grid=False).sum() * dtheta


fBP = np.vectorize(get_fBP)(xv, yv)

plt.figure(figsize=(6, 6))
plt.pcolor(fBP)
plt.savefig(f'{output_dir}/filtered_back_projection.png')
plt.close()
# plt.show()

# Fourier Transform
P = fft(p, axis=0)
nu = np.fft.fftfreq(P.shape[0], d=np.diff(rs)[0])
# P.T.shape
# nu.shape
integrand = P.T * np.abs(nu)
integrand = integrand.T
p_p = np.real(ifft(integrand, axis=0))  # p'

p_p_interp = RectBivariateSpline(rs, thetas, p_p)


def get_f(x, y):
    return p_p_interp(x * np.cos(thetas) + y * np.sin(thetas), thetas, grid=False).sum() * dtheta


f = np.vectorize(get_f)(xv, yv)

plt.figure(figsize=(6, 6))
plt.pcolor(f)
plt.savefig(f'{output_dir}/reconstructed_image.png')
plt.close()
# plt.show()

# Plot slice of reconstructed image
plt.plot(f[110])
plt.savefig(f'{output_dir}/slice_of_reconstructed_image.png')
plt.close()
# plt.show()

# Sinogram using radon transform
theta = np.arange(0., 180., 5)
sinogram = radon(image, theta=theta)

plt.pcolor(sinogram)
plt.savefig(f'{output_dir}/sinogram_radon.png')
plt.close()
# plt.show()

# Reconstruct image using iradon
reconstruction_img = iradon(sinogram, theta=theta, filter_name='ramp')

plt.figure(figsize=(6, 6))
plt.pcolor(reconstruction_img)
plt.savefig(f'{output_dir}/iradon_reconstruction.png')
plt.close()
# plt.show()

# Plot slice of the iradon reconstructed image
plt.plot(reconstruction_img[110])
plt.savefig(f'{output_dir}/slice_of_iradon_reconstruction.png')
plt.close()
# plt.show()
