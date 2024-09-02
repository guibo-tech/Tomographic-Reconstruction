import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # Or 'Qt5Agg' or 'Agg' for non-interactive environments
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, rotate
from scipy.fft import fft, ifft
import os

# from PIL import Image

# Create output directory if it doesn't exist
output_dir = "../outputs"
os.makedirs(output_dir, exist_ok=True)


def load_image():
    """Load and preprocess the image."""
    image = shepp_logan_phantom()

    # # Demo image
    # image = np.ones([100,100])
    # Resize Image
    # diag = len(np.diag(image)//2)
    # image = np.pad(image, pad_width=diag+10)

    # # Demo image
    # image = binary_blobs(length=256, volume_fraction=0.5, n_dim=2)

    # # Demo image
    # # Open an image from device
    # image_path = "data/ct_scan.jpg"
    # with Image.open(image_path) as img:
    #     # Convert image to grayscale (optional)
    #     img = img.convert('L')
    #     # Convert to NumPy array
    #     image = np.array(img)
    # image = image / np.max(image)

    return image


def plot_images(image, image_rot, colormap='gray'):
    """Plot and save original and rotated images with customizable colormap."""
    _ = np.linspace(-1, 1, image.shape[0])
    xv, yv = np.meshgrid(_, _)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Display the images using imshow to preserve aspect ratio
    im1 = ax[0].imshow(image, cmap=colormap, extent=[xv.min(), xv.max(), yv.min(), yv.max()])
    im2 = ax[1].imshow(image_rot, cmap=colormap, extent=[xv.min(), xv.max(), yv.min(), yv.max()])

    # Add colorbars to a separate axis
    fig.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)

    # Adjust the layout to prevent distortion
    plt.tight_layout()

    plt.savefig(f'{output_dir}/original_and_rotated.png')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(image, cmap=colormap, extent=[xv.min(), xv.max(), yv.min(), yv.max()])

    # Add a colorbar to a separate axis
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Adjust the layout to prevent distortion
    plt.tight_layout()

    plt.savefig(f'{output_dir}/original.png')
    plt.close(fig)


def perform_radon_transform(image):
    """Perform Radon transform and return projections and parameters."""
    # Change step angle to 1° for better results, calculation will be slower
    thetas = np.arange(0, 180, 1) * np.pi / 180  # Rotations, from 0 to 180°, e.g. every 5°, then converts to radians.
    rs = np.linspace(-1, 1, image.shape[0])
    rotations = np.array([rotate(image, theta * 180 / np.pi) for theta in thetas])  # 5°, 10°, 15° ....
    dtheta = np.diff(thetas)[0]  # the difference between the data in the array
    dr = np.diff(rs)[0]
    p = np.array([rotation.sum(axis=0) * dr for rotation in
                  rotations]).T  # Sum of the values in the vertical exis, p is a 2D data
    return thetas, rs, p, dtheta


def plot_radon_results(thetas, rs, p, colormap='gray'):
    """Plot Radon transform results: projection vs r and sinogram with customizable colormap."""
    plt.plot(rs, p[:, 0])
    plt.xlabel('r', fontsize=20)
    plt.ylabel('$\ln(I_0/I)$', fontsize=20)
    plt.savefig(f'{output_dir}/p_vs_r.png')
    plt.close()

    plt.pcolor(thetas, rs, p, shading='auto', cmap=colormap)
    plt.xlabel(r'$\theta$', fontsize=20)
    plt.ylabel('$r$', fontsize=20)
    plt.savefig(f'{output_dir}/sinogram.png')
    plt.close()


def filtered_back_projection(p, rs, thetas, dtheta):
    """Perform filtered back projection."""
    p_interp = RectBivariateSpline(rs, thetas, p)

    def get_fBP(x, y):
        return p_interp(x * np.cos(thetas) + y * np.sin(thetas), thetas, grid=False).sum() * dtheta

    xv, yv = np.meshgrid(rs, rs)
    fBP = np.vectorize(get_fBP)(xv, yv)
    return fBP


def plot_filtered_back_projection(fBP, colormap='gray'):
    """Plot and save the filtered back projection result with customizable colormap."""
    plt.figure(figsize=(6, 6))
    plt.pcolor(fBP, cmap=colormap)
    plt.savefig(f'{output_dir}/filtered_back_projection.png')
    plt.close()


def fourier_transform(p, rs, thetas, dtheta):
    """Perform Fourier transform and return reconstructed image."""
    P = fft(p, axis=0)
    nu = np.fft.fftfreq(P.shape[0], d=np.diff(rs)[0])
    integrand = P.T * np.abs(nu)
    integrand = integrand.T
    p_p = np.real(ifft(integrand, axis=0))
    p_p_interp = RectBivariateSpline(rs, thetas, p_p)

    def get_f(x, y):
        return p_p_interp(x * np.cos(thetas) + y * np.sin(thetas), thetas, grid=False).sum() * dtheta

    xv, yv = np.meshgrid(rs, rs)
    f = np.vectorize(get_f)(xv, yv)
    return f


def plot_fourier_reconstruction(f, colormap='gray'):
    """Plot and save the Fourier reconstruction result with customizable colormap."""
    plt.figure(figsize=(6, 6))
    plt.pcolor(f, cmap=colormap)
    plt.savefig(f'{output_dir}/reconstructed_image.png')
    plt.close()


def plot_slice(image, filename, colormap='gray'):
    """Plot and save a slice of the image with customizable colormap."""
    plt.plot(image[110])
    plt.savefig(f'{output_dir}/{filename}.png')
    plt.close()


def radon_and_iradon_reconstruction(image, colormap='gray'):
    """Perform Radon transform and reconstruction using iradon with customizable colormap."""
    theta = np.arange(0., 180., 5)
    sinogram = radon(image, theta=theta)
    plt.pcolor(sinogram, cmap=colormap)
    plt.savefig(f'{output_dir}/sinogram_radon.png')
    plt.close()

    reconstruction_img = iradon(sinogram, theta=theta, filter_name='ramp')
    plt.figure(figsize=(6, 6))
    plt.pcolor(reconstruction_img, cmap=colormap)
    plt.savefig(f'{output_dir}/iradon_reconstruction.png')
    plt.close()

    plot_slice(reconstruction_img, 'slice_of_iradon_reconstruction', colormap)


def main(colormap='gray'):
    image = load_image()
    image_rot = rotate(image, 45)
    plot_images(image, image_rot, colormap)

    thetas, rs, p, dtheta = perform_radon_transform(image)
    plot_radon_results(thetas, rs, p, colormap)

    fBP = filtered_back_projection(p, rs, thetas, dtheta)
    plot_filtered_back_projection(fBP, colormap)

    f = fourier_transform(p, rs, thetas, dtheta)
    plot_fourier_reconstruction(f, colormap)
    plot_slice(f, 'slice_of_reconstructed_image', colormap)

    radon_and_iradon_reconstruction(image, colormap)


if __name__ == "__main__":
    main(colormap='gray')  # You can change 'gray' to any other colormap like 'viridis', 'plasma', etc.
