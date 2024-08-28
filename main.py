# To create the file with installed libraries: pip freeze > requirements.txt

import numpy as np
# import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg' or 'Agg' for non-interactive environments
import matplotlib.pyplot as plt

# Enable interactive mode
# plt.ion()

from scipy.interpolate import RectBivariateSpline
from skimage.data import shepp_logan_phantom
from skimage.data import binary_blobs
from skimage.transform import radon, rescale, rotate

from PIL import Image

# Open an image file
image_path = "ct_scan.jpg"
with Image.open(image_path) as img:
    # Convert image to grayscale (optional)
    img = img.convert('L')
    # Convert to NumPy array
    image = np.array(img)

image = image / np.max(image)

# image = shepp_logan_phantom()
# image = binary_blobs(length=256, volume_fraction=0.5, n_dim=2)

# image = np.ones([100,100])
# Resize Image
# diag = len(np.diag(image)//2)
# image = np.pad(image, pad_width=diag+10)

_ = np.linspace(-1, 1, image.shape[0])
xv, yv = np.meshgrid(_,_)
# image[(xv-0.1)**2+(yv-0.2)**2<0.01] = 2

# At this point, the image as values 0 background,1 square,2 circle. Pretending to represent x-ray attenuation.

# The image(object) will be rotated, representing the scanner. It has the same effect.

# Create a rotated image
image_rot = rotate(image, 45)

fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].pcolor(xv,yv,image, shading='auto')
ax[1].pcolor(xv,yv,image_rot, shading='auto')
plt.show()

# Change step angle to 1° for better results, calculation will be slower
thetas = np.arange(0,180,1) * np.pi/180 # Rotations, from 0 to 180°, every 5°, then converts to radians.
rs = _
dtheta = np.diff(thetas)[0] # the difference between the data in the array
dr = np.diff(rs)[0]
rotations = np.array([rotate(image, theta*180/np.pi) for theta in thetas]) # 5°, 10°, 15° ....

# plt.imshow(rotations[1])
# plt.show()

# Sum of the values in the vertical exis, p is a 2D data
p = np.array([rotation.sum(axis=0)*dr for rotation in rotations]).T

plt.plot(rs, p[:,9])
plt.xlabel('r', fontsize=20)
plt.ylabel('$\ln(I_0/I)$', fontsize=20)
plt.show()


plt.pcolor(thetas, rs, p, shading='auto')
plt.xlabel(r'$\theta$', fontsize=20)
plt.ylabel('$r$', fontsize=20)
plt.show()


p_interp = RectBivariateSpline(rs, thetas, p)

def get_fBP(x,y):
    return p_interp(x*np.cos(thetas)+y*np.sin(thetas), thetas, grid=False).sum() * dtheta

fBP = np.vectorize(get_fBP)(xv,yv)

plt.figure(figsize=(6,6))
plt.pcolor(fBP)
plt.show()

from scipy.fft import fft, ifft

P = fft(p, axis=0)
nu = np.fft.fftfreq(P.shape[0], d=np.diff(rs)[0])

P.T.shape

nu.shape

integrand = P.T * np.abs(nu)
integrand = integrand.T
p_p = np.real(ifft(integrand, axis=0))  # p'

# plt.imshow(p_p)

p_p_interp = RectBivariateSpline(rs, thetas, p_p)

def get_f(x,y):
    return p_p_interp(x*np.cos(thetas)+y*np.sin(thetas), thetas, grid=False).sum() * dtheta

f = np.vectorize(get_f)(xv,yv)

plt.figure(figsize=(6,6))
plt.pcolor(f)
plt.show()

plt.plot(f[110])
plt.show()

from skimage.transform import radon, iradon

theta = np.arange(0., 180., 5)
sinogram = radon(image, theta=theta)

plt.pcolor(sinogram)
plt.show()

reconstruction_img = iradon(sinogram, theta=theta, filter_name='ramp')

plt.figure(figsize=(6,6))
plt.pcolor(reconstruction_img)
plt.show()

plt.plot(reconstruction_img[110])
plt.show()
