# To create the file with installed libraries: pip freeze > requirements.txt

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import RectBivariateSpline
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale, rotate

image = np.ones([100,100])