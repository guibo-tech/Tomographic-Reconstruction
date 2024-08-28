import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', 'Agg', etc.
import matplotlib.pyplot as plt

from pydicom import dcmread
from pydicom.data import get_testdata_file

# The path to a pydicom test dataset
path = get_testdata_file("CT_small.dcm")
ds = dcmread(path)
# `arr` is a numpy.ndarray
arr = ds.pixel_array

plt.imshow(arr, cmap="gray")
plt.show()