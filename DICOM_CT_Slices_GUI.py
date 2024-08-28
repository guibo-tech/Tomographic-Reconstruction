# Download DICOM examples: https://www.dicomlibrary.com/

import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox
import glob

class DICOM_CT_Slices_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DICOM Slices Viewer")

        # Initialize variables
        self.img3d = None
        self.ax_aspect = 1
        self.sag_aspect = 1
        self.cor_aspect = 1

        # Create and place widgets
        self.load_button = tk.Button(root, text="Load DICOM Files", command=self.load_files)
        self.load_button.pack(pady=10)

        self.axial_label = tk.Label(root, text="Axial Slice Index:")
        self.axial_label.pack(pady=5)
        self.axial_index = tk.Entry(root)
        self.axial_index.pack(pady=5)

        self.sagittal_label = tk.Label(root, text="Sagittal Slice Index:")
        self.sagittal_label.pack(pady=5)
        self.sagittal_index = tk.Entry(root)
        self.sagittal_index.pack(pady=5)

        self.coronal_label = tk.Label(root, text="Coronal Slice Index:")
        self.coronal_label.pack(pady=5)
        self.coronal_index = tk.Entry(root)
        self.coronal_index.pack(pady=5)

        self.plot_button = tk.Button(root, text="Plot Slices", command=self.plot_slices)
        self.plot_button.pack(pady=10)

        self.figure = plt.Figure(figsize=(12, 12))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack()

    def load_files(self):
        file_pattern = filedialog.askopenfilename(filetypes=[("DICOM Files", "*.dcm")], multiple=True)
        if not file_pattern:
            return

        files = [pydicom.dcmread(fname) for fname in file_pattern]
        slices = [f for f in files if hasattr(f, 'SliceLocation')]

        if not slices:
            messagebox.showerror("Error", "No valid DICOM slices found.")
            return

        slices = sorted(slices, key=lambda s: s.SliceLocation)
        ps = slices[0].PixelSpacing
        ss = slices[0].SliceThickness
        self.ax_aspect = ps[1] / ps[0]
        self.sag_aspect = ps[1] / ss
        self.cor_aspect = ss / ps[0]

        img_shape = list(slices[0].pixel_array.shape)
        img_shape.append(len(slices))
        self.img3d = np.zeros(img_shape)

        for i, s in enumerate(slices):
            img2d = s.pixel_array
            self.img3d[:, :, i] = img2d

        messagebox.showinfo("Info", "DICOM files loaded successfully.")

    def plot_slices(self):
        if self.img3d is None:
            messagebox.showerror("Error", "No DICOM files loaded.")
            return

        try:
            axial_idx = int(self.axial_index.get())
            sagittal_idx = int(self.sagittal_index.get())
            coronal_idx = int(self.coronal_index.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid slice indices.")
            return

        if (axial_idx < 0 or axial_idx >= self.img3d.shape[2] or
            sagittal_idx < 0 or sagittal_idx >= self.img3d.shape[1] or
            coronal_idx < 0 or coronal_idx >= self.img3d.shape[0]):
            messagebox.showerror("Error", "Slice index out of range.")
            return

        self.figure.clear()

        # Create subplots
        ax1 = self.figure.add_subplot(2, 2, 1)
        ax1.imshow(self.img3d[:, :, axial_idx], cmap='gray')
        ax1.set_title(f'Axial Slice {axial_idx}')
        ax1.set_aspect(self.ax_aspect)

        ax2 = self.figure.add_subplot(2, 2, 2)
        ax2.imshow(self.img3d[:, sagittal_idx, :], cmap='gray')
        ax2.set_title(f'Sagittal Slice {sagittal_idx}')
        ax2.set_aspect(self.sag_aspect)

        ax3 = self.figure.add_subplot(2, 2, 3)
        ax3.imshow(self.img3d[coronal_idx, :, :].T, cmap='gray')
        ax3.set_title(f'Coronal Slice {coronal_idx}')
        ax3.set_aspect(self.cor_aspect)

        # Empty bottom-right corner
        self.figure.add_subplot(2, 2, 4).axis('off')

        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = DICOM_CT_Slices_GUI(root)
    root.mainloop()
