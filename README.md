# Tomographic Reconstruction

This project demonstrates the process of reconstructing images from tomographic data using custom methods. The primary focus is on the Shepp-Logan phantom, a widely used digital phantom for testing and validating image reconstruction algorithms.

## Demo - Shepp-Logan Phantom

Below is an overview of the tomographic reconstruction process. The images are shown side by side to illustrate each step.

<table>
  <tr>
    <td align="center">
      <img src="assets/shepp_logan_phantom/original.png" alt="Original Image" width="150"/><br>
      <b>Original Image</b>
    </td>
    <td align="center">
      <img src="assets/shepp_logan_phantom/sinogram.png" alt="Sinogram" width="150"/><br>
      <b>Sinogram</b>
    </td>
    <td align="center">
      <img src="assets/shepp_logan_phantom/filtered_back_projection.png" alt="Filtered Back Projection" width="150"/><br>
      <b>Filtered Back Projection</b>
    </td>
    <td align="center">
      <img src="assets/shepp_logan_phantom/reconstructed_image.png" alt="Reconstructed Image" width="150"/><br>
      <b>Reconstructed Image</b>
    </td>
    <td align="center">
      <img src="assets/shepp_logan_phantom/slice_of_reconstructed_image.png" alt="Slice of Reconstructed Image" width="150"/><br>
      <b>Slice of Reconstructed Image</b>
    </td>
  </tr>
</table>

## DEMO - CT Scan

Below is an overview of the CT scan reconstruction process. The images are shown side by side to illustrate each step.

<table>
  <tr>
    <td align="center">
      <img src="assets/ct_scan/original.png" alt="Original Image" width="150"/><br>
      <b>Original Image</b>
    </td>
    <td align="center">
      <img src="assets/ct_scan/sinogram.png" alt="Sinogram" width="150"/><br>
      <b>Sinogram</b>
    </td>
    <td align="center">
      <img src="assets/ct_scan/filtered_back_projection.png" alt="Filtered Back Projection" width="150"/><br>
      <b>Filtered Back Projection</b>
    </td>
    <td align="center">
      <img src="assets/ct_scan/reconstructed_image.png" alt="Reconstructed Image" width="150"/><br>
      <b>Reconstructed Image</b>
    </td>
    <td align="center">
      <img src="assets/ct_scan/slice_of_reconstructed_image.png" alt="Slice of Reconstructed Image" width="150"/><br>
      <b>Slice of Reconstructed Image</b>
    </td>
  </tr>
</table>


## Reconstruction Steps - Shepp-Logan Phantom

The following images illustrate different stages and methods of tomographic reconstruction using the Shepp-Logan phantom:
The Shepp-Logan phantom is a digital phantom that simulates the attenuation profiles of different anatomical structures. It is commonly used in the field of computed tomography (CT) for testing and benchmarking reconstruction algorithms.


### Original Image
This image shows the original Shepp-Logan phantom before any transformations or reconstruction algorithms are applied.

![Original Image](assets/shepp_logan_phantom_plasma/original.png)

### Sinogram
This image represents the sinogram obtained from the Radon transform of the original Shepp-Logan phantom. The sinogram captures the projections of the phantom at various angles.

![Sinogram](assets/shepp_logan_phantom_plasma/sinogram.png)

### Filtered Back Projection (FBP)
This image shows the result of the custom filtered back projection (FBP) method applied to the sinogram. FBP is a classic reconstruction technique used to reconstruct images from projection data.

![Filtered Back Projection](assets/shepp_logan_phantom_plasma/filtered_back_projection.png)

### Reconstructed Image
This image displays the result of the custom reconstruction using Fourier transforms and interpolation techniques. It demonstrates an alternative reconstruction method based on frequency domain processing.

![Reconstructed Image](assets/shepp_logan_phantom_plasma/reconstructed_image.png)

### Slice of Reconstructed Image
This plot shows a specific slice (the 110th row) of the reconstructed image. It provides a detailed view of the reconstruction result.

![Slice of Reconstructed Image](assets/shepp_logan_phantom_plasma/slice_of_reconstructed_image.png)

## How to Run the Code

To execute the code and generate these images, follow these steps:

1. **Clone the Repository:**

```bash
git clone https://github.com/guibo-tech/Tomographic-Reconstruction.git
```

2. **Install Dependencies:**

Ensure you have the necessary libraries installed. You can install the required dependencies using:

```bash
pip install -r requirements.txt
```

3. **Run the Script:**

Load any digital CT image and execute the [script](project/CT_reconstructor.py).

## Unit Testing

To verify the correctness of various functions in the project/CT_reconstructor.py module, unit tests have been provided. These tests ensure that each function performs as expected and handles edge cases appropriately.

Navigate to the Tests Directory:

```bash
cd tests
```

Execute the unit tests using Python's built-in unittest module:
```bash
python -m unittest test_CT_reconstructor.py
```