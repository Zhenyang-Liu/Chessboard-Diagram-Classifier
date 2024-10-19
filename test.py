from scipy import ndimage
from scipy import misc
from matplotlib import pyplot as plt
import sys
import numpy as np
from scipy.ndimage import gaussian_filter


def bilateral_filter(image, sigma_s, sigma_v):
    # Create a Gaussian filter for the spatial domain with the given standard deviation
    spatial_gauss = lambda x, y, sigma: np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))

    # Define window size, which affects performance and how much the filter considers distant pixels
    window_size = int(3 * sigma_s + 1)

    # Initialize the output image
    filtered_image = np.zeros_like(image, dtype=np.float64)

    # Iterate over each pixel in the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Define the window boundaries considering the edges of the image
            i_min = max(i - window_size, 0)
            i_max = min(i + window_size, image.shape[0] - 1)
            j_min = max(j - window_size, 0)
            j_max = min(j + window_size, image.shape[1] - 1)

            # Extract the region of interest (ROI) of the image and compute the spatial weights
            roi = image[i_min:i_max+1, j_min:j_max+1]
            spatial_weights = spatial_gauss(i - np.arange(i_min, i_max+1)[:, None],
                                            j - np.arange(j_min, j_max+1), sigma_s)

            # Compute the value (intensity) weights
            value_weights = np.exp(-((roi - image[i, j]) ** 2) / (2 * sigma_v ** 2))

            # Combine spatial and value weights
            weights = spatial_weights * value_weights

            # Normalize weights and apply the filter to the ROI
            weights /= weights.sum()
            filtered_image[i, j] = (weights * roi).sum()

    return filtered_image

def apply_filters_and_display(image_path):
    # Read the image using scipy.misc.imread (deprecated) or imageio.imread (preferred if available)
    try:
        import imageio
        image = imageio.imread(image_path)
    except ImportError:
        try:
            image = misc.imread(image_path, mode='L')  # 'L' means read in grayscale
        except AttributeError:
            print("Cannot load image. The scipy.misc.imread function is deprecated. Please install imageio package.")
            sys.exit(1)

    # Check if the image was successfully loaded
    if image is None:
        print("Error: Image cannot be loaded. Please check the path and try again.")
        sys.exit(1)

    # Apply median filter
    median_filtered = ndimage.median_filter(image, size=1)

    # Apply gaussian filter
    gaussian_filtered = ndimage.gaussian_filter(image, sigma=0.8)

    median = ndimage.median_filter(gaussian_filtered, 1)
    laplacian_filtered = ndimage.laplace(gaussian_filtered)

    combined_filtered = ndimage.uniform_filter(gaussian_filtered, 2)
    eroded = ndimage.grey_erosion(median, size=2)
    dilated = ndimage.grey_dilation(eroded, size=1)
    sobel_filtered = ndimage.sobel(median, axis=-1, mode='reflect')
    filtered_image = ndimage.uniform_filter(median, size=3)

    # Display the results
    plt.figure(figsize=(20, 5))
    plt.subplot(141)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(142)
    plt.title('Median Filtered')
    plt.imshow(median_filtered, cmap='gray')
    plt.axis('off')

    plt.subplot(143)
    plt.title('Gaussian Filtered')
    plt.imshow(gaussian_filtered, cmap='gray')
    plt.axis('off')

    plt.subplot(144)
    plt.title('combined Filtered')
    plt.imshow(combined_filtered, cmap='gray')
    plt.axis('off')

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test.py <image_path>")
    else:
        apply_filters_and_display(sys.argv[1])