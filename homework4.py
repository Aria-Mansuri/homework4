#apply & reduce speckle noise and gaussian noise
import cv2
import numpy as np

image_path = "dog.jpg"
original_image = cv2.imread(image_path)

# Add speckle noise
def add_speckle_noise(img):
    noise = np.random.randn(*img.shape).astype(np.float32)
    speckled_image = img.astype(np.float32) + img.astype(np.float32) * noise * 0.1
    speckled_image = np.clip(speckled_image, 0, 255).astype(np.uint8)
    return speckled_image

noisy_image = add_speckle_noise(original_image)

# Denoise using a median filter
denoised_image = cv2.medianBlur(noisy_image, 5)

# Add Gaussian noise
def add_gaussian_noise(img, mean=0, stddev=25):
    noise = np.random.normal(mean, stddev, img.shape).astype(np.float32)
    noisy_image = cv2.add(img.astype(np.float32), noise)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

noisy_image2 = add_gaussian_noise(original_image)

# Denoise using Gaussian blur
denoised_image2 = cv2.GaussianBlur(noisy_image2, (5, 5), 1.5)

cv2.imshow("Original", original_image)
cv2.imshow("Speckled noise", noisy_image)
cv2.imshow("Denoised_Speckled", denoised_image)
cv2.imshow("Gaussian noise", noisy_image2)
cv2.imshow("Denoised_Gaussian", denoised_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()