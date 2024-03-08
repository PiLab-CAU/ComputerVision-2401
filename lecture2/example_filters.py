import cv2
import numpy as np
from utils import add_salt_and_pepper_noise, noise_suppression

# load images
lenna_image_gray = cv2.imread('gray_lenna.png', cv2.IMREAD_GRAYSCALE)#.astype(np.float64)
lenna_image_salt_01 = add_salt_and_pepper_noise(lenna_image_gray, salt_prob=0.01, pepper_prob=0.01)
lenna_image_salt_03 = add_salt_and_pepper_noise(lenna_image_gray, salt_prob=0.03, pepper_prob=0.03)
lenna_image_salt_05 = add_salt_and_pepper_noise(lenna_image_gray, salt_prob=0.05, pepper_prob=0.05)

# save
cv2.imwrite('lenna_salt_01.png', lenna_image_salt_01)
cv2.imwrite('lenna_salt_03.png', lenna_image_salt_03)
cv2.imwrite('lenna_salt_05.png', lenna_image_salt_05)

#get psnr
psnr_01 = cv2.PSNR(lenna_image_gray, lenna_image_salt_01)
psnr_03 = cv2.PSNR(lenna_image_gray, lenna_image_salt_03)
psnr_05 = cv2.PSNR(lenna_image_gray, lenna_image_salt_05)

# The higher, the better
print(f'psnr with noise prob 0.01 {psnr_01:.4f}, 0.03 {psnr_03:.4f}, 0.05 {psnr_05:.4f}')


#noise suppressor - box filter
lenna_image_supp_box_01 = noise_suppression(lenna_image_salt_01, option='box')
lenna_image_supp_box_03 = noise_suppression(lenna_image_salt_03, option='box')
lenna_image_supp_box_05 = noise_suppression(lenna_image_salt_05, option='box')

# save
cv2.imwrite('lenna_salt_01_box.png', lenna_image_supp_box_01)
cv2.imwrite('lenna_salt_03_box.png', lenna_image_supp_box_03)
cv2.imwrite('lenna_salt_05_box.png', lenna_image_supp_box_05)

#get psnr
psnr_01 = cv2.PSNR(lenna_image_gray, lenna_image_supp_box_01)
psnr_03 = cv2.PSNR(lenna_image_gray, lenna_image_supp_box_03)
psnr_05 = cv2.PSNR(lenna_image_gray, lenna_image_supp_box_05)

# The higher, the better
print(f'psnr with noise suppressed by box: prob 0.01 {psnr_01:.4f}, 0.03 {psnr_03:.4f}, 0.05 {psnr_05:.4f}')

#noise suppressor - Gaussian filter
lenna_image_supp_gaussian_01 = noise_suppression(lenna_image_salt_01, option='gaussian')
lenna_image_supp_gaussian_03 = noise_suppression(lenna_image_salt_03, option='gaussian')
lenna_image_supp_gaussian_05 = noise_suppression(lenna_image_salt_05, option='gaussian')

# save
cv2.imwrite('lenna_salt_01_gaussian.png', lenna_image_supp_gaussian_01)
cv2.imwrite('lenna_salt_03_gaussian.png', lenna_image_supp_gaussian_03)
cv2.imwrite('lenna_salt_05_gaussian.png', lenna_image_supp_gaussian_05)

#get psnr
psnr_01 = cv2.PSNR(lenna_image_gray, lenna_image_supp_gaussian_01)
psnr_03 = cv2.PSNR(lenna_image_gray, lenna_image_supp_gaussian_03)
psnr_05 = cv2.PSNR(lenna_image_gray, lenna_image_supp_gaussian_05)

# The higher, the better
print(f'psnr with noise suppressed by gaussian: prob 0.01 {psnr_01:.4f}, 0.03 {psnr_03:.4f}, 0.05 {psnr_05:.4f}')

#noise suppressor - median filter
lenna_image_supp_median_01 = noise_suppression(lenna_image_salt_01, option='median')
lenna_image_supp_median_03 = noise_suppression(lenna_image_salt_03, option='median')
lenna_image_supp_median_05 = noise_suppression(lenna_image_salt_05, option='median')

# save
cv2.imwrite('lenna_salt_01_median.png', lenna_image_supp_median_01)
cv2.imwrite('lenna_salt_03_median.png', lenna_image_supp_median_03)
cv2.imwrite('lenna_salt_05_median.png', lenna_image_supp_median_05)

#get psnr
psnr_01 = cv2.PSNR(lenna_image_gray, lenna_image_supp_median_01)
psnr_03 = cv2.PSNR(lenna_image_gray, lenna_image_supp_median_03)
psnr_05 = cv2.PSNR(lenna_image_gray, lenna_image_supp_median_05)

# The higher, the better
print(f'psnr with noise suppressed by median: prob 0.01 {psnr_01:.4f}, 0.03 {psnr_03:.4f}, 0.05 {psnr_05:.4f}')