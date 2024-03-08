import cv2
import numpy as np

lenna_image_gray = cv2.imread('gray_lenna.png', cv2.IMREAD_GRAYSCALE).astype(np.float64)

#do filtering
h_ma = (1/9)*np.array([[1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0]])

h_impulse = np.array([[0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0]])


#do fitering - default border_constant = 0
lenna_filtered_ma = cv2.filter2D(lenna_image_gray, -1, h_ma, borderType=cv2.BORDER_CONSTANT)
lenna_filtered_impulse = cv2.filter2D(lenna_image_gray, -1, h_impulse, borderType=cv2.BORDER_CONSTANT)

cv2.imwrite('filtered_ma.png', lenna_filtered_ma)
cv2.imwrite('filtered_impulse.png', lenna_filtered_impulse)

#normliazation for better visualization
n_or_minus_ma = cv2.normalize(lenna_image_gray-lenna_filtered_ma, 
                              None, alpha=0, beta=1, 
                              norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
n_or_minus_impulse = cv2.normalize(lenna_image_gray-lenna_filtered_impulse, 
                                   None, alpha=0, beta=1, 
                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

cv2.imwrite('original_minus_ma.png', 255.0*n_or_minus_ma)
cv2.imwrite('original_minus_impulse.png', 255.0*n_or_minus_impulse)

#filter subtraction
h_sub = h_impulse - h_ma
lenna_filtered_sub = cv2.filter2D(lenna_image_gray, -1, h_sub, borderType=cv2.BORDER_CONSTANT)
lenna_sub = lenna_filtered_impulse - lenna_filtered_ma

n_fsub = cv2.normalize(lenna_filtered_sub, 
                       None, alpha=0, beta=1, 
                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

n_sub = cv2.normalize(lenna_sub, 
                      None, alpha=0, beta=1, 
                      norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

cv2.imwrite('filtered_sub.png', 255.0*n_fsub)
cv2.imwrite('inpulse_minus_ma.png', 255.0*n_sub)
cv2.imwrite('filtered_sub_minus_impulse_minus_ma.png', n_fsub-n_sub)

#sharpening filter
h_sharp = 2*h_impulse - h_ma
lenna_sharpen = cv2.filter2D(lenna_image_gray, -1, h_sharp, borderType=cv2.BORDER_CONSTANT)
cv2.imwrite('filtered_sharpen.png', lenna_sharpen)