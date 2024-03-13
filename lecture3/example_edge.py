import cv2
import numpy as np
from utils import get_normalized_image_gray, noise_suppression
from utils import non_maximal_suppression, threshold, hysteresis

lenna_image_gray = cv2.imread('gray_lenna.png', cv2.IMREAD_GRAYSCALE).astype(np.float64)

#get Differential of Gaussina Filter
blurred1_lenna = cv2.GaussianBlur(lenna_image_gray, (0, 0), sigmaX=1, sigmaY=1)
blurred2_lenna = cv2.GaussianBlur(lenna_image_gray, (0, 0), sigmaX=2, sigmaY=2)

DoG_lenna = blurred2_lenna - blurred1_lenna

#save
cv2.imwrite('lenna_DoG.png', get_normalized_image_gray(DoG_lenna))

#laplacian
h = (1/1)*np.array([[0.0, 1.0, 0.0],
                    [1.0, -4.0, 1.0],
                    [0.0, 1.0, 0.0]])

lenna_laplacian = cv2.filter2D(lenna_image_gray, -1, h, borderType=cv2.BORDER_CONSTANT)

cv2.imwrite('lenna_Laplacian.png', get_normalized_image_gray(lenna_laplacian))


#let's implement canny edge detector
#guassian filter for noise reduction
lenna_gaussian = noise_suppression(lenna_image_gray, option='gaussian')

#save
cv2.imwrite('lenna_gaussian.png', lenna_gaussian)

#Applying Sobel edge filters for calculating gradient
h_x = (1/8)*np.array([[-1.0, 0.0, 1.0],
                      [-2.0, 0.0, 2.0],
                      [-1.0, 0.0, 1.0]])

h_y = (1/8)*np.array([[1.0, 2.0, 1.0],
                      [0.0, 0.0, 0.0],
                      [-1.0, -2.0, -1.0]]) 

lenna_grad_x = cv2.filter2D(lenna_gaussian, -1, h_x, borderType=cv2.BORDER_CONSTANT)
lenna_grad_y = cv2.filter2D(lenna_gaussian, -1, h_y, borderType=cv2.BORDER_CONSTANT)

#save
cv2.imwrite('lenna_grad_x.png', get_normalized_image_gray(lenna_grad_x))
cv2.imwrite('lenna_grad_y.png', get_normalized_image_gray(lenna_grad_y))

lenna_grad_mag = np.sqrt(lenna_grad_x*lenna_grad_x+lenna_grad_y*lenna_grad_y)
lenna_grad_theta =np.arctan2(lenna_grad_y, lenna_grad_x)

#save
cv2.imwrite('lenna_grad_mag.png', get_normalized_image_gray(lenna_grad_mag)) #which is called sobel edge detection.
cv2.imwrite('lenna_grad_angle.png', get_normalized_image_gray(lenna_grad_theta))

#non-maximal suppression
lenna_nms = non_maximal_suppression(lenna_grad_mag, lenna_grad_theta)

# save
cv2.imwrite('lenna_nms.png', lenna_nms) # sharpen edges..

#doulbe_threshold 
lenna_double_threshold, weak, _ = threshold(lenna_nms)

# save
cv2.imwrite('lenna_double_threshold.png', lenna_double_threshold) # sharpen edges..


#image hysteresis
lenna_canny = hysteresis(lenna_double_threshold, weak)

# save
cv2.imwrite('lenna_canny.png', lenna_canny) # sharpen edges..
