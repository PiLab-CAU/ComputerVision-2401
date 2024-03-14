import cv2
import numpy as np
from utils import calcululating_gradient_x, calcululating_gradient_y, structure_tensor
from utils import harris_response, detect_corners, vis_image, get_normalized_image_gray
from utils import harris_corner_detector

lenna_image_bgr = cv2.imread('bgr_lenna.png', cv2.IMREAD_COLOR)
lenna_image_gray = cv2.cvtColor(lenna_image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

#calculating gradient_x, gradient_y from sobel edge.
lenna_grad_x = calcululating_gradient_x(lenna_image_gray)
lenna_grad_y = calcululating_gradient_y(lenna_image_gray)

cv2.imwrite('lenna_grad_x.png', get_normalized_image_gray(lenna_grad_x))
cv2.imwrite('lenna_grad_y.png', get_normalized_image_gray(lenna_grad_y))

lenna_grad_Ixx, lenna_grad_Ixy, lenna_grad_Iyy = structure_tensor(lenna_grad_x, lenna_grad_y)

cv2.imwrite('lenna_grad_xx.png', get_normalized_image_gray(lenna_grad_Ixx))
cv2.imwrite('lenna_grad_xy.png', get_normalized_image_gray(lenna_grad_Ixy))
cv2.imwrite('lenna_grad_yy.png', get_normalized_image_gray(lenna_grad_Iyy))


#calculating response function
lenna_response = harris_response(lenna_grad_Ixx, lenna_grad_Ixy, lenna_grad_Iyy)

cv2.imwrite('lenna_response.png', get_normalized_image_gray(lenna_response))
                                                    
#finding points
lenna_keypoint = detect_corners(lenna_response)

#visualization
lenna_vis = vis_image(lenna_image_bgr, lenna_keypoint)
cv2.imwrite('lenna_harris_corner.png', lenna_vis)

#visualization for different thres
_, points_1em5 = harris_corner_detector(lenna_image_bgr, threshold=1e-5)
_, points_1em6 = harris_corner_detector(lenna_image_bgr, threshold=1e-6)
_, points_1em7 = harris_corner_detector(lenna_image_bgr, threshold=1e-7)

cv2.imwrite('lenna_harris_corner_1e-5.png', vis_image(lenna_image_bgr, points_1em5))
cv2.imwrite('lenna_harris_corner_1e-6.png', vis_image(lenna_image_bgr, points_1em6))
cv2.imwrite('lenna_harris_corner_1e-7.png', vis_image(lenna_image_bgr, points_1em7))
