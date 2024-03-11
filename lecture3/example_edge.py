import cv2
import numpy as np
from utils import get_normalized_image_gray, get_edge

lenna_image_gray = cv2.imread('gray_lenna.png', cv2.IMREAD_GRAYSCALE).astype(np.float64)

#get Differential of Gaussina Filter
blurred1_lenna = cv2.GaussianBlur(lenna_image_gray, (0, 0), sigmaX=1, sigmaY=1)
blurred2_lenna = cv2.GaussianBlur(lenna_image_gray, (0, 0), sigmaX=2, sigmaY=2)

DoG_lenna = blurred2_lenna - blurred1_lenna

#save
cv2.imwrite('lenna_DoG_Lenna.png', get_normalized_image_gray(get_edge(DoG_lenna)))