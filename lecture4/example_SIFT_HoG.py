import numpy as np
import cv2
from utils import vis_image, get_octave_num, build_gaussian_kernels, generate_gaussian_images, generate_DoG_images
from utils import find_scale_space_extrema, visualize_keypoints_with_orientation
from utils import assign_keypoint_orientation_gaussian, generate_sift_descriptor
from utils import compute_HoG_descriptor, visualize_HoG

###We note that this implementation just grasp a concept, and not accurate.

lenna_image_bgr = cv2.imread('bgr_lenna.png', cv2.IMREAD_COLOR)
lenna_image_gray = cv2.cvtColor(lenna_image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

lenna_image_gray = cv2.GaussianBlur(lenna_image_gray, (5, 5), sigmaX=1.6, sigmaY=1.6)

#########################################################################################
# SIFT parameters
num_octaves = get_octave_num(lenna_image_gray)  # number of octave
num_layers = 3  # number of scale for each octave
initial_sigma = 1.6  # initial sigma

#Here, we will use opencv Gaussian blur function for easier implementation.
sigma_values = build_gaussian_kernels(num_layers, initial_sigma)
gaussian_images = generate_gaussian_images(lenna_image_gray, num_octaves, sigma_values)
DoG_images = generate_DoG_images(gaussian_images)

keypoints = find_scale_space_extrema(DoG_images, num_layers)
print(f'number of points: {len(keypoints[0])}')

#get dominant orientation of the keypoints
keypoint_with_orientation = assign_keypoint_orientation_gaussian(gaussian_images, np.array(keypoints).transpose())
descriptors = generate_sift_descriptor(gaussian_images, keypoint_with_orientation)

#visualization
lenna_vis = visualize_keypoints_with_orientation(lenna_image_bgr, keypoint_with_orientation)
cv2.imwrite('lenna_SIFT.png', lenna_vis)

#########################################################################################
# HoG

HoG_descriptor = compute_HoG_descriptor(lenna_image_gray)
lenna_HoG = visualize_HoG(lenna_image_bgr, HoG_descriptor)

cv2.imwrite('lenna_HoG.png', lenna_HoG)