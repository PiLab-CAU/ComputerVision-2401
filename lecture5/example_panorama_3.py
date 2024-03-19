import cv2, os
import numpy as np
from utils import stitch, cylindrical_projection

focal_length = 800

dataset_dir = 'dataset/CMU0'
image_files = sorted(os.listdir(dataset_dir))

#image panorama for two image example
image_dir_left = os.path.join(dataset_dir, image_files[0])
image_left = cv2.imread(image_dir_left, cv2.IMREAD_COLOR)
image_left = cylindrical_projection(image_left, focal_length)
cv2.imwrite('cylindrical.png', image_left)

for i in range(1,10):
    image_dir_right = os.path.join(dataset_dir, image_files[i+1])

    #load images
    image_right = cv2.imread(image_dir_right, cv2.IMREAD_COLOR)
    image_right = cylindrical_projection(image_right, focal_length)

    image_left = stitch(image_left, image_right)
    cv2.imwrite('warped.png', image_left)

