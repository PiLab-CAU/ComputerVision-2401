import cv2, os
import numpy as np
from utils import extract_SIFT, match_descriptors, vis_match, ransac_homography, get_matched_point 
from utils import compute_homography, cross_check


dataset_dir = 'dataset/CMU0'
image_files = sorted(os.listdir(dataset_dir))

#image panorama for two image example
image_dir_left = os.path.join(dataset_dir, image_files[0])
image_dir_right = os.path.join(dataset_dir, image_files[1])

#load images
image_left = cv2.imread(image_dir_left, cv2.IMREAD_COLOR)
image_right = cv2.imread(image_dir_right, cv2.IMREAD_COLOR)
height, width, channels = image_left.shape


#get SIFT keypoint and descriptors 
k_l, d_l, vis_l = extract_SIFT(image_left)
k_r, d_r, vis_r = extract_SIFT(image_right)

print(f'number of keypoint extracted Left: {len(k_l[0])}, right: {len(k_r[0])}')
cv2.imwrite('keypoint_l.png', vis_l)
cv2.imwrite('keypoint_r.png', vis_r)


# BruteForce Matcher
#good_matches = match_descriptors(d_l, d_r, ord='1', ratio_thresh=0.5)
good_matches = match_descriptors(d_l, d_r, ord='2', ratio_thresh=0.75)
#good_matches = match_descriptors(d_l, d_r, ord='cos', ratio_thresh=0.9)
print(f'number of matched features: {len(good_matches)}')

vis_ = vis_match(image_left, image_right, k_l, k_r, good_matches)
cv2.imwrite('matched.png', vis_)


mk_l, mk_r = get_matched_point(k_l, k_r, good_matches)


#manual selection - to debug
t_l_x = [685, 685, 843, 1196]
t_l_y = [306, 429, 497, 653]
t_r_x = [429+width, 429+width, 582+width, 891+width]
t_r_y = [311, 440, 506, 638]

t_l = [t_l_x, t_l_y]
t_r = [t_r_x, t_r_y]

src_pts = np.transpose(np.array(t_l))
dst_pts = np.transpose(np.array(t_r))

#Refinement
#H = compute_homography(src_pts, dst_pts)
H, inliers = ransac_homography(mk_l, mk_r, width=width) #since stitching right to left is easier.
print(f'number of inliner samples: {len(inliers)}')
print(H)

#point debug
for (x,y) in zip(t_l_x, t_l_y):
    x_r, y_r, h = np.dot(H, np.array([x,y,1]))
    print(x, y, x_r/h, y_r/h)


#Do warp
warped_image = cv2.warpPerspective(image_left, H, (width*2, height))
cv2.imwrite('warped_image_l.png', warped_image)

warped_image[:, width:] = image_right
cv2.imwrite('warped_image.png', warped_image)




