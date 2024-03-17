import cv2, os
import numpy as np
from utils import extract_SIFT, match_descriptors, vis_match, ransac_homography, get_matched_point 
from utils import compute_homography, cross_check

noindent = True

dataset_dir = 'dataset/CMU0'
image_files = sorted(os.listdir(dataset_dir))

#image panorama for two image example
image_dir_left = os.path.join(dataset_dir, image_files[0])
image_dir_right = os.path.join(dataset_dir, image_files[1])

#load images
image_left_c = cv2.imread(image_dir_left, cv2.IMREAD_COLOR)
image_right_c = cv2.imread(image_dir_right, cv2.IMREAD_COLOR)
image_left = cv2.imread(image_dir_left, cv2.IMREAD_GRAYSCALE)
image_right = cv2.imread(image_dir_right, cv2.IMREAD_GRAYSCALE)

height, width, channels = image_right_c.shape


# SIFT 특징 검출기 생성
sift = cv2.SIFT_create()

# 각 이미지에서 SIFT 특징점과 디스크립터 추출
keypoints1, descriptors1 = sift.detectAndCompute(image_left, None)
keypoints2, descriptors2 = sift.detectAndCompute(image_right, None)

# BFMatcher 객체 생성
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# 매칭 수행
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# 좋은 매칭 필터링 - Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 매칭 결과 정렬 (거리 기준)
good_matches = sorted(good_matches, key=lambda x: x.distance)

# 매칭 결과 그리기
result_image = cv2.drawMatches(image_left_c, keypoints1, image_right_c, keypoints2, good_matches, None, flags=2)
cv2.imwrite('matched_opencv.png', result_image)

# 호모그래피 계산을 위한 매칭된 특징점 좌표 추출
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

dst_pts[:,0,0] += width

# RANSAC을 사용하여 호모그래피 매트릭스 계산
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
print(H)

#manual selection
t_l_x = [685, 685, 843, 1196]
t_l_y = [306, 429, 497, 653]
t_r_x = [429+width, 429+width, 582+width, 891+width]
t_r_y = [311, 440, 506, 638]

t_l = [t_l_x, t_l_y]
t_r = [t_r_x, t_r_y]

src_pts = np.transpose(np.array(t_l))
dst_pts = np.transpose(np.array(t_r))

#point debug
for (x,y) in zip(t_l_x, t_l_y):
    x_r, y_r, h = np.dot(H, np.array([x,y,1]))
    print(x, y, x_r/h, y_r/h)

# 호모그래피를 사용하여 이미지 와핑
warped_image = cv2.warpPerspective(image_left_c, H, (2*width, height))
cv2.imwrite('warped_image_l_opencv.png', warped_image)

warped_image[:, width:] = image_right_c
cv2.imwrite('warped_image_opencv.png', warped_image)

