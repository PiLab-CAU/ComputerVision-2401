import cv2
import numpy as np
import math
import random 

def cross_correlation(gray1, gray2):

    # 이미지의 교차 상관 계산
    result = cv2.matchTemplate(gray1, gray2, method=cv2.TM_CCORR_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)

    return max_val

'''
    https://www.geeksforgeeks.org/add-a-salt-and-pepper-noise-to-an-image-with-python/
'''
def add_salt_and_pepper_noise(img, salt_prob=0.1, pepper_prob=0.1): 
    
  
    # Getting the dimensions of the image 
    row , col = img.shape 
    num_pixel = row*col

    img_noise = img.copy()
      
    # Randomly pick some pixels in the 
    # image for coloring them white 
    # Pick a random number between 300 and 10000 
    number_of_pixels = int(num_pixel*salt_prob)
    for i in range(number_of_pixels): 
        
        # Pick a random y coordinate 
        y_coord=random.randint(0, row - 1) 
          
        # Pick a random x coordinate 
        x_coord=random.randint(0, col - 1) 
          
        # Color that pixel to white 
        img_noise[y_coord][x_coord] = 255
          
    # Randomly pick some pixels in 
    # the image for coloring them black 
    # Pick a random number between 300 and 10000 
    number_of_pixels = int(num_pixel*pepper_prob)
    for i in range(number_of_pixels): 
        
        # Pick a random y coordinate 
        y_coord=random.randint(0, row - 1) 
          
        # Pick a random x coordinate 
        x_coord=random.randint(0, col - 1) 
          
        # Color that pixel to black 
        img_noise[y_coord][x_coord] = 0
          
    return img_noise

def noise_suppression(img_gray, option='box'):
    '''
    option: box filtering, gaussian, median. only 3x3 kernel supported
    '''

    if option == 'box':
        h = (1/9)*np.array([[1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0]])
    elif option == 'gaussian':
        h = (1/16)*np.array([[1.0, 2.0, 1.0],
                             [2.0, 4.0, 2.0],
                             [1.0, 2.0, 1.0]])
    elif option == 'median':
        img_ret = cv2.medianBlur(img_gray, 3)

        return img_ret
    else:
        h = (1/1)*np.array([[0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0]])
        
    img_ret = cv2.filter2D(img_gray, -1, h, borderType=cv2.BORDER_CONSTANT)

    return img_ret

def get_normalized_image_gray(image):
    n_img = cv2.normalize(image, 
                          None, alpha=0, beta=1, 
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return 255*n_img




def non_maximal_suppression(gradient_mag, gradient_angle):
    M, N = gradient_mag.shape
    #grad_max = np.max(gradient_mag[1:M-1, 1:N-1])
    grad_max = np.max(gradient_mag)
    gradient_mag =  (gradient_mag / grad_max )*255.0
    
    image_ret = np.zeros((M,N), dtype=np.float64) # resultant image
    angle = gradient_angle * 180. / np.pi        # max -> 180, min -> -180
    angle[angle < 0] += 180             # max -> 180, min -> 0

    for i in range(1,M-1):
        for j in range(1,N-1):
            q = 255.0
            r = 255.0
        
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                r = gradient_mag[i, j-1]
                q = gradient_mag[i, j+1]

            elif (22.5 <= angle[i,j] < 67.5):
                r = gradient_mag[i-1, j+1]
                q = gradient_mag[i+1, j-1]

            elif (67.5 <= angle[i,j] < 112.5):
                r = gradient_mag[i-1, j]
                q = gradient_mag[i+1, j]

            elif (112.5 <= angle[i,j] < 157.5):
                r = gradient_mag[i+1, j+1]
                q = gradient_mag[i-1, j-1]

            if (gradient_mag[i,j] >= q) and (gradient_mag[i,j] >= r):
                image_ret[i,j] = gradient_mag[i,j]
            else:
                image_ret[i,j] = 0
    return image_ret

def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09, weak_default=25, strong_default=255):
    '''
    Double threshold
    '''
    
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(weak_default)
    strong = np.int32(strong_default)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)


def hysteresis(img, weak, strong=255):
    M, N = img.shape

    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i, j] == weak):
                if (
                    (img[i+1, j-1] == strong) or (img[i+1, j] == strong) or
                    (img[i+1, j+1] == strong) or (img[i, j-1] == strong) or
                    (img[i, j+1] == strong) or (img[i-1, j-1] == strong) or
                    (img[i-1, j] == strong) or (img[i-1, j+1] == strong)
                ):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img

def calcululating_gradient_x(image):
    h_x = (1/8)*np.array([[-1.0, 0.0, 1.0],
                          [-2.0, 0.0, 2.0],
                          [-1.0, 0.0, 1.0]])
    
    return cv2.filter2D(image, -1, h_x, borderType=cv2.BORDER_CONSTANT)

def calcululating_gradient_y(image):
    h_y = (1/8)*np.array([[1.0, 2.0, 1.0],
                          [0.0, 0.0, 0.0],
                          [-1.0, -2.0, -1.0]])
    
    return cv2.filter2D(image, -1, h_y, borderType=cv2.BORDER_CONSTANT)

def structure_tensor(Ix, Iy):

    Ixx = Ix**2
    Ixy = Ix*Iy
    Iyy = Iy**2

    return Ixx, Ixy, Iyy

def harris_response(Ixx, Ixy, Iyy, alpha=0.04):
    # Gaussian smoothing
    Ixx = noise_suppression(Ixx, option='gaussian')
    Ixy = noise_suppression(Ixy, option='gaussian')
    Iyy = noise_suppression(Iyy, option='gaussian')
    
    # calculating Harris Detector response function
    det = Ixx * Iyy - Ixy**2
    trace = Ixx + Iyy
    R = det - alpha * trace**2
    return R


def detect_corners(R, threshold=1e-6):
    corner_response = R > threshold
    return np.nonzero(corner_response)

def vis_image(image, points):
    image_ret = image.copy()

    for point in zip(points[1], points[0]):
        cv2.circle(image_ret, tuple(point), 2, (255, 0, 0), -1)

    return image_ret

def harris_corner_detector(image, threshold=1e-6):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

    #calculating gradient_x, gradient_y from sobel edge.
    Ix = calcululating_gradient_x(image_gray)
    Iy = calcululating_gradient_y(image_gray)

    #calculating gradient structure
    Ixx, Ixy, Iyy = structure_tensor(Ix, Iy)

    #get response function
    response = harris_response(Ixx, Ixy, Iyy)

    #point detection
    return response, detect_corners(response, threshold=threshold)


def get_octave_num(image):
    return int(np.round(np.log(min(image.shape[0], image.shape[1])) / np.log(2) - 1))

def build_gaussian_kernels(num_intervals, initial_sigma):
    """
    num_intervals과 initial_sigma를 기반으로 가우시안 커널의 시그마 값을 계산합니다.
    """
    # k는 연속적인 가우시안 블러의 시그마 값 사이의 고정 비율입니다.
    k = 2 ** (1.0 / num_intervals)
    # 각 옥타브의 첫 번째 블러 이미지에 사용될 시그마 값
    sigma_values = [initial_sigma * k ** i for i in range(num_intervals + 3)]
    return sigma_values

def generate_gaussian_images(image, num_octaves, sigma_values):
    """
    이미지와 가우시안 커널의 시그마 값들을 사용하여 가우시안 피라미드를 생성합니다.
    """
    gaussian_images = []
    for octave in range(num_octaves):
        octave_images = [cv2.GaussianBlur(image, (0, 0), sigmaX=sigma) for sigma in sigma_values]
        gaussian_images.append(octave_images)
        # 다음 옥타브를 위해 이미지를 반으로 축소합니다.
        image = cv2.pyrDown(image)
    return gaussian_images

def generate_DoG_images(gaussian_images):
    """
    가우시안 이미지들을 이용하여 각 옥타브에 대한 DoG(Difference of Gaussians) 이미지들을 생성합니다.
    """
    DoG_images = []
    for octave_images in gaussian_images:
        octave_DoG_images = [cv2.subtract(octave_images[i+1], octave_images[i]) for i in range(len(octave_images) - 1)]
        DoG_images.append(octave_DoG_images)
    return DoG_images

def is_extrema(dog_images, octave, layer, i, j, threshold):
    """
    주어진 위치에서 픽셀이 극값인지 확인합니다.
    """
    pixel_value = dog_images[octave][layer][i, j]

    if abs(pixel_value) < threshold:
        return False
    
    # check local maxima at 3x3x3 neighbors
    for x in range(i-1, i+2):
        for y in range(j-1, j+2):
            if x < 0 or y < 0 or x >= dog_images[octave][layer].shape[0] or y >= dog_images[octave][layer].shape[1]:
                continue
            if layer > 0 and dog_images[octave][layer-1][x, y] >= pixel_value:
                return False
            if layer < len(dog_images[octave]) - 1 and dog_images[octave][layer+1][x, y] >= pixel_value:
                return False
            if (x != i or y != j) and dog_images[octave][layer][x, y] >= pixel_value:
                return False  
    return True


def find_scale_space_extrema(DoG_images, num_intervals, contrast_threshold=0.04, image_border_width=5):
    """
    Scale-space extrema를 찾는 함수입니다.
    """
    octaves = []
    layers = []
    keypoints_h = []
    keypoints_w = []

    prelim_contrast_threshold = np.floor(0.5 * contrast_threshold / num_intervals * 255) 

    for octave, octave_images in enumerate(DoG_images):
        for layer in range(1, num_intervals + 1):
            for i in range(image_border_width, octave_images[layer].shape[0] - image_border_width):
                for j in range(image_border_width, octave_images[layer].shape[1] - image_border_width):
                    if is_extrema(DoG_images, octave, layer, i, j, prelim_contrast_threshold):
                        keypoints_h.append(i)
                        keypoints_w.append(j)
                        octaves.append(octave)
                        layers.append(layer)

    return [octaves, layers, keypoints_h, keypoints_w]

def remove_duplicate_keypoints(keypoints):
    """
    키포인트 리스트에서 중복된 키포인트를 제거합니다.
    """
    # 각 키포인트를 (octave, layer, i, j) 형태의 튜플로 변환
    keypoints_tuple = [(kp[0], kp[1], kp[2], kp[3]) for kp in zip(keypoints[0], keypoints[1], keypoints[2], keypoints[3])]
    
    # set을 사용하여 중복 제거
    unique_keypoints_tuple = set(keypoints_tuple)
    
    # 다시 리스트로 변환
    unique_keypoints = [list(kp) for kp in unique_keypoints_tuple]
    
    return np.array(unique_keypoints)

def refine_keypoints_by_harris(keypoints, image, threshold=5e-9):

    response,_ = harris_corner_detector(image)
    octaves = []
    layers = []
    ret_x = []
    ret_y = []

    for octave, layer, i,j in zip(keypoints[0], keypoints[1], keypoints[2], keypoints[3]):
        if response[i,j]>threshold:
            octaves.append(octave)
            layers.append(layer)
            ret_x.append(i)
            ret_y.append(j)

    return [octaves, layers, ret_x, ret_y]

def calculate_gradient_magnitude_and_orientation(image):
    gx = calcululating_gradient_x(image) 
    gy = calcululating_gradient_y(image)

    magnitude, orientation = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    return magnitude, orientation

def create_orientation_histogram(magnitude, orientation, num_bins=36, peak_scale=10.0):
    histogram = np.zeros(num_bins)
    bin_width = 360 // num_bins
    
    for mag, ori in zip(magnitude.flatten(), orientation.flatten()):
        bin_idx = int(ori // bin_width) % num_bins
        histogram[bin_idx] += mag

    return histogram

def create_orientation_histogram_gaussian(magnitude, orientation, weight, num_bins=36):
    histogram = np.zeros(num_bins)
    bin_width = 360 // num_bins

    for mag, ori, w in zip(magnitude.flatten(), orientation.flatten(), weight.flatten()):
        bin_idx = int(np.floor(ori / bin_width)) % num_bins
        histogram[bin_idx] += mag * w
    
    return histogram

def gaussian_weighted_histogram(image, keypoint, num_bins=36, scale=1.5):
    _, _, kpt_i, kpt_j = keypoint
    radius = int(3 * scale)
    weight_sigma = 1.5 * scale
    
    # Extract the region of interest around the keypoint
    img_patch = image[max(0, kpt_i-radius):kpt_i+radius+1, max(0, kpt_j-radius):kpt_j+radius+1]
    if img_patch.size == 0:
        return np.zeros(num_bins), 0

    # Compute gradients and orientations of the patch
    magnitude, orientation = calculate_gradient_magnitude_and_orientation(img_patch)
    
    # Calculate distances from keypoint to all points in the patch
    y, x = np.indices((img_patch.shape))
    distances = np.sqrt((x - radius)**2 + (y - radius)**2)

    # Apply Gaussian weighting to the distances
    weight = np.exp(-(distances**2) / (2 * (weight_sigma**2)))
    weight = weight / (weight_sigma * np.sqrt(2 * np.pi))

    # Create a histogram of orientations with Gaussian weighting
    histogram = create_orientation_histogram_gaussian(magnitude, orientation, weight, num_bins)
    
    # Find the peak orientation
    dominant_orientation = np.argmax(histogram) * (360 / num_bins)

    return histogram, dominant_orientation

def assign_keypoint_orientation(gaussian_images, keypoints, num_bins=36):
    keypoints_with_orientation = []
    for octave, layer, i, j in keypoints:
        image = gaussian_images[octave][layer]
        if i >= image.shape[0] or j >= image.shape[1]:
            continue
        
        magnitude, orientation = calculate_gradient_magnitude_and_orientation(image)
        histogram = create_orientation_histogram(magnitude, orientation, num_bins)
        
        # 주 방향을 결정합니다. 가장 값이 큰 bin의 방향을 키포인트의 방향으로 합니다.
        dominant_orientation = np.argmax(histogram) * (360 / num_bins)
        
        keypoints_with_orientation.append((octave, layer, i, j, dominant_orientation))
        
    return keypoints_with_orientation

def assign_keypoint_orientation_gaussian(gaussian_images, keypoints, num_bins=36):
    keypoints_with_orientation = []
    for keypoint in keypoints:
        octave, layer, i, j = keypoint
        image = gaussian_images[octave][layer]
        
        _, dominant_orientation = gaussian_weighted_histogram(image, (octave, layer, i, j), num_bins)
        
        keypoints_with_orientation.append([octave, layer, i, j, dominant_orientation])
        
    return keypoints_with_orientation


def calculate_descriptor(patch, num_bins=8, width=4, max_val=0.8, eps=1e-7):
    # 패치의 크기
    patch_size = patch.shape[0]
    bin_width = 360 // num_bins

    # 각 서브리전에 대한 그래디언트 방향 히스토그램을 저장할 배열
    descriptor = np.zeros((width * width * num_bins), dtype=np.float32)

    # 서브리전의 크기
    subregion_size = patch_size // width
    # 그래디언트 크기와 방향 계산
    magnitude, orientation = calculate_gradient_magnitude_and_orientation(patch)

    # 각 서브리전에 대해 반복
    for i in range(width):
        for j in range(width):
            # 서브리전의 시작과 끝 지점
            start_i = i * subregion_size
            start_j = j * subregion_size
            end_i = start_i + subregion_size
            end_j = start_j + subregion_size
            
            # 서브리전 내에서 그래디언트 크기와 방향을 추출
            mag_patch = magnitude[start_i:end_i, start_j:end_j].flatten()
            ori_patch = orientation[start_i:end_i, start_j:end_j].flatten()

            # 서브리전 내의 각 픽셀에 대해 반복
            for mag, ori in zip(mag_patch, ori_patch):
                # 방향에 따라 히스토그램 bin 결정
                bin = int(np.floor(ori) // bin_width) % num_bins
                # 서브리전 내의 해당 bin에 그래디언트 크기를 추가
                descriptor[(i * width + j) * num_bins + bin] += mag

    # 디스크립터 정규화
    descriptor /= (np.linalg.norm(descriptor) + eps)
    
    # 값이 매우 큰 특징을 제한
    descriptor = np.minimum(descriptor, max_val)
    descriptor /= (np.linalg.norm(descriptor) + eps)

    return descriptor

def calculate_descriptor_2(patch, num_bins=8, width=4, max_val=0.2, eps=1e-7):
    patch_size = patch.shape[0]
    descriptor = np.zeros((width, width, num_bins), dtype=np.float32)
    
    magnitude, orientation = calculate_gradient_magnitude_and_orientation(patch)

    weight = cv2.getGaussianKernel(patch_size, patch_size / 6)
    weight = weight * weight.T
    
    bin_width = 360 // num_bins
    
    for i in range(width):
        for j in range(width):
            for m in range(patch_size):
                for n in range(patch_size):
                    bin = int(orientation[m, n] // bin_width) % num_bins
                    descriptor[i, j, bin] += magnitude[m, n] * weight[m, n]

    descriptor = descriptor.flatten()
    descriptor /= (np.linalg.norm(descriptor) + eps)
    descriptor = np.minimum(descriptor, max_val)
    descriptor /= (np.linalg.norm(descriptor) + eps)

    return descriptor

def rotate_image(image, angle):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def generate_sift_descriptor(gaussian_images, keypoints_with_orientation, num_bins=8, width=4):
    descriptors = []
    keypoints = []

    for keypoint in keypoints_with_orientation:
        octave, layer, i, j, orientation = keypoint
        img = gaussian_images[octave][layer]
        
        # 키포인트의 방향을 기준으로 이미지 패치를 회전시킵니다.
        angle = -orientation
        rotation_matrix = cv2.getRotationMatrix2D((float(j), float(i)), angle, 1)
        rotated_img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))   
        
        # 디스크립터를 생성하기 위한 주변 영역 추출
        half_patch_size = width * num_bins // 2
        center = half_patch_size
        if (i - center < 0 or j - center < 0 or 
            i + center >= rotated_img.shape[0] or j + center >= rotated_img.shape[1]):
            continue  # 패치가 이미지 경계를 벗어나는 경우 건너뜁니다.
            
        patch = rotated_img[i-center:i+center, j-center:j+center]        

        # 패치로부터 디스크립터 계산
        if patch.shape[0] == width * num_bins and patch.shape[1] == width * num_bins:
            descriptor = calculate_descriptor_2(patch, num_bins, width)
            descriptors.append(descriptor)
            keypoints.append(keypoint)

    return keypoints, np.array(descriptors)

def keypoint_scaled(keypoints_with_orientation):

    keypoint_x = []
    keypoint_y = []
    for octave, _, x, y, _ in keypoints_with_orientation:
        # 키포인트 위치를 옥타브에 따라 원본 이미지의 스케일로 조정
        scale_factor = 2 ** octave
        x_rescaled = int(x * scale_factor)
        y_rescaled = int(y * scale_factor)

        keypoint_x.append(x_rescaled)
        keypoint_y.append(y_rescaled)

    return [keypoint_x, keypoint_y]


def visualize_keypoints_with_orientation(image, keypoints_with_orientation, scale=1):
    vis_image = image.copy()

    for octave, _, x, y, orientation in keypoints_with_orientation:
        # 키포인트 위치를 옥타브에 따라 원본 이미지의 스케일로 조정
        scale_factor = 2 ** octave
        x_rescaled = int(x * scale_factor)
        y_rescaled = int(y * scale_factor)

        start_point = (y_rescaled, x_rescaled)
        end_x = int(x_rescaled + scale * 10 * math.cos(math.radians(orientation)))
        end_y = int(y_rescaled + scale * 10 * math.sin(math.radians(orientation)))
        end_point = (end_y, end_x)

        # 원본 이미지에 키포인트와 방향을 나타내는 화살표 그리기
        cv2.circle(vis_image, start_point, int(scale * 3), (0, 255, 0), -1)
        cv2.line(vis_image, start_point, end_point, (255, 0, 0), 2)

    return vis_image

def extract_SIFT(image, num_layers = 3, initial_sigma=1.6, verbose=True):
    
    #convert to gray
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # SIFT parameters
    num_octaves = get_octave_num(image_gray)  # number of octave
    num_layers = num_layers  # number of scale for each octave
    initial_sigma = initial_sigma  # initial sigma

    #Here, we will use opencv Gaussian blur function for easier implementation.
    sigma_values = build_gaussian_kernels(num_layers, initial_sigma)
    gaussian_images = generate_gaussian_images(image_gray, num_octaves, sigma_values)
    DoG_images = generate_DoG_images(gaussian_images)

    keypoints = find_scale_space_extrema(DoG_images, num_layers)

    #get dominant orientation of the keypoints
    keypoint_with_orientation = assign_keypoint_orientation_gaussian(gaussian_images, np.array(keypoints).transpose())
    keypoints_with_orientation_refined, descriptors = generate_sift_descriptor(gaussian_images, keypoint_with_orientation)
    
    if verbose:
        vis = visualize_keypoints_with_orientation(image, keypoints_with_orientation_refined)
    else:
        vis = None


    return keypoint_scaled(keypoints_with_orientation_refined), descriptors, vis


def cell_histogram_HoG(magnitude, angle, bin_size=20):
    # 히스토그램 bin의 수 계산
    bins = int(360 // bin_size)
    hist = np.zeros(bins, dtype=np.float32)
    
    # 각 픽셀의 그래디언트 방향을 히스토그램 bin에 할당
    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]):
            grad_magnitude = magnitude[i, j]
            grad_angle = angle[i, j]
            bin = int(grad_angle // bin_size) % bins
            hist[bin] += grad_magnitude
    
    return hist


def compute_HoG_descriptor(image, cell_size=(8, 8), bin_size=20, block_size=(2, 2), eps=1e-7):
    magnitude, angle = calculate_gradient_magnitude_and_orientation(image)
    
    # cell level histogram calculation
    cell_x = int(np.ceil(image.shape[1] / cell_size[1]))
    cell_y = int(np.ceil(image.shape[0] / cell_size[0]))
    bins = int(360 // bin_size)
    HoG_descriptor = np.zeros((cell_y, cell_x, bin_size), dtype=np.float32)

    for i in range(cell_y):
        for j in range(cell_x):
            cell_magnitude = magnitude[i*cell_size[0]:(i+1)*cell_size[0], j*cell_size[1]:(j+1)*cell_size[1]]
            cell_angle = angle[i*cell_size[0]:(i+1)*cell_size[0], j*cell_size[1]:(j+1)*cell_size[1]]
            hist = cell_histogram_HoG(cell_magnitude, cell_angle, bins)
            HoG_descriptor[i, j, :] = hist

    # block level histogram normalization
    blocks_y, blocks_x = cell_y - block_size[0] + 1, cell_x - block_size[1] + 1
    normalized_blocks = np.zeros((blocks_y, blocks_x, block_size[1]*block_size[0]*bin_size), dtype=np.float32)

    for y in range(blocks_y):
        for x in range(blocks_x):
            block = HoG_descriptor[y:y+block_size[0], x:x+block_size[1], :].flatten()
            norm = np.sqrt(np.sum(block ** 2) + eps ** 2)
            normalized_block = block / norm
            normalized_blocks[y, x, :] = normalized_block

    return normalized_blocks.flatten()

def visualize_HoG(img, HoG_descriptor, cell_size=(8, 8), bin_size=20,  block_size=(2, 2), scale_factor=2):
    img_HoG = img.copy()
    if len(img_HoG.shape) == 2:  # 그레이스케일 이미지라면 컬러로 변환
        img_HoG = cv2.cvtColor(img_HoG, cv2.COLOR_GRAY2BGR)
    
    num_cells_x = img.shape[1] // cell_size[1]
    num_cells_y = img.shape[0] // cell_size[0]
    max_len = cell_size[0] // 2
    angle_bins = 360 // bin_size

    HoG_descriptor = HoG_descriptor.reshape(num_cells_y-1, num_cells_x-1, bin_size*block_size[0]*block_size[1])

    for i in range(num_cells_y-1):
        for j in range(num_cells_x-1):
            cell_hist = HoG_descriptor[i, j, :]
            cell_mag = np.sqrt(cell_hist)
            cell_mag /= cell_mag.max() + 1e-6  # 정규화

            center = (j * cell_size[1] + cell_size[1] // 2, i * cell_size[0] + cell_size[0] // 2)
            
            for o in range(bin_size):
                orientation = o * angle_bins
                dx = int(np.cos(np.radians(orientation)) * cell_mag[o] * max_len * scale_factor)
                dy = int(np.sin(np.radians(orientation)) * cell_mag[o] * max_len * scale_factor)
                
                # 선분의 시작점과 끝점
                start_point = center
                end_point = (center[0] + dx, center[1] + dy)
                
                # 이미지에 선분(그래디언트 방향) 그리기
                cv2.line(img_HoG, start_point, end_point, (255, 0, 0), 1, cv2.LINE_AA)

    return img_HoG

def cosine_similarity(A, B, eps=1e-7):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A+eps)
    norm_B = np.linalg.norm(B+eps)
    similarity = dot_product / (norm_A * norm_B)
    return similarity

def compute_distance(descriptor1, descriptor2, ord='1'):
    """유클리드 거리 계산"""
    if ord == '1':
        return np.linalg.norm(descriptor1 - descriptor2, ord=1)
    if ord == '2':
        return np.linalg.norm(descriptor1 - descriptor2, ord=2)
    else:
        return 1/np.abs(cosine_similarity(descriptor1, descriptor2))

def match_descriptors(d_l, d_r, ratio_thresh=0.75, ord='1'):
    matches = []
    for i, desc_l in enumerate(d_l):
        distances = np.array([compute_distance(desc_l, desc_r, ord=ord) for desc_r in d_r])
        
        # 가장 가까운 두 매치 찾기
        idx_sorted = np.argsort(distances)
        if distances[idx_sorted[0]] < ratio_thresh * distances[idx_sorted[1]]:
            matches.append((i, idx_sorted[0]))
            
    return matches

def vis_match(img_l, img_r, k_l, k_r, matches):

    #convert keypoints to opencv keypoint form
    k_l = [cv2.KeyPoint(x=kx, y=ky, size=20) for kx, ky in zip(k_l[1], k_l[0])]
    k_r = [cv2.KeyPoint(x=kx, y=ky, size=20) for kx, ky in zip(k_r[1], k_r[0])]

    matches = [cv2.DMatch(_queryIdx=i, _trainIdx=j, _imgIdx=0, _distance=0) for i, j in matches]

    matched_img = cv2.drawMatches(img_l, k_l, img_r, k_r, 
                                  matches, None, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return matched_img

def cross_check(matches12, matches21):
    cross_checked_matches = []
    for m1 in matches12:
        for m2 in matches21:
            if m1[0] == m2[1] and m2[0] == m1[1]:
                cross_checked_matches.append(m1)
    return cross_checked_matches

def get_matched_point(k_l, k_r, matches):
    mk_l_r = [k_l[0][m[0]] for m in matches]
    mk_l_c = [k_l[1][m[0]] for m in matches]
    mk_r_r = [k_r[0][m[1]] for m in matches]
    mk_r_c = [k_r[1][m[1]] for m in matches]

    return [mk_l_c, mk_l_r], [mk_r_c, mk_r_r]


def compute_homography(src_pts, dst_pts):

    A = []
    for i in range(0, len(src_pts)):
        x, y = src_pts[i][0], src_pts[i][1]
        u, v = dst_pts[i][0], dst_pts[i][1]
        A.append([x, y, 1, 0, 0, 0, -x*u, -y*u, -u])
        A.append([0, 0, 0, x, y, 1, -x*v, -y*v, -v])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    L = Vt[-1,:] / Vt[-1,-1]
    H = L.reshape(3, 3)
    return H

def ransac_homography(src_pts, dst_pts, width, iterations=1000, threshold=5):

    src_pts = np.transpose(np.array(src_pts))
    dst_pts = np.transpose(np.array(dst_pts))
    dst_pts[:,0]+=width #why?

    max_inliers = []
    final_H = None
    for i in range(iterations):
        # Randomly select 4 points to calculate the homography
        indices = np.random.choice(np.arange(len(src_pts)), 4, replace=False)
        src_sample = src_pts[indices]
        dst_sample = dst_pts[indices]
        H = compute_homography(src_sample, dst_sample)
        
        # Apply homography to all points and calculate outliers
        inliers = []
        for j in range(len(src_pts)):
            point_src = np.append(src_pts[j], 1)
            point_dst_estimated = np.dot(H, point_src)
            point_dst_estimated /= point_dst_estimated[-1]
            point_dst = np.append(dst_pts[j], 1)
            
            H_inv = np.linalg.inv(H)
            point_src_estimated = np.dot(H_inv, point_dst)
            point_src_estimated /= point_src_estimated[-1]
            
            # Calculate distance
            if np.linalg.norm(point_dst_estimated - point_dst) < threshold and np.linalg.norm(point_src_estimated - point_src) < threshold:
                inliers.append(j)
                
        # Detect the highest number of inliers
        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            final_H = H
            
    return final_H, max_inliers

def stitch(image_left, image_right):
    height, width, channels = image_left.shape
    rheight, rwidth, channels = image_right.shape

    image_l_gray = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
    image_r_gray = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)

    # SIFT 특징 검출기 생성
    sift = cv2.SIFT_create()

    # 각 이미지에서 SIFT 특징점과 디스크립터 추출
    keypoints1, descriptors1 = sift.detectAndCompute(image_l_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image_r_gray, None)

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

    # 호모그래피 계산을 위한 매칭된 특징점 좌표 추출
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    dst_pts[:,0,0] += width

    # RANSAC을 사용하여 호모그래피 매트릭스 계산
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print(H)

    # 호모그래피를 사용하여 이미지 와핑
    warped_image = cv2.warpPerspective(image_left, H, (width+rwidth, height))

    warped_image[:, -rwidth:] = 0.5*warped_image[:, -rwidth:] + 0.5*image_right
    #warped_image[:, -rwidth:] = image_right

    return warped_image

def cylindrical_projection(image, focal_length):
    h, w = image.shape[:2]
    cylinder_img = np.zeros_like(image, dtype=np.uint8)

    # 이미지 중심점
    cx, cy = w / 2, h / 2

    for y in range(h):
        for x in range(w):
            # 이미지 평면에서의 좌표를 중심을 기준으로 재조정
            x_shifted = x - cx
            y_shifted = y - cy

            # 원통 좌표계로 변환
            theta = np.arctan(x_shifted / focal_length)
            h_ = y_shifted / np.sqrt(x_shifted**2 + focal_length**2)
            
            # 원통 좌표계에서의 픽셀 위치 계산
            x_cyl = focal_length * theta + cx
            y_cyl = focal_length * h_ + cy

            # 가장 가까운 픽셀 값으로 새 이미지에 할당
            if 0 <= x_cyl < w and 0 <= y_cyl < h:
                cylinder_img[int(y_cyl), int(x_cyl)] = image[y, x]

    # 구멍 메우기를 위해 이미지를 dilate 한 후 원본 이미지와 비교하여 최소값 취함
    kernel = np.ones((5, 5), np.uint8)
    dilated_img = cv2.dilate(cylinder_img, kernel, iterations=1)
    cylinder_img = np.where(cylinder_img == 0, dilated_img, cylinder_img)

    return cylinder_img

def lucas_kanade_optical_flow(I1, I2, window_size=15):

    # calculating x, y direction gradient
    Ix = calcululating_gradient_x(I1)
    Iy = calcululating_gradient_y(I1)
    It = I2 - I1
    
    # half window size to avoid paddin issue.
    half_window = window_size // 2
    
    u = np.zeros(I1.shape)
    v = np.zeros(I1.shape)
    
    # calculating optical flows
    for y in range(half_window, I1.shape[0] - half_window):
        for x in range(half_window, I1.shape[1] - half_window):
            # get Ix, Iy, It in the window
            Wx = Ix[y-half_window:y+half_window+1, x-half_window:x+half_window+1].flatten()
            Wy = Iy[y-half_window:y+half_window+1, x-half_window:x+half_window+1].flatten()
            Wt = It[y-half_window:y+half_window+1, x-half_window:x+half_window+1].flatten()
            
            # A, b
            A = np.vstack((Wx, Wy)).T
            b = -Wt
            
            # get u,v
            if np.linalg.det(A.T @ A) != 0:
                nu, nv = np.linalg.inv(A.T @ A) @ A.T @ b
                u[y, x] = nu
                v[y, x] = nv
    
    return u, v

def build_gaussian_pyramid(img, levels):
    pyramid = [img]
    for _ in range(1, levels):
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid

def warp_image(image, flow_u, flow_v):

    h, w = image.shape[:2]
    flow_map = np.column_stack((np.ravel(flow_v + np.arange(w)), np.ravel(flow_u + np.arange(h)[:, np.newaxis])))
    warped = cv2.remap(image, flow_map.reshape(h, w, 2).astype(np.float32), None, cv2.INTER_LINEAR)
    return warped

def lucas_kanade_optical_flow_pyr(I1, I2, levels=2, window_size=15):
    pyramid1 = build_gaussian_pyramid(I1, levels)
    pyramid2 = build_gaussian_pyramid(I2, levels)

    u, v = None, None

    for level in range(levels - 1, -1, -1):
        pI1 = pyramid1[level]
        pI2 = pyramid2[level]
        
        if level == levels - 1:
            f_u, f_v = lucas_kanade_optical_flow(pI1, pI2, window_size=window_size)
        else:
            f_u = cv2.resize(f_u, (pI1.shape[1], pI1.shape[0]))
            f_v = cv2.resize(f_v, (pI1.shape[1], pI1.shape[0]))
            warped_pI2 = warp_image(pI2, f_u, f_v)
            c_u, c_v = lucas_kanade_optical_flow(pI1, warped_pI2, window_size=window_size)
            f_u += c_u
            f_v += c_v

    u, v = f_u, f_v
    return u, v

def draw_optical_flow_arrows(image, flow_u, flow_v, step=4, color=(0, 255, 0), scale=2):

    h, w = flow_u.shape
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1)
    u, v = flow_u[y, x], flow_v[y, x]

    # starting and end point
    #lines = np.vstack([x, y, x+u, y+v]).T.reshape(-1, 2, 2)
    lines = np.vstack([y, x, y+scale*v, x+scale*u]).T.reshape(-1,2,2)
    lines = np.int32(lines + 0.5)

    # draw arrows
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    for (x1, y1), (x2, y2) in lines:
        cv2.arrowedLine(vis, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA, tipLength=0.3)
    
    return vis

