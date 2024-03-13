import cv2
import numpy as np
 
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
    return detect_corners(response, threshold=threshold)
