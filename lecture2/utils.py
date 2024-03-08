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





