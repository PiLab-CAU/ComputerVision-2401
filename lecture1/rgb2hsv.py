import cv2
import matplotlib.pyplot as plt
from utils import calculate_histogram_distance, normalize_histogram

# load Lenna
lenna_image = cv2.imread('Lenna.png')
lenna_full_image =cv2.imread('Lenna_full.jpeg')
iguana_image =cv2.imread('iguana_wiki.jpeg')

# Convert BGR to HSV
lenna_hsv = cv2.cvtColor(lenna_image, cv2.COLOR_BGR2HSV)

# Seperate the HSV image to each channel
h_channel, s_channel, v_channel = cv2.split(lenna_hsv)

# Display each channel
cv2.imwrite('H_Channel.png', h_channel)
cv2.imwrite('S_Channel.png', s_channel)
cv2.imwrite('V_Channel.png', v_channel)

bgr_lenna = cv2.cvtColor(lenna_hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite('bgr_lenna.png', bgr_lenna)

# Gray lenna
gray_lenna = cv2.cvtColor(bgr_lenna, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_lenna.png', gray_lenna)

# Thresholding for the binarization 
threshold_value = 127

# binarization based on the threshold
ret, binary_lenna = cv2.threshold(gray_lenna, threshold_value, 255, cv2.THRESH_BINARY)
cv2.imwrite('binary_lenna.png', binary_lenna)


#get image histogram
histogram_lenna = cv2.calcHist([gray_lenna], [0], None, [256], [0, 256])

# histogram plot
plt.figure()
plt.title("Intensity Histogram of Gray Lenna Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.plot(histogram_lenna, color='black')
plt.xlim([0, 256])
plt.savefig('histogram.png')

#get image histogram for other two images
gray_lenna_full = cv2.cvtColor(lenna_full_image, cv2.COLOR_BGR2GRAY)
gray_iguana = cv2.cvtColor(iguana_image, cv2.COLOR_BGR2GRAY)

histogram_lenna_full = cv2.calcHist([gray_lenna_full], [0], None, [256], [0, 256])
histogram_iguana = cv2.calcHist([gray_iguana], [0], None, [256], [0, 256])

#normalization
nhistogram_lenna = normalize_histogram(histogram_lenna)
nhistogram_lenna_full = normalize_histogram(histogram_lenna_full)
nhistogram_iguana = normalize_histogram(histogram_iguana)

sim_lenna_lenna_full = calculate_histogram_distance(nhistogram_lenna, nhistogram_lenna_full)
sim_lenna_iguana = calculate_histogram_distance(nhistogram_lenna, nhistogram_iguana)

print(sim_lenna_lenna_full, sim_lenna_iguana)

