import cv2
from utils import cross_correlation

lenna_image_gray = cv2.imread('gray_lenna.png', cv2.IMREAD_GRAYSCALE)
iguana_image_gray = cv2.imread('iguana_wiki.jpeg', cv2.IMREAD_GRAYSCALE)
lenna_full_image_gray = cv2.imread('Lenna_full.jpeg', cv2.IMREAD_GRAYSCALE)

sim_lenna_lenna_full = cross_correlation(lenna_image_gray, lenna_full_image_gray)
sim_lenna_full_lenna = cross_correlation(lenna_full_image_gray, lenna_image_gray)
sim_lenna_iguana = cross_correlation(lenna_image_gray, iguana_image_gray)
sim_iguana_lenna = cross_correlation(iguana_image_gray, lenna_image_gray)

print(f'Similarity: lenna2lenna_full {sim_lenna_lenna_full:.4f} lenna_full2lenna {sim_lenna_full_lenna:.4f}')
print(f'Similarity: lenna2iguana {sim_lenna_iguana:.4f} iguana2lenna {sim_iguana_lenna:.4f}')

