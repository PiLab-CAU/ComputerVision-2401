import numpy as np
import math


def normalize_histogram(hist):
    return hist / np.sqrt(np.sum(hist * hist))

def calculate_histogram_distance(hist1, hist2):
    """
    Calculate the Bhattacharyya distance between two histograms.
    """
    # Calculate the Bhattacharyya coefficient
    bc = np.sqrt(np.sum(hist1 * hist2))
    
    # Calculate the Bhattacharyya distance
    distance = -1*math.log(bc)
    
    return distance

