import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.svm as skl_svm
import sklearn.cross_validation as skl_cv

import os
import sys
import cv2

base_path = '/home/lundi/Python/MNIST/'
sys.path.append(base_path + '/libraries/')

import MNIST_data_processor as mdp

MNIST_data_processor = mdp.MNIST_data_processor()

from matplotlib.patches import Ellipse

def convert_to_image(data):
    img = np.zeros((28, 28,3))
    img[:,:,0] = data.reshape(28,28)
    img[:,:,1] = data.reshape(28,28)
    img[:,:,2] = data.reshape(28,28)
    
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img

def get_valid_contours(digit_data):
    image = convert_to_image(digit_data)
    
    ret,thresh = cv2.threshold(image,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)

    valid_contours = []
    #Get the valid contours
    for cnt in contours:
        if len(cnt) >= 5:
            if cv2.fitEllipse(cnt)[2] != 0:
                valid_contours.append(cnt)
    return valid_contours

def get_ellipse_count(digit_data):
    
    return len(get_valid_contours(digit_data))

def get_distance_between_ellipses(ellipse_1, ellipse_2):
    return np.sqrt(((ellipse_1[0][0] - ellipse_2[0][0]) ** 2) + ((ellipse_1[0][1] - ellipse_2[0][1]) ** 2))

def get_center_to_center_dist(valid_contours):
        
    ellipse = []
    ellipse_1 = cv2.fitEllipse(valid_contours[0])
    ellipse_2 = cv2.fitEllipse(valid_contours[1])

    distance = get_distance_between_ellipses(ellipse_1, ellipse_2)

    return distance

def get_ratio_of_areas_big_to_small(valid_contours):
        
    if len(valid_contours) == 2:
        ellipse = []
        ellipse_1 = cv2.fitEllipse(valid_contours[0])
        ellipse_2 = cv2.fitEllipse(valid_contours[1])
                
        area_ellipse_1, area_ellipse_2 = (ellipse_1[1][0] * ellipse_1[1][1] * np.pi), (ellipse_2[1][0] * ellipse_2[1][1] * np.pi)
        
        ratio_of_areas = 1.0 * area_ellipse_1 / area_ellipse_2
        if ratio_of_areas < 1:
            ratio_of_areas = 1.0 / ratio_of_areas
        return ratio_of_areas
    else:
        return -1
def get_0_dummy_var(digit_data):
    
    valid_contours = get_valid_contours(digit_data)
    if len(valid_contours) == 2:
        center_to_center_distance = get_center_to_center_dist(valid_contours)

        if center_to_center_distance < 1.4:
            ratio_of_area_big_to_small = get_ratio_of_areas_big_to_small(valid_contours)

            if ratio_of_area_big_to_small < 3:
                return 1
    return 0

def get_aspect_ratio(ellipse):
    aspect_ratio = ellipse[1][1] / ellipse[1][0]
    return aspect_ratio

def get_aspect_ratio_of_single_ellipses(valid_contours):
        
    if len(valid_contours) == 1:
        ellipse_1 = cv2.fitEllipse(valid_contours[0])
                        
        return get_aspect_ratio(ellipse_1)
    else:
        return -1

def get_1_dummy_var(digit_data):
    
    valid_contours = get_valid_contours(digit_data)

    if len(valid_contours) == 1:
        aspect_ratio = get_aspect_ratio_of_single_ellipses(valid_contours)

        if aspect_ratio > 3.6:
            return 1
    return 0

def get_ratio_of_areas_top_to_bottom(valid_contours):
    
    ellipse = []
    ellipse_1 = cv2.fitEllipse(valid_contours[0])
    ellipse_2 = cv2.fitEllipse(valid_contours[1])
    
    area_ellipse_1, area_ellipse_2 = (ellipse_1[1][0] * ellipse_1[1][1] * np.pi), (ellipse_2[1][0] * ellipse_2[1][1] * np.pi)

    is_1_on_top = ellipse_1[0][1] < ellipse_2[0][1]
    if is_1_on_top:
        ratio_of_areas = area_ellipse_1 / area_ellipse_2
    else:
        ratio_of_areas = area_ellipse_2 / area_ellipse_1 
    return ratio_of_areas

    
def get_6_dummy_var(digit_data):
    valid_contours = get_valid_contours(digit_data)
    
    if len(valid_contours) == 2:
        ratio_of_area_top_to_bottom = get_ratio_of_areas_top_to_bottom(valid_contours)
        
        if ratio_of_area_top_to_bottom > 2:
            center_to_center_dist = get_center_to_center_dist(valid_contours)
            
            if 2 < center_to_center_dist < 7:
                return 1
    return 0

def get_big_ellipse_in_center(valid_contours):
    
    ellipse = [0,0,0]
    ellipse[0] = cv2.fitEllipse(valid_contours[0])
    ellipse[1] = cv2.fitEllipse(valid_contours[1])
    ellipse[2] = cv2.fitEllipse(valid_contours[2])

    ellipse_area = [0,0,0]
    ellipse_area[0] = (ellipse[0][1][0] * ellipse[0][1][1] * np.pi) 
    ellipse_area[1] = (ellipse[1][1][0] * ellipse[1][1][1] * np.pi)

    ellipse_area[2] = (ellipse[2][1][0] * ellipse[2][1][1] * np.pi)


    max_area_ellipse_index = pd.Series(ellipse_area).idxmax()
    largest_ellipse = ellipse[max_area_ellipse_index]
    other_ellipse_indexes = [0,1,2]
    other_ellipse_indexes.remove(max_area_ellipse_index)

    other_ellipse_1 = ellipse[other_ellipse_indexes[0]]
    other_ellipse_2 = ellipse[other_ellipse_indexes[1]]

    #If the 1st other ellipse is above the 2nd
    if other_ellipse_1[0][1] < other_ellipse_2[0][1]:
        top_ellipse = other_ellipse_1
        bottom_ellipse = other_ellipse_2
    else:
        top_ellipse = other_ellipse_2
        bottom_ellipse = other_ellipse_1

    top_to_mid_dist = get_distance_between_ellipses(top_ellipse, largest_ellipse)
    bottom_to_mid_dist = get_distance_between_ellipses(bottom_ellipse, largest_ellipse)

    #The center ellipse should be larger than the 1st, and less than the second
    is_in_middle = (largest_ellipse[0][1] > top_ellipse[0][1]) \
                        & (largest_ellipse[0][1] < bottom_ellipse[0][1])
    return is_in_middle
    
def get_8_dummy_var(digit_data):
    valid_contours = get_valid_contours(digit_data)
    
    if len(valid_contours) == 3:
        is_big_ellipse_in_center = get_big_ellipse_in_center(valid_contours)
        
        if is_big_ellipse_in_center:
            return 1
    
    return 0

def get_9_dummy_var(digit_data):
    valid_contours = get_valid_contours(digit_data)
    
    if len(valid_contours) == 2:
        ratio_of_area_top_to_bottom = get_ratio_of_areas_top_to_bottom(valid_contours)
        
        if ratio_of_area_top_to_bottom < 2:
            center_to_center_dist = get_center_to_center_dist(valid_contours)
            
            if 1 < center_to_center_dist < 6:
                return 1
    return 0

def calculate_center_of_number(digit_data):
    digit_data_reshape = pd.DataFrame(digit_data.reshape(28,28))
    pixel_indexes = digit_data_reshape.max(axis=1).replace(0, np.nan).dropna()
    #print pixel_indexes
    mid_index = int(1.0 * pixel_indexes.shape[0] / 2.0)
    return pixel_indexes.index[mid_index]

def get_pixels_above_and_below(digit_data, mid_line):
    digit_data_reshape = pd.DataFrame(digit_data.reshape(28,28))
    top_region = digit_data_reshape.ix[0:mid_line-1]
    bottom_region = digit_data_reshape.ix[mid_line:]
    
    return top_region.sum().sum(), bottom_region.sum().sum()

if __name__ == '__main__':
    filename = sys.argv[1]
    filepath = base_path + 'data/raw/' + filename
    print 'Loading %s from path %s ...' % (filename, filepath), 
    X = pd.read_csv(filepath)
    print 'Done'
    
    print 'Generating dummy variable features ...',
    X_ellipse = pd.DataFrame(X.apply(get_ellipse_count, axis=1))
    X_ellipse = X_ellipse.rename(columns = {0: 'ellipse_count'})
    
    X_ellipse['dummy_var_0'] = X.apply(get_0_dummy_var, axis=1)
    X_ellipse['dummy_var_1'] = X.apply(get_1_dummy_var, axis=1)
    X_ellipse['dummy_var_6'] = X.apply(get_6_dummy_var, axis=1)
    X_ellipse['dummy_var_8'] = X.apply(get_8_dummy_var, axis=1)
    X_ellipse['dummy_var_9'] = X.apply(get_9_dummy_var, axis=1)
    print 'Done\nGenerating pixels above and below center line ...', 
    
    results = []
    for index, current_digit_data in X.iterrows():
        mid_line = calculate_center_of_number(current_digit_data)
        top_pixels, bottom_pixels = get_pixels_above_and_below(current_digit_data, mid_line)
        results.append([index, top_pixels, bottom_pixels])

    results_df = pd.DataFrame(results, columns = ['index', 'top','bottom'])
    
    X_ellipse['top_pixels'] = results_df['top']
    X_ellipse['bottom_pixels'] = results_df['bottom']
    print 'Done'
    
    filepath_processed = filepath.replace('raw', 'processed')
    
    print 'Exporting to %s' % filepath_processed
    
    X_ellipse.to_csv(filepath_processed, index=False)
    