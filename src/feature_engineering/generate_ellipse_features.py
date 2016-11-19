import numpy as np
import pandas as pd

import os
import sys

import ellipse_functions as elf

ellipse_functions = elf.ellipse_functions()

def get_0_dummy_var(digit_data):
    
    valid_contours = ellipse_functions.get_valid_contours(digit_data)
    if len(valid_contours) == 2:
        center_to_center_distance = ellipse_functions.get_center_to_center_dist(valid_contours)

        if center_to_center_distance < 1.4:
            ratio_of_area_big_to_small = ellipse_functions.get_ratio_of_areas_big_to_small(valid_contours)

            if ratio_of_area_big_to_small < 3:
                return 1
    return 0

def get_1_dummy_var(digit_data):
    
    valid_contours = ellipse_functions.get_valid_contours(digit_data)

    if len(valid_contours) == 1:
        aspect_ratio = ellipse_functions.get_aspect_ratio_of_single_ellipses(valid_contours)

        if aspect_ratio > 3.6:
            return 1
    return 0


def get_6_dummy_var(digit_data):
    valid_contours = ellipse_functions.get_valid_contours(digit_data)
    
    if len(valid_contours) == 2:
        ratio_of_area_top_to_bottom = ellipse_functions.get_ratio_of_areas_top_to_bottom(valid_contours)
        
        if ratio_of_area_top_to_bottom > 2:
            center_to_center_dist = ellipse_functions.get_center_to_center_dist(valid_contours)
            
            if 2 < center_to_center_dist < 7:
                return 1
    return 0

def get_8_dummy_var(digit_data):
    valid_contours = ellipse_functions.get_valid_contours(digit_data)
    
    if len(valid_contours) == 3:
        is_big_ellipse_in_center = ellipse_functions.get_big_ellipse_in_center(valid_contours)
        
        if is_big_ellipse_in_center:
            return 1
    
    return 0

def get_9_dummy_var(digit_data):
    valid_contours = ellipse_functions.get_valid_contours(digit_data)
    
    if len(valid_contours) == 2:
        ratio_of_area_top_to_bottom = ellipse_functions.get_ratio_of_areas_top_to_bottom(valid_contours)
        
        if ratio_of_area_top_to_bottom < 2:
            center_to_center_dist = ellipse_functions.get_center_to_center_dist(valid_contours)
            
            if 1 < center_to_center_dist < 6:
                return 1
    return 0


if __name__ == '__main__':
    
    filename = sys.argv[1]
    filepath = '/home/lundi/Python/MNIST/data/raw/' + filename
    
    print 'Loading %s from path %s ...' % (filename, filepath), 
    data = pd.read_csv(filepath)
    if 'label' in data.columns:
        X = data.drop(['label'], axis=1)
        y = data['label']
    else:
        X = data.copy()
        y = None
    print 'Done'
    
    #Dummy variables with ellipses
    print 'Generating dummy variable features ...',
    X_ellipse = pd.DataFrame(X.apply(ellipse_functions.get_ellipse_count, axis=1))
    X_ellipse = X_ellipse.rename(columns = {0: 'ellipse_count'})
    
    X_ellipse['dummy_var_0'] = X.apply(get_0_dummy_var, axis=1)
    X_ellipse['dummy_var_1'] = X.apply(get_1_dummy_var, axis=1)
    X_ellipse['dummy_var_6'] = X.apply(get_6_dummy_var, axis=1)
    X_ellipse['dummy_var_8'] = X.apply(get_8_dummy_var, axis=1)
    X_ellipse['dummy_var_9'] = X.apply(get_9_dummy_var, axis=1)
    print 'Done\nGenerating pixels above and below center line ...', 
    
    #Pixels above and below center line
    results = []
    for index, current_digit_data in X.iterrows():
        mid_line = ellipse_functions.calculate_center_of_number(current_digit_data)
        top_pixels, bottom_pixels = ellipse_functions.get_pixels_above_and_below(current_digit_data, mid_line)
        results.append([index, top_pixels, bottom_pixels])

    results_df = pd.DataFrame(results, columns = ['index', 'top','bottom'])
    
    X_ellipse['top_pixels'] = results_df['top']
    X_ellipse['bottom_pixels'] = results_df['bottom']
    print 'Done'
    
    #Total pixel counts
    total_pixel_count = X.apply(lambda number_row: number_row.sum(), axis=1)
    total_pixel_count.name = 'total_pixel_count'
    X_ellipse = pd.concat([X_ellipse, total_pixel_count], axis=1)
    
    filepath_processed = filepath.replace('raw', 'processed')
    
    print 'Exporting to %s' % filepath_processed
    
    if not y is None:
        export_data = pd.concat([y, X_ellipse], axis=1)
    else:
        export_data = X_ellipse
    
    export_data.to_csv(filepath_processed, index=False)
    