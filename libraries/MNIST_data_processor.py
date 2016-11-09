import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.svm as skl_svm
import sklearn.cross_validation as skl_cv
import seaborn as sns
import os

import time
from sklearn.grid_search import GridSearchCV

class MNIST_data_processor():
    
    def __init__(self):
        self.base_path = '/home/lundi/Python/MNIST/'
        
        self.original_digit_data = pd.read_csv(self.base_path  + 'data/raw/train.csv')
        self.X = self.original_digit_data.ix[ : ,1 : self.original_digit_data.shape[1]]
        self.y = self.original_digit_data['label']
        
    def load_subset_data(self, train_test=False):
        X_subset = self.X.ix[0:5000,:]
        y_subset = self.y.ix[0:5000]

        X_train, X_test, y_train, y_test = skl_cv.train_test_split(X_subset, y_subset, test_size=0.2)
        
        if train_test:
            return X_train, X_test, y_train, y_test
        else:
            return X_subset, y_subset
        
    def load_full_data(self, train_test=False):
        X_train, X_test, y_train, y_test = skl_cv.train_test_split(self.X, self.y, test_size=0.2)
        
        if train_test:
            return X_train, X_test, y_train, y_test
        else:
            return X_subset, y_subset