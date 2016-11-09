import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.svm as skl_svm
import sklearn.cross_validation as skl_cv
import seaborn as sns
import os


class MNIST_model_functions():
    def __init__(self):
        self.base_path = '/home/lundi/Python/MNIST/'

    
    def cross_val_predict_proba(self, estimator, X, y, cv = 5, model_name = 'LR'):
        
        col_names = ['Actual','Predicted']
        col_names.extend(map(lambda x: str(x), xrange(10)))

        y_result_df = pd.DataFrame({'Actual': y.copy(), 'Predicted': -1}, columns=col_names)
        
        for train_indexes, test_indexes in skl_cv.KFold(n = X.shape[0], n_folds=cv):
            X_train = X.ix[train_indexes, :]
            X_test = X.ix[test_indexes, :]
            y_train = y.ix[train_indexes]
            y_test = y.ix[test_indexes]

            estimator.fit(X_train, y_train)
            y_result_df.ix[X_test.index, 'Predicted'] = estimator.predict(X_test)
            y_result_df.ix[X_test.index, map(lambda x: str(x), xrange(10))] = estimator.predict_proba(X_test)
        
        y_result_df = y_result_df.applymap(lambda x: round(x,5))
        
        y_result_df['Actual'] = y_result_df['Actual'].astype(int)
        y_result_df['Predicted'] = y_result_df['Predicted'].astype(int)
        
        y_result_df['is_misclassified'] = (y_result_df['Actual'] != y_result_df['Predicted'])
        
        y_result_df['Model'] = model_name
        return y_result_df

    def calculate_prediction_by_probs_with_weights(self, prediction_data, weights):
        self.model_weights_by_number = weights
        
        stacked_prediction = prediction_data.groupby(['datum_index']).apply(self.__multiply_weights).reset_index()
        stacked_prediction = stacked_prediction.rename(columns = {0: 'Predicted'})
        
        return stacked_prediction
        
    def __multiply_weights(self, data):
        data_v1 = data.drop(['datum_index','Actual','Predicted'], axis=1).set_index('Model') * self.model_weights_by_number
        return data_v1.mean(axis=0).idxmax()
    
    
    
    def calculate_misclassifications_by_number(self, actual, predicted):
        
        data = pd.merge(actual, predicted, on = ['datum_index'])
        
        data['is_misclassified'] = data['Actual'].astype(str) != data['Predicted'].astype(str)
        
        return data
        
        
        
        