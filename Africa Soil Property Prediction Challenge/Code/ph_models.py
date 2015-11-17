'''
Created on Sep 18, 2014

@author: P_Kravik
'''

import numpy as np
import pandas as pd
import math
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR

def pHModel(X_train, Y_train, X_test, Y_test, X_eval, cv_iterator):
    svr_model, test_error_svr = pH_SVR(X_train, Y_train, X_test, Y_test, cv_iterator)
    
    final_prediction = svr_model.predict(X_eval)
    
    return final_prediction, {'final':test_error_svr, 'svr':test_error_svr}

def pH_SVR(X_train, Y_train, X_test, Y_test, cv_iterator):
    
    #===========================================================================
    # param_grid = {'C':[100,500,1000, 5000, 10000, 100000],
    #               'epsilon':[0.075,0.1, 0.125]
    #               }
    # 
    # svr = SVR(random_state=42)
    # search = GridSearchCV(svr, param_grid, scoring="mean_squared_error", cv=cv_iterator)
    # search.fit(X_train, Y_train["pH"])
    # #search.grid_scores_
    # svr = search.best_estimator_
    #===========================================================================
     
    #svr.fit(X_train, Y_train["pH"])
#     
    #test = cross_val_score(svr, X_train.astype('float64'), Y_train["Ca"].astype('float64'), scoring="mean_squared_error", cv=cv_iterator)
    
    svr = SVR(C=10000)
    svr.fit(X_train, Y_train["pH"])
    
    yhat_svr = svr.predict(X_test)
    test_error = math.sqrt(mean_squared_error(Y_test["pH"], yhat_svr))
    
    return svr, test_error