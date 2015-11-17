'''
Created on Sep 16, 2014

@author: P_Kravik
'''

import pandas as pd
import numpy as np
import random
import math
import time
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import LeaveOneLabelOut
from sklearn.cross_validation import LeavePLabelOut
from sklearn.cross_validation import ShuffleSplit
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC

from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from sklearn.preprocessing import StandardScaler

from sklearn.svm import OneClassSVM

from sklearn.ensemble import GradientBoostingClassifier

import matplotlib.pyplot as plt

def PModel(X_train, Y_train, X_test, Y_test, X_eval, cv_iterator):
    #ridge_model_p, test_error_p = P_RidgeModel()
    #gbm_model_p, test_error_gbm = P_GBMModel()
    #pls_model_p, test_error_pls = P_PLSModel()
    svr_model_p, test_error_svr = P_SVRModel(X_train, Y_train, X_test, Y_test, cv_iterator)
    
    final_prediction = svr_model_p.predict(X_eval)
    return final_prediction, {'final':test_error_svr, 'svr':test_error_svr}

def P_GBMModel(X_train, Y_train, X_test, Y_test, cv_itertor):
    params = {'verbose':2, 'n_estimators':100, 'max_depth':200, 'min_samples_leaf':2, 'learning_rate':0.01, 'loss':'ls', 'max_features':None, "subsample":1}
    gbm2 = GradientBoostingRegressor(**params)
    gbm2.fit(X_train, Y_train["P"])
    yhat_gbm = gbm2.predict(X_test)
    search_gbm.grid_scores_
    
def P_RidgeModel(X_train, Y_train, X_test, Y_test, cv_iterator):
    modelCV = Ridge()
    param_grid = {'alpha': [0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 2.5, 5, 7, 10, 15, 20, 50, 100, 150],
                  'normalize': [True],
                  'fit_intercept': [True]}
    search = GridSearchCV(modelCV, param_grid, scoring="mean_squared_error", n_jobs=2, cv=cv_iterator)
    search.fit(X_train, Y_train["P"])
    search.grid_scores_
    search_ridge = search
    model = search.best_estimator_
    mse = search.best_score_
    cv_score = math.sqrt(-1*mse)
    yhat_ridge = model.predict(X_test)

def P_PLSModel(X_train, Y_train, X_test, Y_test, cv_iterator):
    param_grid = {'n_components':[1,2,3,4,5,6,7,8,9,10,25]}
    modelCV = PLSRegression(scale=True)
    search = GridSearchCV(modelCV, param_grid, n_jobs=2, scoring="mean_squared_error", cv=cv_iterator, iid=False)
    search.fit(X_train, Y_train["P"])
    search.grid_scores_

    model = search.best_estimator_

    yhat_pls = model.predict(X_test)
    test_error = math.sqrt(mean_squared_error(Y_test["P"], yhat_pls))
    
    return model, test_error
    
    
def P_SVRModel(X_train, Y_train, X_test, Y_test, cv_iterator):
    
    #===========================================================================
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    # 
    # 
    # param_grid = {'C':[0.0001, 0.001, 0.01, 0.1],
    #               'epsilon':[0.1, 0.01]
    #               }
    #   
    # svr = SVR(random_state=42, verbose = 2)
    # search = GridSearchCV(svr, param_grid, scoring="mean_squared_error", n_jobs=1, cv=cv_iterator, iid=False)
    # search.fit(X_train, Y_train["P"])
    # #search.grid_scores_
    # #svr = search.best_estimator_
    #===========================================================================
    
    svr = SVR(C=10000, epsilon=0.1)
    svr.fit(X_train, Y_train["P"])
    #test = cross_val_score(svr, X_train.astype('float64'), Y_train["Ca"].astype('float64'), scoring="mean_squared_error", cv=cv_iterator)
    
    yhat_svr = svr.predict(X_test)
    test_error = math.sqrt(mean_squared_error(Y_test["P"], yhat_svr))
    
    return svr, test_error
    
def PClassification():
    
    #===========================================================================
    # normalize = StandardScaler()
    # normalized_x = normalize.fit_transform(X_train)
    # normalized_test = normalize.transform(X_test)
    # 
    # Is_cray_cray = (Y_train["P"] > 2).apply(lambda x: 1 if x else 0)
    # modelCV = LogisticRegression()
    # param_grid = {'penalty':['l2'],
    #               'C': [0.00000001, 0.0000001, 0.000001, 0.0001, 0.001, 0.01],
    #               'fit_intercept': [True]}
    # search = GridSearchCV(modelCV, param_grid, scoring="recall", cv=cv_iterator)
    # search.fit(normalized_x, Is_cray_cray)
    # model = search.best_estimator_
    # search.grid_scores_
    # 
    # yhat_p = search.best_estimator_.predict(normalized_test)
    # 
    # param_grid = {'C': [0.001, 10, 1000, 100000],
    #               'gamma': [0.01, 0],
    #               'kernel':['rbf'],
    #               'class_weight':['auto']
    #               }
    # search = GridSearchCV(svc, param_grid, scoring="precision", cv=cv_iterator)
    # search.fit(normalized_x, Is_cray_cray)
    # search.grid_scores_
    # model = search.best_estimator_
    # yhat_p = model.predict(normalized_test)
    # 
    # recall_score(Is_cray_cray, model.predict(normalized_x))
    # precision_score(Is_cray_cray, model.predict(normalized_x))
    # tmp = model.predict(normalized_x)
    # 
    # test_cray = (Y_test["P"]>2).apply(lambda x: 1 if x else 0)
    # 
    # blob = model.predict(normalized_test)
    # recall_score(test_cray, model.predict(normalized_test))
    # precision_score(test_cray, model.predict(normalized_test))
    # 
    # svc = SVC()
    # svc.fit(X_train, Is_cray_cray)
    # cross_val_score(svc, X_train.astype('float64'), Is_cray_cray.astype('float64'), scoring="recall", cv=cv_iterator)
    # 
    # tmp = svc.predict(X_test)
    # tmp.mean()
    # 
    # test = svc.predict(X_train)
    # 
    # train_p = model.predict(normalized_x)
    # plt.scatter(train_p, Y_train["P"].values, c='b')
    # 
    plt.figure(figsize=(12, 6))
    plt.title('Residual')
    plt.scatter(yhat_train, Y_train["P"].values, c='b')
    plt.scatter(yhat_pls, yhat_svr, c='r')
    #plt.scatter(huh, Y_train["pH"].values, c='r')
    #plt.scatter(huh, Y_train["SOC"].values, c='y')
    #plt.scatter(huh, Y_train["Sand"].values, c='g')
    plt.xlabel('Ca')
    plt.ylabel('P')
    # 
    # plt.show()
    # 
    # oneSVM = OneClassSVM(kernel='rbf', nu=0.8, gamma=.01, random_state = 42)
    # oneSVM.fit(X_train)
    # huh = oneSVM.decision_function(X_train)
    # huh.mean()   
    # np.corrcoef([yhat_p,  Y_test["P"].values])
    # 
    # pgbm = GradientBoostingClassifier(random_state=42, verbose=2, subsample=0.8)
    # pgbm.fit(X_train, Is_cray_cray)
    # yhat_pgbm = pgbm.predict_proba(X_test)[:,1]
    #===========================================================================
    return 1