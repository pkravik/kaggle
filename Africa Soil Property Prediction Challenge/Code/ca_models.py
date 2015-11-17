'''
Created on Sep 16, 2014

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

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler



def CaModel(X_train, Y_train, X_test, Y_test, X_eval, cv_iterator):
#     print "Ca Ridge"
#     ridge_model_ca, test_error_ridge = CaRidgeModel(X_train, Y_train, X_test, Y_test, cv_iterator)
#     #gbm_model_ca, test_error_gbm = CaGBMModel(X_train, Y_train, X_test, Y_test, cv_iterator)
#     print "Ca PLS"
#     pls_model_ca, test_error_pls = CaPLSModel(X_train, Y_train, X_test, Y_test, cv_iterator)
#     print "Ca SVR"
#     svr_model_ca, test_error_svr = CaSVRModel(X_train, Y_train, X_test, Y_test, cv_iterator)
#     print "Combine!"
#     final_model, final_error = Ca_final_model(ridge_model_ca, pls_model_ca, svr_model_ca, X_test, Y_test, X_eval)
#     
#     test_error = {'ridge':test_error_ridge, 'pls':test_error_pls, 'final':final_error, 'svr':test_error_svr}
    #np.corrcoef([ridge_model_ca, gbm_model_ca, pls_model_ca, svr_model_ca, final_model])
    
    svr_model_ca, test_error_svr = CaSVRModel(X_train, Y_train, X_test, Y_test, cv_iterator)
    
    final_model = svr_model_ca.predict(X_eval)
    test_error = {'final':test_error_svr, 'svr':test_error_svr}
    
    return final_model, test_error

def stackModels(ridge, pls, gbm, svr, X_test, Y_test):
    yhat_ridge_test = ridge.predict(X_test)
    yhat_pls_test = np.reshape(pls.predict(X_test), len(X_test))
    yhat_gbm_test = gbm.predict(X_test)
    yhat_svr_test = svr.predict(X_test)
    
    new_train = pd.DataFrame({'ridge':yhat_ridge_test, 'pls':yhat_pls_test,'gbm':yhat_gbm_test,'svr':yhat_svr_test})
    
    stack = Ridge()
    param_grid = {'alpha': [0.001, 0.05, 0.1, 0.5, 1, 1.5, 2,10, 100],
                  'normalize': [True],
                  'fit_intercept': [True, False]}
    search = GridSearchCV(stack, param_grid, scoring="mean_squared_error", cv=5)
    search.fit(new_train, Y_test["Ca"])
    search.grid_scores_
    model = search.best_estimator_
    
    test_average = np.mean([yhat_ridge_test,yhat_pls_test,yhat_gbm_test,yhat_svr_test], axis=0)
    mean_squared_error(Y_test["Ca"], test_average)
    mean_squared_error(Y_test["Ca"], new_train['ridge'])
    mean_squared_error(Y_test["Ca"], new_train['pls'])
    mean_squared_error(Y_test["Ca"], new_train['gbm'])
    mean_squared_error(Y_test["Ca"], new_train['svr'])
    
    clf = RandomForestRegressor(max_depth = 4, min_samples_leaf = 5, n_estimators = 1000, oob_score = True, verbose = 2)
    clf.fit(new_train, Y_test["Ca"])
    mean_squared_error(Y_test["Ca"], clf.predict(new_train))
    
    plt.figure(figsize=(12, 6))
    plt.title('Residual')
    plt.scatter(Y_test["Ca"], yhat_ridge_test, c='b')
    plt.scatter(Y_test["Ca"], yhat_pls_test, c='r')
    plt.scatter(Y_test["Ca"], yhat_gbm_test, c='y')
    plt.scatter(Y_test["Ca"], yhat_svr_test, c='g')
    plt.scatter(Y_test["Ca"], yhat_svr_test, c='g')
    plt.scatter(Y_test["Ca"], Y_test["Ca"], c='black')
    
    plt.xlabel('Prediction')
    plt.ylabel('Residual')
    
    plt.show()
    

def Ca_final_model(ridge, pls, svr, X_test, Y_test, X_eval):
    
#===============================================================================
#     yhat_ridge = pd.Series(yhat_ridge)
#     yhat_gbm = pd.Series(yhat_gbm)
#     yhat_pls = pd.Series(np.reshape(yhat_pls, yhat_pls.size))
#     
#     ridge_res = Y_test["Ca"] - yhat_ridge.values
#     gbm_res = Y_test["Ca"] - yhat_gbm.values
#     pls_res = Y_test["Ca"] - yhat_pls.values
#     combo_res = pd.Series(np.mean([ridge_res, gbm_res], axis=0))
# 
#     plt.figure(figsize=(12, 6))
#     plt.title('Residual')
#     plt.scatter(yhat_ridge, ridge_res.values, c='b')
#     plt.scatter(yhat_ridge, gbm_res.values, c='r')
#     plt.scatter(yhat_ridge, pls_res.values, c='y')
#     plt.xlabel('Prediction')
#     plt.ylabel('Residual')
#     
#     plt.show()
#===============================================================================
   
    yhat_ridge_test = ridge.predict(X_test)
    yhat_pls_test = pls.predict(X_test)
    #yhat_gbm_test = gbm.predict(X_test)
    yhat_svr_test = svr.predict(X_test)
    
    yhat_ridge_eval = ridge.predict(X_eval)
    yhat_pls_eval = pls.predict(X_eval)
    #yhat_gbm_eval = gbm.predict(X_eval)
    yhat_svr_eval = svr.predict(X_eval)
    
    #mean_squared_error(Y_test["Ca"], yhat_pls)
    #mean_squared_error(Y_test["Ca"], yhat_ridge)
    #mean_squared_error(Y_test["Ca"], yhat_gbm)
    #mean_squared_error(Y_test["Ca"], np.mean([yhat_ridge, yhat_gbm], axis=0))
    #mean_squared_error(Y_test["Ca"], test)
    final_test = np.mean([yhat_ridge_test, yhat_pls_test, yhat_svr_test], axis=0)
    final_test_error = math.sqrt(mean_squared_error(Y_test["Ca"], final_test))
    
    final = np.mean([yhat_ridge_eval, yhat_pls_eval, yhat_svr_eval], axis=0)
    
    return final, final_test_error

def CaRidgeModel(X_train, Y_train, X_test, Y_test, cv_iterator):
    
    modelCV = Ridge()
    param_grid = {'alpha': [0.05, 0.1, 0.5, 1, 1.5, 2],
                  'normalize': [True],
                  'fit_intercept': [True]}
    search = GridSearchCV(modelCV, param_grid, scoring="mean_squared_error", cv=cv_iterator)
    search.fit(X_train, Y_train["Ca"])
    #search.grid_scores_
    model = search.best_estimator_
    #mse = search.best_score_
    #cv_score = math.sqrt(-1*mse)
    
    #model.fit(X_train, Y_train["Ca"])
    
    yhat_ridge = model.predict(X_test)
    test_error = math.sqrt(mean_squared_error(Y_test["Ca"], yhat_ridge))
    
    return model, test_error

def CaGBMModel(X_train, Y_train, X_test, Y_test, cv_iterator):
    
    #===========================================================================
    # modelCV = GradientBoostingRegressor(subsample = 1, random_state = 42)
    # param_grid = {'loss':['ls'],
    #               'learning_rate':[0.1],
    #               'n_estimators':[100],
    #               'max_depth':[5, 50, 150],
    #               'min_samples_split':[2],
    #               'min_samples_leaf':[5, 15, 30],
    #               'max_features':["auto"]
    #               }
    # 
    # search = GridSearchCV(modelCV, param_grid, scoring="mean_squared_error", cv=cv_iterator, n_jobs = -1)
    # search.fit(X_train, Y_train["P"])
    # search.grid_scores_
    # model = search.best_estimator_
    # mse = search.best_score_
    # print (time.strftime("%H:%M:%S"))
    #===========================================================================
    gbm = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, max_depth=50, min_samples_leaf=20, max_features=None, random_state=76)
    gbm.fit(X_train, Y_train["Ca"])
    
    yhat_gbm = gbm.predict(X_test)
    test_error = math.sqrt(mean_squared_error(Y_test["Ca"], yhat_gbm))

    return gbm, test_error

def CaPLSModel(X_train, Y_train, X_test, Y_test, cv_iterator):
    
    #tmp_X_train = X_train
    #tmp_Y_train = Y_train
    #tmp_X_test = X_test
    
    param_grid = {'n_components':[10,11,12,13,14,15,16,17,18,19,20]}
    modelCV = PLSRegression(scale=True)
    search = GridSearchCV(modelCV, param_grid, scoring="mean_squared_error", cv=cv_iterator)
    search.fit(X_train, Y_train["Ca"])
    search.grid_scores_
    model = search.best_estimator_
    #mse = search.best_score_
    #cv_score = math.sqrt(-1*mse)
    
    #model.fit(X_train, Y_train["Ca"])
    
    
    yhat_pls = model.predict(X_test)
    test_score = math.sqrt(mean_squared_error(Y_test["Ca"], yhat_pls))

    return model, test_score

def CaSVRModel(X_train, Y_train, X_test, Y_test, cv_iterator):
#     
#     param_grid = {'C':[10000],
#                    'epsilon':[0.001, 0.01, 0.05, 0.1, 0.15, 1]
#                    }
#       
#     svr = SVR(random_state=42, cache_size=1000, verbose=2)
#     search = GridSearchCV(svr, param_grid, scoring="mean_squared_error", n_jobs= 1, iid=True, cv=cv_iterator)
#     search.fit(X_train, Y_train["Ca"])
#     #search.grid_scores_
#       
#     model = search.best_estimator_

    #scaler = StandardScaler()

    model = SVR(C=10000, epsilon = 0.01, cache_size=1000)
    model.fit(X_train, Y_train["Ca"])
    #model.fit(X_train, Y_train["Ca"])
    
    #model.fit(X_train, Y_train["Ca"])
    
    #test = cross_val_score(svr, X_train.astype('float64'), Y_train["Ca"].astype('float64'), scoring="mean_squared_error", cv=cv_iterator)
    
    yhat_svr = model.predict(X_test)
    test_error = math.sqrt(mean_squared_error(Y_test["Ca"], yhat_svr))
    
    return model, test_error

def testingGBM(X_train, Y_train, X_test, Y_test):
    params = {'verbose':2, 'n_estimators':100, 'max_depth':50, 'min_samples_leaf':20, 'learning_rate':0.1, 'loss':'ls', 'max_features':None}
    test_init = Ridge(alpha = 0.1, normalize = True, fit_intercept=True)
    gbm2 = GradientBoostingRegressor(**params)
    gbm2.fit(X_train, Y_train["Ca"])
    yhat_gbm = gbm2.predict(X_test)
    mean_squared_error(Y_test["Ca"], yhat_gbm)
    math.sqrt(mean_squared_error(Y_test["Ca"], yhat_gbm))
    
    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
    
    for i, y_pred in enumerate(gbm2.staged_decision_function(X_test)):
        test_score[i]=mean_squared_error(Y_test["Ca"], y_pred)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, gbm2.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
             label='Test Set Deviance')
    
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
    plt.show()
