'''
Created on Jun 20, 2014

@author: P_Kravik
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import ensemble
from scipy import optimize
from sklearn import grid_search
from sklearn import decomposition
from sklearn.svm import SVC

def newOutcomeVariable(clean_data_path, dataset):
    data = pd.read_csv(clean_data_path + dataset + '.csv')
    num_outcomes = data['at_least_1_teacher_referred_donor'] + data['fully_funded'] + data['at_least_1_green_donation'] + data['great_chat']  
    return num_outcomes

def getOutcome(clean_data_path,dataset,outcome_var):
    data = pd.read_csv(clean_data_path+ dataset +'.csv')
    return data[outcome_var]

def getEssayXValues(x_values_path):

    X_train_essay = pd.read_csv(x_values_path + 'essay training X values.csv',index_col=0)
    X_cross_essay = pd.read_csv(x_values_path + 'essay cross validation data X values.csv',index_col=0)
    X_test_essay = pd.read_csv(x_values_path + 'essay test data X values.csv',index_col=0)
    X_eval_essay = pd.read_csv(x_values_path + 'essay evaluation data X values.csv',index_col=0)
    
    #!!!!!!!!!!!!!!!!!!!!!!!WRONG WRONG TRAIN!!!!!!!!!!!!!!!!!!!!!!!!!!
    X_predict_essay = pd.read_csv(x_values_path + 'essay prediction X values.csv',index_col=0) 
    #FIX FIX FIX OH PLEASE

    return X_train_essay, X_cross_essay, X_test_essay, X_eval_essay, X_predict_essay

def getProjectXValues(x_values_path):
    
    X_train_project = pd.read_csv(x_values_path + 'project training X values.csv',index_col=0)
    X_cross_project = pd.read_csv(x_values_path + 'project cross validation X values.csv',index_col=0)
    X_test_project = pd.read_csv(x_values_path + 'project test X values.csv',index_col=0)
    X_eval_project = pd.read_csv(x_values_path + 'project evaluation X values.csv',index_col=0)
    X_predict_project = pd.read_csv(x_values_path + 'project prediction X values.csv',index_col=0)

    return X_train_project, X_cross_project, X_test_project, X_eval_project, X_predict_project

def getResourcesXValues(x_values_path):
    X_train_resources = pd.read_csv(x_values_path + 'resources training X values.csv',index_col=0)
    X_cross_resources = pd.read_csv(x_values_path + 'resources cross validation X values.csv',index_col=0)
    X_test_resources = pd.read_csv(x_values_path + 'resources test X values.csv',index_col=0)
    X_eval_resources = pd.read_csv(x_values_path + 'resources evaluation X values.csv',index_col=0)
    X_predict_resources = pd.read_csv(x_values_path + 'resources prediction X values.csv',index_col=0)

    return X_train_resources, X_cross_resources, X_test_resources, X_eval_resources, X_predict_resources

def generatePrediction(clean_data_path,subPath, Y_eval, name):
    print "Generating prediction"
    print "Reading evaluation data"
    evalData = pd.read_csv(clean_data_path+'evaluation data.csv')
    
    #Need to figure this part out
    print "Saving submission for Kaggle..."
    submission = pd.DataFrame(data= {'projectid':evalData['projectid'].as_matrix(), 'is_exciting': Y_eval, 'date_posted':evalData['date_posted']})
    submission['date_posted'] = pd.to_datetime(submission['date_posted'])
    submission.sort('date_posted',inplace=True)
    submission['is_exciting'] = submission['is_exciting'] * np.linspace(1,0.3,submission['is_exciting'].size)
    submission.drop('date_posted',inplace=True,axis=1)
    
    submission.to_csv(subPath + name+'.csv', index=False, cols=['projectid','is_exciting'])
    
def finalModel():
    
    x_values_path = 'C:/Users/P_Kravik/Desktop/Kaggle/KDD Cup 2014 - Predicting Excitement at DonorsChoose.org/Data/Clean/Derived/X values/'
    subPath = 'C:/Users/P_Kravik/Desktop/Kaggle/KDD Cup 2014 - Predicting Excitement at DonorsChoose.org/Submissions/'
    clean_data_path = 'C:/Users/P_Kravik/Desktop/Kaggle/KDD Cup 2014 - Predicting Excitement at DonorsChoose.org/Data/Clean/Derived/'
    
    X_train_essay, X_cross_essay, X_test_essay, X_eval_essay, X_predict_essay = getEssayXValues(x_values_path)
    X_train_project, X_cross_project, X_test_project, X_eval_project, X_predict_project = getProjectXValues(x_values_path)
    X_train_resources, X_cross_resources, X_test_resources, X_eval_resources, X_predict_resources = getResourcesXValues(x_values_path)
    
    X_train_essay.shape
    X_train_project.shape
    X_train_resources.shape
    
    X_cross_essay.shape
    X_cross_project.shape
    X_cross_resources.shape
    
    X_eval_essay.shape
    X_eval_project.shape
    X_eval_resources.shape
    
    X_predict_essay.shape
    X_predict_project.shape
    X_predict_resources.shape
    
    Y_train = getOutcome(clean_data_path,'two year train','is_exciting')
    Y_cross = getOutcome(clean_data_path,'cross validation data','is_exciting')    
    Y_test = getOutcome(clean_data_path,'test data','is_exciting')        
    Y_predict = getOutcome(clean_data_path, 'prediction train','is_exciting')
    
    standardize = preprocessing.StandardScaler()
    X_train_essay = standardize.fit_transform(X_train_essay)
    X_cross_essay = standardize.transform(X_cross_essay)
    X_test_essay = standardize.transform(X_test_essay)
    X_eval_essay = standardize.transform(X_eval_essay)
    
    decomp = decomposition.PCA(17)
    X_train_essay = decomp.fit_transform(X_train_essay)
    X_cross_essay = decomp.transform(X_cross_essay)
    X_test_essay = decomp.transform(X_test_essay)
    X_eval_essay = decomp.transform(X_eval_essay)
    
    X_train_essay = pd.DataFrame(X_train_essay)
    X_cross_essay = pd.DataFrame(X_cross_essay)
    X_test_essay = pd.DataFrame(X_test_essay)
    X_eval_essay = pd.DataFrame(X_eval_essay)
    
    X_train = pd.concat([X_train_essay,X_train_project,X_train_resources], axis=1)
    X_cross = pd.concat([X_cross_essay, X_cross_project,X_cross_resources], axis=1)
    X_test = pd.concat([X_test_essay,X_test_project,X_test_resources],axis=1)
    X_predict = pd.concat([X_predict_essay, X_predict_project,X_predict_resources], axis=1)
    X_eval = pd.concat([X_eval_essay, X_eval_project,X_eval_resources], axis=1)
    
#     X_train = pd.concat([X_train_essay,X_train_project], axis=1)
#     X_cross = pd.concat([X_cross_essay, X_cross_project], axis=1)
#     X_test = pd.concat([X_test_essay,X_test_project],axis=1)
#     X_predict = pd.concat([X_predict_essay, X_predict_project], axis=1)
#     X_eval = pd.concat([X_eval_essay, X_eval_project], axis=1)
#     
#     X_train = X_train_project
#     X_cross = X_cross_project
#     X_test = X_test_project
#     X_eval = X_eval_project
    
    
    standardize = preprocessing.StandardScaler()
    X_train_project = standardize.fit_transform(X_train_project)
    X_cross_project = standardize.transform(X_cross_project)
    X_test_project = standardize.transform(X_test_project)
    X_eval_project = standardize.transform(X_eval_project)
    #X_eval = standardize.transform(X_eval)
    
    standardize = preprocessing.StandardScaler()
    X_train_resources = standardize.fit_transform(X_train_resources)
    X_cross_resources = standardize.transform(X_cross_resources)
    X_test_resources = standardize.transform(X_test_resources)
    X_eval_resources = standardize.transform(X_eval_resources)
    
    standardize = preprocessing.StandardScaler()
    X_train = standardize.fit_transform(X_train)
    X_cross = standardize.transform(X_cross)
    X_test = standardize.transform(X_test)
    X_eval = standardize.transform(X_eval)
    
    #standardize_test = preprocessing.StandardScaler()
    #X_test = standardize_test.fit_transform(X_test)
    
    standardize_predict = preprocessing.StandardScaler()
    #X_predict = standardize_predict.fit_transform(X_predict)
    X_eval = standardize_predict.fit_transform(X_eval)
    
    print "May the odds be ever in your favor..."
    
    #Logit
    #logit = LogisticRegression(penalty='l1',dual=False,tol=0.0001,fit_intercept=True, C=.4325, intercept_scaling=1, class_weight='auto', random_state=423)
    
    logit = LogisticRegression(penalty='l2',dual=False,tol=0.0001,fit_intercept=True, C=5, intercept_scaling=1, class_weight='auto', random_state=423)
    
    logit.fit(X_train,Y_train)
     
    predict = logit.predict_proba(X_train)
    predict_logit = predict[:,1]
    
    inScore = roc_auc_score(Y_train,predict_logit)
    print "Train Logistic: Area under auc curve is %f" % (inScore)
    
    cross_predict = logit.predict_proba(X_cross)
    cross_predict_logit = cross_predict[:,1]
    
    inScore = roc_auc_score(Y_cross,cross_predict_logit)
    print "Cross Logistic: Area under auc curve is %f" % (inScore)
    
    test_predict = logit.predict_proba(X_test)
    test_predict_logit = test_predict[:,1]
    
    inScore = roc_auc_score(Y_test,test_predict_logit)
    print "Test logistic: Area under auc curve is %f" % (inScore)
    
    logit_predict = LogisticRegression(penalty='l2',dual=False,tol=1,fit_intercept=True, C=.0004325, intercept_scaling=1, class_weight='auto', random_state=423)
    logit_predict.fit(X_predict,Y_train)
    
    temp = logit.predict_proba(X_eval)
    Y_predict_logit = temp[:,1]
    generatePrediction(clean_data_path, subPath,Y_predict_logit,'Final logit model')
    
    #GRADIENT BOOSTING PROJECT MODEL
    gbm = ensemble.GradientBoostingClassifier(random_state=43,verbose = 1,max_depth=1)
    gbm.fit(X_train,Y_train)
    
    predict = gbm.predict_proba(X_train)
    predict_gbm = predict[:,1]
    
    inScore = roc_auc_score(Y_train,predict_gbm)
    print "Train AdaBoost: Area under auc curve is %f" % (inScore)
    
    cross_predict = gbm.predict_proba(X_cross)
    cross_predict_gbm = cross_predict[:,1]
    
    inScore = roc_auc_score(Y_cross,cross_predict_gbm)
    print "Cross AdaBoost: Area under auc curve is %f" % (inScore)
    
    test_predict = gbm.predict_proba(X_test)
    test_predict_gbm = test_predict[:,1]
    
    inScore = roc_auc_score(Y_test,test_predict_gbm)
    print "Test AdaBoost: Area under auc curve is %f" % (inScore)
    
    temp = gbm.predict_proba(X_eval)
    Y_predict_gbm = temp[:,1]
    generatePrediction(clean_data_path, subPath, Y_predict_gbm,'Project GBM model')
    
    #ADABOOST STUMPS PROJECT MODEL
    ada_weight = Y_train*4+1
    
    ada = ensemble.AdaBoostClassifier(n_estimators=200,learning_rate = .5,random_state = 42)
    ada.fit(X_train,Y_train,ada_weight.values)
    
    predict = ada.predict_proba(X_train)
    predict_ada = predict[:,1]
    
    inScore = roc_auc_score(Y_train,predict_ada)
    print "Train AdaBoost: Area under auc curve is %f" % (inScore)
    
    cross_predict = ada.predict_proba(X_cross)
    cross_predict_ada = cross_predict[:,1]
    
    inScore = roc_auc_score(Y_cross,cross_predict_ada)
    print "Cross AdaBoost: Area under auc curve is %f" % (inScore)
    
    test_predict = ada.predict_proba(X_test)
    test_predict_ada = test_predict[:,1]
    
    inScore = roc_auc_score(Y_test,test_predict_ada)
    print "Test AdaBoost: Area under auc curve is %f" % (inScore)
    
    ada_weight_predict = Y_predict*4+1
    
    ada2 = ensemble.AdaBoostClassifier(n_estimators=100,learning_rate = 1,random_state = 42)
    ada2.fit(X_predict,Y_predict,ada_weight_predict.values)
    
    temp = ada.predict_proba(X_eval)
    Y_predict_ada = temp[:,1]
    generatePrediction(clean_data_path, subPath, Y_predict_ada,'Project AdaBoost model')
    
    #RANDOM FOREST PROJECT MODEL
    #Should cross validate this value
    rf_weight = Y_train*4+1
    
    clf = ensemble.RandomForestClassifier(n_estimators=100,max_depth=10,max_features=25,bootstrap=True,oob_score=True,n_jobs=2,verbose=1)
    clf.fit(X_train,Y_train,rf_weight.values)
    
    predict = clf.predict_proba(X_train)
    predict_rf = predict[:,1]
    
    inScore = roc_auc_score(Y_train,predict_rf)
    print "Train Random Forest: Area under auc curve is %f" % (inScore)
    
    cross_predict = clf.predict_proba(X_cross)
    cross_predict_rf = cross_predict[:,1]
    
    inScore = roc_auc_score(Y_cross,cross_predict_rf)
    print "Cross Random Forest: Area under auc curve is %f" % (inScore)
    
    test_predict = clf.predict_proba(X_test)
    test_predict_rf = test_predict[:,1]
    
    inScore = roc_auc_score(Y_test,test_predict_rf)
    print "Test Random Forest: Area under auc curve is %f" % (inScore)
    
    rf_predict_weight = Y_predict*4+1
    
    #clf2 = ensemble.RandomForestClassifier(n_estimators=100,max_depth=10,max_features=25,oob_score=True,n_jobs=2)
    #clf2.fit(X_predict,Y_predict,rf_predict_weight.values)
    
    temp = clf.predict_proba(X_eval)
    Y_predict_rf = temp[:,1]
    generatePrediction(clean_data_path, subPath,Y_predict_rf,'Project random forest model')

    
    
    
    
    
    
    
    
    
    final_standardize = preprocessing.StandardScaler()
    logit_score = final_standardize.fit_transform(cross_predict_logit)
    ada_score = final_standardize.fit_transform(cross_predict_ada)
    rf_score = final_standardize.fit_transform(cross_predict_rf)
    
    cross_final_score = ada_score + rf_score + logit_score
    inScore = roc_auc_score(Y_cross,cross_final_score)
    print "Cross Blend: Area under auc curve is %f" % (inScore)
    
    final_standardize2 = preprocessing.StandardScaler()
    logit_test_score = final_standardize2.fit_transform(test_predict_logit)
    ada_test_score = final_standardize2.fit_transform(test_predict_ada)
    rf_test_score = final_standardize2.fit_transform(test_predict_rf)
    
    test_final_score = ada_test_score + rf_test_score + logit_test_score
    inScore = roc_auc_score(Y_test,test_final_score)
    print "Test Blend: Area under auc curve is %f" % (inScore)
    
    cross_x_values = pd.DataFrame({'logit':logit_score, 'ada':ada_score, 'rf':rf_score})
    test_x_values = pd.DataFrame({'logit':logit_test_score, 'ada':ada_test_score, 'rf':rf_test_score})
    logit = LogisticRegression(penalty='l2',dual=False,tol=0.0001,fit_intercept=True, C=0.0005, intercept_scaling=1, class_weight='auto', random_state=423)
    logit.fit(cross_x_values, Y_cross)
    
    inScore = roc_auc_score(Y_cross,logit.predict_proba(cross_x_values)[:,1])
    print "Cross Blend Logit: Area under auc curve is %f" % (inScore)
    
    inScore = roc_auc_score(Y_test,logit.predict_proba(test_x_values)[:,1])
    print "Test Blend Logit: Area under auc curve is %f" % (inScore)

    

    
    #Project RF
    
    rf_weight = Y_train*4+1
    
    clf_proj = ensemble.RandomForestClassifier(n_estimators=100,criterion = 'entropy',max_depth=10,max_features=25,bootstrap=True,oob_score=True,n_jobs=2)
    clf_proj.fit(X_train_project,Y_train,rf_weight.values)
    
    predict = clf_proj.predict_proba(X_train_project)
    predict_project_rf = predict[:,1]
    
    inScore = roc_auc_score(Y_train,predict_project_rf)
    print "Train Random Forest: Area under auc curve is %f" % (inScore)
    
    cross_predict = clf_proj.predict_proba(X_cross_project)
    cross_predict_project_rf = cross_predict[:,1]
    
    inScore = roc_auc_score(Y_cross,cross_predict_project_rf)
    print "Cross Random Forest: Area under auc curve is %f" % (inScore)
    
    test_predict = clf_proj.predict_proba(X_test_project)
    test_predict_project_rf = test_predict[:,1]
    
    inScore = roc_auc_score(Y_test,test_predict_project_rf)
    print "Test Random Forest: Area under auc curve is %f" % (inScore)
    
    eval_predict_project_rf = clf_proj.predict_proba(X_eval_project)[:,1]
    
    eval_predict_project_rf = eval_predict_project_rf * np.linspace(1,0.3,eval_predict_project_rf.size)
    generatePrediction(clean_data_path, subPath,eval_predict_project_rf,'Project RF model')
    
    #ADABOOST STUMPS PROJECT MODEL
    ada_weight = Y_train*4+1
    
    ada = ensemble.AdaBoostClassifier(n_estimators=100,learning_rate = 1,random_state = 42)
    ada.fit(X_train_project,Y_train,ada_weight.values)
    
    predict = ada.predict_proba(X_train_project)
    predict_project_ada = predict[:,1]
    
    inScore = roc_auc_score(Y_train,predict_project_ada)
    print "Train AdaBoost: Area under auc curve is %f" % (inScore)
    
    cross_predict = ada.predict_proba(X_cross_project)
    cross_predict_project_ada = cross_predict[:,1]
    
    inScore = roc_auc_score(Y_cross,cross_predict_project_ada)
    print "Cross AdaBoost: Area under auc curve is %f" % (inScore)
    
    test_predict = ada.predict_proba(X_test_project)
    test_predict_project_ada = test_predict[:,1]
    
    inScore = roc_auc_score(Y_test,test_predict_project_ada)
    print "Test AdaBoost: Area under auc curve is %f" % (inScore)
    
    eval_predict_project_ada = ada.predict_proba(X_eval_project)[:,1]

    
    
    
    #Project Logit
    
    logit = LogisticRegression(penalty='l2',dual=False,tol=1,fit_intercept=True, C=.0004325, intercept_scaling=1, class_weight='auto', random_state=423)
    
    logit.fit(X_train_project,Y_train)
     
    predict = logit.predict_proba(X_train_project)
    predict_logit_project = predict[:,1]
    
    inScore = roc_auc_score(Y_train,predict_logit_project)
    print "Train Logistic: Area under auc curve is %f" % (inScore)
    
    cross_predict = logit.predict_proba(X_cross_project)
    cross_predict_logit_project = cross_predict[:,1]
    
    inScore = roc_auc_score(Y_cross,cross_predict_logit_project)
    print "Cross Logistic: Area under auc curve is %f" % (inScore)
    
    test_predict = logit.predict_proba(X_test_project)
    test_predict_logit_project = test_predict[:,1]
    
    inScore = roc_auc_score(Y_test,test_predict_logit_project)
    print "Test logistic: Area under auc curve is %f" % (inScore)
  

    
    standardize = preprocessing.StandardScaler()
    project_cross_ada = standardize.fit_transform(cross_predict_project_ada)
    project_cross_rf = standardize.fit_transform(cross_predict_project_rf)
    
    weight = 3
    
    project_cross_score = weight*project_cross_ada + project_cross_rf
    
    inScore = roc_auc_score(Y_cross,project_cross_ada)
    print "Project Cross AdaBoost: Area under auc curve is %f" % (inScore)
    inScore = roc_auc_score(Y_cross,project_cross_rf)
    print "Project Cross RF: Area under auc curve is %f" % (inScore)
    inScore = roc_auc_score(Y_cross,project_cross_score)
    print "Project Cross Blend: Area under auc curve is %f" % (inScore)
    
    project_test_ada = standardize.fit_transform(test_predict_project_ada)
    project_test_rf = standardize.fit_transform(test_predict_project_rf)
    
    project_test_score = weight*project_test_ada + project_test_rf
    
    inScore = roc_auc_score(Y_test,project_test_ada)
    print "Project Test AdaBoost: Area under auc curve is %f" % (inScore)
    inScore = roc_auc_score(Y_test,project_test_rf)
    print "Project Test RF: Area under auc curve is %f" % (inScore)
    inScore = roc_auc_score(Y_test,project_test_score)
    print "Project Test Blend: Area under auc curve is %f" % (inScore)
    
    project_eval_ada = standardize.fit_transform(eval_predict_project_ada)
    project_eval_rf = standardize.fit_transform(eval_predict_project_rf)
    
    project_eval_score = weight*project_eval_ada + project_eval_rf
    

    
    #Essay logit
    
    logit = LogisticRegression(penalty='l2',dual=False,tol=.5,fit_intercept=True, C=.0004325, intercept_scaling=1, class_weight='auto', random_state=423)
    
    logit.fit(X_train_essay,Y_train)
     
    predict = logit.predict_proba(X_train_essay)
    predict_logit_essay = predict[:,1]
    
    inScore = roc_auc_score(Y_train,predict_logit_essay)
    print "Train Logistic: Area under auc curve is %f" % (inScore)
    
    cross_predict = logit.predict_proba(X_cross_essay)
    cross_predict_logit_essay = cross_predict[:,1]
    
    inScore = roc_auc_score(Y_cross,cross_predict_logit_essay)
    print "Cross Logistic: Area under auc curve is %f" % (inScore)
    
    test_predict = logit.predict_proba(X_test_essay)
    test_predict_logit_essay = test_predict[:,1]
    
    inScore = roc_auc_score(Y_test,test_predict_logit_essay)
    print "Test logistic: Area under auc curve is %f" % (inScore)
    
    eval_predict_logit_essay = logit.predict_proba(X_eval_essay)[:,1]
    
    #Essay Random Forest
    rf_weight = Y_train*4+1
    
    clf_essay = ensemble.RandomForestClassifier(n_estimators=100,max_depth=4,bootstrap=True,oob_score=True,n_jobs=2)
    clf_essay.fit(X_train_essay,Y_train,rf_weight.values)
    
    predict = clf_essay.predict_proba(X_train_essay)
    predict_rf_essay = predict[:,1]
    
    inScore = roc_auc_score(Y_train,predict_rf_essay)
    print "Train Random Forest: Area under auc curve is %f" % (inScore)
    
    cross_predict = clf_essay.predict_proba(X_cross_essay)
    cross_predict_rf_essay = cross_predict[:,1]
    
    inScore = roc_auc_score(Y_cross,cross_predict_rf_essay)
    print "Cross Random Forest: Area under auc curve is %f" % (inScore)
    
    test_predict = clf_essay.predict_proba(X_test_essay)
    test_predict_rf_essay = test_predict[:,1]
    
    inScore = roc_auc_score(Y_test,test_predict_rf_essay)
    print "Test Random Forest: Area under auc curve is %f" % (inScore)
    
    eval_predict_rf_essay = clf_essay.predict_proba(X_eval_essay)[:,1]
    
    standardize = preprocessing.StandardScaler()
    essay_cross_logit = standardize.fit_transform(cross_predict_logit_essay)
    essay_cross_rf = standardize.fit_transform(cross_predict_rf_essay)    
    
    weight = 1
    
    essay_cross_score = weight*essay_cross_rf + essay_cross_logit
    
    inScore = roc_auc_score(Y_cross,essay_cross_logit)
    print "Essay Cross Logit: Area under auc curve is %f" % (inScore)
    inScore = roc_auc_score(Y_cross,essay_cross_rf)
    print "Essay Cross RF: Area under auc curve is %f" % (inScore)
    inScore = roc_auc_score(Y_cross,essay_cross_score)
    print "Essay Cross Blend: Area under auc curve is %f" % (inScore)
    
    essay_test_logit = standardize.fit_transform(test_predict_logit_essay)
    essay_test_rf = standardize.fit_transform(test_predict_rf_essay)
    
    essay_test_score = weight*essay_test_logit + essay_test_rf
    
    inScore = roc_auc_score(Y_test,essay_test_logit)
    print "Essay Test Logit: Area under auc curve is %f" % (inScore)
    inScore = roc_auc_score(Y_test,essay_test_rf)
    print "Essay Test RF: Area under auc curve is %f" % (inScore)
    inScore = roc_auc_score(Y_test,essay_test_score)
    print "Essay Test Blend: Area under auc curve is %f" % (inScore)
    
    essay_eval_logit = standardize.fit_transform(eval_predict_logit_essay)
    essay_eval_rf = standardize.fit_transform(eval_predict_rf_essay)
    
    essay_eval_score = weight*essay_eval_logit + essay_eval_rf
    
  
    #Resource Logit
    
    logit = LogisticRegression(penalty='l2',dual=False,tol=1,fit_intercept=True, C=.004325, intercept_scaling=1, class_weight='auto', random_state=423)
    
    logit.fit(X_train_resources,Y_train)
     
    predict = logit.predict_proba(X_train_resources)
    predict_logit_resources = predict[:,1]
    
    inScore = roc_auc_score(Y_train,predict_logit_resources)
    print "Train Logistic: Area under auc curve is %f" % (inScore)
    
    cross_predict = logit.predict_proba(X_cross_resources)
    cross_predict_logit_resources = cross_predict[:,1]
    
    inScore = roc_auc_score(Y_cross,cross_predict_logit_resources)
    print "Cross Logistic: Area under auc curve is %f" % (inScore)
    
    test_predict = logit.predict_proba(X_test_resources)
    test_predict_logit_resources = test_predict[:,1]
    
    inScore = roc_auc_score(Y_test,test_predict_logit_resources)
    print "Test logistic: Area under auc curve is %f" % (inScore)
    
    eval_predict_logit_resources = logit.predict_proba(X_eval_resources)[:,1]
    
    standardize = preprocessing.StandardScaler()
    resources_cross_score = standardize.fit_transform(cross_predict_logit_resources)
    resources_test_score = standardize.fit_transform(test_predict_logit_resources)
    resources_eval_score = standardize.fit_transform(eval_predict_logit_resources)
    
    
    
    
    final_standardize = preprocessing.StandardScaler()
    essay_cross_final_score = final_standardize.fit_transform(essay_cross_score)
    project_cross_final_score = final_standardize.fit_transform(project_cross_score)
    resources_cross_final_score = final_standardize.fit_transform(resources_cross_score)
    
    cross_final_score = 3*essay_cross_final_score + 6*project_cross_final_score + 1*resources_cross_final_score
    inScore = roc_auc_score(Y_cross,cross_final_score)
    print "Cross Blend: Area under auc curve is %f" % (inScore)
    
    final_standardize2 = preprocessing.StandardScaler()
    essay_test_final_score = final_standardize2.fit_transform(essay_test_score)
    project_test_final_score = final_standardize2.fit_transform(project_test_score)
    resources_test_final_score = final_standardize2.fit_transform(resources_test_score)
    
    test_final_score = 3*essay_test_final_score + 6*project_test_final_score + 1*resources_test_final_score
    inScore = roc_auc_score(Y_test,test_final_score)
    print "Test Blend: Area under auc curve is %f" % (inScore)
    
    final_standardize3 = preprocessing.StandardScaler()
    essay_eval_final_score = final_standardize3.fit_transform(essay_eval_score)
    project_eval_final_score = final_standardize3.fit_transform(project_eval_score)
    resources_eval_final_score = final_standardize3.fit_transform(resources_eval_score)
    
    eval_final_score = 3*essay_eval_final_score + 6*project_eval_final_score + 1*resources_eval_final_score
     
    min_max_scalar = preprocessing.MinMaxScaler()
    eval_final_score = min_max_scalar.fit_transform(eval_final_score)
    
    eval_final_score = eval_final_score * np.linspace(1,0.3,eval_final_score.size)
    generatePrediction(clean_data_path, subPath,eval_final_score,'Final Blended model')
    
    
    
    
    
    cross_x_values = pd.DataFrame({'essay':cross_predict_logit_essay, 'project':cross_predict_project_rf, 'resources':cross_predict_logit_resources})
    test_x_values = pd.DataFrame({'essay':test_predict_logit_essay, 'project':test_predict_project_rf, 'resources':test_predict_logit_resources})
    eval_x_values = pd.DataFrame({'essay':eval_predict_logit_essay, 'project':eval_predict_project_rf, 'resources':eval_predict_logit_resources})
    
    logit = LogisticRegression(penalty='l2',dual=False,tol=0.0001,fit_intercept=True, C=500, intercept_scaling=1, class_weight='auto', random_state=423)
    logit.fit(cross_x_values, Y_cross)
    
    ada_weight = Y_cross*4+1
    
    ada = ensemble.AdaBoostClassifier(n_estimators=500,learning_rate = .1,random_state = 42)
    ada.fit(cross_x_values,Y_cross,ada_weight.values)
    
    
    inScore = roc_auc_score(Y_cross,ada.predict_proba(cross_x_values)[:,1])
    print "Cross Blend Logit: Area under auc curve is %f" % (inScore)
    
    inScore = roc_auc_score(Y_test,ada.predict_proba(test_x_values)[:,1])
    print "Test Blend Logit: Area under auc curve is %f" % (inScore)
    
    generatePrediction(clean_data_path, subPath,ada.predict_proba(eval_x_values)[:,1],'Final Ada Blended model')
    
    #logit_predict = LogisticRegression(penalty='l2',dual=False,tol=1,fit_intercept=True, C=.0004325, intercept_scaling=1, class_weight='auto', random_state=423)
    #logit_predict.fit(X_predict,Y_predict)
    
    #temp = logit_predict.predict_proba(X_eval)
    #Y_predict_logit = temp[:,1]
    #generatePrediction(clean_data_path, subPath,Y_predict_logit,'Project logit model')

    train_projid = getOutcome(clean_data_path,'two year train','projectid')
    cross_projid = getOutcome(clean_data_path,'cross validation data','projectid')    
    test_projid = getOutcome(clean_data_path,'test data','projectid')        
    
    cross_predictions = pd.DataFrame({'projectid': cross_projid, 'pk_prediction': cross_final_score})
    test_predictions = pd.DataFrame({'projectid': test_projid, 'pk_prediction':test_final_score})
    
    cross_predictions.to_csv(subPath + 'cross validation predictions.csv')
    test_predictions.to_csv(subPath + 'test predictions.csv')
    
    
    Y_predict = getOutcome(clean_data_path, 'prediction train','is_exciting')



if __name__ == '__main__':
    finalModel()