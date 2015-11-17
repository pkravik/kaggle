'''
Created on Jul 3, 2014

@author: P_Kravik
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn import ensemble
from scipy import optimize
from sklearn import grid_search
from sklearn import decomposition
from sklearn.metrics import r2_score

def getFeatures(text):
    #Screw it, going to use sk_learn
    vectorizer = CountVectorizer(min_df=.01, max_df=0.95, ngram_range=(1,2))
    X_train = vectorizer.fit_transform(text)
    num_samples, num_features = X_train.shape
    print "#samples: %d, #features: %d" % (num_samples, num_features)
    
    return (vectorizer)

def bowFitAndPrediction(predictData, textSeries, outcome,typeModel='binary'):
    print "Bag of words for %s" % (textSeries.name)
    
    if typeModel == 'continuous':
        bowModel = Ridge(alpha = 0.001)
    else:
        bowModel = LogisticRegression(penalty='l2',dual=False,tol=0.0001,fit_intercept=True, C=1, intercept_scaling=1, class_weight=None, random_state=423) 
    
    
    vectorizer = getFeatures(textSeries)
    
    X_train = vectorizer.transform(predictData)
        
    #Outcomes
    Y_train = outcome
    
    #Logistic regression, not sure if best
    bowModel.fit(X_train,Y_train)
    
    #Comment out later, fitting on CV data
    
    if typeModel == 'continuous':
        predict = bowModel.predict(X_train)
        yhat = predict
    else:
        predict = bowModel.predict_proba(X_train)
        yhat = predict[:,1]
    
    
    return (yhat, vectorizer, bowModel)

def bagofwords(X_train, X_cross, X_test, X_predict, X_eval, Y_train, Y_cross, Y_predict,variable = 'test', typeModel='binary',name='test'):

    X_train_text = X_train[variable]
    X_cross_text = X_cross[variable]
    X_test_text = X_test[variable]
    X_predict_text = X_predict[variable]
    X_eval_text = X_eval[variable]
    
    train_vec = getFeatures(X_train_text)
    X_train_text = train_vec.transform(X_train_text)
    X_cross_text = train_vec.transform(X_cross_text)
    X_test_text = train_vec.transform(X_test_text)
    
    predict_vec = getFeatures(X_predict_text)
    X_predict_text = predict_vec.transform(X_predict_text)
    X_eval_text = predict_vec.transform(X_eval_text)
   
    if typeModel == 'continuous':
        bowModel = Ridge(alpha = 0.001)
        bowModel2 = Ridge(alpha = 0.001)
        bowModel.fit(X_train_text,Y_train)
        bowModel2.fit(X_predict_text,Y_predict)
        
        inScore = r2_score(Y_train,bowModel.predict(X_train_text))
        print "Train Ridge: r2 score is %f" % (inScore)
        
        inScore = r2_score(Y_cross,bowModel.predict(X_cross_text))
        print "Cross Ridge: r2 score is %f" % (inScore)
        
        X_train[name] = bowModel.predict(X_train_text)
        X_cross[name] = bowModel.predict(X_cross_text)
        X_test[name] = bowModel.predict(X_test_text)
        X_predict[name] = bowModel2.predict(X_test_text)
        X_eval[name] = bowModel2.predict(X_eval_text)
        
    else:
        bowModel = LogisticRegression(penalty='l2',dual=False,tol=0.0001,fit_intercept=True, C=0.0005, intercept_scaling=1, class_weight=None, random_state=423) 
        bowModel2 = LogisticRegression(penalty='l2',dual=False,tol=0.0001,fit_intercept=True, C=.0005, intercept_scaling=1, class_weight=None, random_state=423) 
        bowModel.fit(X_train_text,Y_train)
        bowModel2.fit(X_predict_text,Y_predict)
        
        inScore = roc_auc_score(Y_train,bowModel.predict_proba(X_train_text)[:,1])
        print "Train Logistic: Area under auc curve is %f" % (inScore)
        
        inScore = roc_auc_score(Y_cross,bowModel.predict_proba(X_cross_text)[:,1])
        print "Cross Logistic: Area under auc curve is %f" % (inScore)
        
        X_train[name] = bowModel.predict_proba(X_train_text)[:,1]
        X_cross[name] = bowModel.predict_proba(X_cross_text)[:,1]
        X_test[name] = bowModel.predict_proba(X_test_text)[:,1]
        X_predict[name] = bowModel2.predict_proba(X_predict_text)[:,1]
        X_eval[name] = bowModel2.predict_proba(X_eval_text)[:,1]
        
    return X_train, X_cross, X_test, X_predict, X_eval

def bowModels(X_train, X_cross, X_test, X_predict, X_eval, Y_train_outcomes, Y_cross_outcomes, Y_predict_outcomes):
    
    for outcome in ['fully_funded','great_chat','is_exciting','at_least_1_teacher_referred_donor','at_least_1_green_donation','three_or_more_non_teacher_referred_donors','one_non_teacher_referred_donor_giving_100_plus','donation_from_thoughtful_donor']:
        Y_train = Y_train_outcomes[outcome]
        Y_cross = Y_cross_outcomes[outcome]
        Y_predict = Y_predict_outcomes[outcome]   
        X_train, X_cross, X_test, X_predict, X_eval = bagofwords(X_train, X_cross, X_test, X_predict, X_eval, Y_train, Y_cross, Y_predict,'item_names','binary',outcome+'_model')
    
    return X_train, X_cross, X_test, X_predict, X_eval

def getSamples(data):
    evalData = data[data['date_posted'] >= pd.datetime(2014,1,1)]
    insampledata = data[data['date_posted'] < pd.datetime(2014,1,1)]
    test_data = insampledata[insampledata['date_posted']>=pd.datetime(2013,10,1)]
    cv_data = insampledata[(insampledata['date_posted']>=pd.datetime(2013,7,1)) & (insampledata['date_posted']<pd.datetime(2013,10,1))]
    train_data = insampledata[(insampledata['date_posted']>=pd.datetime(2011,7,1)) & (insampledata['date_posted']<pd.datetime(2013,7,1))]
    prediction_train_data = insampledata[(insampledata['date_posted']<pd.datetime(2014,1,1)) & (insampledata['date_posted']>=pd.datetime(2012,1,1))]
    
    #train_data = train_data.drop('date_posted',axis=1)
    train_data.reset_index(drop=True,inplace=True)
    #cv_data = cv_data.drop('date_posted',axis=1)
    cv_data.reset_index(drop=True,inplace=True)
    #test_data = test_data.drop('date_posted',axis=1)
    test_data.reset_index(drop=True,inplace=True)
    #evalData = evalData.drop('date_posted',axis=1)
    evalData.reset_index(drop=True,inplace=True)
    #prediction_train_data = prediction_train_data.drop('date_posted',axis=1)
    prediction_train_data.reset_index(drop=True,inplace=True)
    
    return train_data, cv_data, test_data, evalData, prediction_train_data
def getOutcomes(allData):
    
    outcomes = ['fully_funded','great_chat','is_exciting','at_least_1_teacher_referred_donor','at_least_1_green_donation','three_or_more_non_teacher_referred_donors','one_non_teacher_referred_donor_giving_100_plus','donation_from_thoughtful_donor']
    data = allData[outcomes]
    train = data[(allData['date_posted']>=pd.datetime(2011,7,1)) & (allData['date_posted']<pd.datetime(2013,7,1))]
    cv = data[(allData['date_posted']>=pd.datetime(2013,7,1)) & (allData['date_posted']<pd.datetime(2013,10,1))]
    predict = data[(allData['date_posted']<pd.datetime(2014,1,1)) & (allData['date_posted']>=pd.datetime(2012,1,1))]
    test = data[(allData['date_posted']>=pd.datetime(2013,10,1)) & (allData['date_posted']<pd.datetime(2014,1,1))]
    
    return train, cv, predict, test

def avgItem(group):
    item = group['item_unit_price']
    number = group['item_quantity']
    return (item * number).sum() / group['item_quantity'].sum()

def generatePrediction(clean_data_path,subPath, Y_eval, name):
    print "Generating prediction"
    print "Reading evaluation data"
    evalData = pd.read_csv(clean_data_path+'evaluation data.csv')
    
    #Need to figure this part out
    print "Saving submission for Kaggle..."
    submission = pd.DataFrame(data= {'projectid':evalData['projectid'].as_matrix(), 'is_exciting': Y_eval})
    submission.to_csv(subPath + name+'.csv', index=False, cols=['projectid','is_exciting'])

def resourceModel():
    
    print "Prediction using project data"
    clean_data_path = 'C:/Users/P_Kravik/Desktop/Kaggle/KDD Cup 2014 - Predicting Excitement at DonorsChoose.org/Data/Clean/Derived/'
    raw_data_path = 'C:/Users/P_Kravik/Desktop/Kaggle/KDD Cup 2014 - Predicting Excitement at DonorsChoose.org/Data/raw/csv/'
    subPath = 'C:/Users/P_Kravik/Desktop/Kaggle/KDD Cup 2014 - Predicting Excitement at DonorsChoose.org/Submissions/'
    
    resources = pd.read_csv(raw_data_path+'resources.csv')
    
    resources['vendorid'].fillna(0, inplace=True)
    resources['vendor_name'].fillna(' ', inplace=True)
    resources['project_resource_type'].fillna(' ', inplace=True)
    resources['item_name'].fillna(' ', inplace=True)
    resources['item_number'].fillna(' ', inplace=True)
    resources['item_unit_price'].fillna(0, inplace=True)
    resources['item_quantity'].fillna(0, inplace=True)
    
    #resources = resources[0:10000]
    
    df = resources.groupby('projectid')['item_name'].apply(lambda x: (x+' ').sum())
    df = df.reset_index()
    df.columns = ['projectid','item_names']
    
    #Number of items
    df['num_items'] = resources.groupby('projectid')['item_quantity'].sum().values
    
    #Average items cost, function defined above
    df['average_item'] = resources.groupby('projectid').apply(avgItem).values
    
    #Expensive items
    resources['expensive_item_over_100'] = (resources['item_unit_price']>=100).apply(lambda x: 1 if x else 0)
    df['num_expensive_over_100'] = resources.groupby('projectid')['expensive_item_over_100'].sum().values
    df['percent_expensive_100'] = 100 * df['num_expensive_over_100'] / df['num_items']
    
    
    #Very expensive item
    resources['expensive_item_over_250'] = resources['item_unit_price']>=250
    df['num_expensive_over_250'] = resources.groupby('projectid')['expensive_item_over_250'].sum().values
    df['percent_expensive_250'] = 100 * df['num_expensive_over_250'] / df['num_items']
    
    
    master = pd.read_csv(clean_data_path + 'all master data.csv')
    date = master[['projectid','teacher_acctid_x','date_posted','fully_funded','great_chat','is_exciting','at_least_1_teacher_referred_donor','at_least_1_green_donation','three_or_more_non_teacher_referred_donors','one_non_teacher_referred_donor_giving_100_plus','donation_from_thoughtful_donor']]
    
    allData = pd.merge(df, date, on='projectid', how='outer')
    allData['date_posted'] = pd.to_datetime(allData['date_posted'])
    allData.sort(['teacher_acctid_x','date_posted'], inplace = True)
    
    del master
    del df
    del date
    
    X_train, X_cross, X_test, X_eval, X_predict  = getSamples(allData)
    Y_train_outcomes, Y_cross_outcomes, Y_predict_outcomes, Y_test_outcomes = getOutcomes(allData)
    Y_train_outcomes.fillna(0,inplace=True)
    Y_cross_outcomes.fillna(0,inplace=True)
    Y_predict_outcomes.fillna(0,inplace=True)
    Y_test_outcomes.fillna(0,inplace=True)
    X_train, X_cross, X_test, X_predict, X_eval = bowModels(X_train, X_cross, X_test, X_predict, X_eval, Y_train_outcomes, Y_cross_outcomes, Y_predict_outcomes)
    
    covar = ['num_items','average_item','num_expensive_over_100','percent_expensive_100','num_expensive_over_250','percent_expensive_250']
    modelcovar = ['fully_funded_model','great_chat_model','is_exciting_model','at_least_1_teacher_referred_donor_model','at_least_1_green_donation_model','three_or_more_non_teacher_referred_donors_model','one_non_teacher_referred_donor_giving_100_plus_model','donation_from_thoughtful_donor_model']
    covar = covar + modelcovar
    X_train = X_train[covar]
    X_cross = X_cross[covar]
    X_test = X_test[covar]
    X_predict = X_predict[covar]
    X_eval = X_eval[covar]
    
    X_train['no_dollars']=X_train['average_item'].isnull().apply(lambda x: 1 if x else 0)
    X_train.fillna(0.0, inplace=True)
    X_cross['no_dollars']=X_train['average_item'].isnull().apply(lambda x: 1 if x else 0)
    X_cross.fillna(0.0, inplace=True)
    X_test['no_dollars']=X_train['average_item'].isnull().apply(lambda x: 1 if x else 0)
    X_test.fillna(0.0, inplace=True)
    X_predict['no_dollars']=X_train['average_item'].isnull().apply(lambda x: 1 if x else 0)
    X_predict.fillna(0.0, inplace=True)
    X_eval['no_dollars']=X_train['average_item'].isnull().apply(lambda x: 1 if x else 0)
    X_eval.fillna(0.0, inplace=True)
    
    X_train.to_csv(clean_data_path+'/X values/resources training X values.csv')
    X_cross.to_csv(clean_data_path+'/X values/resources cross validation X values.csv')
    X_test.to_csv(clean_data_path+'/X values/resources test X values.csv')
    X_predict.to_csv(clean_data_path+'/X values/resources prediction X values.csv')
    X_eval.to_csv(clean_data_path+'/X values/resources evaluation X values.csv')
    
    
    Y_train = Y_train_outcomes['is_exciting']
    Y_cross = Y_cross_outcomes['is_exciting']    
    Y_test = Y_test_outcomes['is_exciting']
    Y_predict = Y_predict_outcomes['is_exciting']
    
    standardize = preprocessing.StandardScaler()
    X_train = standardize.fit_transform(X_train)
    X_cross = standardize.transform(X_cross)
    X_test = standardize.transform(X_test)
    
    standardize_predict = preprocessing.StandardScaler()
    X_predict = standardize_predict.fit_transform(X_predict)
    X_eval = standardize_predict.transform(X_eval)
    
    logit = LogisticRegression(penalty='l2',dual=False,tol=1,fit_intercept=True, C=.004325, intercept_scaling=1, class_weight='auto', random_state=423)
    
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
    
    logit_predict = LogisticRegression(penalty='l2',dual=False,tol=1,fit_intercept=True, C=.004325, intercept_scaling=1, class_weight='auto', random_state=423)
    logit_predict.fit(X_predict,Y_predict)
    
    temp = logit_predict.predict_proba(X_eval)
    Y_predict_logit = temp[:,1]
    generatePrediction(clean_data_path, subPath,Y_predict_logit,'Resource logit model')
    
    rf_weight = Y_train*4+1
    
    clf = ensemble.RandomForestClassifier(n_estimators=100,max_depth=10,max_features=4,bootstrap=True,oob_score=True,n_jobs=2)
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
    
    clf2 = ensemble.RandomForestClassifier(n_estimators=100,max_depth=10,max_features=4,oob_score=True,n_jobs=2)
    clf2.fit(X_predict,Y_predict,rf_predict_weight.values)
    
    temp = clf2.predict_proba(X_eval)
    Y_predict_rf = temp[:,1]
    generatePrediction(clean_data_path, subPath,Y_predict_rf,'Resources random forest model')
    
    #ADABOOST STUMPS PROJECT MODEL
    ada_weight = Y_train*4+1
    
    ada = ensemble.AdaBoostClassifier(n_estimators=100,learning_rate = .1,random_state = 42)
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
    
    temp = ada2.predict_proba(X_eval)
    Y_predict_ada = temp[:,1]
    generatePrediction(clean_data_path, subPath, Y_predict_ada,'Resources AdaBoost model')
    
    
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
    
    final_standardize3 = preprocessing.StandardScaler()
    logit_eval_score = final_standardize3.fit_transform(Y_predict_logit)
    ada_eval_score = final_standardize3.fit_transform(Y_predict_ada)
    rf_eval_score = final_standardize3.fit_transform(Y_predict_rf)
    
    eval_final_score = logit_eval_score + ada_eval_score + rf_eval_score
    generatePrediction(clean_data_path, subPath, eval_final_score,'Resources Blended model')
    
    cross_x_values = pd.DataFrame({'logit':logit_score, 'ada':ada_score, 'rf':rf_score})
    test_x_values = pd.DataFrame({'logit':logit_test_score, 'ada':ada_test_score, 'rf':rf_test_score})
    logit = LogisticRegression(penalty='l2',dual=False,tol=0.0001,fit_intercept=True, C=0.0005, intercept_scaling=1, class_weight='auto', random_state=423)
    logit.fit(cross_x_values, Y_cross)
    
    inScore = roc_auc_score(Y_cross,logit.predict_proba(cross_x_values)[:,1])
    print "Cross Blend Logit: Area under auc curve is %f" % (inScore)
    
    inScore = roc_auc_score(Y_test,logit.predict_proba(test_x_values)[:,1])
    print "Test Blend Logit: Area under auc curve is %f" % (inScore)
    
    
    
    return 1

if __name__ == '__main__':
    resourceModel()