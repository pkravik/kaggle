'''
Created on Sep 11, 2014

@author: P_Kravik
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import sparse

from sklearn.grid_search import GridSearchCV


from sklearn.ensemble import GradientBoostingClassifier

from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit


import random


def getOrdinalCategorical(train, test, need_dums):
    #train: training set (in sample)
    #test: test set (out of sample)
    #need_dums: array of variable names to generate dummies for

    #Initialize empty dataframes
    X_train= pd.DataFrame()
    X_test = pd.DataFrame()
    
    #Combine to get dummies with all the data to avoid issues of new things popping up in test
    for d in need_dums:
        #Combine the data to ensure that you get all of the unique values
        test_dummies = pd.concat([train[d], test[d]], axis=0)
        
        #This returns a DataFrame with the column being a number corresponding to a unique categorical value
        test2 = pd.DataFrame({'cat'+d:np.unique(test_dummies, return_inverse=True)[1]})
        
        #Split into train and test
        X_train = pd.concat([X_train, test2.iloc[0:len(train)]], axis=1)
        X_test =  pd.concat([X_test, test2.iloc[len(train):]], axis=1)
  
    #Reset the Dataframe indices because the concat messes it up  
    X_test.reset_index(inplace=True, drop=True)
    
    return X_train, X_test

def genNewVariables(train, test):
    #Random things I would think are relevant
    
    #Last digit of odomoter. Looks actually kind of significant, at least 0 and 1
    train['lastDigitOdo0'] = (train['VehOdo'] % 10 == 0).apply(lambda x: 1 if x else 0)
    test['lastDigitOdo0'] = (test['VehOdo'] % 10 == 0).apply(lambda x: 1 if x else 0)
    
    #Code below return average outcome and count by last digit of odometer
    #tmp = train.groupby('lastDigitOdo').agg({'IsBadBuy':np.mean ,'RefId':pd.Series.nunique})
    
    #Is the odometer high for the age
    
    #This returns a dataframe at the vehicle age level with average and std of number of miles
    odometer_age = pd.DataFrame(train.groupby("VehicleAge").agg({'VehOdo':[np.mean, np.std]})['VehOdo'])
    odometer_age.columns=['vehicleAgeOdoMean','vehicleAgeOdoStd']
    
    #Merge this back on the dataset, and resort to fix any index issues
    train = train.join(odometer_age, on=['VehicleAge'])
    test = test.join(odometer_age, on=['VehicleAge'])
    train.sort(inplace=True)
    test.sort(inplace=True)
    
    #Calculate standardized odomoeter reading by vehicle age. 
    train['vehicleAgeOdo'] = (train['VehOdo']-train['vehicleAgeOdoMean'])/train['vehicleAgeOdoStd']
    test['vehicleAgeOdo'] = (test['VehOdo']-test['vehicleAgeOdoMean'])/test['vehicleAgeOdoStd']
    
    #Is the odometer high for the make/model
    
    #This returns a dataframe at the make and model level with average 
    odometer_make_model = pd.DataFrame(train.groupby(["Make","Model"]).agg({'VehOdo':[np.mean, np.std]})['VehOdo'])
    odometer_make_model.columns=['vehicleMakeModelOdoMean','vehicleMakeModelOdoStd']
    #Fill in missing values for when there is no std
    odometer_make_model.fillna(100)
    train = train.join(odometer_make_model, on=["Make","Model"])
    test = test.join(odometer_make_model, on=["Make","Model"])
    train.sort(inplace=True)
    test.sort(inplace=True)
    
    #Is the purchase price high for the make/age
     
    train['vehicleMakeModelOdo'] = (train['VehOdo']-train['vehicleMakeModelOdoMean'])/train['vehicleMakeModelOdoStd']
    test['vehicleMakeModelOdo'] = (test['VehOdo']-test['vehicleMakeModelOdoMean'])/test['vehicleMakeModelOdoStd']
    
    #Change in average acquisition price from time of purchase to current date
    train['changeAveragePrice'] = train['MMRCurrentAuctionAveragePrice'] - train['MMRAcquisitionAuctionAveragePrice']
    train['percentChangeAveragePrice'] = train['changeAveragePrice']/train['MMRAcquisitionAuctionAveragePrice']
    test['changeAveragePrice'] = test['MMRCurrentAuctionAveragePrice'] - test['MMRAcquisitionAuctionAveragePrice']
    test['percentChangeAveragePrice'] = test['changeAveragePrice']/test['MMRAcquisitionAuctionAveragePrice']
    
    #Change in above average auction
    train['changeCleanPrice'] = train['MMRCurrentAuctionCleanPrice'] - train['MMRAcquisitionAuctionCleanPrice']
    train['percentChangeCleanPrice'] = train['changeCleanPrice']/train['MMRAcquisitionAuctionCleanPrice']
    test['changeCleanPrice'] = test['MMRCurrentAuctionCleanPrice'] - test['MMRAcquisitionAuctionCleanPrice']
    test['percentChangeCleanPrice'] = test['changeCleanPrice']/test['MMRAcquisitionAuctionCleanPrice']
    
    #Change in retail average
    train['changeRetailPrice'] = train['MMRCurrentRetailAveragePrice'] - train['MMRAcquisitionRetailAveragePrice']
    train['percentChangeRetailPrice'] = train['changeRetailPrice']/train['MMRAcquisitionRetailAveragePrice']
    test['changeRetailPrice'] = test['MMRCurrentRetailAveragePrice'] - test['MMRAcquisitionRetailAveragePrice']
    test['percentChangeRetailPrice'] = test['changeRetailPrice']/test['MMRAcquisitionRetailAveragePrice']
    
    #Change in above average retail
    train['changeRetailCleanPrice'] = train['MMRCurrentRetailCleanPrice'] - train['MMRAcquisitonRetailCleanPrice']
    train['percentChangeRetailCleanPrice'] = train['changeRetailCleanPrice']/train['MMRAcquisitonRetailCleanPrice']
    test['changeRetailCleanPrice'] = test['MMRCurrentRetailCleanPrice'] - test['MMRAcquisitonRetailCleanPrice']
    test['percentChangeRetailCleanPrice'] = test['changeRetailCleanPrice']/test['MMRAcquisitonRetailCleanPrice']

    #might need to divide by year
    
    #Compare acquisition bought cost to the MMR model values
    train['paidDifferenceAverage'] = train['VehBCost']-train['MMRAcquisitionAuctionAveragePrice']
    train['paidDifferenceClean'] = train['VehBCost']-train['MMRAcquisitionAuctionCleanPrice']
    train['paidDifferenceRetail'] = train['VehBCost']-train['MMRAcquisitionRetailAveragePrice']
    test['paidDifferenceAverage'] = test['VehBCost']-test['MMRAcquisitionAuctionAveragePrice']
    test['paidDifferenceClean'] = test['VehBCost']-test['MMRAcquisitionAuctionCleanPrice']
    test['paidDifferenceRetail'] = test['VehBCost']-test['MMRAcquisitionRetailAveragePrice']
    
    #Odometer/cost
    train['costPerMile'] = train['VehOdo']/train['VehBCost']
    test['costPerMile'] = test['VehOdo']/test['VehBCost']
    
    #Warranty cost as a % of purchase cost
    train['warrantyCostPctPaid'] = train['WarrantyCost']/train['VehBCost']
    test['warrantyCostPctPaid'] = test['WarrantyCost']/test['VehBCost']
    
    #Warranty cost as a percent of MMR cost
    train['warrantyCostPctMMR'] = train['WarrantyCost']/train['MMRAcquisitionAuctionAveragePrice']
    test['warrantyCostPctMMR'] = test['WarrantyCost']/test['MMRAcquisitionAuctionAveragePrice']
    
    #Time trends
    train = train.set_index(pd.DatetimeIndex(pd.to_datetime(train['PurchDate'])))
    test = test.set_index(pd.DatetimeIndex(pd.to_datetime(test['PurchDate'])))
    
    #Day of week 0-6, 0 is monday
    train['dayOfWeek'] = train.index.dayofweek
    test['dayOfWeek'] = test.index.dayofweek
    
    #week of year
    train['week'] = train.index.week
    test['week'] = test.index.week
    
    #Month.
    train['month'] = train.index.week
    test['month'] = test.index.week
    
    #year
    train['year'] = train.index.year
    test['year'] = test.index.year
    
    #reset the index and get rid of date
    train.reset_index(inplace=True)
    test.reset_index(inplace=True)
    
    train.replace([np.inf, -np.inf], -100, inplace=True)
    test.replace([np.inf, -np.inf], -100, inplace=True)
    
    #Test interaction of dealer and wheel type?
    #mmm = train.groupby(["WheelType", "WheelTypeID"]).agg({'IsBadBuy':np.mean, 'RefId':pd.Series.nunique})
    
    return train, test;
    #Make MMR variables

def clean(train, test):
    
    #Generate new variables to include
    train, test = genNewVariables(train, test)
    
    #'year', 'month', 'week', 'dayOfWeek',
    #Get dummy variables
    need_dums = ['Make', 'VNST', 'VNZIP1', 'Auction', 'Transmission', 'Color', 'WheelType', 'Nationality', 'Size', 'VehYear']
    need_dums = need_dums + ['Model', 'SubModel','Trim', 'TopThreeAmericanName', 'PRIMEUNIT', 'AUCGUART', 'BYRNO']

    X_train_dummies, X_test_dummies = getOrdinalCategorical(train, test, need_dums)
    
    #Get continuous variables
    
    #Select continuous variables
    continuous_vars = ['VehicleAge','VehOdo', 'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice', 'VehBCost', 'IsOnlineSale', 'WarrantyCost']
    newVarContinuous = ['year', 'month', 'week', 'dayOfWeek','lastDigitOdo0', 'vehicleAgeOdo','vehicleMakeModelOdo']
    changeVar = ['changeAveragePrice','changeCleanPrice','changeRetailPrice','changeRetailCleanPrice']
    pctChangeVar = ['percentChangeAveragePrice']
    diffPaidVar = ['paidDifferenceAverage']
    adjustedCostVar = ['costPerMile','warrantyCostPctPaid']
    continuous_vars = continuous_vars + changeVar + pctChangeVar + diffPaidVar + adjustedCostVar + newVarContinuous
    
    #Keep only the selected variables
    X_train_continuous = train[continuous_vars]
    X_test_continuous = test[continuous_vars]
    
    #Some missing values, for a tiny amount
    X_train_continuous.isnull().any()
    X_train_dummies.isnull().any()
    X_train_continuous.isnull().sum()
    
    #Just filling in with the mean. NOT SURE IF THIS IS WORKING RIGHT, but a small amount of missing anyway
    X_train_continuous = X_train_continuous.fillna(X_train_continuous.mean())
    X_test_continuous = X_test_continuous.fillna(X_test_continuous.mean())
    
    #Combine to create the final dataset
    X_train = pd.concat([X_train_dummies, X_train_continuous], axis=1)
    X_test = pd.concat([X_test_dummies, X_test_continuous], axis=1)

    return X_train, X_test

def gini(estimator, X, y):
    #This function takes an estimator and a dataset and outcome and resturns the gini coefficient
    df = pd.DataFrame({'actual':y, 'predicted':estimator.predict_proba(X)[:,1]})
    df.sort('predicted', ascending=False, inplace=True)
    pop_delta = 1/float(len(df['actual']))
    total_losses = np.sum(df['actual'])
    null_losses = [pop_delta] * len(df['actual'])
    accum_losses = df['actual']/total_losses
    gini_sum = (accum_losses - null_losses).cumsum()
    return sum(gini_sum)/len(df['actual'])

def giniNoEstimator(y, y_pred):
    df = pd.DataFrame({'actual':y, 'predicted':y_pred})
    df.sort('predicted', ascending=False, inplace=True)
    pop_delta = 1/float(len(df['actual']))
    total_losses = np.sum(df['actual'])
    null_losses = [pop_delta] * len(df['actual'])
    accum_losses = df['actual']/total_losses
    gini_sum = (accum_losses - null_losses).cumsum()
    return sum(gini_sum)/len(df['actual'])

def saveObject(file_path, file_name, objectToSave):
    filehandler = open(file_path+file_name, 'w')
    pickle.dump(objectToSave, filehandler)
    print file_name + " Saved"
    
def openObject(file_path, file_name):
    filehandler = open(file_path+file_name, 'r')
    return pickle.load(filehandler)

def gradientBoostingModel(derived_data_path,X_train, Y_train, X_test):
    print "Gradient Boosting"
    random.seed(21)
    
    gbm = GradientBoostingClassifier(loss='deviance')
    #gbm.fit(X_train, Y_train)
    
    #Perform 5-fold Stratified cross validation
    modelCV = gbm
    param_grid = {'n_estimators': [200,500, 1000,1500],
                  'learning_rate':[0.1, 0.008],
                  'subsample':[1],
                  'min_samples_leaf':[5],
                  'max_depth':[3, 7],
                  'max_features':[0.2]}
    
    cv_iterator = StratifiedKFold(Y_train, 5, shuffle=True, random_state = 11)
    search = GridSearchCV(modelCV, param_grid, scoring=gini, n_jobs=1, iid=True, cv=cv_iterator, verbose = 2)
    
    #Not sure where to set the seed, here is a good a place as any
    random.seed(42)
    
    #Run the grid search
    search.fit(X_train, Y_train)
    
    #Return the grid scores
    search.grid_scores_
    
    #Feature importance
    varlist = pd.DataFrame({'variable':X_train.columns.values, 'importance':search.best_estimator_.feature_importances_})
    varlist.sort('importance', ascending=False, inplace=True)
    varlist
    
    #Save the results
    saveObject(derived_data_path, 'GBM_gridsearch_example.obj', search)
    results = openObject(derived_data_path, 'GBM_gridsearch_example.obj')
    
    model = results.best_estimator_
    
    #print the results
    print "gini score CV: %f" % (results.best_score_)
    print "gini score train: %f" % (gini(model, X_train, Y_train))
    
    return search

def ensembleGBMTest(derived_data_path, X_train, Y_train, X_test, Y_test):
    random.seed(60)
    GBM1 = GradientBoostingClassifier(n_estimators = 1500, learning_rate = 0.008, min_samples_leaf = 5, max_features=0.2, max_depth=7)
    GBM2 = GradientBoostingClassifier(n_estimators = 1700, learning_rate = 0.007, min_samples_leaf = 5, max_features=0.2, max_depth=7)
    GBM3 = GradientBoostingClassifier(n_estimators = 1600, learning_rate = 0.0075, min_samples_leaf = 5, max_features=0.2, max_depth=7)
    GBM4 = GradientBoostingClassifier(n_estimators = 1650, learning_rate = 0.007, min_samples_leaf = 5, max_features=0.2, max_depth=8)
    GBM5 = GradientBoostingClassifier(n_estimators = 1750, learning_rate = 0.00725, min_samples_leaf = 6, max_features=0.2, max_depth=7)
    GBM6 = GradientBoostingClassifier(n_estimators = 1550, learning_rate = 0.00775, min_samples_leaf = 4, max_features=0.2, max_depth=7)
    GBM7 = GradientBoostingClassifier(n_estimators = 1850, learning_rate = 0.00725, min_samples_leaf = 5, max_features=0.2, max_depth=6)

    GBM1.fit(X_train, Y_train)
    GBM2.fit(X_train, Y_train)
    GBM3.fit(X_train, Y_train)
    GBM4.fit(X_train, Y_train)
    GBM5.fit(X_train, Y_train)
    GBM6.fit(X_train, Y_train)
    GBM7.fit(X_train, Y_train)
    
    print "GBM1: %f" % (gini(GBM1, X_test, Y_test))
    print "GBM2: %f" % (gini(GBM2, X_test, Y_test))
    print "GBM3: %f" % (gini(GBM3, X_test, Y_test))
    print "GBM4: %f" % (gini(GBM4, X_test, Y_test))
    print "GBM5: %f" % (gini(GBM5, X_test, Y_test))
    print "GBM6: %f" % (gini(GBM6, X_test, Y_test))
    print "GBM7: %f" % (gini(GBM7, X_test, Y_test))
    
    #now combine!
    combine = GBM1.predict_proba(X_test)[:,1] + GBM2.predict_proba(X_test)[:,1] + GBM3.predict_proba(X_test)[:,1] +GBM4.predict_proba(X_test)[:,1] +GBM5.predict_proba(X_test)[:,1] 
    combine = combine + GBM6.predict_proba(X_test)[:,1] + GBM7.predict_proba(X_test)[:,1]
    print "With our powers combined: %f" % (giniNoEstimator(Y_test, combine))

    GBMClassifiers = [GBM1, GBM2, GBM3, GBM4, GBM5, GBM6, GBM7]
    saveObject(derived_data_path, 'GBM_classifiers.obj', GBMClassifiers)

def ensembleGBM(derived_data_path, X_train, Y_train, X_test, seed=60):
    random.seed(seed)
    GBM1 = GradientBoostingClassifier(n_estimators = 1500, learning_rate = 0.008, min_samples_leaf = 5, max_features=0.2, max_depth=7)
    GBM2 = GradientBoostingClassifier(n_estimators = 1700, learning_rate = 0.007, min_samples_leaf = 5, max_features=0.2, max_depth=7)
    GBM3 = GradientBoostingClassifier(n_estimators = 1600, learning_rate = 0.0075, min_samples_leaf = 5, max_features=0.2, max_depth=7)
    GBM4 = GradientBoostingClassifier(n_estimators = 1650, learning_rate = 0.007, min_samples_leaf = 5, max_features=0.2, max_depth=8)
    GBM5 = GradientBoostingClassifier(n_estimators = 1750, learning_rate = 0.00725, min_samples_leaf = 6, max_features=0.2, max_depth=7)
    GBM6 = GradientBoostingClassifier(n_estimators = 1550, learning_rate = 0.00775, min_samples_leaf = 4, max_features=0.2, max_depth=7)
    GBM7 = GradientBoostingClassifier(n_estimators = 1850, learning_rate = 0.00725, min_samples_leaf = 5, max_features=0.2, max_depth=6)

    print "Running Model 1"
    GBM1.fit(X_train, Y_train)
    print "Running Model 2"
    GBM2.fit(X_train, Y_train)
    print "Running Model 3"
    GBM3.fit(X_train, Y_train)
    print "Running Model 4"
    GBM4.fit(X_train, Y_train)
    print "Running Model 5"
    GBM5.fit(X_train, Y_train)
    print "Running Model 6"
    GBM6.fit(X_train, Y_train)
    print "Running Model 7"
    GBM7.fit(X_train, Y_train)
    
    GBMClassifiers = [GBM1, GBM2, GBM3, GBM4, GBM5, GBM6, GBM7]
    saveObject(derived_data_path, 'GBM_classifiers.obj', GBMClassifiers)
    
    combine = float(1)/7*(GBM1.predict_proba(X_test)[:,1] + GBM2.predict_proba(X_test)[:,1] + GBM3.predict_proba(X_test)[:,1] +GBM4.predict_proba(X_test)[:,1] +GBM5.predict_proba(X_test)[:,1] + GBM6.predict_proba(X_test)[:,1] + GBM7.predict_proba(X_test)[:,1])

    return combine

if __name__ == '__main__':
    raw_data_path = "C:/Users/P_Kravik/Desktop/Kaggle/Don't Get Kicked!/Data/Raw/"
    derived_data_path = "C:/Users/P_Kravik/Desktop/Kaggle/Don't Get Kicked!/Data/Derived/"
    submission_data_path = "C:/Users/P_Kravik/Desktop/Kaggle/Don't Get Kicked!/Submisisons/"
    
    #read raw data
    train = pd.read_csv(raw_data_path + "training.csv")
    test = pd.read_csv(raw_data_path + "test.csv")
    
    #train_insample, train_est, y_insample, y_test = train_test_split(train, train['IsBadBuy'], test_size = 0.25, random_state = 775)
    
    #===========================================================================
    # msk = np.random.rand(len(train)) < 0.8
    # train_insample = train[msk]
    # train_test = train[~msk]
    # train_insample.reset_index(inplace=True, drop=True)
    # train_test.reset_index(inplace=True, drop=True)
    # 
    # Y_train = train_insample['IsBadBuy']
    # Y_test = train_test['IsBadBuy']
    # 
    # X_train, X_test = clean(train_insample, train_test, True)
    #===========================================================================
    
    #Final
    Y_train = train["IsBadBuy"]
    X_train, X_test = clean(train, test)
    
    #===========================================================================
    # prediction = ensembleGBM(derived_data_path, X_train, Y_train, X_test, seed=666)
    # 
    # submission = pd.DataFrame({'RefId':test['RefId'], 'IsBadBuy':prediction})
    # submission.to_csv(submission_data_path + "The Mauve Avengers - THE FINAL SHOWDOWN.csv", index=False, columns = ['RefId','IsBadBuy'])
    # #Clean data
    # 
    # final_classifiers = openObject(derived_data_path,'GBM_classifiers.obj')
    # final_classifiers[0].feature_importances_
    # 
    # varlist = pd.DataFrame({'variable':X_train.columns.values, 'importance':final_classifiers[0].feature_importances_})
    # varlist.sort('importance', ascending=False, inplace=True)
    # varlist
    #===========================================================================
    
    #ensembleGBMTest(X_train, Y_train, X_test, Y_test)
    
    classifiers = openObject(derived_data_path, 'GBM_classifiers.obj')
    i = 1;
    for model in classifiers:
        print "Model" + str(i)
        prediction = model.predict_proba(X_test)[:,1]
        submission = pd.DataFrame({'RefId':test['RefId'], 'IsBadBuy':prediction})
        submission.to_csv(submission_data_path + "GBM Model" + str(i) + ".csv", index=False, columns = ['RefId','IsBadBuy'])
        i = i+1
    
    #search = gradientBoostingModel(derived_data_path, X_train, Y_train, X_test)
     
    #===========================================================================
    # xa_train, xb_test, ya_train, yb_test = train_test_split(X_train, Y_train, test_size = 0.25, random_state = 10)
    # results = openObject(derived_data_path, 'GBM_gridsearch_more_trees.obj')
    # best_model = results.best_estimator_.fit(xa_train, ya_train)
    # giniTest(yb_test, best_model.predict_proba(xb_test))
    #  
    # n_estimators = 2000
    # x = np.arange(n_estimators) + 1
    # 
    #  
    # test_score = np.zeros(2000,)
    #  
    # for i, y_pred in enumerate(best_model.staged_predict_proba(xb_test)):
    #     #test_score[i] = 1
    #     test_score[i] = giniTest(yb_test, y_pred)
    #  
    # plt.plot(x, test_score)
    # plt.show()
    #===========================================================================   
    
    pass