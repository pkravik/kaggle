'''
Created on Sep 4, 2014

@author: P_Kravik
'''

import pandas as pd
import numpy as np
import random
import math
import time
import os
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

from ca_models import *
from p_models import *
from ph_models import *
from soc_models import *
from sand_models import *
from read_data import *


def runModel(X_train, Y_train, X_eval, var = "Ca", seed=42):
    #25% in Test Sample, 75% in train and cv
    X_train, Y_train, X_test, Y_test, X_train_location_labels = generateTrainTestSplit(X_train, Y_train, seed)
    
    
    labels = np.array(pd.factorize(X_train_location_labels)[0])
    
    #n_folds = 10
    n_folds = 10
    
    labels2 = (labels * (n_folds+1)/labels.max())
    labels2[-1]=n_folds
    
    #Model is Ridge Regression
    cv_iterator = LeaveOneLabelOut(labels2)
    #test = [0.00001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    #modelCV = RidgeCV(test, normalize=True, score_func="mse", scoring="mse", cv=cv_iterator )
    #modelCV.fit(X_train, Y_train[var])
    
    if var == "Ca":
        prediction, test_score = CaModel(X_train, Y_train, X_test, Y_test, X_eval, cv_iterator)
    elif var == "P":
        prediction, test_score = PModel(X_train, Y_train, X_test, Y_test, X_eval, cv_iterator)
    elif var == "pH":
        prediction, test_score = pHModel(X_train, Y_train, X_test, Y_test, X_eval, cv_iterator)
    elif var == "SOC":
        prediction, test_score = SOCModel(X_train, Y_train, X_test, Y_test, X_eval, cv_iterator)
    elif var == "Sand":
        prediction, test_score = SandModel(X_train, Y_train, X_test, Y_test, X_eval, cv_iterator)
    else:
        print "Error"

    
    return prediction, test_score

def test(n):
    num_models = n
    
    #Create list of random seed
    random.seed(45)
    seeds = random.sample(range(200), num_models)
    print seeds
    
    

def multipleModels(X_train, Y_train, X_eval, n = 10, var = "Ca"):
    
    #If you do more than 10 this fucks up, no idea why
    num_models = n
    
    #Create list of random seed
    random.seed(45)
    seeds = random.sample(range(200), num_models)
    results = []
    predictions = []
    
    #Loop through each seed, create a test/train split and run the model
    for seed in seeds:
        print "Running iteration"
        prediction, test_score = runModel(X_train, Y_train, X_eval, var, seed)
        results.append(test_score)
        predictions.append(prediction)
    
    #Get the average MSE for CV and Test
    test_results = pd.DataFrame(results)

    avg_prediction = np.median(np.array(predictions), axis=0)
    std_prediction = np.array(predictions).std(axis=0)
    
    #print results
    #print "Var: " + var + ", Test, avg: %f, std: %f" % (abs(avg_test), std_test)
    
    return test_results, avg_prediction

if __name__ == '__main__':
    
    raw_data_path = "C:/Users/P_Kravik/Desktop/Kaggle/Africa Soil Property Prediction Challenge/Data/Raw/"
    submission_data_path = "C:/Users/P_Kravik/Desktop/Kaggle/Africa Soil Property Prediction Challenge/Submissions/"
    intermediate_predicton_path = "C:/Users/P_Kravik/Desktop/Kaggle/Africa Soil Property Prediction Challenge/Individual Predictions/"
    train, eval = readData(raw_data_path)
    
    #train.groupby("Depth")[["Ca", "P", "pH", "SOC", "Sand"]].agg([np.mean, np.std])
    Y_train, train = separateOutcome(train)
    
    id_train = train.pop("PIDN")
    id_eval = eval.pop("PIDN")
    #np.corrcoef(train['m7497.96'], Y_train)
    
    X_train, X_eval = getVariables(train, eval)
     
   
    
    #a,b=runModel(X_train, Y_train, X_eval, "P", 71)
    bench_sub = "SVR 40237 prediction/"
    directory = intermediate_predicton_path + bench_sub
    subdir = "Median SVR/"
    directory = intermediate_predicton_path + subdir
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    #["Ca", "P", "pH", "SOC", "Sand"]
    
    #Calcium
    model_results_ca, ca_prediction = multipleModels(X_train, Y_train, X_eval, n=10, var = "Ca")
    
    model_results_ca.to_csv(directory + "Ca model results.csv")
    pd.DataFrame({"PIDN": id_eval, "Ca":ca_prediction}, columns=["PIDN", "Ca"]).to_csv(directory + "Ca.csv", index=False)
    
    
    #Phosphorus
    model_results_P, p_prediction = multipleModels(X_train, Y_train, X_eval, n=10, var = "P")
    
    model_results_P.to_csv(directory + "P model results.csv")
    pd.DataFrame({"PIDN": id_eval, "P":p_prediction}, columns=["PIDN", "P"]).to_csv(directory + "P.csv", index=False)
    
    #pH
    model_results_pH, pH_prediction = multipleModels(X_train, Y_train, X_eval, n=10, var = "pH")
    
    model_results_pH.to_csv(directory + "pH model results.csv")
    pd.DataFrame({"PIDN": id_eval, "pH":pH_prediction}, columns=["PIDN", "pH"]).to_csv(directory + "pH.csv", index=False)
    
    #SOC
    model_results_SOC, SOC_prediction = multipleModels(X_train, Y_train, X_eval, n=10, var = "SOC")
    
    model_results_SOC.to_csv(directory + "SOC model results.csv")
    pd.DataFrame({"PIDN": id_eval, "SOC":SOC_prediction}, columns=["PIDN", "SOC"]).to_csv(directory + "SOC.csv", index=False)

    #Sand
    model_results_Sand, Sand_prediction = multipleModels(X_train, Y_train, X_eval, n=10,var = "Sand")
    
    model_results_Sand.to_csv(directory + "Sand model results.csv")
    pd.DataFrame({"PIDN": id_eval, "Sand":Sand_prediction}, columns=["PIDN", "Sand"]).to_csv(directory + "Sand.csv", index=False)
    
    
    #Read in all of the previous individual predictions
    Ca = pd.read_csv(directory + "Ca.csv")
    P = pd.read_csv(directory + "P.csv")
    pH = pd.read_csv(directory + "pH.csv")
    SOC = pd.read_csv(directory + "SOC.csv")
    Sand = pd.read_csv(directory + "Sand.csv")   
    #Final Submission
    submission = pd.DataFrame({"PIDN": id_eval, "Ca":Ca["Ca"], "P":P["P"], "pH":pH["pH"], "SOC":SOC["SOC"], "Sand":Sand["Sand"]}, columns=["PIDN", "Ca", "P", "pH", "SOC", "Sand"])
    
    #Print errors
    Ca_results = pd.read_csv(directory + "Ca model results.csv")
    P_results = pd.read_csv(directory + "P model results.csv")
    pH_results = pd.read_csv(directory + "pH model results.csv")
    SOC_results = pd.read_csv(directory + "SOC model results.csv")
    Sand_results = pd.read_csv(directory + "Sand model results.csv")
    
    Ca_error = Ca_results['final'].mean()
    P_error = P_results['final'].mean()
    pH_error = pH_results['final'].mean()
    SOC_error = SOC_results['final'].mean()
    Sand_error = Sand_results['final'].mean()
    total_error = np.mean([Ca_error, P_error, pH_error, SOC_error, Sand_error])
    
    print "Final MCRMSE\nTest; Ca: %f, P: %f, pH: %f, SOC: %f, Sand: %f\nTotal: %f" % (Ca_error, P_error,pH_error,SOC_error,Sand_error,total_error)
    
    #Save submission
    submission.to_csv(directory + "Final Submission.csv", index=False)
    
    
    pass