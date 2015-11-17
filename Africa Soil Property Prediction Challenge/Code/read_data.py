'''
Created on Sep 16, 2014

@author: P_Kravik
'''

import pandas as pd
from sklearn.cross_validation import train_test_split

def readData(raw_data_path):
    train = pd.read_csv(raw_data_path + "training.csv")
    test = pd.read_csv(raw_data_path + "sorted_test.csv")
    return train, test

def separateOutcome(train):
    y_train = train[["Ca", "P", "pH", "SOC", "Sand"]]
    train.drop(["Ca", "P", "pH", "SOC", "Sand"], axis=1, inplace=True)
    
    return y_train, train

def generateTrainTestSplit(X_train_full, Y_train_full, seed):
    
    train_columns = X_train_full.columns
    #X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.25, random_state = seed)
    labels_train = X_train_full["location"]
    
    X_train_labels, X_test_labels = train_test_split(labels_train.unique(), test_size = 0.25, random_state = seed)
    
    in_train = X_train_full.apply(lambda x: True if x["location"] in X_train_labels else False, axis=1)
    
    
    X_train = X_train_full[in_train]
    X_test = X_train_full[~in_train]
    Y_train = Y_train_full[in_train]
    Y_test = Y_train_full[~in_train]
    
    Y_train = pd.DataFrame(Y_train, columns = ["Ca", "P", "pH", "SOC", "Sand"])
    Y_test = pd.DataFrame(Y_test, columns = ["Ca", "P", "pH", "SOC", "Sand"])
    X_train = pd.DataFrame(X_train, columns = train_columns)
    X_test = pd.DataFrame(X_test, columns = train_columns)
    
    X_train_location = X_train.pop("location")
    X_test_location = X_test.pop("location")
    
    return X_train, Y_train, X_test, Y_test, X_train_location

def getVariables(train, eval):
    
    nonspectravar = ["BSAN", "BSAS", "BSAV", "CTI", "ELEV", "EVI", "LSTD", "LSTN", "REF1", "REF2", "REF3", "REF7", "RELI", "TMAP", "TMFI"]
    spectravar = train.columns[0:3577]
    X_train = pd.concat([train[nonspectravar], train[spectravar], pd.get_dummies(train['Depth'])], axis=1)
    X_train["location"] = X_train["ELEV"].astype(str) + X_train["EVI"].astype(str)
   
    X_eval = pd.concat([eval[nonspectravar], eval[spectravar],pd.get_dummies(eval['Depth'])], axis=1)
    
    return X_train, X_eval
