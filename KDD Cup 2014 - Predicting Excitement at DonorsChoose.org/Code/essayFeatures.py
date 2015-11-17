'''
Created on Jun 5, 2014

@author: P_Kravik
'''
import pandas as pd
import numpy as np
import random
import nltk
from sklearn.feature_extraction.text import CountVectorizer

def getFeatures(essays):
    #Screw it, going to use sk_learn
    print "Feature extraction..."
    vectorizer = CountVectorizer(min_df=.01, max_df=0.95)
    X_train = vectorizer.fit_transform(essays)
    num_samples, num_features = X_train.shape
    print "#samples: %d, #features: %d" % (num_samples, num_features)
    
    return (X_train, vectorizer)

def tokenize(essays):
    #Should think about how to tokenize
    tokens = essays.apply(lambda d: nltk.word_tokenize(d))
    
    numWords = tokens.apply(lambda d: len(d))
    numUniqueWords = tokens.apply(lambda d: len(set(d)))
    lexicalDiversity = numWords/numUniqueWords
    
    
    
    newFeatures = [numWords, numUniqueWords, lexicalDiversity]

def numCharacters(essays):
    numCharacters = essays.apply(lambda d: len(d))
    return numCharacters


def main(clean_data_path):
    print "Making Essay Features"
    trainEssay = pd.read_csv(clean_data_path+"training data.csv")
    

if __name__ == '__main__':
    clean_data_path = 'C:/Users/P_Kravik/Desktop/Kaggle/KDD Cup 2014 - Predicting Excitement at DonorsChoose.org/Data/Clean/Derived/'
    main(clean_data_path)