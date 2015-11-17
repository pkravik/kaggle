'''
Created on May 23, 2014

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

def testBagofWords(vectorizer,model,outcome,variable):
    clean_data_path = 'C:/Users/P_Kravik/Desktop/Kaggle/KDD Cup 2014 - Predicting Excitement at DonorsChoose.org/Data/Clean/Derived/'
    cvData = pd.read_csv(clean_data_path + 'cross validation data.csv')
    #test each individual model on cross validation
    print "Testing Bag of words for %s on cross validation" % (variable)
    X_cv = vectorizer.transform(cvData[variable])        
    #Outcomes
    Y_cv = cvData[outcome]
        
    #Logistic regression, not sure if best
    predict = model.predict_proba(X_cv)
    yhat = predict[:,1]
    
    inScore = roc_auc_score(Y_cv,yhat)
    print "CV: Area under auc curve is %f" % (inScore)
    

def getFeatures(text):
    #Screw it, going to use sk_learn
    vectorizer = CountVectorizer(min_df=.01, max_df=0.6, ngram_range=(1,2))
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
    
    
    #Fitting on cross
    #testBagofWords(vectorizer,bowLogit,'is_exciting','essay')
    
    return (yhat, vectorizer, bowModel)

def bagofwords(trainData, outcome, variables, time='', mask='all',typeModel='binary'):
      
    X_covar_bow = {}
    X_covar_vocab={}
    
    if mask == 'all':
        predictData = trainData
    else:  
        predictData = trainData
        trainData = trainData[trainData[mask]]
    
    Y_train = predictData[outcome]
    
    if 'title' in variables:
        yhat_bowTitle, title_vectorizer, title_model = bowFitAndPrediction(predictData['title'], trainData['title'],Y_train,typeModel)
        X_covar_bow['bowTitle'+time]=yhat_bowTitle
        X_covar_vocab['bowTitle'+time]=['title',title_vectorizer,title_model]
    if 'short' in variables:
        yhat_bowShortDescription, short_vectorizer, short_model = bowFitAndPrediction(predictData['short_description'], trainData['short_description'],Y_train,typeModel)
        X_covar_bow['bowShort'+time]=yhat_bowShortDescription
        X_covar_vocab['bowShort'+time]=['short_description',short_vectorizer,short_model]
    if 'essay' in variables:
        yhat_bowEssay, essay_vectorizer, essay_model = bowFitAndPrediction(predictData['essay'], trainData['essay'],Y_train,typeModel)
        X_covar_bow['bowEssay'+time]=yhat_bowEssay
        X_covar_vocab['bowEssay'+time]=['essay',essay_vectorizer,essay_model]
    if 'need' in variables:
        yhat_bowNeedStatement, need_vectorizer, need_model = bowFitAndPrediction(predictData['need_statement'], trainData['need_statement'],Y_train,typeModel)
        X_covar_bow['bowNeed'+time]=yhat_bowNeedStatement
        X_covar_vocab['bowNeed'+time]=['need_statement',need_vectorizer,need_model]
        
    return (X_covar_bow,X_covar_vocab)

def runBagOutcome(trainData,outcome,typeModel='binary'):
    #Run bag of words on title, short description, essay, and need statement. Should edit to eventually allow different parameters per model
    
    X_covar_bow3 , X_covar_vocab3 = bagofwords(trainData,outcome,['title','short','essay','need'],'3_'+outcome,'three_months',typeModel)
    X_train_bow3 = pd.DataFrame(X_covar_bow3)
    
    X_covar_bow6 , X_covar_vocab6 = bagofwords(trainData,outcome,['title','short','essay','need'],'6_'+outcome,'six_months',typeModel)
    X_train_bow6 = pd.DataFrame(X_covar_bow6)
    
    X_covar_bow12 , X_covar_vocab12 = bagofwords(trainData,outcome,['title','short','essay','need'],'12_'+outcome,'one_year',typeModel)
    X_train_bow12 = pd.DataFrame(X_covar_bow12)
    
    #X_covar_bow24 , X_covar_vocab24 = bagofwords(trainData,'is_exciting',['title','short','essay','need'],'24','two_year')
    #X_train_bow24 = pd.DataFrame(X_covar_bow24)
        
    #combine all of the bow
    data = [X_train_bow3,X_train_bow6, X_train_bow12]# , X_train_bow24]
    X_covar_vocab = dict(X_covar_vocab3.items()+X_covar_vocab6.items()+X_covar_vocab12.items())#+X_covar_vocab24.items())
   
    X_train_bow = pd.concat(data, axis=1)
    
    return (X_train_bow,X_covar_vocab)

def runBagOfWords(trainData):

    #Additional possible outcomes:
    #Binary:    
    
    #Continuous
    #great_messages_proportion
    #teacher_referred_count
    #non_teacher_referred_count

    #Donation outcomes
    print "Bag of words for total donation to project"
    X_train_bow_project_donation, X_covar_vocab_total_donation = runBagOutcome(trainData,'donation_to_project','continuous')
    print "Bag of words for optional donation to project"
    X_train_bow_optional_donation, X_covar_vocab_optional_donation = runBagOutcome(trainData,'donation_optional_support','continuous')
    print "Bag of words for number of donations"
    X_train_bow_num_donation, X_covar_vocab_num_donation = runBagOutcome(trainData,'donation_count','continuous')
    
    data1 = [X_train_bow_project_donation,X_train_bow_optional_donation,X_train_bow_num_donation]
    X_covar_vocab1 = dict(X_covar_vocab_total_donation.items()+X_covar_vocab_optional_donation.items()+X_covar_vocab_num_donation.items())
    
    print "Bag of words for teacher referred"
    X_train_bow_refer, X_covar_vocab_refer = runBagOutcome(trainData,'at_least_1_teacher_referred_donor','binary')
    print "Bag of words for green donation"
    X_train_bow_green, X_covar_vocab_green = runBagOutcome(trainData,'at_least_1_green_donation','binary')
    print "Bag of words for exciting"
    X_train_bow_excite, X_covar_vocab_excite = runBagOutcome(trainData,'is_exciting','binary')
    print "Bag of words for fully funded"
    X_train_bow_fund, X_covar_vocab_fund = runBagOutcome(trainData,'fully_funded','binary')
    
    data2 = [X_train_bow_refer,X_train_bow_green ,X_train_bow_excite,X_train_bow_fund]
    X_covar_vocab2 = dict(X_covar_vocab_refer.items()+X_covar_vocab_green.items()+X_covar_vocab_excite.items()+X_covar_vocab_fund.items())
    
    print "Bag of words for great chat"
    X_train_bow_great_chat, X_covar_vocab_great_chat = runBagOutcome(trainData,'great_chat','binary')
    print "Bag of words for three or more non teacher referred donors"
    X_train_bow_non_teacher_refer, X_covar_vocab_non_teacher_refer = runBagOutcome(trainData,'three_or_more_non_teacher_referred_donors','binary')
    print "Bag of words for one non teacher referred donor giving 100 plus"
    X_train_bow_non_teacher_large, X_covar_vocab_non_teacher_large = runBagOutcome(trainData,'one_non_teacher_referred_donor_giving_100_plus','binary')
    print "Bag of words for donation from thoughtful donor"
    X_train_bow_thoughtful_donor, X_train_vocab_thoughtful_donor = runBagOutcome(trainData,'donation_from_thoughtful_donor','binary')
    
    data3 = [X_train_bow_great_chat,X_train_bow_non_teacher_refer ,X_train_bow_non_teacher_large, X_train_bow_thoughtful_donor]
    X_covar_vocab3 = dict(X_covar_vocab_great_chat.items()+X_covar_vocab_non_teacher_refer.items()+X_covar_vocab_non_teacher_large.items()+X_train_vocab_thoughtful_donor.items())

    #data = [X_train_bow_fund,X_train_bow_excite,X_train_bow_refer,X_train_bow_green]
    data = data1+data2+data3
    X_train_bow = pd.concat(data, axis=1)
    #X_covar_vocab = dict(X_covar_vocab_fund.items()+X_covar_vocab_excite.items()+X_covar_vocab_refer.items()+X_covar_vocab_green.items())
    X_covar_vocab = dict(X_covar_vocab1.items() + X_covar_vocab2.items() + X_covar_vocab3.items())
    
    return (X_train_bow,X_covar_vocab)

def generatePrediction(clean_data_path,subPath, Y_eval, name):
    print "Generating prediction"
    print "Reading evaluation data"
    evalData = pd.read_csv(clean_data_path+'evaluation data.csv')
    
    #Need to figure this part out
    print "Saving submission for Kaggle..."
    submission = pd.DataFrame(data= {'projectid':evalData['projectid'].as_matrix(), 'is_exciting': Y_eval})
    submission.to_csv(subPath + name+'.csv', index=False, cols=['projectid','is_exciting'])
    
def testSampleEvaluation(clean_data_path, vocabDict, X_covar, model, standardize, data='cross validation data'):
    
    evalData = pd.read_csv(clean_data_path+data+'.csv')
    if len(vocabDict)!=0:
        #Create text variables
        covar = {}
        for var in vocabDict:
            text = evalData[vocabDict[var][0]]
            textDict = vocabDict[var][1]
            bowModel = vocabDict[var][2]
            X_train = textDict.transform(text)
            if isinstance(bowModel,LogisticRegression):
                predict = bowModel.predict_proba(X_train)
                yhat = predict[:,1]
            else:
                predict = bowModel.predict(X_train)
                yhat = predict
            covar[var]=yhat
                    
        X_train_text = pd.DataFrame(covar)
        final = pd.merge(evalData,X_train_text, left_index=True,right_index=True)
    else:
        final = evalData
    
    #should add in standardizing here
    
    X_train_ordered = final[X_covar]
    
    X_train_ordered = standardize.transform(X_train_ordered)
    
    eval_predict = model.predict_proba(X_train_ordered)
    evalPredict_prob = eval_predict[:,1]
    
    Y_train = evalData['is_exciting']
    
    inScore = roc_auc_score(Y_train,evalPredict_prob)
    return inScore

def getTextAttr():
    #Other essay characteristics
    essay_attr = ['numWordsEssay','lexicalDiversityEssay'] #  ['numCharactersEssay','numWordsEssay','numUniqueWordsEssay','lexicalDiversityEssay']
    short_attr = ['numWordsShortDescription','lexicalDiversityShortDescription'] # ['numCharacterShortDescription','numWordsShortDescription','numUniqueWordsShortDescription','lexicalDiversityShortDescription']
    need_attr = ['numWordsNeedStatement'] # ['numCharacterNeedStatement','numWordsNeedStatement','numUniqueWordsNeedStatement']
    title_attr = ['numWordsTitle','titleHasExclamation'] # ['numCharacterTitle','numWordsTitle','numUniqueWordsTitle','titleHasExclamation']
    X_text_attr = essay_attr + short_attr + need_attr + title_attr
    return X_text_attr

def getProjectAttr():
    
    #Geography
    
    #rural or suburban
    school_binary = ['school_charter','school_magnet','school_year_round','school_nlns','school_kipp','school_charter_ready_promise']
    
    #NEED TO ADD TEACHER GENDER
    teacher_attr = ['teacher_teach_for_america','teacher_ny_teaching_fellow']
    
    #Primary focus subject
    
    #Secondary focus subject
    
    #Resource type
    
    #This has missing values, have to figure out how to use
    #students_attr = ['students_reached'] #GRADE LEVEL, 
    
    match_attr = ['eligible_double_your_impact_match','eligible_almost_home_match']
    
    request_attr = ['total_price_excluding_optional_support']#,'optional_support'] #ADD optional support
    
    X_proj_attr = school_binary + teacher_attr + match_attr + request_attr
    
    return X_proj_attr

def generateProjectDummies(trainData):
    
    
    #Geography, should think about how to do this
    #could potentially also bring in census data with lat/long 
    #school metro to dummies
    metro_dummies = pd.get_dummies(trainData['school_metro'], dummy_na=True)
    
    #zipcode dummies for geography
    zipcode_dummies = pd.get_dummies(trainData['school_zip'], dummy_na=True)
    
    #Could convert to male/female
    teacher_prefix_dummies = pd.get_dummies(trainData['teacher_prefix'], dummy_na=True)
    
    data=[trainData, metro_dummies,zipcode_dummies,teacher_prefix_dummies]
    trainData = pd.concat(data,axis=1)
    
    #Generate optional support
    trainData['optional_support'] = trainData['total_price_including_optional_support']-trainData['total_price_excluding_optional_support']
    trainData['optional_support_pct_price']=100*trainData['optional_support']/trainData['total_price_excluding_optional_support']
    
    return trainData

def generatePastSchool(trainData):
    
    #Group by school
    
    #Sort by date
    
    #Sequence
    
    return trainData

def getTimeVariables(xData, trainData):
    
    #trainData['week']=trainData['date_posted'].week
    train_week = pd.get_dummies(pd.to_datetime(trainData['date_posted'].values).week,'week','_')
    
    
    #week = pd.get_dummies(trainData['week'],dummy_na=True)
    #month = pd.get_dummies(trainData['month'], dummy_na=True)
    
    return train_week

def savePredictionXvalues(clean_data_path, vocabDict, X_covar, standardize, data='cross validation data'):

    evalData = pd.read_csv(clean_data_path+data+'.csv')
    if len(vocabDict)!=0:
        #Create text variables
        covar = {}
        for var in vocabDict:
            text = evalData[vocabDict[var][0]]
            textDict = vocabDict[var][1]
            bowModel = vocabDict[var][2]
            X_train = textDict.transform(text)
            if isinstance(bowModel,LogisticRegression):
                predict = bowModel.predict_proba(X_train)
                yhat = predict[:,1]
            else:
                predict = bowModel.predict(X_train)
                yhat = predict
            covar[var]=yhat
                    
        X_train_text = pd.DataFrame(covar)
        final = pd.merge(evalData,X_train_text, left_index=True,right_index=True)
    else:
        final = evalData
    
    #should add in standardizing here
    
    X_train_ordered = final[X_covar]
    
    X_train_ordered = standardize.transform(X_train_ordered)
    
    X_train_ordered=pd.DataFrame(X_train_ordered, columns=X_covar)
    
    X_train_ordered.to_csv(clean_data_path + 'X values/' + data +' X values.csv')

def fillMissing(trainData):
    
    #Should do this more elegantly
    trainData['at_least_1_teacher_referred_donor'].fillna(0,inplace=True)
    trainData['at_least_1_green_donation'].fillna(0,inplace=True)
    trainData['fully_funded'].fillna(0,inplace=True)
    trainData['is_exciting'].fillna(0,inplace=True)
    
    #Binary outcomes
    trainData['great_chat'].fillna(0,inplace=True)
    trainData['three_or_more_non_teacher_referred_donors'].fillna(0,inplace=True)
    trainData['one_non_teacher_referred_donor_giving_100_plus'].fillna(0,inplace=True)
    trainData['donation_from_thoughtful_donor'].fillna(0,inplace=True)
    
    #Continuous
    #Donation dollar amount
    #Number of donations
    #Continuous outcomes
    trainData['great_messages_proportion'].fillna(0,inplace=True)
    trainData['teacher_referred_count'].fillna(0,inplace=True)
    trainData['non_teacher_referred_count'].fillna(0,inplace=True)
    
    trainData['donation_to_project'].fillna(0,inplace=True)
    trainData['donation_optional_support'].fillna(0,inplace=True)
    trainData['donation_total'].fillna(0,inplace=True)
    trainData['donation_count'].fillna(0,inplace=True)
    
    return trainData

def getOutcome(clean_data_path,dataset,outcome_var):
    data = pd.read_csv(clean_data_path+ dataset +'.csv')
    return data[outcome_var]

def getXvalues(clean_data_path,dataset):
    data = pd.read_csv(clean_data_path + 'X values/essay '+ dataset +' X values.csv',index_col=0)
    return data   

def cvScore(x, *params):
    penalty = x
    X_train,Y_train,X_cross,Y_cross,model=params
    
    decomp = decomposition.PCA(17)
    X_train_decom = decomp.fit_transform(X_train)
    X_cross_decom = decomp.transform(X_cross)
    
    penalty_param = 10**(-1*penalty)
    
    model.set_params(C=penalty_param)

    model.fit(X_train_decom,Y_train)
    predict = model.predict_proba(X_cross_decom)
    predict = predict[:,1]
    score = -1*roc_auc_score(Y_cross,predict)
    print "Penalty parameter %f" % (penalty_param)
    print "AUC score %f" % (score)
    return score
            

def essayModel(purpose='test'):
    print "Prediction using essay data"
    clean_data_path = 'C:/Users/P_Kravik/Desktop/Kaggle/KDD Cup 2014 - Predicting Excitement at DonorsChoose.org/Data/Clean/Derived/'
    subPath = 'C:/Users/P_Kravik/Desktop/Kaggle/KDD Cup 2014 - Predicting Excitement at DonorsChoose.org/Submissions/' 
    
    print "Running model on training time data..."
    trainData = pd.read_csv(clean_data_path + 'train.csv')
    trainData = fillMissing(trainData)
    
    
    trainData['date_posted'] = pd.to_datetime(trainData['date_posted'])
    trainData.sort(['teacher_acctid_x','date_posted'], inplace=True)
    
    if purpose == 'test':
        #Should create a function to do this
        trainData['date_posted'] = pd.to_datetime(trainData['date_posted'])
        trainData = trainData[trainData['date_posted'] < pd.datetime(2013,7,1)]
        trainData['date_posted'] = pd.to_datetime(trainData['date_posted'])
        trainData['three_months'] = trainData['date_posted']>pd.datetime(2013,4,1)
        trainData['six_months'] = (trainData['date_posted']>pd.datetime(2013,1,1)) & (trainData['date_posted']<=pd.datetime(2013,4,1)) 
        trainData['one_year'] = (trainData['date_posted']>pd.datetime(2012,7,1)) & (trainData['date_posted']<=pd.datetime(2013,1,1)) 
        trainData['two_year'] = (trainData['date_posted']>pd.datetime(2011,7,1)) & (trainData['date_posted']<=pd.datetime(2012,7,1)) 
        trainData = trainData[trainData['date_posted']>=pd.datetime(2011,7,1)]
        trainData.reset_index(drop=True,inplace=True)
    else:
        #Should create a function to do this
        trainData['date_posted'] = pd.to_datetime(trainData['date_posted'])
        trainData = trainData[trainData['date_posted'] < pd.datetime(2014,1,1)]
        trainData['three_months'] = trainData['date_posted']>pd.datetime(2013,10,1)
        trainData['six_months'] = (trainData['date_posted']>pd.datetime(2013,6,1)) & (trainData['date_posted']<=pd.datetime(2013,10,1)) 
        trainData['one_year'] = (trainData['date_posted']>pd.datetime(2013,1,1)) & (trainData['date_posted']<=pd.datetime(2013,6,1)) 
        trainData['two_year'] = (trainData['date_posted']>=pd.datetime(2012,1,1)) & (trainData['date_posted']<=pd.datetime(2013,1,1)) 
        trainData = trainData[trainData['date_posted']>=pd.datetime(2012,1,1)]
        trainData.reset_index(drop=True,inplace=True)
          
    #Run bag of words analysis on each text field individually, get a prediction, which will be covar in final regression
    X_train_bow , X_covar_vocab = runBagOfWords(trainData) 
    
    #save the bag of words data
    
    #Other text attributes
    X_text_attr = getTextAttr()
    X_train_text_attr = trainData[X_text_attr]
    
    
    
    if len(X_train_bow)==0:
        X_train = X_train_text_attr
    else:
        X_train = pd.concat([X_train_bow, X_train_text_attr],axis=1)
    
    #X_project_attr = getProjectAttr()
    #X_train_proj_attr = trainData[X_project_attr]
    
    #X_train = pd.concat([X_train_bow, X_train_text_attr,X_train_proj_attr],axis=1)
    
    #Should run a scaling/normalizing step here
    X_covar_names = X_train.columns.values.tolist()
    
    standardize = preprocessing.StandardScaler()
    X_train = standardize.fit_transform(X_train)
    
    X_train_df = pd.DataFrame(X_train, columns = X_covar_names)
    
    if purpose == 'test':
        X_train_df.to_csv(clean_data_path+ 'X values/' + 'essay training X values.csv')
        savePredictionXvalues(clean_data_path, X_covar_vocab, X_covar_names, standardize, 'cross validation data')
        savePredictionXvalues(clean_data_path, X_covar_vocab, X_covar_names, standardize, 'test data')
        savePredictionXvalues(clean_data_path, X_covar_vocab, X_covar_names, standardize, 'evaluation data')
    else:
        X_train_df.to_csv(clean_data_path+ 'X values/' + 'essay prediction X values.csv')
        savePredictionXvalues(clean_data_path, X_covar_vocab, X_covar_names, standardize, 'essay final evaluation data')

    X_train = getXvalues(clean_data_path,'training')
    Y_train = trainData['is_exciting']
    X_cross = getXvalues(clean_data_path,'cross validation data')
    X_test = getXvalues(clean_data_path,'test data')
    Y_cross = getOutcome(clean_data_path,'cross validation data','is_exciting')
    Y_test = getOutcome(clean_data_path,'test data','is_exciting')
    X_eval = getXvalues(clean_data_path,'evaluation data')
    
    
    
    
    #Logit model part
    logit = LogisticRegression(penalty='l2',dual=False,tol=1,fit_intercept=True, C=.000000001, intercept_scaling=1, class_weight='auto', random_state=423)
    print "Final prediction"
    
    #Try to implement grid search
    rranges = slice(1, 10, 1)
    params = (X_train,Y_train,X_cross,Y_cross,logit)
    resbrute = optimize.brute(cvScore, (rranges,), args=params, full_output=True, finish=optimize.fmin)
    
    decomp = decomposition.PCA(17)
    X_train = decomp.fit_transform(X_train)
    X_cross = decomp.transform(X_cross)
    X_test = decomp.transform(X_test)
    X_eval = decomp.transform(X_eval)
    
    #This is apparantly bad
#     standardize = preprocessing.StandardScaler()
#     X_train = standardize.fit_transform(X_train)
#     X_cross = standardize.transform(X_cross)
#     X_test = standardize.transform(X_test)
#     X_eval = standardize.transform(X_eval)
        
    logit = LogisticRegression(penalty='l2',dual=False,tol=1,fit_intercept=True, C=6.5632*10**-7, intercept_scaling=1, class_weight='auto', random_state=423)
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
    
    temp = logit.predict_proba(X_eval)
    Y_predict_logit = temp[:,1]
    generatePrediction(clean_data_path, subPath,Y_predict_logit,'Essay logit model')
    
    #Random Forest
    
    #Should cross validate this value
    rf_weight = Y_train*4+1
    
    clf = ensemble.RandomForestClassifier(n_estimators=100,max_depth=10, max_features=25,oob_score=True,n_jobs=2)
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
    
    temp = clf.predict_proba(X_eval)
    Y_predict_rf = temp[:,1]
    generatePrediction(clean_data_path, subPath,Y_predict_rf,'Essay random forest model')
    
    #Gradient Boosting
    gbm = ensemble.GradientBoostingClassifier(random_state=43,verbose = 1)
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
    generatePrediction(clean_data_path, subPath, Y_predict_gbm,'Essay GBM model')
    
    #Adaboost stumps
    
    ada_weight = Y_train*1+1
    
    ada = ensemble.AdaBoostClassifier(n_estimators=5,learning_rate = .1,random_state = 42)
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
    
    temp = ada.predict_proba(X_eval)
    Y_predict_ada = temp[:,1]
    generatePrediction(clean_data_path, subPath, Y_predict_ada,'Essay AdaBoost model')
    
    Model_train_xvalues = np.column_stack((predict_logit, predict_rf))
    Model_cross_xvalues = np.column_stack((cross_predict_logit, cross_predict_rf))
    Model_test_xvalues = np.column_stack((test_predict_logit, test_predict_rf))
    Model_eval_xvalues = np.column_stack((Y_predict_logit,Y_predict_rf))
    
    Model_train_xvalues = pd.DataFrame(Model_train_xvalues, columns=['logit','forest'])
    Model_cross_xvalues = pd.DataFrame(Model_cross_xvalues, columns=['logit','forest'])
    Model_test_xvalues = pd.DataFrame(Model_test_xvalues, columns=['logit','forest'])
    Model_eval_xvalues = pd.DataFrame(Model_eval_xvalues, columns=['logit','forest'])
    
    final_standardize = preprocessing.StandardScaler()
    Model_train_xvalues=final_standardize.fit_transform(Model_train_xvalues)
    Model_cross_xvalues=final_standardize.transform(Model_cross_xvalues)
    Model_test_xvalues=final_standardize.transform(Model_test_xvalues)
    Model_eval_xvalues = final_standardize.transform(Model_eval_xvalues)
    
    finaldecomp = decomposition.PCA()
    Model_train_xvalues = finaldecomp.fit_transform(Model_train_xvalues)
    Model_cross_xvalues = finaldecomp.transform(Model_cross_xvalues)
    Model_test_xvalues = finaldecomp.transform(Model_test_xvalues)
    Model_eval_xvalues = finaldecomp.transform(Model_eval_xvalues)
    #Final combine
    
    logitWrapper = LogisticRegression(penalty='l2',dual=False,tol=1,fit_intercept=True, C=.0000001, intercept_scaling=1, class_weight=None, random_state=423)
    print "Final prediction"
    logitWrapper.fit(Model_train_xvalues,Y_train)
         
    final_predict = logitWrapper.predict_proba(Model_train_xvalues)
    final_predict_logit = final_predict[:,1]
    
    inScore = roc_auc_score(Y_train,final_predict_logit)
    print "Train Logistic: Area under auc curve is %f" % (inScore)
    
    final_cross_predict = logitWrapper.predict_proba(Model_cross_xvalues)
    final_cross_predict_logit = final_cross_predict[:,1]
    
    inScore = roc_auc_score(Y_cross,final_cross_predict_logit)
    print "Cross Logistic: Area under auc curve is %f" % (inScore)
    
    final_test_predict = logitWrapper.predict_proba(Model_test_xvalues)
    final_test_predict_logit = final_test_predict[:,1]
    
    inScore = roc_auc_score(Y_test,final_test_predict_logit)
    print "Test logistic: Area under auc curve is %f" % (inScore)
    
    temp = logitWrapper.predict_proba(Model_eval_xvalues)
    Y_predict = temp[:,1]
    generatePrediction(clean_data_path, subPath, Y_predict,'Essay final model')
    
    
#     if purpose == 'test':
#         
#         inScore = testSampleEvaluation(clean_data_path, X_covar_vocab, X_covar_names, logit, standardize, 'cross validation data')
#         print "Cross: Area under auc curve is %f" % (inScore)
#         
#         inScore = testSampleEvaluation(clean_data_path, X_covar_vocab, X_covar_names, logit, standardize, 'test data')
#         print "Test: Area under auc curve is %f" % (inScore)
#     else:
#         subPath = 'C:/Users/P_Kravik/Desktop/Kaggle/KDD Cup 2014 - Predicting Excitement at DonorsChoose.org/Submissions/' 
#         generatePrediction(clean_data_path, subPath, X_covar_vocab,X_covar_names,logit,standardize,'Bad data lots of bags')
#     
    print "Complete!"
    
    
if __name__ == '__main__':
    essayModel('predict')