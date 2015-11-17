'''
Created on Jun 19, 2014

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
from sklearn.datasets import dump_svmlight_file

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
    
    trainData['students_reached_na'] = trainData['students_reached'].isnull()
    trainData['students_reached_na']= trainData['students_reached_na'].apply(lambda x: 1 if x else 0)
    
    return trainData

def fixStudentsReached(dataset):
    
    dataset['students_reached_na'] = dataset['students_reached'].isnull()
    dataset['students_reached_na']= dataset['students_reached_na'].apply(lambda x: 1 if x else 0)
    dataset['students_reached'].fillna(0,inplace=True)
    
    return dataset

def getNumberPastSubmission(allData):
    allData['counter']=1
    
    #allData['fully_funded'][allData['date_posted']>=pd.datetime(2013,10,1)] = 0
    
    cumNoSubmissionTeacher = allData.groupby('teacher_acctid_x')['counter'].cumsum()
    cumNoFundedSubmissionTeacher = allData.groupby('teacher_acctid_x')['fully_funded'].cumsum()
    cumNoExcitingSubmissionTeacher = allData.groupby('teacher_acctid_x')['is_exciting'].cumsum()
    
    cumNoSubmissionSchool = allData.groupby('schoolid')['counter'].cumsum()
    cumNoFundedSubmissionSchool = allData.groupby('schoolid')['fully_funded'].cumsum()
    cumNoExcitingSubmissionSchool = allData.groupby('schoolid')['is_exciting'].cumsum()
    
    #Fixing expost issue
    cumNoFundedSubmissionTeacher[cumNoFundedSubmissionTeacher>0] = cumNoFundedSubmissionTeacher[cumNoFundedSubmissionTeacher>0]-1
    cumNoExcitingSubmissionTeacher[cumNoExcitingSubmissionTeacher>0] = cumNoExcitingSubmissionTeacher[cumNoExcitingSubmissionTeacher>0]-1  
    cumNoFundedSubmissionSchool[cumNoFundedSubmissionSchool>0] = cumNoFundedSubmissionSchool[cumNoFundedSubmissionSchool>0]-1
    cumNoExcitingSubmissionSchool[cumNoExcitingSubmissionSchool>0] = cumNoExcitingSubmissionSchool[cumNoExcitingSubmissionSchool>0]-1
    
    #going to cap teacher at 25, 50, 100
    cumNoSubmissionTeacher[(cumNoSubmissionTeacher >= 25) & (cumNoSubmissionTeacher<50)] = 25
    cumNoSubmissionTeacher[(cumNoSubmissionTeacher >= 50) & (cumNoSubmissionTeacher<100)] = 50
    cumNoSubmissionTeacher[cumNoSubmissionTeacher >= 100] = 100
    
    cumNoFundedSubmissionTeacher[(cumNoFundedSubmissionTeacher >= 25) & (cumNoFundedSubmissionTeacher<50)] = 25
    cumNoFundedSubmissionTeacher[(cumNoFundedSubmissionTeacher >= 50) & (cumNoFundedSubmissionTeacher<100)] = 50
    cumNoFundedSubmissionTeacher[cumNoFundedSubmissionTeacher >= 100] = 100
    
    cumNoExcitingSubmissionTeacher[(cumNoExcitingSubmissionTeacher >= 25) & (cumNoExcitingSubmissionTeacher<50)] = 25
    cumNoExcitingSubmissionTeacher[(cumNoExcitingSubmissionTeacher >= 50) & (cumNoExcitingSubmissionTeacher<100)] = 50
    cumNoExcitingSubmissionTeacher[cumNoExcitingSubmissionTeacher >= 100] = 100
    
    #going to cap school at 25, 50, 100
    cumNoSubmissionSchool[(cumNoSubmissionSchool >= 25) & (cumNoSubmissionSchool<50)] = 25
    cumNoSubmissionSchool[(cumNoSubmissionSchool >= 50) & (cumNoSubmissionSchool<100)] = 50
    cumNoSubmissionSchool[(cumNoSubmissionSchool >= 100) & (cumNoSubmissionSchool<250)] = 100
    cumNoSubmissionSchool[(cumNoSubmissionSchool >= 250) & (cumNoSubmissionSchool<500)] = 250
    cumNoSubmissionSchool[cumNoSubmissionSchool >= 500] = 500
    
    cumNoFundedSubmissionSchool[(cumNoFundedSubmissionSchool >= 25) & (cumNoFundedSubmissionSchool<50)] = 25
    cumNoFundedSubmissionSchool[(cumNoFundedSubmissionSchool >= 50) & (cumNoFundedSubmissionSchool<100)] = 50
    cumNoFundedSubmissionSchool[(cumNoFundedSubmissionSchool >= 100) & (cumNoFundedSubmissionSchool<250)] = 100
    cumNoFundedSubmissionSchool[(cumNoFundedSubmissionSchool >= 250) & (cumNoFundedSubmissionSchool<500)] = 250
    cumNoFundedSubmissionSchool[cumNoFundedSubmissionSchool >= 500] = 500
    
    cumNoExcitingSubmissionSchool[(cumNoExcitingSubmissionSchool >= 10) & (cumNoExcitingSubmissionSchool<15)] = 10
    cumNoExcitingSubmissionSchool[(cumNoExcitingSubmissionSchool >= 15) & (cumNoExcitingSubmissionSchool<25)] = 25
    cumNoExcitingSubmissionSchool[(cumNoExcitingSubmissionSchool >= 25) & (cumNoExcitingSubmissionSchool<50)] = 25
    cumNoExcitingSubmissionSchool[cumNoExcitingSubmissionSchool >= 50] = 50
    
    #Convert to dummies
    #Should consider adding other outcomes, could do pct, like great chat
    cumNoSubmissionTeacherDummy = pd.get_dummies(cumNoSubmissionTeacher, 'cumNoTeacher','_', dummy_na=True)
    cumNoFundedSubmissionTeacherDummy = pd.get_dummies(cumNoFundedSubmissionTeacher,  'cumNoFundedTeacher','_', dummy_na=True)
    cumNoExcitingSubmissionTeacherDummy = pd.get_dummies(cumNoExcitingSubmissionTeacher,  'cumNoExcitingTeacher','_', dummy_na=True)
    
    
    cumNoSubmissionSchoolDummy = pd.get_dummies(cumNoSubmissionSchool, 'cumNoSchool','_', dummy_na=True)
    cumNoFundedSubmissionSchoolDummy = pd.get_dummies(cumNoFundedSubmissionSchool, 'cumNoFundedSchool','_', dummy_na=True)
    cumNoExcitingSubmissionSchoolDummy = pd.get_dummies(cumNoExcitingSubmissionSchool, 'cumNoExcitingSchool','_', dummy_na=True)
    
    first_submission_teacher = (cumNoSubmissionTeacher == 1).apply(lambda x: 1 if x else 0)
    first_submission_school = (cumNoSubmissionSchool == 1).apply(lambda x: 1 if x else 0)
    
    previousFundedTeacher = (cumNoFundedSubmissionTeacher >=1).apply(lambda x: 1 if x else 0)
    
    cumNoSubmissionTeacherSq = cumNoSubmissionTeacher**2
    
    previousFunded = noExpostData(allData)
    noPreviousFunded = previousFunded.apply(lambda x: 1 if x == 0 else 0)
    
    data = [cumNoSubmissionTeacher]#, cumNoFundedSubmissionTeacherDummy]#pd.DataFrame(previousFundedTeacher,columns=['previous_funded'])]
    pastSubmissionVariables = pd.concat(data, axis=1)
    #cumNoSubmission = allData.groupby('teacher_acctid_x').cumcount(ascending=False)
    #allData['first_submission'] = (cumNoSubmission == 0).apply(lambda x: 1 if x else 0)
    #allData['second_submission'] = (cumNoSubmission == 1).apply(lambda x: 1 if x else 0)
    #allData['third_submission'] = (cumNoSubmission == 1).apply(lambda x: 1 if x else 0)
    return pastSubmissionVariables

def getProjectAttr(dataset):
        
    #Geography
    
    #rural or suburban
    school_binary = ['school_charter','school_magnet','school_year_round','school_nlns','school_kipp','school_charter_ready_promise']
    
    #NEED TO ADD TEACHER GENDER
    teacher_attr = ['teacher_teach_for_america','teacher_ny_teaching_fellow','teacher_acctid_x']
    
    #Primary focus subject
    
    #Secondary focus subject
    
    #Resource type
    
    #This has missing values, have to figure out how to use. na is missing and made in other function
    students_attr = ['students_reached','students_reached_na'] #GRADE LEVEL, 
    
    match_attr = ['eligible_double_your_impact_match','eligible_almost_home_match']
    
    request_attr = ['total_price_excluding_optional_support']#,'optional_support'] #ADD optional support
    
    X_proj_attr = school_binary + teacher_attr + match_attr + request_attr + students_attr
    
    return dataset[X_proj_attr]

def noExpostData(allData):
    
    teacher = allData['teacher_acctid_x'].values
    date = allData['date_posted']
    outcome = allData['fully_funded'].values
    cum = [0] * len(allData)
    
    for i in range(1,len(allData)-1):
        if outcome[i-1]==1:
            if teacher[i] != teacher[i-1]:
                cum[i] = 0
            else:
                cum[i] = cum[i-1]+1
        else:
            cum[i]=cum[i-1]
        
    cumNoPreviousFunded = pd.Series(cum)
    
    return cumNoPreviousFunded

def fixedEffectVariables(allData,group,outcome_var):
    
    date = allData['date_posted']
    teacher = allData[group]
    outcome = allData[outcome_var]
    date_offset = pd.DateOffset(months=6, days=0)
    newVariable = [0]*len(allData)
    
    for i in range(0, len(allData)-1):
        newVariable[i] = outcome[(date<(date[i] - date_offset)) & (teacher == teacher[i])].mean()

    return newVariable

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


def getProjectVariables(allData):

    
    dataset = allData
          
    #week
    month = pd.get_dummies(pd.to_datetime(dataset['date_posted'].values).month,'month','_')
    year = pd.get_dummies(pd.to_datetime(dataset['date_posted'].values).year, 'year','_')
    
    #school metro to dummies (urban,suburban,rural
    metro_dummies = pd.get_dummies(dataset['school_metro'],'metro','_', dummy_na=True)
    
    #state dummies for geography. Could go to a different level (state, lat/long, etc), school district, etc.
    #Should look into zipcode dummies, but I was getting a memory error?
    #can do one hot encoding for zip, 12500 or so zipcodes
    state_dummies = pd.get_dummies(dataset['school_state'],'state','_', dummy_na=True)
    
    #Teacher prefix Mr,Mrs,Ms
    teacher_prefix_dummies = pd.get_dummies(dataset['teacher_prefix'],'teacherprefix','_', dummy_na=True)
    
    #Primary and secondary focus e.g. Mathematics, health, sports
    primary_focus_dummies = pd.get_dummies(dataset['primary_focus_subject'],'primaryfocus','_', dummy_na=True)
    secondary_focus_dummies = pd.get_dummies(dataset['secondary_focus_subject'],'secondaryfocus','_', dummy_na=True)
    
    #Resource type e.g. books, supplies, technology
    resource_type_dummies = pd.get_dummies(dataset['resource_type'],'resource','_',dummy_na=True)
    
    #poverty level e.g.high, highest, moderate
    poverty_level_dummy = pd.get_dummies(dataset['poverty_level'],'poverty','_',dummy_na=True)
    
    #Grade level e.g. 3-5,9-12
    grade_level_dummy = pd.get_dummies(dataset['grade_level'],'grade','_',dummy_na=True)
    
    dataset=fixStudentsReached(dataset)
    
    #Existing variables that are ready to use
    existVar = getProjectAttr(dataset)
    
    #Past Submission information at teacher and school level
    pastSubmissionVariables = getNumberPastSubmission(dataset)
    
    combine = [existVar, month, year, metro_dummies, state_dummies,teacher_prefix_dummies,primary_focus_dummies,secondary_focus_dummies,resource_type_dummies,poverty_level_dummy,pastSubmissionVariables]
    dataset = pd.concat(combine, axis=1)
    dataset['date_posted'] = allData['date_posted']
    return dataset

def createSamples(data):
    evalData = data[data['date_posted'] >= pd.datetime(2014,1,1)]
    insampledata = data[data['date_posted'] < pd.datetime(2014,1,1)]
    test_data = insampledata[insampledata['date_posted']>=pd.datetime(2013,10,1)]
    cv_data = insampledata[(insampledata['date_posted']>=pd.datetime(2013,7,1)) & (insampledata['date_posted']<pd.datetime(2013,10,1))]
    train_data = insampledata[(insampledata['date_posted']>=pd.datetime(2011,7,1)) & (insampledata['date_posted']<pd.datetime(2013,7,1))]
    prediction_train_data = insampledata[(insampledata['date_posted']<pd.datetime(2014,1,1)) & (insampledata['date_posted']>=pd.datetime(2012,1,1))]
    
    train_data = train_data.drop('date_posted',axis=1)
    train_data.reset_index(drop=True,inplace=True)
    cv_data = cv_data.drop('date_posted',axis=1)
    cv_data.reset_index(drop=True,inplace=True)
    test_data = test_data.drop('date_posted',axis=1)
    test_data.reset_index(drop=True,inplace=True)
    evalData = evalData.drop('date_posted',axis=1)
    evalData.reset_index(drop=True,inplace=True)
    prediction_train_data = prediction_train_data.drop('date_posted',axis=1)
    prediction_train_data.reset_index(drop=True,inplace=True)
    
    return train_data, cv_data, test_data, evalData, prediction_train_data

def getOutcome(clean_data_path,dataset,outcome_var):
    data = pd.read_csv(clean_data_path+ dataset +'.csv')
    return data[outcome_var]

def generatePrediction(clean_data_path,subPath, Y_eval, name):
    print "Generating prediction"
    print "Reading evaluation data"
    evalData = pd.read_csv(clean_data_path+'evaluation data.csv')
    
    #Need to figure this part out
    print "Saving submission for Kaggle..."
    submission = pd.DataFrame(data= {'projectid':evalData['projectid'].as_matrix(), 'is_exciting': Y_eval})
    submission.to_csv(subPath + name+'.csv', index=False, cols=['projectid','is_exciting'])

def cvScore(x, *params):
    penalty = x
    X_train,Y_train,X_cross,Y_cross,model=params
    
    penalty_param = 10**(-1*penalty)
    
    model.set_params(C=penalty_param)

    model.fit(X_train,Y_train)
    predict = model.predict_proba(X_cross)
    predict = predict[:,1]
    score = -1*roc_auc_score(Y_cross,predict)
    print "Penalty parameter %f" % (penalty_param)
    print "AUC score %f" % (score)
    return score

def generateTeacherScore(X_train, X_cross, X_test, X_eval, X_predict, Y_train,Y_predict):

    teacher = pd.DataFrame(X_train['teacher_acctid_x'])
    teacher_cross = pd.DataFrame(X_cross['teacher_acctid_x'])
    teacher_test = pd.DataFrame(X_test['teacher_acctid_x'])
    teacher_eval = pd.DataFrame(X_eval['teacher_acctid_x'])
    teacher_predict = pd.DataFrame(X_predict['teacher_acctid_x'])
    
    vectorizer = DictVectorizer(sparse = True)
    teacher_dummies_train = vectorizer.fit_transform(teacher.T.to_dict().values())
    teacher_dummies_cross = vectorizer.transform(teacher_cross.T.to_dict().values())
    teacher_dummies_test = vectorizer.transform(teacher_test.T.to_dict().values())
    
    vectorizer2 = DictVectorizer(sparse = True)
    teacher_dummies_predict = vectorizer2.fit_transform(teacher_predict.T.to_dict().values())
    teacher_dummies_eval = vectorizer2.transform(teacher_eval.T.to_dict().values())
    
    logit = LogisticRegression(penalty='l2',dual=False,tol=1,fit_intercept=True, C=0.00000001, intercept_scaling=1, class_weight='auto', random_state=423)
    logit.fit(teacher_dummies_train,Y_train)
    X_train_teacher = logit.predict_proba(teacher_dummies_train)[:,1]
    X_cross_teacher = logit.predict_proba(teacher_dummies_cross)[:,1]
    X_test_teacher = logit.predict_proba(teacher_dummies_test)[:,1]
    
    logit2 = LogisticRegression(penalty='l2',dual=False,tol=1,fit_intercept=True, C=0.00000001, intercept_scaling=1, class_weight='auto', random_state=423)
    logit2.fit(teacher_dummies_predict,Y_predict)
    X_predict_teacher = logit2.predict_proba(teacher_dummies_predict)[:,1]
    X_eval_teacher = logit2.predict_proba(teacher_dummies_eval)[:,1]
    
    return X_train_teacher, X_cross_teacher, X_test_teacher, X_eval_teacher, X_predict_teacher

def makeFixedEffects(allData, clean_data_path):
    #print "teacher exciting"
    #teacher_exciting_dummies = fixedEffectVariables(allData,'teacher_acctid_x','is_exciting')
    print "teacher funded"
    teacher_funded_dummies = fixedEffectVariables(allData,'teacher_acctid_x','fully_funded')
    print "teacher chat"
    teacher_chat_dummies = fixedEffectVariables(allData,'teacher_acctid_x','great_chat')
    
    #print "school exciting"
    #school_exciting_dummies = fixedEffectVariables(allData,'schoolid','is_exciting')
    print "school funded"
    school_funded_dummies = fixedEffectVariables(allData,'schoolid','fully_funded')
    #print "school chat"
    #school_chat_dummies = fixedEffectVariables(allData,'schoolid','great_chat')
    fixedthings = allData[['teacher_acctid_x','schoolid','date_posted']]
    #teacher_exc_dum = pd.DataFrame(pd.Series(teacher_exciting_dummies), columns = ['teacher_exciting_mean'])
    #school_exc_dum = pd.DataFrame(pd.Series(school_exciting_dummies), columns = ['school_exciting_mean'])
    teacher_fun_dum = pd.DataFrame(pd.Series(teacher_funded_dummies), columns = ['teacher_funded_mean'])
    school_fun_dum = pd.DataFrame(pd.Series(school_funded_dummies), columns = ['school_funded_mean'])
    teacher_chat_dum = pd.DataFrame(pd.Series(teacher_chat_dummies), columns = ['teacher_chat_mean'])
    data = [teacher_chat_dum]
    
    fixed_effects = pd.concat(data,axis=1)
    fixed_effects.to_csv(clean_data_path + 'fixed_effects_chat.csv',index=False)

def genTimeWeight(allData):
    date = allData['date_posted']
    min_max_scalar = preprocessing.MinMaxScaler()
    X_train_date = pd.to_datetime(date[(date < pd.datetime(2013,7,1)) & (date >=pd.datetime(2011,7,1))])
    X_predict_date = pd.to_datetime(date[(date < pd.datetime(2014,1,1)) & (date >=pd.datetime(2012,1,1))])
    
    X_train_dif_days = (X_train_date-X_train_date.min()).apply(lambda x: float(x)/(3600*24*1000000000))
    X_predict_dif_days = (X_predict_date-X_predict_date.min()).apply(lambda x: float(x)/(3600*24*1000000000))
    
    X_train_weight = (X_train_dif_days-X_train_dif_days.min())/(X_train_dif_days.max()-X_train_dif_days.min())
    X_predict_weight = (X_predict_dif_days-X_predict_dif_days.min())/(X_predict_dif_days.max()-X_predict_dif_days.min())
    
def otherOutcomeModel(clean_data_path,X_train,X_cross,X_test,X_predict,X_eval,outcome,name='default'):
    #Fit logit models for now, might want to do others later
    
    print "Running intermediate model on " + outcome 
    
    Y_predict = getOutcome(clean_data_path, 'prediction train',outcome)
    Y_train = getOutcome(clean_data_path,'two year train',outcome)
    Y_cross = getOutcome(clean_data_path,'cross validation data',outcome)
    
    Y_cross.fillna(0, inplace=True)
    Y_predict.fillna(0, inplace=True)
    Y_train.fillna(0,inplace=True)
    
    standardize = preprocessing.StandardScaler()
    X_train_predict = standardize.fit_transform(X_train)
    X_cross_predict = standardize.transform(X_cross)
    X_test_predict = standardize.transform(X_test)    
    standardize_predict = preprocessing.StandardScaler()
    X_predict_predict = standardize_predict.fit_transform(X_predict)
    X_eval_predict = standardize_predict.transform(X_eval)
    
    if ((Y_train == 1) | (Y_train == 0)).all():
        #Binary variable
        logit = LogisticRegression(penalty='l2',dual=False,tol=1,fit_intercept=True, C=.0004325, intercept_scaling=1, class_weight='auto', random_state=423)
        logit.fit(X_train_predict,Y_train)
        logit2 = LogisticRegression(penalty='l2',dual=False,tol=1,fit_intercept=True, C=.0004325, intercept_scaling=1, class_weight='auto', random_state=423)
        logit2.fit(X_predict_predict,Y_predict)
        
    
        inScore = roc_auc_score(Y_cross,logit.predict_proba(X_cross_predict)[:,1])
        print "Cross Logistic: Area under auc curve is %f" % (inScore)
            
        
        X_train[name] = logit.predict_proba(X_train_predict)[:,1]
        X_cross[name] = logit.predict_proba(X_cross_predict)[:,1]
        X_test[name] = logit.predict_proba(X_test_predict)[:,1]
        X_predict[name] = logit2.predict_proba(X_predict_predict)[:,1]
        X_eval[name] = logit2.predict_proba(X_eval_predict)[:,1]
    else:
        #Continuous variable
        ridge = Ridge(alpha=.001)
        ridge.fit(X_train_predict,Y_train)
        ridge2 = Ridge(alpha=0.001)
        ridge2.fit(X_predict_predict, Y_predict)
        
        inScore = r2_score(Y_train,ridge.predict(X_train_predict))
        print "Train Ridge: r2 score is %f" % (inScore)
        
        inScore = r2_score(Y_cross,ridge.predict(X_cross_predict))
        print "Cross Ridge: r2 score is %f" % (inScore)
        
        X_train[name] = ridge.predict(X_train_predict)
        X_cross[name] = ridge.predict(X_cross_predict)
        X_test[name] = ridge.predict(X_test_predict)
        X_predict[name] = ridge2.predict(X_predict_predict)
        X_eval[name] = ridge2.predict(X_eval_predict)
    
    
    return X_train,X_cross,X_test,X_predict,X_eval

def genOtherOutcomeVariables(clean_data_path,X_train,X_cross,X_test,X_predict,X_eval):
    X_train,X_cross,X_test,X_predict,X_eval = otherOutcomeModel(clean_data_path,X_train,X_cross,X_test,X_predict,X_eval,'fully_funded','predict_fund')
    X_train,X_cross,X_test,X_predict,X_eval = otherOutcomeModel(clean_data_path,X_train,X_cross,X_test,X_predict,X_eval,'great_chat','predict_chat')
    X_train,X_cross,X_test,X_predict,X_eval = otherOutcomeModel(clean_data_path,X_train,X_cross,X_test,X_predict,X_eval,'donation_from_thoughtful_donor','predict_thoughtful')
    X_train,X_cross,X_test,X_predict,X_eval = otherOutcomeModel(clean_data_path,X_train,X_cross,X_test,X_predict,X_eval,'at_least_1_teacher_referred_donor','predict_refer')
    X_train,X_cross,X_test,X_predict,X_eval = otherOutcomeModel(clean_data_path,X_train,X_cross,X_test,X_predict,X_eval,'at_least_1_green_donation','predict_green')
    
    #Five above have been tested and improve the model
    
    #X_train,X_cross,X_test,X_predict,X_eval = otherOutcomeModel(clean_data_path,X_train,X_cross,X_test,X_predict,X_eval,'three_or_more_non_teacher_referred_donors','predict_non_refer')
    #X_train,X_cross,X_test,X_predict,X_eval = otherOutcomeModel(clean_data_path,X_train,X_cross,X_test,X_predict,X_eval,'one_non_teacher_referred_donor_giving_100_plus','predict_large')
    #X_train,X_cross,X_test,X_predict,X_eval = otherOutcomeModel(clean_data_path,X_train,X_cross,X_test,X_predict,X_eval,'great_messages_proportion','predict_gm_proportion')
    #X_train,X_cross,X_test,X_predict,X_eval = otherOutcomeModel(clean_data_path,X_train,X_cross,X_test,X_predict,X_eval,'teacher_referred_count','predict_refer_count')
    #X_train,X_cross,X_test,X_predict,X_eval = otherOutcomeModel(clean_data_path,X_train,X_cross,X_test,X_predict,X_eval,'non_teacher_referred_count','predict_non_refer_ct')
    #X_train,X_cross,X_test,X_predict,X_eval = otherOutcomeModel(clean_data_path,X_train,X_cross,X_test,X_predict,X_eval,'donation_count','predict_donation_ct')
    #X_train,X_cross,X_test,X_predict,X_eval = otherOutcomeModel(clean_data_path,X_train,X_cross,X_test,X_predict,X_eval,'donation_total','predict_donation_total')
    
    return X_train,X_cross,X_test,X_predict,X_eval
    
def testingSVMstuff(trainData,Y_train ,crossData, Y_cross, testData, Y_test):
    
    dump_svmlight_file(X_train, Y_train.apply(lambda x: 1 if x else -1), 'C:/Users/P_Kravik/Desktop/mlsofia/sofia-ml/Kaggle/kaggle.train')
    dump_svmlight_file(X_cross, Y_cross.apply(lambda x: 1 if x else -1), 'C:/Users/P_Kravik/Desktop/mlsofia/sofia-ml/Kaggle/kaggle.cross')
    dump_svmlight_file(X_test, Y_test.apply(lambda x: 1 if x else -1), 'C:/Users/P_Kravik/Desktop/mlsofia/sofia-ml/Kaggle/kaggle.test')
    
    Y_eval = [1]*X_eval.shape[0]
    dump_svmlight_file(X_eval, Y_eval, 'C:/Users/P_Kravik/Desktop/mlsofia/sofia-ml/Kaggle/kaggle.eval')
    
    rank = genNewOutcome(clean_data_path)
    
    dump_svmlight_file(X_train, rank, 'C:/Users/P_Kravik/Desktop/mlsofia/sofia-ml/Kaggle/kaggle.rank')
    
    train_prediction = pd.read_csv('C:/Users/P_Kravik/Desktop/mlsofia/sofia-ml/Kaggle/trainprediction.txt',sep='\t')
    cross_prediction = pd.read_csv('C:/Users/P_Kravik/Desktop/mlsofia/sofia-ml/Kaggle/crossprediction.txt',sep='\t')
    test_prediction = pd.read_csv('C:/Users/P_Kravik/Desktop/mlsofia/sofia-ml/Kaggle/testprediction.txt',sep='\t')
    eval_prediction = pd.read_csv('C:/Users/P_Kravik/Desktop/mlsofia/sofia-ml/Kaggle/evalprediction.txt',sep='\t',header=None)
    
    generatePrediction(clean_data_path, subPath, eval_prediction[[0]].values.reshape((44772,)),'Learning to rank model')
    
    return 1

def genNewOutcome(clean_data_path):
    
    is_exciting = getOutcome(clean_data_path,'two year train','is_exciting')
    fully_funded = getOutcome(clean_data_path,'two year train','fully_funded')
    at_least_1_teacher_referred_donor = getOutcome(clean_data_path,'two year train','at_least_1_teacher_referred_donor')
    at_least_1_green_donation = getOutcome(clean_data_path,'two year train','at_least_1_green_donation')
    great_chat = getOutcome(clean_data_path,'two year train','great_chat')
    three_or_more_non_teacher_referred_donors = getOutcome(clean_data_path,'two year train','three_or_more_non_teacher_referred_donors')
    one_non_teacher_referred_donor_giving_100_plus = getOutcome(clean_data_path,'two year train','one_non_teacher_referred_donor_giving_100_plus')
    donation_from_thoughtful_donor = getOutcome(clean_data_path,'two year train','donation_from_thoughtful_donor')

    is_exciting.fillna(0,inplace=True)
    fully_funded.fillna(0, inplace = True)
    great_chat.fillna(0, inplace=True)
    at_least_1_teacher_referred_donor.fillna(0, inplace=True)
    at_least_1_green_donation.fillna(0, inplace=True)
    
    rank = 1+at_least_1_green_donation+fully_funded + great_chat + 10*is_exciting+at_least_1_teacher_referred_donor

    return rank
def createFeatures(purpose='test'):
    print "Prediction using project data"
    clean_data_path = 'C:/Users/P_Kravik/Desktop/Kaggle/KDD Cup 2014 - Predicting Excitement at DonorsChoose.org/Data/Clean/Derived/'
    subPath = 'C:/Users/P_Kravik/Desktop/Kaggle/KDD Cup 2014 - Predicting Excitement at DonorsChoose.org/Submissions/'
    
    #print "Running model on training time data..."
    #trainData = pd.read_csv(clean_data_path + 'train.csv')   
    #trainData = fillMissing(trainData)
    
    allData = pd.read_csv(clean_data_path + 'all master data.csv')
    allData = fillMissing(allData)
    allData = allData.drop('essay',1)
    allData = allData.drop('short_description',1)
    allData = allData.drop('need_statement',1)
    allData = allData.drop('title',1)
      
    allData['date_posted'] = pd.to_datetime(allData['date_posted'])
    allData.sort(['teacher_acctid_x','date_posted'], inplace=True)
    
    Y_train = getOutcome(clean_data_path,'two year train','is_exciting')
    Y_cross = getOutcome(clean_data_path,'cross validation data','is_exciting')    
    Y_test = getOutcome(clean_data_path,'test data','is_exciting')        
    Y_predict = getOutcome(clean_data_path, 'prediction train','is_exciting')
    
    #Y_train_funded = getOutcome(clean_data_path,'two year train','fully_funded')
    
    #allData = allData[allData['date_posted']>=pd.datetime(2011,7,1)]
    #allData = allData[(allData['date_posted']>=pd.datetime(2011,7,1)) & (allData['date_posted']<pd.datetime(2013,7,1))] 
    
    #makeFixedEffects(allData, clean_data_path)
    
    
    allData = getProjectVariables(allData)
    allData.shape
    #bring back teacherid
    
    fixedEffects = pd.read_csv(clean_data_path + 'fixed_effects.csv')
    fixedEffects['teacher_no_history'] = (fixedEffects['teacher_exciting_mean'].isnull()).apply(lambda x: 1 if x else 0)
    fixedEffects['teacher_exciting_mean'].fillna(0, inplace=True)
    fixedEffects['school_no_history'] = (fixedEffects['school_exciting_mean'].isnull()).apply(lambda x: 1 if x else 0)
    fixedEffects['school_exciting_mean'].fillna(0, inplace=True)
    fixedEffects2 = pd.read_csv(clean_data_path + 'fixed_effects_funded.csv')
    fixedEffects2['teacher_no_funded'] = (fixedEffects2['teacher_funded_mean'].isnull()).apply(lambda x: 1 if x else 0)
    fixedEffects2['teacher_funded_mean'].fillna(0, inplace=True)
    
    
    allData['teacher_mean_effect'] = fixedEffects['teacher_exciting_mean']
    allData['teacher_mean_no_history'] = fixedEffects['teacher_no_history']
    
    #allData['teacher_mean_funded'] = fixedEffects2['teacher_funded_mean']
    #allData['teacher_no_funded'] = fixedEffects2['teacher_no_funded']
    
    allData['school_mean_effect'] = fixedEffects['school_exciting_mean']
    allData['school_mean_no_history'] = fixedEffects['school_no_history']
    
    #allData['promising'] = ((allData['teacher_mean_effect']>0) & (allData['noSubmission']>1)).apply(lambda x: 1 if x else 0)
    #allData['very_promising'] = ((allData['teacher_mean_effect']>0.9) & (allData['noSubmission']>1)).apply(lambda x: 1 if x else 0)
    #allData.drop('teacher_mean_effect',inplace=True, axis=1)
    
    #Create interaction for first submission
    
    allData.shape
    
    
    #allData.drop('teacher_acctid_x',inplace=True)
    columns = allData.columns
    columns.values[-6] = 'noSubmission'
    allData.columns = columns.values
    
    X_train, X_cross, X_test, X_eval, X_predict = createSamples(allData)
    
    #X_train_score, X_cross_score, X_test_score, X_eval_score, X_predict_score = generateTeacherScore(X_train, X_cross, X_test, X_eval, X_predict, Y_train, Y_predict)
    X_train.drop('teacher_acctid_x', inplace=True, axis=1)
    X_cross.drop('teacher_acctid_x', inplace=True, axis=1)
    X_test.drop('teacher_acctid_x', inplace=True, axis=1)
    X_eval.drop('teacher_acctid_x', inplace=True, axis=1)
    X_predict.drop('teacher_acctid_x', inplace=True, axis=1)
    

    
#     
#     X_train['teacher_score'] = X_train_score
#     X_cross['teacher_score'] = X_cross_score
#     X_test['teacher_score'] = X_test_score
#     X_eval['teacher_score'] = X_eval_score
#     X_predict['teacher_score'] = X_predict_score
    
#     X_train['teacher_mean_effect'] = (X_train['teacher_mean_effect']>0).apply(lambda x: 1 if x else 0)
#     X_train['promising'] = ((X_train['teacher_mean_effect']>0) & (X_train['noSubmission']>1)).apply(lambda x: 1 if x else 0)
#     X_train['very_promising'] = ((X_train['teacher_mean_effect']>0.9) & (X_train['noSubmission']>2)).apply(lambda x: 1 if x else 0)
#     #X_train.drop('teacher_mean_effect',inplace=True, axis=1)
#     X_cross['teacher_mean_effect'] = (X_cross['teacher_mean_effect']>0).apply(lambda x: 1 if x else 0)
#     X_cross['promising'] = ((X_cross['teacher_mean_effect']>0) & (X_cross['noSubmission']>1)).apply(lambda x: 1 if x else 0)
#     X_cross['very_promising'] = ((X_cross['teacher_mean_effect']>0.9) & (X_cross['noSubmission']>2)).apply(lambda x: 1 if x else 0)
#    
#     X_test['teacher_mean_effect'] = (X_test['teacher_mean_effect']>0).apply(lambda x: 1 if x else 0)
#     X_test['promising'] = ((X_test['teacher_mean_effect']>0) & (X_test['noSubmission']>1)).apply(lambda x: 1 if x else 0)
#     X_test['very_promising'] = ((X_test['teacher_mean_effect']>0.9) & (X_test['noSubmission']>2)).apply(lambda x: 1 if x else 0)
#    
#     X_predict['teacher_mean_effect'] = (X_predict['teacher_mean_effect']>0).apply(lambda x: 1 if x else 0)
#     X_predict['promising'] = ((X_predict['teacher_mean_effect']>0) & (X_predict['noSubmission']>1)).apply(lambda x: 1 if x else 0)
#     X_predict['very_promising'] = ((X_predict['teacher_mean_effect']>0.9) & (X_predict['noSubmission']>2)).apply(lambda x: 1 if x else 0)
#     
#     X_eval['teacher_mean_effect'] = (X_eval['teacher_mean_effect']>0).apply(lambda x: 1 if x else 0)
#     X_eval['promising'] = ((X_eval['teacher_mean_effect']>0) & (X_eval['noSubmission']>1)).apply(lambda x: 1 if x else 0)
#     X_eval['very_promising'] = ((X_eval['teacher_mean_effect']>0.9) & (X_eval['noSubmission']>2)).apply(lambda x: 1 if x else 0)
# 
#    
#     X_train.shape
#     firstSubmission = (X_train['noSubmission'] == 1).apply(lambda x: 1 if x else 0)
#     #notFirstSubmission = (X_train['noSubmission'] == 0).apply(lambda x: 1 if x else 0)
#     firstSubData = X_train.mul(firstSubmission, axis=0)
#     #notFirstSubData = X_train.mul(notFirstSubmission, axis=0)
#     promisingSubData = X_train.mul(X_train['promising'],axis=0)
#     veryPromisingSubData = X_train.mul(X_train['very_promising'],axis=0)
#     X_train = pd.concat([X_train, firstSubData,promisingSubData,veryPromisingSubData], axis=1)
#     
#     firstSubmission = (X_cross['noSubmission'] == 1).apply(lambda x: 1 if x else 0)
#     #notFirstSubmission = (X_train['noSubmission'] == 0).apply(lambda x: 1 if x else 0)
#     firstSubData = X_cross.mul(firstSubmission, axis=0)
#     #notFirstSubData = X_train.mul(notFirstSubmission, axis=0)
#     promisingSubData = X_cross.mul(X_cross['promising'],axis=0)
#     veryPromisingSubData = X_cross.mul(X_cross['very_promising'],axis=0)
#     X_cross = pd.concat([X_cross, firstSubData,promisingSubData,veryPromisingSubData], axis=1)
#     
#     firstSubmission = (X_test['noSubmission'] == 1).apply(lambda x: 1 if x else 0)
#     #notFirstSubmission = (X_train['noSubmission'] == 0).apply(lambda x: 1 if x else 0)
#     firstSubData = X_test.mul(firstSubmission, axis=0)
#     #notFirstSubData = X_train.mul(notFirstSubmission, axis=0)
#     promisingSubData = X_test.mul(X_test['promising'],axis=0)
#     veryPromisingSubData = X_test.mul(X_test['very_promising'],axis=0)
#     X_test = pd.concat([X_test, firstSubData,promisingSubData,veryPromisingSubData], axis=1)
#     
#     firstSubmission = (X_eval['noSubmission'] == 1).apply(lambda x: 1 if x else 0)
#     #notFirstSubmission = (X_train['noSubmission'] == 0).apply(lambda x: 1 if x else 0)
#     firstSubData = X_eval.mul(firstSubmission, axis=0)
#     #notFirstSubData = X_train.mul(notFirstSubmission, axis=0)
#     promisingSubData = X_eval.mul(X_eval['promising'],axis=0)
#     veryPromisingSubData = X_eval.mul(X_eval['very_promising'],axis=0)
#     X_eval = pd.concat([X_eval, firstSubData,promisingSubData,veryPromisingSubData], axis=1)
#     
#     firstSubmission = (X_predict['noSubmission'] == 1).apply(lambda x: 1 if x else 0)
#     #notFirstSubmission = (X_train['noSubmission'] == 0).apply(lambda x: 1 if x else 0)
#     firstSubData = X_predict.mul(firstSubmission, axis=0)
#     #notFirstSubData = X_train.mul(notFirstSubmission, axis=0)
#     promisingSubData = X_predict.mul(X_predict['promising'],axis=0)
#     veryPromisingSubData = X_predict.mul(X_predict['very_promising'],axis=0)
#     X_predict = pd.concat([X_predict, firstSubData,promisingSubData,veryPromisingSubData], axis=1)
#     
#     del firstSubmission
#     del firstSubData
#     del promisingSubData
#     del veryPromisingSubData
    
    X_train.drop('noSubmission', inplace=True, axis=1)
    X_cross.drop('noSubmission', inplace=True, axis=1)
    X_test.drop('noSubmission', inplace=True, axis=1)
    X_eval.drop('noSubmission', inplace=True, axis=1)
    X_predict.drop('noSubmission', inplace=True, axis=1)
    
        
    del allData
    X_train.shape
    
    X_train, X_cross, X_test, X_predict, X_eval = genOtherOutcomeVariables(clean_data_path,X_train, X_cross, X_test, X_predict, X_eval)
    
    print "Train data size match %s" % (len(X_train) == len(Y_train))
    print "Cross data size match %s" % (len(X_cross) == len(Y_cross))
    print "Test data size match %s" % (len(X_test) == len(Y_test))
    print "prediction training data size match %s" % (len(X_predict) == len(Y_predict))

    X_train.to_csv(clean_data_path+'/X values/project training X values.csv')
    X_cross.to_csv(clean_data_path+'/X values/project cross validation X values.csv')
    X_test.to_csv(clean_data_path+'/X values/project test X values.csv')
    X_predict.to_csv(clean_data_path+'/X values/project prediction X values.csv')
    X_eval.to_csv(clean_data_path+'/X values/project evaluation X values.csv')


    standardize = preprocessing.StandardScaler()
    X_train = standardize.fit_transform(X_train)
    X_cross = standardize.transform(X_cross)
    X_test = standardize.transform(X_test)
    X_eval = standardize.transform(X_eval)
    #X_eval = standardize.transform(X_eval)
    
    #standardize_test = preprocessing.StandardScaler()
    #X_test = standardize_test.fit_transform(X_test)
    
    standardize_predict = preprocessing.StandardScaler()
    X_predict = standardize_predict.fit_transform(X_predict)
    X_eval = standardize_predict.transform(X_eval)
    
#     decomp = decomposition.PCA(75)
#     decomp2 = decomposition.PCA(75)
#     X_train = decomp.fit_transform(X_train)
#     X_cross = decomp.transform(X_cross)
#     X_test = decomp.transform(X_test)
#     X_predict = decomp2.fit_transform(X_predict)
#     X_eval = decomp2.transform(X_eval)
    
    #Fit the penalty parameter
    #Logit model part
#     logit = LogisticRegression(penalty='l2',dual=False,tol=1,fit_intercept=True, C=1, intercept_scaling=1, class_weight='auto', random_state=423)
#     print "Final prediction"
#     
#     #Try to implement grid search
#     rranges = slice(-2, 10, 1)
#     params = (X_train,Y_train,X_cross,Y_cross,logit)
#     resbrute = optimize.brute(cvScore, (rranges,), args=params, full_output=True, finish=optimize.fmin)
    
    #LOGIT PROJECT MODEL
    
    
    logit = LogisticRegression(penalty='l2',dual=False,tol=1,fit_intercept=True, C=.0004325, intercept_scaling=1, class_weight='auto', random_state=423)
    
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
    logit_predict.fit(X_predict,Y_predict)
    
    temp = logit_predict.predict_proba(X_eval)
    Y_predict_logit = temp[:,1]
    generatePrediction(clean_data_path, subPath,Y_predict_logit,'Project logit model')
    
    #SVM Model

#     svm = SVC(probability=True,class_weight='auto',random_state = 45)
#     svm.fit(X_train,Y_train)
#      
#     predict_svm = logit.predict_proba(X_train)[:,1]
#      
#     inScore = roc_auc_score(Y_train,predict_svm)
#     print "Train Logistic: Area under auc cur ve is %f" % (inScore)
#      
#     cross_predict_svm = logit.predict_proba(X_cross)[:,1]
#      
#     inScore = roc_auc_score(Y_cross,cross_predict_svm)
#     print "Cross Logistic: Area under auc curve is %f" % (inScore)
#      
#     test_predict_svm = logit.predict_proba(X_test)[:,1]
#      
#     inScore = roc_auc_score(Y_test,test_predict_svm)
#     print "Test logistic: Area under auc curve is %f" % (inScore)
    
    #SGD MODEL
    sgd = SGDClassifier(loss='hinge',alpha=10, class_weight={1:18,0:1})
    sgd.fit(X_train,Y_train)
    
    predict = sgd.predict(X_train)
    predict.mean()
    predict_sgd = predict
    
    inScore = roc_auc_score(Y_train,predict)
    print "Train SGD: Area under auc curve is %f" % (inScore)
    
    cross_predict = sgd.predict(X_cross)
    cross_predict_sgd = cross_predict
    
    inScore = roc_auc_score(Y_cross,cross_predict_sgd)
    print "Cross Random Forest: Area under auc curve is %f" % (inScore)
    
    test_predict = sgd.predict(X_test)
    test_predict_sgd = test_predict
    
    inScore = roc_auc_score(Y_test,test_predict_sgd)
    print "Test Random Forest: Area under auc curve is %f" % (inScore)
    
    #RANDOM FOREST PROJECT MODEL
    #Should cross validate this value
    rf_weight = Y_train*4+1
    
    clf = ensemble.RandomForestClassifier(n_estimators=100, criterion='entropy',max_depth=10,max_features=25,bootstrap=True,oob_score=True,n_jobs=2,random_state = 42)
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
    
    clf2 = ensemble.RandomForestClassifier(n_estimators=100,criterion='entropy',max_depth=10,max_features=150,oob_score=True,n_jobs=2,random_state = 42)
    clf2.fit(X_predict,Y_predict,rf_predict_weight.values)
    
    temp = clf2.predict_proba(X_eval)
    Y_predict_rf = temp[:,1]
    generatePrediction(clean_data_path, subPath,Y_predict_rf,'Project random forest model')
    
    #GRADIENT BOOSTING PROJECT MODEL
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
    generatePrediction(clean_data_path, subPath, Y_predict_gbm,'Project GBM model')
    
    #ADABOOST STUMPS PROJECT MODEL
    ada_weight = Y_train*4+1
    
    ada = ensemble.AdaBoostClassifier(n_estimators=100,learning_rate = 1,random_state = 42)
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
    generatePrediction(clean_data_path, subPath, Y_predict_ada,'Project AdaBoost model')
    

if __name__ == '__main__':
    createFeatures()