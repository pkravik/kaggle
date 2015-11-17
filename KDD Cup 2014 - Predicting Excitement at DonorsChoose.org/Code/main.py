'''
Created on May 22, 2014

@author: P_Kravik
'''
import pandas as pd
import numpy as np
import random
import nltk
from sklearn.feature_extraction.text import CountVectorizer

def dataExplorationProjects():
    raw_data_path =  'C:/Users/P_Kravik/Desktop/Kaggle/KDD Cup 2014 - Predicting Excitement at DonorsChoose.org/Data/raw/csv/'
    projects = pd.read_csv(raw_data_path+'projects.csv')
    
    projects['date_posted'] = pd.to_datetime(projects['date_posted'])
      
    #Convert t/f values into 1/0 binary
    tffields = ['school_charter','school_magnet','school_year_round','school_nlns','school_kipp','school_charter_ready_promise','teacher_teach_for_america','teacher_ny_teaching_fellow','eligible_double_your_impact_match','eligible_almost_home_match']
    for var in tffields:
        projects[var].replace('t',1,inplace=True)
        projects[var].replace('f',0,inplace=True)
        
    timeTrend = projects.groupby('date_posted')
    

def cleanProjects(raw_data_path,clean_data_path):
    #there are 664098 rows for projects.csv. They are all unique projects, on the project level
    print "Reading raw projects data set..."
    projects = pd.read_csv(raw_data_path+'projects.csv')
    len(projects.index)
    projects['projectid'].nunique()
    
    print "Cleaning projects data set..."
    #Convert date into pandas date
    projects['date_posted'] = pd.to_datetime(projects['date_posted'])
      
    #Convert t/f values into 1/0 binary
    tffields = ['school_charter','school_magnet','school_year_round','school_nlns','school_kipp','school_charter_ready_promise','teacher_teach_for_america','teacher_ny_teaching_fellow','eligible_double_your_impact_match','eligible_almost_home_match']
    for var in tffields:
        projects[var].replace('t',1,inplace=True)
        projects[var].replace('f',0,inplace=True)
    
    print "Saving cleaned projects data set..."
    projects.to_csv(clean_data_path+'projects_clean.csv')
    
    return projects

def cleanOutcomes(raw_data_path,clean_data_path):
    #Open the outcomes data. There are 619326 rows. All unique projects, on the project level
    print "Reading raw outcomes data set..."
    outcomes = pd.read_csv(raw_data_path+'outcomes.csv')
    len(outcomes.index)
    outcomes['projectid'].nunique()
    
    print "Cleaning outcomes data set..."
    #Create list of t/f values
    tffields = ['is_exciting','at_least_1_teacher_referred_donor','fully_funded','at_least_1_green_donation','great_chat','three_or_more_non_teacher_referred_donors','one_non_teacher_referred_donor_giving_100_plus','donation_from_thoughtful_donor']
    
    for var in tffields:
        #replace t/f values with 1/0
        outcomes[var].replace('t',1,inplace=True)
        outcomes[var].replace('f',0,inplace=True)
    
    print "Saving cleaned outcomes data set..."
    outcomes.to_csv(clean_data_path+'outcomes_clean.csv')
    
    return outcomes

def cleanEssays(raw_data_path,clean_data_path):
    #open the essay data. There are 664098 rows. All unique projects, on the project level
    print "Reading raw essay data..."
    essays = pd.read_csv(raw_data_path+'essays.csv')
    len(essays.index)
    essays['projectid'].nunique()
    
    #There are 249555 unique teachers
    essays['teacher_acctid'].nunique()
    
    #Really need to think about this encoding issue because I have no idea what is happening, but i'll keep moving forward
    
    print "Cleaning essay data..."
    #Replace NaN values for text with blank space
    essays.replace(np.nan,' ',inplace=True)
    
    #Remove whitespace and new line shenanigancs which is coming up as  \\n
    essays['essay'] = essays['essay'].str.replace('\\\\n', '')
    essays.replace('\s+', ' ',inplace=True,regex = True)
    
    #Create essay variables
    print "Creating essay variables..."
    
    tokens = essays['essay'].apply(lambda d: nltk.word_tokenize(d))
    
    essays['numCharactersEssay'] = essays['essay'].apply(lambda d: len(d))
    essays['numWordsEssay'] = tokens.apply(lambda d: len(d))
    essays['numUniqueWordsEssay'] = tokens.apply(lambda d: len(set(d)))
    essays['lexicalDiversityEssay'] = essays['numWordsEssay']/essays['numUniqueWordsEssay']
    essays['lexicalDiversityEssay'][essays['lexicalDiversityEssay']>10]=10
    
    
    
    print "Creating short description variables..."
    
    tokens = essays['short_description'].apply(lambda d: nltk.word_tokenize(d))
    
    essays['numCharacterShortDescription'] = essays['short_description'].apply(lambda d: len(d))
    essays['numWordsShortDescription'] = tokens.apply(lambda d: len(d))
    essays['numUniqueWordsShortDescription'] = tokens.apply(lambda d: len(set(d)))
    essays['lexicalDiversityShortDescription'] = essays['numWordsShortDescription']/essays['numUniqueWordsShortDescription']
    essays['lexicalDiversityShortDescription'][essays['lexicalDiversityShortDescription']>10]=10
    
    print "Creating need statement variables..."
    
    tokens = essays['need_statement'].apply(lambda d: nltk.word_tokenize(d))
    
    essays['numCharacterNeedStatement'] = essays['need_statement'].apply(lambda d: len(d))
    essays['numWordsNeedStatement'] = tokens.apply(lambda d: len(d))
    essays['numUniqueWordsNeedStatement'] = tokens.apply(lambda d: len(set(d)))
    
    
    print "Creating title variables"
    
    tokens = essays['title'].apply(lambda d: nltk.word_tokenize(d))
    
    essays['numCharacterTitle'] = essays['title'].apply(lambda d: len(d))
    essays['numWordsTitle'] = tokens.apply(lambda d: len(d))
    essays['numUniqueWordsTitle'] = tokens.apply(lambda d: len(set(d)))
    essays['titleHasExclamation'] = essays['title'].apply(lambda d: d.find("!"))
    essays['titleHasExclamation'][essays['titleHasExclamation']>=0] = 1
    essays['titleHasExclamation'][essays['titleHasExclamation']<0] = 0
    
    
    print "Saving cleaned essay data set..."
    essays.to_csv(clean_data_path+'essays_clean.csv')
    #testEssay = essays
    #features = getFeatures(testEssay)
    
    #So it looks like I can't join everything together at once, so going to do in bits
    return essays

def cleanDonations(raw_data_path,clean_data_path):
    print "Reading raw donations data..."
    donations = pd.read_csv(raw_data_path+'donations.csv')
    donations['donation_count'] = 1
    
    #3,097,989 donations
    len(donations.index)
    
    group = donations.groupby('projectid',as_index=False)
    
    donations_proj = group.aggregate(np.sum)
    
    #525,042 projects with donations
    len(donations_proj.index)
    
    donations_proj = donations_proj.drop('donor_zip',axis=1)
    
    return donations_proj


def getFeatures(essays):
    #Screw it, going to use sk_learn
    print "Feature extraction..."
    vectorizer = CountVectorizer(min_df=.01, max_df=0.95)
    X_train = vectorizer.fit_transform(essays)
    num_samples, num_features = X_train.shape
    print "#samples: %d, #features: %d" % (num_samples, num_features)
    
    return (X_train, vectorizer)

def createMaster(raw_data_path,clean_data_path):
    
    #Get the cleaned data
    projects = cleanProjects(raw_data_path,clean_data_path)
    outcomes = cleanOutcomes(raw_data_path,clean_data_path)
    
    
    #try the merge. Nothing wrong with the data, have 664098 unique projects
    print "Merging outcomes and project data set..."
    project_with_outcome = pd.merge(projects, outcomes, how='outer', on='projectid')
    project_with_outcome['projectid'].nunique()
    
    del projects
    del outcomes
    
    #Merge in essay data, have 664098 unique projects
    essays = cleanEssays(raw_data_path,clean_data_path)
    print "Merging in essay data..."
    master = pd.merge(project_with_outcome,essays, how='outer', on='projectid')
    del essays
    master['projectid'].nunique()
    
    print "Saving master data..."
    master.to_csv(clean_data_path+'master.csv')
    
    print "Saving subsample of master"
    firstThousandMaster = master[0:999]
    firstThousandMaster.to_csv(clean_data_path+'firstThousandMaster.csv')
    
    return master

def genTestTrainEvalCV(data, save_path):
    print "Saving evaluation data..."
    data['date_posted']=pd.to_datetime(data['date_posted'])
    evalData = data[data['date_posted'] >= pd.datetime(2014,1,1)]
    evalData.to_csv(save_path + 'evaluation data.csv')
    
    insampledata = data[data['date_posted'] < pd.datetime(2014,1,1)]
    #Create training, test, and cross validation. 60% training, 20% test, 20 CV
    num_row = len(insampledata.index)
    pctTrain = 0.6
    pctCross = 0.2
    #pctTest = 0.2
    print "Saving prediction data..."
    insampledata.to_csv(save_path + 'prediction data.csv')
    
    print "Saving training data..."
    #Create the training data. Has 371,596 observations
    random.seed(10)
    rows = random.sample(insampledata.index, int(round(num_row * pctTrain)))
    len(rows)
    trainSample = insampledata.ix[rows]
    trainSample.to_csv(save_path + 'training data.csv')
    insampledata=insampledata.drop(rows)
    
    print "Saving CV data..."
    #Create cross validation data. Has 123,865 observations
    random.seed(30)
    rows = random.sample(insampledata.index,int(round(num_row*pctCross)))
    len(rows)
    crossSample = insampledata.ix[rows]
    crossSample.to_csv(save_path + 'cross validation data.csv')
    insampledata=insampledata.drop(rows)
    
    print "Saving test data..."
    #Create test sample data. Has 123,865 observations
    testSample = insampledata
    testSample.to_csv(save_path + 'test data.csv')

def genTestTrainEvalCVTime(data, save_path):
    print "Saving evaluation data..."
    data['date_posted']=pd.to_datetime(data['date_posted'])
    evalData = data[data['date_posted'] >= pd.datetime(2014,1,1)]
    evalData.to_csv(save_path + 'evaluation data.csv')
    
    insampledata = data[data['date_posted'] < pd.datetime(2014,1,1)]
    #Create training, test, and cross validation. 60% training, 20% test, 20 CV
    num_row = len(insampledata.index)
    #pctTrain = 0.6
    pctCross = 0.2
    pctTest = 0.2
    
    numTest = int(round(pctTest*num_row))
    numCross = int(round(pctCross*num_row))+numTest
    testSample = insampledata[0:numTest-1] 
    cvSample = insampledata[numTest:numCross]
    trainSample = insampledata[numCross+1:]
        
    testSample.to_csv(save_path + 'test time data.csv')
    trainSample.to_csv(save_path + 'training time data.csv')
    cvSample.to_csv(save_path + 'cross validation time data.csv')
    
def genExperimentalTestSet(data, save_path):
    print "Saving evaluation data..."
    data['date_posted']=pd.to_datetime(data['date_posted'])
    evalData = data[data['date_posted'] >= pd.datetime(2014,1,1)]
    evalData.to_csv(save_path + 'evaluation data.csv')
    
    insampledata = data[data['date_posted'] < pd.datetime(2014,1,1)]
    
    test_data = insampledata[insampledata['date_posted']>=pd.datetime(2013,10,1)]
    cv_data = insampledata[(insampledata['date_posted']>=pd.datetime(2013,7,1)) & (insampledata['date_posted']<pd.datetime(2013,10,1))]
    train_data = insampledata[(insampledata['date_posted']>=pd.datetime(2011,7,1)) & (insampledata['date_posted']<pd.datetime(2013,7,1))]
    all_train_data = insampledata[insampledata['date_posted']<pd.datetime(2014,1,1)]
    predict_data = insampledata[(insampledata['date_posted']<pd.datetime(2014,1,1)) & (insampledata['date_posted']>=pd.datetime(2012,1,1))]
    
    train_data.to_csv(save_path + 'two year train.csv')
    cv_data.to_csv(save_path + 'cross validation data.csv')      
    test_data.to_csv(save_path + 'test data.csv')
    all_train_data.to_csv(save_path + 'train.csv')
    predict_data.to_csv(save_path + 'prediction train.csv')
    
    
        
def main():
    print "KDD Cup 2014 code"
    raw_data_path =  'C:/Users/P_Kravik/Desktop/Kaggle/KDD Cup 2014 - Predicting Excitement at DonorsChoose.org/Data/raw/csv/'
    clean_data_path = 'C:/Users/P_Kravik/Desktop/Kaggle/KDD Cup 2014 - Predicting Excitement at DonorsChoose.org/Data/Clean/Derived/'
        
    #master = createMaster(raw_data_path,clean_data_path)
    donations = cleanDonations(raw_data_path, clean_data_path)
    master = pd.read_csv(clean_data_path+'master.csv')
    master = pd.merge(master,donations, how='outer', on='projectid')
    master['date_posted']=pd.to_datetime(master['date_posted'])
    master.sort(['teacher_acctid_x','date_posted'], inplace = True)
    master.to_csv(clean_data_path+'all master data.csv')
    #genTestTrainEvalCV(master, clean_data_path)
    #genTestTrainEvalCVTime(master, clean_data_path)
    
    
    
    genExperimentalTestSet(master, clean_data_path)
    print "Complete!"
    
    
if __name__ == '__main__':
    main()