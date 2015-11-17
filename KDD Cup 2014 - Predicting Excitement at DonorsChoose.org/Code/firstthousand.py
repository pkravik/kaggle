'''
Created on May 22, 2014

@author: P_Kravik
'''


import csv
import pandas as pd



    
def csv_reader(file_obj):
    #read a CSV file
    
    reader = csv.reader(file_obj)
    temp = list()
    for i in range(100):
        temp.append(reader.next())
     
    return temp
    
def csv_writer(data, path):
    with open(path, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)
    
def outcomes():
    
#     files = ['donations','essays',]
#     csv_path = 'C:/Users/P_Kravik/Desktop/Kaggle/KDD Cup 2014 - Predicting Excitement at DonorsChoose.org/Data/raw/csv/outcomes.csv'
#     with open(csv_path,"rb") as f_obj:
#         data = csv_reader(f_obj)
#     write_path = 'C:/Users/P_Kravik/Desktop/Kaggle/KDD Cup 2014 - Predicting Excitement at DonorsChoose.org/Data/raw/first 1000/outcomes.csv'
#     csv_writer(data, write_path
    raw_data_path = 'C:/Users/P_Kravik/Desktop/Kaggle/KDD Cup 2014 - Predicting Excitement at DonorsChoose.org/Data/raw/csv/'
    subset_save_path = 'C:/Users/P_Kravik/Desktop/Kaggle/KDD Cup 2014 - Predicting Excitement at DonorsChoose.org/Data/raw/first 1000/'
    files = ['outcomes.csv','essays.csv','donations.csv','projects.csv','resources.csv','sampleSubmission.csv']
    for datafile in files:
        print "reading "+datafile
        df = pd.read_csv(raw_data_path+datafile)
        print "writing "+datafile
        df[0:1000].to_csv(subset_save_path+datafile)
    
    
if __name__ == '__main__':
    outcomes()
    
    
    