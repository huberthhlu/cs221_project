# -*- coding: utf-8 -*-

'''
File_name: svm_baseline.py:
---------------------------
Use the simple bag-of-words to train a svm model.
In line 69, choose one of the three model. 

1. Without using TF-IDF uni-gram: 0.4993
2. Without using TF-IDF bi-gram: 0.5067
3. Using TF-IDF uni-gram: 0.5026
4. Using TF-IDF bi-gram: 0.5102

'''
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import jieba
import random
import joblib
from sklearn.svm import SVC

import sys
import imp
imp.reload(sys)

def newloadfile():
    obj_filename = './wiki_data/wiki_sentence_12381.csv' # 12381
    sub_filename = './ptt_data/ptt_data_18607.csv' #18607
    p = 0.66  # 20% of ptt_data lines
    obj = pd.read_csv(obj_filename,header = 0, skip_blank_lines=True)
    obj.dropna(how="all", inplace=True)
    sub = pd.read_csv(sub_filename, header = 0, skiprows = lambda i: i>0 and random.random() > p, index_col = None, error_bad_lines = False, skip_blank_lines=True)
    sub.dropna(how="all", inplace=True)
    cw = lambda x: ' '.join(jieba.cut(x))
    # print('======================================== ')
    # print(obj.head(5))
    # print(sub.head(5))
    # print('======================================== ')
    obj['words'] = obj['sentences'].apply(cw)
    sub['words'] = sub['sentences'].apply(cw)
    # print('======================================== ')
    # print(obj.words.head(5))
    # print(sub.words.head(5))
    # print('======================================== ')
    data_y = np.concatenate((np.ones(len(sub)), np.zeros(len(obj)))) # subjective == 1, objective == 0
    data_x = np.concatenate((sub['words'], obj['words']))
    # np.save('svm_data/baseline_data_y.npy',data_y)
    # np.save('svm_data/baseline_data_x.npy',data_x)
    return data_x, data_y


## Trian svm model
def svm_train(x_train, y_train, x_test, y_test):
    clf=SVC(kernel='rbf',verbose=True)
    clf.fit(x_train,y_train)
    # joblib.dump(clf, 'svm_data/baseline_model.pkl')         # save the trained model
    # clf_load = joblib.load('svm_data/clf.pkl')   # load the trained model
    print (clf.score(x_test,y_test))


if __name__=='__main__':
    # Load data and do word segmentation
    data_x, data_y = newloadfile()

    # Choose one of these below
    # word count transform to vector
    # count_vect = CountVectorizer(min_df=1, token_pattern='(?u)\\b\\w+\\b') #不過濾單個字 
    # count_vect = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b', min_df=1)

    # TF-IDF: "Term Frequency - Inverse Documentation Frequency" 
    # count_vect = TfidfVectorizer(min_df=1, token_pattern='(?u)\\b\\w+\\b') #不過濾單個字 
    count_vect = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b', min_df=1)  #不過率單個字 + bigram   

    # Fist fit all the data then divide into traning and testing data
    count_vect.fit(data_x)
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)
    
    x_train = count_vect.transform(x_train)
    x_test = count_vect.transform(x_test)

    svm_train(x_train, y_train, x_test, y_test)






