# -*- coding: utf-8 -*-

'''
File_name: multinomial_NB.py:
---------------------------
Use the bag-of-words to train a multinomial Naive Bayes model.

1. Without using TF-IDF uni-gram: 0.9443
2. Without using TF-IDF bi-gram: 0.9538
3. Using TF-IDF uni-gram: 0.9486
4. Using TF-IDF bi-gram: 0.9520

'''
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
import jieba
import random
import joblib

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
def MNB_train(x_train, y_train, x_test, y_test):
    clf=MultinomialNB()
    clf.fit(x_train,y_train)
    joblib.dump(clf, 'MNB_data/MNB_model.pkl')         # save the trained model
    # clf_load = joblib.load('MNB_data/MNB_model.pkl')   # load the trained model
    # print(x_test.shape)
    # print(y_test.shape)
    print (clf.score(x_test,y_test))

def MNB_finaltest():

    # Load final test data and pre-process
    data = './final_test_data/merge.csv'
    final = pd.read_csv(data, header = 0, skip_blank_lines=True)
    final.dropna(how="all", inplace=True)
    cw = lambda x: ' '.join(jieba.cut(x))
    final['words'] = final['sentences'].apply(cw)

    # final_test_x and final_test_y 
    count_vect = joblib.load('MNB_data/count_vect_model.pkl')
    final_x = count_vect.transform(final['words'])
    final_y_sy = final['label_shaoyuan'].astype(int)
    final_y_h = final['label_hubert'].astype(int)
    
    # print(final_x.shape)
    # print(final_y_sy.shape)
    # print(final_y_h.shape)

    clf = joblib.load('MNB_data/MNB_model.pkl')
    print("predict ShaoYuan_hand_label data:")
    print("====================")
    print(clf.score(final_x, final_y_sy))
    print("predict Hubert_hand_label data:")
    print("====================")
    print(clf.score(final_x, final_y_h))

if __name__=='__main__':
    # Load data and do word segmentation
    data_x, data_y = newloadfile()

    # Choose one of these below
    # word count transform to vector
    # count_vect = CountVectorizer(min_df=1, token_pattern='(?u)\\b\\w+\\b') #不過濾單個字 
    count_vect = CountVectorizer(ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b', min_df=1)

    # TF-IDF: "Term Frequency - Inverse Documentation Frequency" 
    # count_vect = TfidfVectorizer(min_df=1, token_pattern='(?u)\\b\\w+\\b') #不過濾單個字 
    # count_vect = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b', min_df=1)  #不過率單個字 + bigram   

    # Fist fit all the data then divide into traning and testing data
    count_vect.fit(data_x)
    joblib.dump(count_vect, 'MNB_data/count_vect_model.pkl')
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)
    
    x_train = count_vect.transform(x_train)
    x_test = count_vect.transform(x_test)

    MNB_train(x_train, y_train, x_test, y_test)
    MNB_finaltest()

