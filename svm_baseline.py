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
# from gensim.models.word2vec import Word2Vec
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
    obj_filename = './wiki_data/wiki_sentence_59239.csv' # 59239
    sub_filename = './ptt_data/ptt_data_45516.csv' #45516
    p = 0.768 
    obj = pd.read_csv(obj_filename,header = 0, skip_blank_lines=True, skiprows = lambda i: i>0 and random.random() > p)
    obj.dropna(inplace=True)
    sub = pd.read_csv(sub_filename, header = 0,  index_col = None, error_bad_lines = False, skip_blank_lines=True)
    sub.dropna(inplace=True)
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
    joblib.dump(clf, 'svm_data/baseline_model.pkl')         # save the trained model
    # clf_load = joblib.load('svm_data/baseline_model.pkl')   # load the trained model

    print("cross validation:")
    print("====================")
    print (clf.score(x_test,y_test))

def svm_finaltest():

    # Load final test data and pre-process
    data = './final_test_data/merge.csv'
    final = pd.read_csv(data, header = 0, skip_blank_lines=True)
    final.dropna(how="all", inplace=True)
    cw = lambda x: ' '.join(jieba.cut(x))
    final['words'] = final['sentences'].apply(cw)

    # final_test_x and final_test_y 
    count_vect = joblib.load('svm_data/count_vect_model_svm.pkl')
    final_x = count_vect.transform(final['words'])
    final_y_sy = final['label_shaoyuan'].astype(int)
    final_y_h = final['label_hubert'].astype(int)
    
    # print(final_x.shape)
    # print(final_y_sy.shape)
    # print(final_y_h.shape)

    clf = joblib.load('svm_data/baseline_model.pkl')
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
    # count_vect = CountVectorizer(ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b', min_df=1)

    # TF-IDF: "Term Frequency - Inverse Documentation Frequency" 
    # count_vect = TfidfVectorizer(min_df=1, token_pattern='(?u)\\b\\w+\\b') #不過濾單個字 
    count_vect = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b', min_df=1)  #不過率單個字 + bigram   

    # Fist fit all the data then divide into traning and testing data
    count_vect.fit(data_x)
    joblib.dump(count_vect, 'svm_data/count_vect_model_svm.pkl')
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)
    
    x_train = count_vect.transform(x_train)
    x_test = count_vect.transform(x_test)

    svm_train(x_train, y_train, x_test, y_test)
    svm_finaltest()






