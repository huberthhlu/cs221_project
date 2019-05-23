# -*- coding: utf-8 -*-
"""
'./ptt_data/ptt_data_18607.csv': 
    18607 subjecctive sentences from PTT HatePolitics board
'./wiki_data/wiki_sentence_12381.csv': 
    12381 objective sentences from Wiki based on the keyword "臺灣政治"

"""
from sklearn.model_selection import train_test_split
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


# load file, word segmentation 
def newloadfile():
    obj_filename = './wiki_data/wiki_sentence_12381.csv' # 12381
    sub_filename = './ptt_data/ptt_data_18607.csv' #18607
    p = 0.66  # 20% of ptt_data lines
    obj = pd.read_csv(obj_filename,header = 0, skip_blank_lines=True)
    obj.dropna(how="all", inplace=True)
    sub = pd.read_csv(sub_filename, header = 0, skiprows = lambda i: i>0 and random.random() > p, index_col = None, error_bad_lines = False, skip_blank_lines=True)
    sub.dropna(how="all", inplace=True)
    cw = lambda x: list(jieba.cut(x))
    # print('======================================== ')
    # print(obj.head(5))
    # print(sub.head(5))
    # print('======================================== ')
    obj['words'] = obj['sentences'].apply(cw)
    sub['words'] = sub['sentences'].apply(cw)

    y = np.concatenate((np.ones(len(sub)), np.zeros(len(obj)))) # subjective == 1, objective == 0
    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((sub['words'], obj['words'])), y, test_size=0.2)
    np.save('svm_data/y_train.npy',y_train)
    np.save('svm_data/y_test.npy',y_test)
    return x_train, x_test, y_train, y_test

def buildWordVector(text, size,imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec
    
## Trian svm model
def svm_train(train_vecs, y_train, test_vecs, y_test):
    clf=SVC(kernel='rbf',verbose=True)
    clf.fit(train_vecs,y_train)
    joblib.dump(clf, 'svm_data/clf.pkl')         # save the trained model
    # clf_load = joblib.load('svm_data/clf.pkl')   # load the trained model
    print (clf.score(test_vecs,y_test))

if __name__=='__main__':

    # Load data and do word segmentation
    x_train, x_test, y_train, y_test = newloadfile()
    
    # calculate the vector using Word2Vec
    n_dim = 300 # people usually use n_dim form 200 to 300
    model = Word2Vec(size=n_dim, min_count=5)
    model.build_vocab(x_train, progress_per=1000)
    # Train the model  (this may take several minutes)
    model.train(x_train, total_examples=model.corpus_count, epochs=model.iter)
    train_vecs = np.concatenate([buildWordVector(z, n_dim, model) for z in x_train])
    np.save('svm_data/train_vecs.npy',train_vecs)
    print ("train_vects.shape = {}".format(train_vecs.shape))

    #Train word2vec on test tweets
    model.train(x_test,total_examples=model.corpus_count, epochs=model.iter)
    model.save('svm_data/w2v_model.pkl')
    #Build test tweet vectors then scale
    test_vecs = np.concatenate([buildWordVector(z, n_dim,model) for z in x_test])
    #test_vecs = scale(test_vecs)
    np.save('svm_data/test_vecs.npy',test_vecs)
    print ("test_vects.shape = {}".format(test_vecs.shape))


    svm_train(train_vecs, y_train, test_vecs, y_test)



