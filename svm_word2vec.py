# -*- coding: utf-8 -*-
'''
File: svm_word2vec.py
---------------------
Using word2vec to
'''
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
    print('======================================== ')
    print(obj.words.head(20))
    print(sub.words.head(20))
    print('======================================== ')

    y = np.concatenate((np.ones(len(sub)), np.zeros(len(obj)))) # subjective == 1, objective == 0
    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((sub['words'], obj['words'])), y, test_size=0.2)
    # np.save('svm_data/y_train.npy',y_train)
    # np.save('svm_data/y_test.npy',y_test)
    return x_train, x_test, y_train, y_test

def buildWordVector(text, size, model):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += model[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

# given a list of words, return a dataframe showing top10 similar words in the model
def most_similar(w2v_model, words, topn=10):
    similar_df = pd.DataFrame()
    for word in words:
        try:
            similar_words = pd.DataFrame(w2v_model.wv.most_similar(word, topn=topn), columns=[word, 'cos'])
            similar_df = pd.concat([similar_df, similar_words], axis=1)
        except:
            print(word, "not found in Word2Vec model!")
    return similar_df

## Trian svm model
def svm_train(train_vecs, y_train, test_vecs, y_test, model,n_dim):
    clf=SVC(kernel='rbf',verbose=True)
    clf.fit(train_vecs,y_train)
    # joblib.dump(clf, 'svm_data/svm_w2v_model.pkl')         # save the trained model
    # clf_load = joblib.load('svm_data/clf.pkl')   # load the trained model

    print("cross validation:")
    print("====================")
    print (clf.score(test_vecs,y_test))

    data = './final_test_data/merge.csv'
    final = pd.read_csv(data, header = 0, skip_blank_lines=True)
    final.dropna(how="all", inplace=True)

    final_x = final['sentences']
    final_y_sy = final['label_shaoyuan'].astype(int)
    final_y_h = final['label_hubert'].astype(int)
    model.train(final_x, total_examples=model.corpus_count, epochs=model.iter)
    final_vecs = np.concatenate([buildWordVector(z, n_dim, model) for z in final_x])
    
    # print(final_vecs.shape)
    # print(final_vecs)
    # print(final_y_sy)
    print("predict ShaoYuan_hand_label data:")
    print("====================")
    print(clf.score(final_vecs, final_y_sy))
    print("predict Hubert_hand_label data:")
    print("====================")
    H_data = './final_test_data/hubert_hand_label.csv'
    print(clf.score(final_vecs, final_y_h))



    


if __name__=='__main__':

    # Load data and do word segmentation
    x_train, x_test, y_train, y_test = newloadfile()
    
    # calculate the vector using Word2Vec
    n_dim = 250 # people usually use n_dim form 200 to 300
    model=Word2Vec(size=n_dim, min_count=10, sg=1, iter=10)
    # test min_count if data set increase, increase iter a little bit if data are larger
    model.build_vocab(x_train, progress_per=1000)
    # Train the model  (this may take several minutes)
    model.train(x_train, total_examples=model.corpus_count, epochs=model.iter)
    train_vecs = np.concatenate([buildWordVector(z, n_dim, model) for z in x_train])
    np.save('svm_data/train_vecs.npy',train_vecs)
    print ("train_vects.shape = {}".format(train_vecs.shape))

    #Train word2vec on test tweets
    model.train(x_test,total_examples=model.corpus_count, epochs=model.iter)
    # model.save('svm_data/w2v_model.pkl')
    #Build test tweet vectors then scale
    test_vecs = np.concatenate([buildWordVector(z, n_dim,model) for z in x_test])
    #test_vecs = scale(test_vecs)
    np.save('svm_data/test_vecs.npy',test_vecs)
    print ("test_vects.shape = {}".format(test_vecs.shape))

    similar_df = most_similar(model, ['中華','民國', '中國','大陸','根據','政府', '柯文', '行政院', '只要', '我','明天', '同婚', '就', '這邊', '小英', '明年','柯黑','台灣','法律','選舉','《','草包','韓國瑜'])
    similar_df.to_csv('similar_words_Skip-gram.csv')

    svm_train(train_vecs, y_train, test_vecs, y_test, model, n_dim)




