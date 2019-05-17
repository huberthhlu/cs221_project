# -*- coding: utf-8 -*-
"""
28027 subjecctive sentences from PTT HatePolitics board
	  from page 4100 to page 4166
	  sentences_Data_Ptt_large.csv 

	  https://www.ptt.cc/bbs/HatePolitics/index4100.html
	  ...
	  https://www.ptt.cc/bbs/HatePolitics/index4166.html


886 subjecctive sentences from Wiki based on the keyword "臺灣政治"
wiki_sentence_small.csv

19219 subjecctive sentences from Wiki based on the keyword "臺灣政治" 	"-l 150"
wiki_sentence_medium.csv
"""

# import os
# import pandas as pd
# import random
# import numpy as np
# import jieba
# from gensim.models.word2vec import Word2Vec
# from gensim.corpora.dictionary import Dictionary
# from keras.preprocessing import sequence
# import multiprocessing
# # from sklearn.cross_validation import train_test_split
# from sklearn.model_selection import train_test_split

# from keras.models import Sequential
# from keras.layers.embeddings import Embedding
# # from keras.layers.recurrent import LSTM
# # from keras.layers.core import Dense, Dropout,Activation
# # from keras.models import model_from_yaml
# import sys
# # import yaml
# import keras

from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import jieba
import random
from sklearn.externals import joblib
from sklearn.svm import SVC


import sys
import imp
imp.reload(sys)








# combined = np.concatenate((obj, sub))
# print(combined.shape)

# # decoder = np.vectorize(lambda x: x.decode('UTF-8'))
# # combined = decoder(combined)
# # combined = [el.decode('UTF-8') for el in combined]
# # combined = os.linesep.join([s for s in combined.splitlines() if s])

# # obj data are labeled as "1" and sub data are labeled as "-1"
# y = np.concatenate((np.ones(len(obj), dtype=int), -1*np.ones(len(sub),dtype=int)))
# print(y.shape)


############################################################def loadfile():
# def loadfile(combined):
#     cw = lambda x: list(jieba.cut(x))
    
#     sentences = []
#     for i in range(len(combined)):
#     	sentence = str(combined[i])
#     	# snetence.encode('utf-8')
#     	sentence = jieba.cut(sentence)
#     	# print("Full Mode: " + "/ ".join(sentence))  # 全模式
#     	sentences.append(sentence)
#     combined = sentences
#     # pos['words'] = pos[0].apply(cw)
#     # neg['words'] = neg[0].apply(cw)
#     x_train, x_test, y_train, y_test = train_test_split((combined), y, test_size=0.2)
#     # np.save('./y_train.npy',y_train)
#     # np.save('./y_test.npy',y_test)
#     return x_train,x_test, y_train, y_test



# 加载文件，导入数据,分词
def newloadfile():
	obj_filename = './sentences_Data_Ptt_large.csv'
	sub_filename = './wiki_sentence_medium.csv'
	p = 0.68  # 68% of the lines (19219/28027)
	# keep the header, then take only 3.16% of lines
	# if random from [0,1] interval is greater than 0.01 the row will be skipped
	# obj = pd.read_csv(obj_filename, header = 0, skiprows = lambda i: i>0 and random.random() > p)
	# obj = pd.read_csv(obj_filename, usecols=['sentences'],header = 0, skiprows = lambda i: i>0 and random.random() > p, skip_blank_lines=True)
	obj = pd.read_csv(obj_filename,header = 0, skiprows = lambda i: i>0 and random.random() > p, skip_blank_lines=True)
	obj.dropna(how="all", inplace=True)
	# sub = pd.read_csv(sub_filename, usecols=['sentences'], header = 0, index_col = None, error_bad_lines = False, skip_blank_lines=True)
	sub = pd.read_csv(sub_filename, header = 0, index_col = None, error_bad_lines = False, skip_blank_lines=True)
	sub.dropna(how="all", inplace=True)
	cw = lambda x: list(jieba.cut(x))
	print('==== ',obj.dtypes)
	# obj['words'] = obj[2].apply(cw)
	# sub['words'] = sub[2].apply(cw)




	# df = pd.read_csv('data/RBefh.csv', dtype=str)
	# keys = list(df['to_search'].dropna())
	# values = list(df['value_to_copy'].dropna())
	# map_values = dict(zip(keys, values))
	# mapper = df.to_replace.isin(map_values)
	# df.loc[mapper, 'to_replace'] = df.loc[mapper, 'to_replace'].apply(lambda row: map_values[row])
	# df.fillna('', inplace=True)
	# #print pos['words']
	# #use 1 for objective sentences, 0 for subjective sentences
	# y = np.concatenate((np.ones(len(obj)), np.zeros(len(sub))))
	# x_train, x_test, y_train, y_test = train_test_split(np.concatenate((obj['words'], sub['words'])), y, test_size=0.2)
	# # np.save('../y_train.npy',y_train)
	# # np.save('../y_test.npy',y_test)
	# return x_train,x_test



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
    

#计算词向量
def get_train_vecs(x_train,x_test):
    n_dim = 300
    #Initialize model and build vocab
    imdb_w2v = Word2Vec(size=n_dim, min_count=10)
    imdb_w2v.build_vocab(x_train)
    
    #Train the model over train_reviews (this may take several minutes)
    imdb_w2v.train(x_train)
    
    train_vecs = np.concatenate([buildWordVector(z, n_dim,imdb_w2v) for z in x_train])
    #train_vecs = scale(train_vecs)
    
    np.save('./train_vecs.npy',train_vecs)
    print (train_vecs.shape)
    #Train word2vec on test tweets
    imdb_w2v.train(x_test)
    imdb_w2v.save('./w2v_model.pkl')
    #Build test tweet vectors then scale
    test_vecs = np.concatenate([buildWordVector(z, n_dim,imdb_w2v) for z in x_test])
    #test_vecs = scale(test_vecs)
    np.save('./test_vecs.npy',test_vecs)
    print (test_vecs.shape)
    return train_vecs, test_vecs



##训练svm模型
def svm_train(train_vecs, y_train, test_vecs, y_test):
    clf=SVC(kernel='rbf',verbose=True)
    clf.fit(train_vecs,y_train)
    # joblib.dump(clf, './model.pkl')
    print (clf.score(test_vecs,y_test))


# x_train, x_test, y_train, y_test = newloadfile()
newloadfile()
# train_vecs, test_vecs = get_train_vecs(x_train, x_test)
# svm_train(train_vecs, y_train, test_vecs, y_test)