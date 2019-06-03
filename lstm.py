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

import os
import pandas as pd
import random
import numpy as np
import jieba
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
import multiprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
"""
TF-IDF: “Term Frequency – Inverse Document”

One issue with simple counts is that some words like “the” 
will appear many times and their large counts will not be very meaningful in the encoded vectors.
Without going into the math, TF-IDF are word frequency scores that try to highlight words that are more interesting, 
    e.g. frequent in a document but not across documents.


"""
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout,Activation
from keras.models import model_from_yaml
import sys
import yaml
import keras
import imp

from xlsxwriter.workbook import Workbook
import glob
import csv


imp.reload(sys)


obj_filename = './wiki_data/wiki_sentence_12381.csv' # 12381
sub_filename = './ptt_data/ptt_data_18607.csv' #18607
p = 0.66  # 20% of ptt_data lines
obj = pd.read_csv(obj_filename,header = 0, skip_blank_lines=True)
obj.dropna(how="all", inplace=True)
sub = pd.read_csv(sub_filename, header = 0, skiprows = lambda i: i>0 and random.random() > p, index_col = None, error_bad_lines = False, skip_blank_lines=True)
sub.dropna(how="all", inplace=True)
cw = lambda x: list(jieba.cut(x))
obj['words'] = obj['sentences'].apply(cw)
sub['words'] = sub['sentences'].apply(cw)

pn=pd.concat([obj,sub],ignore_index=None) #合并语料

d2v_train = pn['words']


w = [] #将所有词语整合在一起
for i in d2v_train:
    w.extend(i)

dict = pd.DataFrame(pd.Series(w).value_counts()) #统计词的出现次数
del w,d2v_train
dict['id']=list(range(1,len(dict)+1))

get_sent = lambda x: list(dict['id'][x])
pn['sentences'] = pn['words'].apply(get_sent) #速度太慢

maxlen = 50

print("Pad sequences (samples x time)")
pn['sentences'] = list(sequence.pad_sequences(pn['sentences'], maxlen=maxlen))

x = np.array(list(pn['sentences']))[::2] #训练集
y = np.array(list(pn['label']))[::2]
xt = np.array(list(pn['sentences']))[1::2] #测试集
yt = np.array(list(pn['label']))[1::2]
xa = np.array(list(pn['sentences'])) #全集
ya = np.array(list(pn['label']))

print('Build model...')
model = Sequential()
model.add(Embedding(len(dict)+1, 256))
model.add(LSTM(128)) # try using a GRU instead, for fun
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x, y, batch_size=16, nb_epoch=5) 

classes = model.predict_classes(xt)
acc = np_utils.accuracy(classes, yt)
print('Test accuracy:', acc)


######## For test hand-labeling data
new_filename = './xxx.csv' # 12381
 
final = pd.read_csv(new_filename,header = 0, skip_blank_lines=True)
final.dropna(how="all", inplace=True)
cw = lambda x: list(jieba.cut(x))
final['words'] = final['sentences'].apply(cw)
classes = model.predict_classes(xt)
final_acc = np_utils.accuracy(classes, obj['label'])
print('Final Accuracy',final_acc)
############################################################


# """
# Requirement:

# https://www.anaconda.com/tensorflow-in-anaconda/

# conda activate tensorflow_env
# conda deactivate

# """