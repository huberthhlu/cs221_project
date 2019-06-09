#! /bin/env python
# -*- coding: utf-8 -*-
"""
训练网络，并保存模型，其中LSTM的实现采用Python中的keras库
"""
import pandas as pd 
import numpy as np 
import jieba
import multiprocessing

from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Dropout,Activation
from keras.models import model_from_yaml
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import keras
import random

np.random.seed(1337)  # For Reproducibility
import sys
sys.setrecursionlimit(1000000)
import yaml

# set parameters:
cpu_count = multiprocessing.cpu_count() # 4
vocab_dim = 250
n_iterations = 10  # ideally more..
n_exposures = 10 # 所有频数超过10的词语
window_size = 7
# n_epoch = 4
n_epoch = 1
# n_epoch = 2
input_length = 100
maxlen = 100

batch_size = 32


def loadfile():

    # 19321 # python wikiCrawler.py -l 600
    # 17329 # python pttCrawler.py -b HatePolitics -i  3850 4113
    # obj_filename = './wiki_data/wiki_sentence_59239.csv' # 59239
    # sub_filename = './ptt_data/ptt_data_45516.csv' #45516
    # p = 0.753
    # ps = 1

    obj_filename = './wiki_data/wiki_sentence_19321.csv' # 59239
    sub_filename = './ptt_data/ptt_data_17329.csv' #45516

    p = 0.8969
    ps = 1
    obj = pd.read_csv(obj_filename,header = 0, skip_blank_lines=True, skiprows = lambda i: i>0 and random.random() > p)
    obj.dropna(inplace=True)
    sub = pd.read_csv(sub_filename, header = 0, skip_blank_lines=True, skiprows = lambda i: i>0 and random.random() > ps)
    sub.dropna(inplace=True)
    print('=='*10)
    print(obj.shape[0], sub.shape[0])
    print('=='*10)
    # neg=pd.read_csv('test_data/neg.csv',header=None,index_col=None)
    # pos=pd.read_csv('test_data/pos.csv',header=None,index_col=None,error_bad_lines=False)
    # neu=pd.read_csv('test_data/neutral.csv', header=None, index_col=None)

    combined = pd.concat([obj,sub],ignore_index=None) #合并语料
    # combined = np.concatenate((sub['sentences'], obj['sentences']))
    
    
    # combined = np.concatenate((pos[0], neu[0], neg[0]))
    # y = np.concatenate((np.ones(len(sub), dtype=int), np.zeros(len(obj), dtype=int)))

    # return combined,y
    return combined['sentences'], combined['label']


#对句子经行分词，并去掉换行符
def tokenizer(text):
    ''' Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace
    '''
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text


def create_dictionaries(model=None,
                        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries
    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        # freqxiao10->0 所以k+1
        w2indx = {v: k+1 for k, v in gensim_dict.items()} #所有频数超过10的词语的索引,(k->v)=>(v->k)
        w2vec = {word: model[word] for word in w2indx.keys()} #所有频数超过10的词语的词向量, (word->model(word))

        def parse_dataset(combined): # 闭包-->临时使用
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0) # freqxiao10->0
                data.append(new_txt)
            return data # word=>index
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec,combined
    else:
        print ('No data provided...')


#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):

    model = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     iter=n_iterations)

    model.build_vocab(combined) # input: list

    model.train(combined, total_examples = vocab_dim, epochs = n_iterations)
    model.save('test_data/Word2vec_model.pkl')
    index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
    return   index_dict, word_vectors,combined


def get_data(index_dict,word_vectors,combined,y):

    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim)) # 初始化 索引为0的词语，词向量全为0
    for word, index in index_dict.items(): # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    # y_train = keras.utils.to_categorical(y_train,num_classes=2) 
    # y_test = keras.utils.to_categorical(y_test,num_classes=2)
    # print x_train.shape,y_train.shape
    return n_symbols,embedding_weights,x_train,y_train,x_test,y_test


##定义网络结构
def train_model(n_symbols,embedding_weights,x_train,y_train,x_test,y_test, model_type):
    print ('Defining a Simple Keras Model...')
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))  # Adding Input Length
    if model_type == 'LSTM':
        # model.add(LSTM(output_dim=50, activation='tanh'))
        model.add(LSTM(100))
    else:
        model.add(GRU(output_dim=50, activation='tanh'))
    # model.add(LSTM(output_dim=50, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    # model.add(Dense(1)) # Dense=>全连接层,输出维度=3
    # model.add(Activation('sigmoid'))

    print ('Compiling the Model...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=['accuracy'])


    print ('Train...') # batch_size=32
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch,verbose=1)

    print ("Evaluate...")
    score = model.evaluate(x_test, y_test, batch_size=batch_size)

    yaml_string = model.to_yaml()
    if model_type == 'LSTM':
        with open('test_data/lstm.yml', 'w') as outfile:
            outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
        model.save_weights('test_data/lstm.h5')
    else:
        with open('test_data/gru.yml', 'w') as outfile:
            outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
        model.save_weights('test_data/gru.h5')
    print ('Test score:', score)

    #################################
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
    print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    #################################


#训练模型，并保存
print ('Loading Data...')
combined, y = loadfile()
# print (len(combined),len(y))
print ('Tokenising...')
combined = tokenizer(combined)
print ('Training a Word2vec model...')
index_dict, word_vectors,combined=word2vec_train(combined)

print ('Setting up Arrays for Keras Embedding Layer...')
n_symbols,embedding_weights,x_train,y_train,x_test,y_test = get_data(index_dict, word_vectors,combined, y)
print ("x_train.shape and y_train.shape:")
print (x_train.shape,y_train.shape)
model_type = 'LSTM'
# model_type = 'GRU'
train_model(n_symbols,embedding_weights,x_train,y_train,x_test,y_test, model_type)
