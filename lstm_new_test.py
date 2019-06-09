#! /bin/env python
# -*- coding: utf-8 -*-
"""
预测
"""
import jieba
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
import yaml
from keras.models import model_from_yaml
import sys
from keras import backend as K
import keras
import os
import random
np.random.seed(1337)  # For Reproducibility

sys.setrecursionlimit(1000000)

# define parameters
maxlen = 100

os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
        #  freqxiao10->0 所以k+1
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引,(k->v)=>(v->k)
        w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量, (word->model(word))

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


def input_transform(string):
    words=jieba.lcut(string)
    words=np.array(words).reshape(1,-1)
    model=Word2Vec.load('test_data/Word2vec_model.pkl')
    _,_,combined=create_dictionaries(model,words)
    return combined

def lstm_predict(model_type):
# def lstm_predict(string):
    # print ('loading model......')
    if model_type == 'LSTM':
        with open('test_data/lstm.yml', 'r') as f:
            yaml_string = yaml.load(f)
        model = model_from_yaml(yaml_string)
        model.load_weights('test_data/lstm.h5')
    else:
        with open('test_data/gru.yml', 'r') as f:
            yaml_string = yaml.load(f)
        model = model_from_yaml(yaml_string)
        model.load_weights('test_data/gru.h5')
    

    # print ('loading weights......')
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    # data=input_transform(string)
    # data.reshape(1,-1)
    # #print data
    # result=model.predict_classes(data)
    # # print result # [[1]]

    # if result[0]==1:
    #     print (string,' subjective')
    # elif result[0]==0:
    #     print (string,' objective')
    # else:
    #     print (string,' ERROR')

    ###########
    p = 0.02
    # p = 1.0
    data1 = pd.read_csv('./final_test_data/merge1.csv', header = 0, index_col = None, error_bad_lines = False, skip_blank_lines=True, skiprows = lambda i: i>0 and random.random() > p)
    data2 = pd.read_csv('./final_test_data/merge2.csv', header = 0, index_col = None, error_bad_lines = False, skip_blank_lines=True, skiprows = lambda i: i>0 and random.random() > p)
    # cw = lambda x: model.predict_classes(input_transform(x).reshape(1,-1))[0]


    cw = lambda x: model.predict_classes(input_transform(x))[0] > 0.5
    # cw = lambda x: model.predict(input_transform(x))

    
    data1['predict'] = data1['sentences'].apply(cw)
    data2['predict'] = data2['sentences'].apply(cw)
    # output = model.predict_classes(input_transform(data1['sentences'][0]))
    
    data1['final'] = data1['predict'] == data1['label_hubert']
    # data1['final'] = data1['predict'] == data1['label_hubert']
    for index, row in data1.iterrows():
        print(row['predict'], ' ? ', row['label_hubert'], ' = ', row['final'], '\n')

    # hubert1_acc = sum(data1['predict'] == data1['label_hubert'])/data1.shape[0]
    # print(hubert1_acc)
    


    hubert1_acc = sum(data1['predict'] == data1['label_hubert'])/data1.shape[0]
    ShaoYuan1_acc = sum(data1['predict'] == data1['label_shaoyuan'])/data1.shape[0]

    hubert2_acc = sum(data2['predict'] == data2['label_hubert'])/data2.shape[0]
    ShaoYuan2_acc = sum(data2['predict'] == data2['label_shaoyuan'])/data2.shape[0]
    print('Accuracy based on Hubert and ShaoYuan hand-label data 1 is ',hubert1_acc, ' & ',ShaoYuan1_acc)
    print('Accuracy based on Hubert and ShaoYuan hand-label data 2 is ',hubert2_acc, ' & ',ShaoYuan2_acc)
    
    


if __name__=='__main__':
    # string='酒店的环境非常好，价格也便宜，值得推荐'
    # string='手机质量太差了，傻逼店家，赚黑心钱，以后再也不会买了'
    # string = "这是我看过文字写得很糟糕的书，因为买了，还是耐着性子看完了，但是总体来说不好，文字、内容、结构都不好"
    # string = "虽说是职场指导书，但是写的有点干涩，我读一半就看不下去了！"
    # string = "书的质量还好，但是内容实在没意思。本以为会侧重心理方面的分析，但实际上是婚外恋内容。"
    # string = "不是太好"
    # string = "不错不错"
    # string = "真的一般，没什么可以学习的"
    # lstm_predict('GRU')
    lstm_predict('LSTM')
    # lstm_predict(string)
    #########################
    

    
