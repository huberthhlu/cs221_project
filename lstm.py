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





imp.reload(sys)

def newloadfile():
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


## Trian LSTM model
def LSTM_train(train_vecs, y_train, test_vecs, y_test, word_vectors):
    model = Sequential()
    model.add(Embedding(len(word_vectors.vocab)+1, 256))
    model.add(LSTM(128)) # try using a GRU instead, for fun
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_vecs, y_train, batch_size=16, nb_epoch=10)  #训练时间为若干个小时
    # model.fit(train_vecs, y_train, batch_size=16, nb_epoch=10)  #训练时间为若干个小时
    classes = model.predict_classes(test_vecs)
    acc = np_utils.accuracy(classes, y_test)
    print('Test accuracy:', acc)
    # print (model.score(test_vllecs,y_test))
    score = model.evaluate(test_vecs, y_test, batch_size=batch_size)
    print ('Test score:', score)

## Trian svm model
def svm_train(train_vecs, y_train, test_vecs, y_test):
    clf=SVC(kernel='rbf',verbose=True)
    clf.fit(train_vecs,y_train)
    # joblib.dump(clf, 'svm_data/svm_w2v_model.pkl')         # save the trained model
    # clf_load = joblib.load('svm_data/clf.pkl')   # load the trained model
    print (clf.score(test_vecs,y_test))

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

    # print(model.wv[])
    LSTM_train(train_vecs, y_train, test_vecs, y_test, model.wv)





############################################################
# def segment_text(source_corpus, train_corpus, coding, punctuation):
#     '''
#     切词,去除标点符号
#     :param source_corpus: 原始语料
#     :param train_corpus: 切词语料
#     :param coding: 文件编码
#     :param punctuation: 去除的标点符号
#     :return:
#     '''
#     with open(source_corpus, 'r', encoding=coding) as f, open(train_corpus, 'w', encoding=coding) as w:
#         for line in f:
#             # 去除标点符号
#             line = re.sub('[{0}]+'.format(punctuation), '', line.strip())
#             # 切词
#             words = jieba.cut(line)
#             w.write(' '.join(words))
# # 严格限制标点符号
# strict_punctuation = '。，、＇：∶；?‘’“”〝〞ˆˇ﹕︰﹔﹖﹑·¨….¸;！´？！～—ˉ｜‖＂〃｀@﹫¡¿﹏﹋﹌︴々﹟#﹩$﹠&﹪%*﹡﹢﹦﹤‐￣¯―﹨ˆ˜﹍﹎+=<­­＿_-\ˇ~﹉﹊（）〈〉‹›﹛﹜『』〖〗［］《》〔〕{}「」【】︵︷︿︹︽_﹁﹃︻︶︸﹀︺︾ˉ﹂﹄︼'
# # 简单限制标点符号
# simple_punctuation = '’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
# # 去除标点符号
# punctuation = simple_punctuation + strict_punctuation

# segment_text(source_corpus_text, train_corpus_text, coding, punctuation)

############################################################
# cpu_count = multiprocessing.cpu_count() # 4
# vocab_dim = 100
# n_iterations = 10  # ideally more..
# n_exposures = 1 # 所有频数超过10的词语
# window_size = 7
# n_epoch = 4
# input_length = 100
# maxlen = 100

# def create_dictionaries(model=None,
#                         combined=None):
#     ''' Function does are number of Jobs:
#         1- Creates a word to index mapping
#         2- Creates a word to vector mapping
#         3- Transforms the Training and Testing Dictionaries

#     '''
#     if (combined is not None) and (model is not None):
#         gensim_dict = Dictionary()        
#         # gensim_dict.doc2bow(model.vocab.keys(),
#         #                     allow_update=True)
#         gensim_dict.doc2bow(model.wv.vocab.keys(),
#                             allow_update=True)
#         w2indx = {v: k+1 for k, v in gensim_dict.items()} #所有频数超过10的词语的索引
#         w2vec = {word: model[word] for word in w2indx.keys()} #所有频数超过10的词语的词向量

#         def parse_dataset(combined):
#             ''' Words become integers
#             '''
#             data=[]
#             for sentence in combined:
#                 new_txt = []
#                 for word in sentence:
#                     try:
#                         new_txt.append(w2indx[word])
#                     except:
#                         new_txt.append(0)
#                 data.append(new_txt)
#             return data
#         combined=parse_dataset(combined)
#         combined= sequence.pad_sequences(combined, maxlen=maxlen) #每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
#         return w2indx, w2vec,combined
#     else:
#         print ('No data provided...')

# #创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
# def word2vec_train(combined):

#     model = Word2Vec(size=vocab_dim,
#                      min_count=n_exposures,
#                      window=window_size,
#                      workers=cpu_count,
#                      iter=n_iterations)
#     model.build_vocab(combined)
#     # model.train(combined)
#     model.train(combined,total_examples=len(combined), epochs=n_iterations)
#     model.save('./Word2vec_model.pkl')
#     index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
#     return   index_dict, word_vectors,combined

# print ('Training a Word2vec model...')
# index_dict, word_vectors,combined = word2vec_train(combined)


# ###############################################################################################
# ###############################################################################################
# ###############################################################################################

# np.random.seed(1337)  # For Reproducibility
# sys.setrecursionlimit(1000000)
# batch_size = 32


# def get_data(index_dict,word_vectors,combined,y):
# 	print('=='*10,combined.shape)
# 	print('=='*10,len(y))
# 	print('=='*10,index_dict)

# 	n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
# 	embedding_weights = np.zeros((n_symbols, vocab_dim)) # 初始化 索引为0的词语，词向量全为0
# 	for word, index in index_dict.items(): # 从索引为1的词语开始，对每个词语对应其词向量
# 		embedding_weights[index, :] = word_vectors[word]
# 	x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
# 	y_train = keras.utils.to_categorical(y_train,num_classes=2) 
# 	y_test = keras.utils.to_categorical(y_test,num_classes=2)
# 	# print x_train.shape,y_train.shape
# 	return n_symbols,embedding_weights,x_train,y_train,x_test,y_test


# ##定义网络结构
# def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
#     print ('Defining a Simple Keras Model...')
#     model = Sequential()  # or Graph or whatever
#     model.add(Embedding(output_dim=vocab_dim,
#                         input_dim=n_symbols,
#                         mask_zero=True,
#                         weights=[embedding_weights],
#                         input_length=input_length))  # Adding Input Length
#     model.add(LSTM(output_dim=50, activation='tanh', inner_activation='hard_sigmoid'))
#     model.add(Dropout(0.5))
#     model.add(Dense(3, activation='softmax')) # Dense=>全连接层,输出维度=1
#     model.add(Activation('softmax'))

#     print('Compiling the Model...')
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adam',metrics=['accuracy'])

#     print("Train...") # batch_size=32
#     model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch,verbose=1)

#     print ("Evaluate...")
#     score = model.evaluate(x_test, y_test,
#                                 batch_size=batch_size)

#     yaml_string = model.to_yaml()
#     with open('../model/lstm.yml', 'w') as outfile:
#         outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
#     model.save_weights('../model/lstm.h5')
#     print ('Test score:', score)

# print ('Setting up Arrays for Keras Embedding Layer...')
# n_symbols,embedding_weights,x_train,y_train,x_test,y_test = get_data(index_dict, word_vectors,combined,y)
# print ("x_train.shape and y_train.shape:")
# print (x_train.shape,y_train.shape)
# train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test)


# """
# Requirement:

# https://www.anaconda.com/tensorflow-in-anaconda/

# conda activate tensorflow_env
# conda deactivate

# """