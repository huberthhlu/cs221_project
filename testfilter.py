# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import os
import re
import sys
import json
import csv
import requests
import argparse
import time
import codecs
from bs4 import BeautifulSoup
from six import u
from filter_text import Filter_Text  

new3_content = '台灣總統蔡英文22日簽署立法院日前三讀通過的《司法院釋字第748號解釋施行法》，而在17日立院通過該法的當天，除了英國外交大臣杭特、加拿大外交部均推文祝賀台灣人民，法國常駐聯合國副代表古根也在聯合國國際反恐同日慶祝活動中，恭賀台灣LGBTI伴侶終於有權依法結婚。此外，法國人權大使克羅凱特也說，台灣同婚合法化是國際反恐同日最美麗的象徵。每年5月17日是國際反恐同日，因為世界衛生組織（WHO）在1990年的這天，正式把同性戀傾向從疾病分類表中除名，自此同性戀不再是心理或精神疾病。聯合國也在這天舉行慶祝活動「正義與保障人人平等」，法國駐聯合國副代表古根公開讚揚台灣同婚合法化。「每個往正確方向踏出一步的時刻，我們都應同感歡欣鼓舞」古根表示，「從這個角度來看，我要祝賀所有台灣LGBTI伴侶，從今（5月17日）開始他們有權依法結婚了」。此外，2019年2月曾訪台的克羅凱特（François Croquette）也在17日貼文祝賀：「今天，台灣向前邁進一大步，賦予所有人結婚權利，這是國際反恐同日最美麗的象徵。」台灣成為亞洲首個同婚合法化的國家引起國際關注，英國外相杭特（Jeremy Hunt）推文稱：「恭喜台灣人民通過同婚！這是台灣LGBT社群的大好消息，也是亞洲LGBT平權向前邁進的一大步。」加拿大外交部也推文寫道：「祝福台灣人民，亞洲同婚合法化首例。」奧地利台北辦事處處長陸德飛（Roland Rudorfer）也說，此舉確立台灣在促進與保護人權的領先角色。'

newfilename = 'test_filter'
f = csv.writer(open(newfilename, "w"))
f.writerow(["label","id", "title", "sentences"])

def have_start_quote(s):
	q = {u'「': u'」', 
		u'（': u'）',
		u'(':u')',
		u'[':u']',
		u'{':u'}',  
		u'｢': u'｣', 
		u'『': u'』',
		u'【':u'】',
		u'〔':u'〕',
		u'〖':u'〗',
		u'〘': u'〙',
		u'〚':u'〛',
		u'《':u'》'}
	startlist = [u'（',u'(',u'[',u'{',u'【',u'〔',u'〖',u'〘',u'〚',u'《',u'｢',u'『',u'「']
	# endlist =   [u'」',u'）'u')',u']',u'}',u'｣',u'』',u'】',u'〕',u'〗',u'〙',u'〛',u'》']
	tmp = 'N'
	for c in startlist:
		if c in s:
			# print(c)
			tmp = q[c]
			if tmp in s:
				# print(tmp)
				tmp = 'N'

	if not tmp=='N':
		return True, tmp

	return False, None

def checkend(sentence, symbol):
	if symbol in sentence:
		return True


sentences = re.split(u'(，|。|,| )', new3_content)

quote = False
S = sentences[0]
for i in range(1,len(sentences)):
	sentences[i] = Filter_Text().filtet_text(sentences[i]) # emoji filter!!!!
	if quote:
		flag = checkend(sentences[i], symbol)
		if flag:
			quote = False

	if not quote:
		start, symbol = have_start_quote(sentences[i])
		if start:
			print(sentences[i])
			print(sentences[i+1])
			quote = True

	if len(re.sub("^[0-9]*$", '', sentences[i])) == 0:
		continue
	if len(sentences[i]) < 10 or quote:
		S += sentences[i]
		if i == len(sentences):
			f.writerow([1 ,0, 'None', S])
			S = ''
	else:
		f.writerow([1 ,0, 'None', S]) # 1 means label this sentence as 1(subjective)
		S = sentences[i]
	# ID += 1





