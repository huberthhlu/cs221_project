# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

# import os
# import re
# import sys
# import codecs
# import logging
# from hanziconv import HanziConv
# import csv
# import argparse


# logging.basicConfig(level=logging.INFO)



import os
import re
import sys
import codecs
from bs4 import BeautifulSoup
from six import u
import logging
import wikipediaapi
from hanziconv import HanziConv
import csv
import argparse
import xlrd 
import zhon.hanzi
logging.basicConfig(level=logging.INFO)



def parseArticles(mylist):
	loc = "testset_raw.xlsx"
	wb = xlrd.open_workbook(loc) 
	sheet = wb.sheet_by_index(0) 
	for i in range(sheet.nrows): 
		print(sheet.cell_value(i, 1)) 

		main_content = sheet.cell_value(i, 1)
		# 移除 '※ 發信站:' (starts with u'\u203b'), '◆ From:' (starts with u'\u25c6'), 空行及多餘空白
		# 保留英數字, 中文及中文標點, 網址, 部分特殊符號
		expr = re.compile(u(r'[^\u4e00-\u9fa5\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b\s\w:/-_.?~%()]'))
		main_content = re.sub(expr, '', main_content)
		main_content = re.sub("[A-Za-z]", '', main_content)
		main_content = re.sub('\[.+\]', '', main_content) # to strip out [???] from a string 
		main_content = re.sub('\([^)]*\)', '', main_content) # to strip out <p> and </p> from a string 
		clean = re.compile('<.*?>') # to strip out <p> and </p> from a string 
		main_content = re.sub(clean, ' ', main_content)
		main_content = re.sub(r'(\s)+', ' ', main_content)
		main_content = re.sub(r'http\S+', '', main_content) ###			
		mylist.append((i, main_content, sheet.cell_value(i, 0)))

def sentenceGenerator(mylist):
	newfilename = "wiki_sentence.csv"
	f = csv.writer(open(newfilename, "w"))
	f.writerow(["label", "id", "source", "sentences"]) # Notice that there is no title for the test set!!!!!!!
	id1 = 0

	"""
	Priliminary thought: first use regular expression to scan through the entire article.
	Using the dictionary for signal words of quotation sentences to divide sentences apart.
	Then, we use the orginal rule to divide the remaining sentences.

	"""
	signalDict = ['説','條文','認為','表示','指出','強調','：','透露','喊','驚呼','稱','嘆','回應']

	for row in mylist:
		label = 2 
		"""
		objective == 0
		subjective == 1
		unknown == 2
		"""
		id_article = row[0]
		content = row[1]
		source = row[2]




		# punc = "！？｡。，：；､、"
		# punc.decode('utf-8')
		# , 'utf-8')
		# punc = punc.decode("utf-8")
		# line = "测试。。去除标点。。，、！"
		# print re.sub(ur"[%s]+" %punc, "", line.decode("utf-8"))


		# sentences = re.split('，|。| |!|！|?|？', content) # how about the symbol '、'?????
		# sentences = re.split('，|。| |!|"！"|?|"？"', content) # how about the symbol '、'?????
		# sentences = re.split('，|。| |!|?', content) # how about the symbol '、'?????
		# sentences = re.split('，|。| |!|?'+punc, content) # how about the symbol '、'?????
		# sentences = re.split('，|。| |!|?'+punc, content) # how about the symbol '、'?????

		sentences = re.findall(zhon.hanzi.sentence, content)
		for k in sentences:
			print('==='*8, k)
		print('==='*8, sentences[0])
		# first merge based on the quotation mark "「」"
		i = 0
		quotationFound = False
		while i < range(sentences):
			if sentences[i].contains(u'「'):
				quotationFound = True
				i += 1
				continue
			if quotationFound:
				if sentences[i].contains(u'」'):
					quotationFound = False
				sentences[i:i+1] = [''.join(sentences[i:i+1])]
				

		for i in range(len(sentences)):
			if len(sentences[i]) < 7 and i == len(sentences) - 1:
				continue
			elif len(sentences[i]) < 7:
				sentences[i + 1] = sentences[i] + sentences[i + 1]
			else:
				f.writerow([label, id_article, id1, sentences[i], source])
				id1 += 1


		# for i in range(sentences):
		# 	quotationFound = False
		# 	if sentences[i].contains('「'):
		# 		quotationFound = True
		# 		continue
		# 	if quotationFound:
		# 		sentences[i:i+1] = [''.join(x[3:6])]
		# 		if sentences[i].contains('」')

		# 	for word in signalDict:
		# 		sentences[i].contains(word)

		


	# for a, b, c in self.mylist:
	# 	f.writerow([a, b,c])

	# ########### Sentence-wise Data from Wiki
	# newfilename = "wiki_sentence.csv"
	# f = csv.writer(open(newfilename, "w"))
	# f.writerow(["label", "id", "title", "sentences"])
	# filename = 'wiki_test.csv'
	# path = './'
	# filename = os.path.join(path, filename)

	# id1 = 0
	# with open(filename, 'r') as csvfile: 
	# 	# creating a csv reader object 
	# 	csvreader = csv.reader(csvfile) 
	# 	# extracting field names through first row 
	# 	fields = next(csvreader) 
	# 	for row in csvreader:
	# 		# id1 = row[0]
	# 		title = row[1]
	# 		content = row[2]		
	# 		sentences = re.split('，|。| ', content)
	# 		for i in range(len(sentences)):
	# 		# for s in sentences:
	# 			if len(sentences[i]) < 7 and i == len(sentences) - 1:
	# 				continue
	# 			elif len(sentences[i]) < 7:
	# 				sentences[i + 1] = sentences[i] + sentences[i + 1]
	# 			else:
	# 				f.writerow([0, id1, title, sentences[i]])
	# 				id1 += 1


if __name__ == '__main__':
	
	mylist = []
	parseArticles(mylist)
	# for i in range(mylist): 
	for row in mylist: 
		# print(mylist(i,1))
		print(row[1])
	sentenceGenerator(mylist)
	print("===================================")
	print("DONE!")
	# print("Filename: {} has store in current directory!".format(newfilename))
	



