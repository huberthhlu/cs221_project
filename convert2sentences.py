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
from newspaper import Article 

__version__ = '1.0'

# if python 2, disable verify flag in requests.get()
VERIFY = True
if sys.version_info[0] < 3:
    VERIFY = False
    requests.packages.urllib3.disable_warnings()


"""
filename = "HatePolitics--2-0.json"
with open(filename,encoding='utf-8', errors='ignore') as json_data:
	x = json.load(json_data, strict = False)

f = csv.writer(open("test.csv", "w"))

# Write CSV Header, If you dont need that, remove this line
# f.writerow(["article_id", "title", "author", "date", "content","messages"])
f.writerow(["article_id", "article_title", "author", "date", "content"])
"""
			# 'url': link,
   #          'board': board,
   #          'article_id': article_id,
   #          'article_title': title,
   #          'author': author,
   #          'date': date,
   #          'content': content,
   #          'ip': ip,
   #          'message_count': message_count,
   #          'messages': messages
"""
for x in x:
    f.writerow([x["article_id"],
                x["article_title"],
                x["author"],
                x["date"],
                # x["content"],
                x["content"]])
                # x["messages"]["push_content"]])
"""
class json2csv(object):
    PTT_URL = 'https://www.ptt.cc'

    """docstring for PttWebCrawler"""
    def __init__(self, cmdline=None, as_lib=False):
        self.board = 0
        self.end = 0
        self.start = 0
        self.path = '.'
        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='''
            A crawler for the web version of PTT, the largest online community in Taiwan.
            Input: board name and page indices (or articla ID)
            Output: BOARD_NAME-START_INDEX-END_INDEX.json (or BOARD_NAME-ID.json)
        ''')
        parser.add_argument('-b', metavar='BOARD_NAME', help='Board name', required=True)
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('-i', metavar=('START_INDEX', 'END_INDEX'), type=int, nargs=2, help="Start and end index")
        group.add_argument('-a', metavar='ARTICLE_ID', help="Article ID")
        parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __version__)

        if not as_lib:
            if cmdline:
                args = parser.parse_args(cmdline)
            else:
                args = parser.parse_args()
            self.board = args.b
            if args.i:
                self.start = args.i[0]
                if args.i[1] == -1:
                    self.end = self.getLastPage(board)
                else:
                    self.end = args.i[1]
            else:  # args.a
                article_id = args.a
            
            
    def subFunction(self):
        board = self.board
        path = self.path
        end = self.end
        start = self.start
        newFilename = board + '-' + str(start) + '-' + str(end)
        newFilename = os.path.join(path, newFilename)
        filename = board + '-' + str(start) + '-' + str(end) + '.json'
        filename = os.path.join(path, filename)
        print ('JSON file : ', filename)
        with open(filename,encoding='utf-8', errors='ignore') as json_data:
            x = json.load(json_data, strict = False)
        f = csv.writer(open(newFilename+'.csv', "w"))
        f.writerow(["article_id", "article_title", "author", "date", "content"])
        """
                    'url': link,
                    'board': board,
                    'article_id': article_id,
                    'article_title': title,
                    'author': author,
                    'date': date,
                    'content': content,
                    'ip': ip,
                    'message_count': message_count,
                    'messages': messages
        """
        for x in x:
            f.writerow([x["article_id"],
                        x["article_title"],
                        x["author"],
                        x["date"],
                        x["content"]])

    def extractSentece(self):
        board = self.board
        path = self.path
        end = self.end
        start = self.start
        print ('board = ', board)
        print ('path = ', path)
        print ('end = ', end)
        print ('start = ', start)

        # filename = board + '-' + str(start) + '-' + str(end) + '.csv'
        # filename = os.path.join(path, filename)
        # print ('filename2 = ', filename)
        # # initializing the titles and rows list 
        # fields = [] 
        # rows = [] 
        # # reading csv file 
        # with open(filename, 'r') as csvfile: 
        # 	# creating a csv reader object 
        # 	csvreader = csv.reader(csvfile) 
        # 	# extracting field names through first row 
        # 	fields = next(csvreader) 
        # 	# extracting each data row one by one 
        # 	for row in csvreader: 
        # 		rows.append(row) 
        # 		# get total number of rows 
        # 	print("Total no. of rows: %d"%(csvreader.line_num)) 
        # # printing the field names 
        # print('Field names are:' + ', '.join(field for field in fields)) 
        # #  printing first 5 rows 
        # print('\nFirst 5 rows are:\n') 
        # for row in rows[:5]: 
        # 	# parsing each column of a row 
        # 	for col in row: 
        # 		print("%10s"%col), 
        # 	print('\n') 


        newfilename = "sentences_Data_Ptt.csv"
        f = csv.writer(open(newfilename, "w"))
        f.writerow(["article_id", "article_title", "author", "date", "sentences"])

        filename = board + '-' + str(start) + '-' + str(end) + '.csv'
        filename = os.path.join(path, filename)
        with open(filename, 'r') as csvfile: 
            # creating a csv reader object 
            csvreader = csv.reader(csvfile) 
            # extracting field names through first row 
            fields = next(csvreader) 
            for row in csvreader:
                ID = row[0]
                title = row[1]
                author = row[2]
                date = row[3]
                corpus = row[4]
                sentences = corpus.split('，')
                for s in sentences:
                    f.writerow([ID, title, author, date, s])

        print("===================================")
        print("DONE!")
        print("Filename: {} has store in current directory!".format(newfilename))




if __name__ == '__main__':

    print("Convert json to csv...")
    print("===================================")
    converter = json2csv()
    converter.subFunction()
    converter1 = json2csv()
    converter1.extractSentece()
    


# https://github.com/jwlin/ptt-web-crawler
# python crawler.py -b PublicServan -i 100 200
# 設為負數則以倒數第幾頁計算
"""
1. filter: exclude 公告               OK
2. clear url in content and message  OK
3. seperate the sentences from the stored .json files
4. remove the sentences containing english words
5. 
"""
# 1. exclude sentences with english/chinese 
# 2. questioning sentence /mark
# 3. no 句號 sometimes only uses space
# 4. sentence is too short 於是，
# 5. 、 、
# 6. 難道柯總統要干涉台北市長 繼續卡蛋嗎???
# 7. ????
# 8. 表情符號 =.=?
# 9.
# python isalpha()
