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
from filter_text import Filter_Text   # emoji filter

__version__ = '1.0'

# if python 2, disable verify flag in requests.get()
VERIFY = True
if sys.version_info[0] < 3:
    VERIFY = False
    requests.packages.urllib3.disable_warnings()


class PttWebCrawler(object):

    PTT_URL = 'https://www.ptt.cc'

    """docstring for PttWebCrawler"""
    def __init__(self, cmdline=None, as_lib=False):
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
            board = args.b
            if args.i:
                start = args.i[0]
                if args.i[1] == -1:
                    end = self.getLastPage(board)
                else:
                    end = args.i[1]
                self.parse_articles(start, end, board)
            else:  # args.a
                article_id = args.a
                self.parse_article(article_id, board)

    def parse_articles(self, start, end, board, path='.', timeout=3):
            filename = board + '-' + str(start) + '-' + str(end) + '.json'
            filename = os.path.join(path, filename)
            # self.store(filename, u'{"articles": [', 'w')
            self.store(filename, u'[', 'w')

            flag = 0 # the first valid DATA
            for i in range(end-start+1):
                index = start + i
                print('Processing index:', str(index))
                resp = requests.get(
                    url = self.PTT_URL + '/bbs/' + board + '/index' + str(index) + '.html',
                    cookies={'over18': '1'}, verify=VERIFY, timeout=timeout
                )
                if resp.status_code != 200:
                    print('invalid url:', resp.url)
                    continue
                soup = BeautifulSoup(resp.text, 'html.parser')
                divs = soup.find_all("div", "r-ent")
                for div in divs:
                    try:
                        # ex. link would be <a href="/bbs/PublicServan/M.1127742013.A.240.html">Re: [問題] 職等</a>
                        href = div.find('a')['href']
                        link = self.PTT_URL + href
                        article_id = re.sub('\.html', '', href.split('/')[-1])
                        # if div == divs[-1] and i == 1:  # last div of last page

                        DATA = self.parse(link, article_id, board)
                        # DATA = self.parse(article_id)
                        if not DATA :
                            print(" ==== Title or content contain somthing we don't want ==== ")
                        else:
                            if flag == 0:
                                flag += 1
                                self.store(filename, DATA, 'a')
                            else:
                                self.store(filename, ',\n' + DATA, 'a')
                        # if div == divs[-1] and i == end-start:  # last div of last page
                        #     self.store(filename, DATA, 'a')
                        # else:
                        #     self.store(filename, DATA + ',\n', 'a')
                    except:
                        pass
                time.sleep(0.1)
            # self.store(filename, u']}', 'a')
            self.store(filename, u']', 'a')
            return filename

    def parse_article(self, article_id, board, path='.'):
        link = self.PTT_URL + '/bbs/' + board + '/' + article_id + '.html'
        filename = board + '-' + article_id + '.json'
        filename = os.path.join(path, filename)
        self.store(filename, self.parse(link, article_id, board), 'w')
        return filename

    @staticmethod
    def parse(link, article_id, board, timeout=3):
        print('Processing article:', article_id)
        resp = requests.get(url=link, cookies={'over18': '1'}, verify=VERIFY, timeout=timeout)
        if resp.status_code != 200:
            print('invalid url:', resp.url)
            # return json.dumps({"error": "invalid url"}, sort_keys=True, ensure_ascii=False)
            return json.dumps({"error": "invalid url"}, sort_keys=False, ensure_ascii=False)
        soup = BeautifulSoup(resp.text, 'html.parser')
        main_content = soup.find(id="main-content")
        metas = main_content.select('div.article-metaline')
        author = ''
        title = ''
        date = ''
        if metas:
            author = metas[0].select('span.article-meta-value')[0].string if metas[0].select('span.article-meta-value')[0] else author
            title = metas[1].select('span.article-meta-value')[0].string if metas[1].select('span.article-meta-value')[0] else title
            date = metas[2].select('span.article-meta-value')[0].string if metas[2].select('span.article-meta-value')[0] else date

            # remove meta nodes
            for meta in metas:
                meta.extract()
            for meta in main_content.select('div.article-metaline-right'):
                meta.extract()

        # put which condition that you don't want the entire article
        # return None
        ####################
        if u'公告' in title or u'新聞' in title or u'通報' in title: 
            return None
        ####################
        # remove and keep push nodes
        pushes = main_content.find_all('div', class_='push')
        for push in pushes:
            push.extract()

        try:
            ip = main_content.find(text=re.compile(u'※ 發信站:'))
            ip = re.search('[0-9]*\.[0-9]*\.[0-9]*\.[0-9]*', ip).group()
        except:
            ip = "None"


        ########################### Content Filtering ###########################
        # 移除 '※ 發信站:' (starts with u'\u203b'), '◆ From:' (starts with u'\u25c6'), 空行及多餘空白
        # 保留英數字, 中文及中文標點, 網址, 部分特殊符號

        # filtered = [ v for v in main_content.stripped_strings if v[0] not in [u'※', u'◆'] and v[:2] not in [u'--']]
        # expr = re.compile(u(r'[^\u4e00-\u9fa5\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b\s\w:/-_.?~%()]'))
        # for i in range(len(filtered)):
        #     filtered[i] = re.sub(expr, '', filtered[i])
        # filtered = [_f for _f in filtered if _f]  # remove empty strings
        # filtered = [x for x in filtered if article_id not in x]  # remove last line containing the url of the article
        # content = ' '.join(filtered)
        # content = re.sub(r'(\s)+', ' ', content)
        # content = re.sub(r'http\S+', '', content) ###

        filtered = [ v for v in main_content.stripped_strings if v[0] not in [u'※', u'◆'] and v[:2] not in [u'--']]
        expr = re.compile(u(r'[^\u4e00-\u9fa5\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b\s\w:/-_.?~%()]'))
        for i in range(len(filtered)):
            filtered[i] = re.sub(expr, '', filtered[i])
        filtered = [_f for _f in filtered if _f]  # remove empty strings
        filtered = [x for x in filtered if article_id not in x]  # remove last line containing the url of the article
        content = ' '.join(filtered)
        content = re.sub("[A-Za-z]", '', content)
        content = re.sub('\[.+\]', '', content) # to strip out [???] from a string 
        # content = re.sub('\([^)]*\)', '', content) # to strip out <p> and </p> from a string 
        # clean = re.compile('<.*?>') # to strip out <p> and </p> from a string 
        # content = re.sub(clean, ' ', content)
        content = re.sub(r'(\s)+', ' ', content)
        content = re.sub(r'http\S+', '', content) ###

        ########################### Content Filtering ###########################
        # print 'content', content

        # push messages
        p, b, n = 0, 0, 0
        messages = []
        for push in pushes:
            if not push.find('span', 'push-tag'):
                continue
            push_tag = push.find('span', 'push-tag').string.strip(' \t\n\r')
            push_userid = push.find('span', 'push-userid').string.strip(' \t\n\r')
            # if find is None: find().strings -> list -> ' '.join; else the current way
            push_content = push.find('span', 'push-content').strings
            push_content = ' '.join(push_content)[1:].strip(' \t\n\r')  # remove ':'
            push_content = re.sub(r'http\S+', '', push_content) ###
            push_ipdatetime = push.find('span', 'push-ipdatetime').string.strip(' \t\n\r')
            messages.append( {'push_tag': push_tag, 'push_userid': push_userid, 'push_content': push_content, 'push_ipdatetime': push_ipdatetime} )
            if push_tag == u'推':
                p += 1
            elif push_tag == u'噓':
                b += 1
            else:
                n += 1

        # count: 推噓文相抵後的數量; all: 推文總數
        message_count = {'all': p+b+n, 'count': p-b, 'push': p, 'boo': b, "neutral": n}

        # print 'msgs', messages
        # print 'mscounts', message_count

        # json data
        data = {
            # 'url': link,
            # 'board': board,
            'article_id': article_id,
            'article_title': title,
            'author': author,
            'date': date,
            'content': content
            # 'ip': ip,
            # 'message_count': message_count,
            # 'messages': messages
        }
        # print 'original:', d
        # return json.dumps(data, sort_keys=True, ensure_ascii=False) # modified by Hubert 
        return json.dumps(data, sort_keys=False, ensure_ascii=False)

    @staticmethod
    def getLastPage(board, timeout=3):
        content = requests.get(
            url= 'https://www.ptt.cc/bbs/' + board + '/index.html',
            cookies={'over18': '1'}, timeout=timeout
        ).content.decode('utf-8')
        first_page = re.search(r'href="/bbs/' + board + '/index(\d+).html">&lsaquo;', content)
        if first_page is None:
            return 1
        return int(first_page.group(1)) + 1

    @staticmethod
    def store(filename, data, mode):
        with codecs.open(filename, mode, encoding='utf-8') as f:
            f.write(data)
    

    """
    json and csv converter
    """

    @staticmethod
    def get(filename, mode='r'):
        with codecs.open(filename, mode, encoding='utf-8') as f:
            return json.load(f)



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

        newfilename = "sentences_Data_Ptt.csv"
        f = csv.writer(open(newfilename, "w"))
        # f.writerow(["article_id", "article_title", "author", "date", "sentences"])
        f.writerow(["label","id", "title", "sentences"])
        filename = board + '-' + str(start) + '-' + str(end) + '.csv'
        filename = os.path.join(path, filename)

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

        ID = 0
        with open(filename, 'r') as csvfile: 
            # creating a csv reader object 
            csvreader = csv.reader(csvfile) 
            # extracting field names through first row 
            fields = next(csvreader) 
            for row in csvreader:
                # ID = row[0]
                title = row[1]
                author = row[2]
                date = row[3]
                corpus = row[4]


                sentences = re.split(u'(，|。|,|:| )', corpus)
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
                            # print(sentences[i])
                            # print(sentences[i+1])
                            quote = True

                    if len(re.sub("^[0-9]*$", '', sentences[i])) == 0 or len(re.sub(r"[a-zA-Z0-9./]",'',sentences[i])) == 0:
                        if i == len(sentences): 
                            if S == '':
                                f.writerow([None ,ID, title, S])
                            else:
                                f.writerow([1 ,ID, title, S])
                        continue
                    if len(sentences[i]) < 7 or quote:
                        if sentences[i] == ' ' or sentences[i] == ':':
                            pass
                        else:
                            S += sentences[i]

                        if i == len(sentences):
                            if S == '':
                                f.writerow([None ,ID, title, S])
                            else:
                                f.writerow([1 ,ID, title, S])
                                S = ''
                    else:
                        if S == '':
                            f.writerow([None ,ID, title, S])
                        else:
                            f.writerow([1 ,ID, title, S]) # 1 means label this sentence as 1(subjective)
                            S = sentences[i]
                    ID += 1

                # sentences = re.split(u'，|。|,| ', corpus)
                # quotation = False
                # for i in range(len(sentences)):
                #     sentences[i] = Filter_Text().filtet_text(sentences[i]) # emoji filter!!!!
                #     if len(sentences[i]) < 7 and i == len(sentences) -1:
                #         continue
                #     elif len(sentences[i]) < 7:
                #         sentences[i + 1] = sentences[i] + sentences[i + 1]
                #     else:
                #         if len(re.sub("^[0-9]*$", '', sentences[i])) <= 1: continue  # exclud the string only contains numbers
                #         f.writerow([1 ,ID, title, sentences[i]]) # 1 means label this sentence as 1(subjective)
                #         ID += 1

        print("===================================")
        print("DONE!")
        print("Filename: {} has store in current directory!".format(newfilename))

if __name__ == '__main__':
    c = PttWebCrawler()
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

"""
ref:
1. https://medium.com/@gotraveltoworld/%E7%94%A8python%E9%81%8E%E6%BF%BE%E6%A8%99%E9%BB%9E%E7%AC%A6%E8%99%9F-punctuations-%E5%92%8C%E8%A1%A8%E6%83%85%E7%AC%A6%E8%99%9F-emojis-c31a9a2422d5
"""

