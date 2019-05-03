# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

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
logging.basicConfig(level=logging.INFO)

class wikiCrawler(object):
	def __init__(self, cmdline=None, as_lib=False):
		parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='''
			Wiki Crawler
			Input: number of articles and category name
			Output: wiki.csv 
			''')
		parser.add_argument('-l', metavar='ARTICLE_NUM', help='article num', required=True)
		# group = parser.add_mutually_exclusive_group(required=True)
		# group.add_argument('-c', metavar='CATEGORY_NAME', help='category name', required=True)

		if not as_lib:
			if cmdline:
				args = parser.parse_args(cmdline)
			else:
				args = parser.parse_args()
			limit = args.l
			# category_name = arg.c
			# self.parse_article(limit, category_name)
			self.parse_wiki(limit)
	def parse_wiki(self, limit):
	# def parse_article(limit, category_name):
		pageLimit = int(limit)

		category = u'臺灣政治'
		# wiki_wiki = wikipediaapi.Wikipedia('en')
		wiki_wiki = wikipediaapi.Wikipedia('zh-tw')

		# page_py = wiki_wiki.page('Python_(programming_language)')

		# print("Page - Exists: %s" % page_py.exists())
		# print("Page - Id: %s" % page_py.pageid)
		# print("Page - Title: %s" % page_py.title)
		# print("Page - Summary: %s" % page_py.summary[0:60])


		# def print_sections(sections, level=0):
		#     for s in sections:
		#         print("%s: %s - %s" % ("*" * (level + 1), s.title, s.text[0:40]))
		#         print_sections(s.sections, level + 1)


		# print("Sections:")
		# print_sections(page_py.sections)


		# def print_langlinks(page):
		#     langlinks = page.langlinks
		#     for k in sorted(langlinks.keys()):
		#         v = langlinks[k]
		#         print("%s: %s - %s: %s" % (k, v.language, v.title, v.fullurl))


		# print("Lang links:")
		# print_langlinks(page_py)


		# def print_links(page):
		#     links = page.links
		#     for title in sorted(links.keys()):
		#         print("%s: %s" % (title, links[title]))


		
		wiki_html = wikipediaapi.Wikipedia(
		    language='zh-tw',
		    # language='cs',
		    extract_format=wikipediaapi.ExtractFormat.HTML
		)

		# c

		def print_sections(sections, level=0):
		        for s in sections:
		                print("%s: %s - %s" % ("*" * (level + 1), s.title, s.text[0:40]))
		                print_sections(s.sections, level + 1)

		def print_links(page):
		        links = page.links
		        for title in sorted(links.keys()):
		            print("%s: %s" % (title, links[title]))

		def print_categories(page):
		        categories = page.categories
		        for title in sorted(categories.keys()):
		            print("%s: %s" % (title, categories[title]))

		# def print_categorymembers(categorymembers, level=0, max_level=1):
		#         for c in categorymembers.values():
		#             print("%s: %s (ns: %d)" % ("*" * (level + 1), c.title, c.ns))
		#             if c.ns == wikipediaapi.Namespace.CATEGORY and level < max_level:
		#                 print_categorymembers(c.categorymembers, level=level + 1, max_level=max_level)



		def parseArticles(categorymembers, level=0, max_level=1, idcount = 0, mylist = []):
			print (idcount)
			for c in categorymembers.values():
				if c.ns == 0:
					print('==========', c.title, '\n' )
					page_py = wiki_wiki.page(c.title)
					# print('==========', page_py.text, '\n' )
					# print('==========', c.text, '\n' )
					title = c.title
					# main_content = HanziConv.toTraditional(subcategory.text)
					main_content = c.text
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
					title = HanziConv.toTraditional(title)
					main_content = HanziConv.toTraditional(main_content)
					mylist.append((idcount, title, main_content))
					idcount += 1
					if idcount > pageLimit:
						return
				if c.ns == wikipediaapi.Namespace.CATEGORY and level < max_level:
					idcount += 1
					print_categorymembers(c.categorymembers, level=level + 1, max_level=max_level, idcount = idcount, mylist = mylist)


		# cat = wiki_wiki.page('Category:'+ u'政治')
		cat = wiki_wiki.page('Category:'+ category)
		filename = 'wiki_test'
		f = csv.writer(open(filename+'.csv', "w"))
		f.writerow(["id", "title", "content"])
		idcount = 0
		mylist = []
		parseArticles(cat.categorymembers, 0, 1, 0, mylist)

		for a, b, c in mylist:
			f.writerow([a, b,c])
		"""
		for i in cat.categorymembers:
			# while i.categorymembers is
			subcategory = wiki_html.page(i)
			title = subcategory.title

			main_content = HanziConv.toTraditional(subcategory.text)
			
			# 移除 '※ 發信站:' (starts with u'\u203b'), '◆ From:' (starts with u'\u25c6'), 空行及多餘空白
			# 保留英數字, 中文及中文標點, 網址, 部分特殊符號
			expr = re.compile(u(r'[^\u4e00-\u9fa5\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b\s\w:/-_.?~%()]'))
			main_content = re.sub(expr, '', main_content)
			main_content = re.sub("[A-Za-z]", '', main_content)
			# main_content = re.sub(".*?\((.*?)\)", '', main_content)
			# main_content = re.sub("[\(\[].*?[\)\]]", '', main_content)
			main_content = re.sub('\[.+\]', '', main_content) # to strip out [???] from a string 
			main_content = re.sub('\([^)]*\)', '', main_content) # to strip out <p> and </p> from a string 
			# re.sub(r'<.+?>', '', s)
			# main_content = main_content.decode("utf-8")
			# main_content = str(main_content, 'utf-8')
			# clean = re.compile(u'[^\u4E00-\u9FA5A-Za-z0-9_]')#中文字,字母,下划线
			clean = re.compile('<.*?>') # to strip out <p> and </p> from a string 
			main_content = re.sub(clean, ' ', main_content)
			main_content = re.sub(r'(\s)+', ' ', main_content)
			main_content = re.sub(r'http\S+', '', main_content) ###
			# print('********', main_content)
			
			
			print (title, '\n')
			
			# if u(r'[^\u4e00-\u9fff]') in title:
			# 	print (title, '******')
			# check = False
			# for k in range(len(title)):
			# 	if title[k] != ' ' and and title[k] > u'\u4e00' and title[k] < u'\u9fff':
			# 		check = True
			# 		continue
			# if check == True:
			# 	print ('**********************', title,'\n')
			# 	continue

			f.writerow( [idcount,
						HanziConv.toTraditional(title),
						main_content])

			print ("------------ WRITE ------------", '\n')
			idcount += 1
			if idcount > pageLimit:
				break
		# title_filter = re.compile(u(r'[^\u4e00-\u9fa5\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b\s\w:/-_.?~%()]')))


		"""



		# with open(filename + '.csv', 'r') as csvfile: 
		# 	# creating a csv reader object 
		# 	csvreader = csv.reader(csvfile) 
		# 	# extracting field names through first row 
		# 	fields = next(csvreader) 
		# 	# extracting each data row one by one 
		# 	for row in csvreader: 
		# 		rows.append(row) 
		# 		# get total number of rows 
		# 	print("Total no. of rows: %d"%(csvreader.line_num)) 





		# wiki_zh = wikipediaapi.Wikipedia('zh')
		# # fetch page about Python in Chinese
		# # https://hi.wikipedia.org/wiki/Category:社会

		# p_zh_python_quoted = wiki_zh.article(
		#     title='\u793e\u6703',
		#     unquote=True,
		# )
		# print(p_zh_python_quoted.title)
		# print(p_zh_python_quoted.summary)
		# # print(p_zh_python_quoted.summary[0:60])
		# # Category:社會
		# """ Output: by (https://www.chineseconverter.com/en/convert/unicode)
		# 社會一詞並没有太正式明確定義，一般是指由自我繁殖的個體構建而成的群體，占据一定的空間，具有其獨特的文化和風俗習慣。由於社會通常被認為是人類組成的，所以社會和人類社會一般具有相同的含義。在科學研究和科幻小說等等里面，有時亦可作“外星人社會”。狹義的社會，也叫“社群”，可以只指群體人類活動和聚居的範圍，例如是：鄉、村、鎮、城市、聚居點等等；廣義的社會則可以指一個國家、一個大範圍地區或一個文化圈，例如是英國社會、東亞社會、東南亞或西方世界，均可作為社會的廣義解釋，也可以引申為他們的文化習俗。以人類社會為研究對象的學科叫做社會學。
		# """
if __name__ == '__main__':
    c = wikiCrawler()
