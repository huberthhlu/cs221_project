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
		self.check = False
		parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='''
			Wiki Crawler
			Input: number of articles and category name
			Output: wiki.csv 
			''')
		parser.add_argument('-l', metavar='ARTICLE_NUM', help='article num', required=True)
		if not as_lib:
			if cmdline:
				args = parser.parse_args(cmdline)
			else:
				args = parser.parse_args()
			limit = args.l

			self.parse_wiki(limit)
		
	def parse_wiki(self, limit):
		pageLimit = int(limit)
		category = u'臺灣政治'
		# wiki_wiki = wikipediaapi.Wikipedia('en')
		wiki_wiki = wikipediaapi.Wikipedia(
		    language='zh-tw',
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


		def parseArticles(self, categorymembers, level=0, max_level=1, mylist = []):
			if self.check == True: 
				return
			idcount = id
			for c in categorymembers.values():
				if c.ns == 0:
					print('==========', c.title, '\n' )
					page_py = wiki_wiki.page(c.title)
					print(c.summary, '\n' )

					title = c.title
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
					count = len(mylist)
					print (count)
					if count > pageLimit:
						self.check = True
						return
					mylist.append((count, title, main_content))					
				if c.ns == wikipediaapi.Namespace.CATEGORY and level < max_level and self.check == False:
					parseArticles(self, c.categorymembers, level=level + 1, max_level=max_level, mylist = mylist)

		cat = wiki_wiki.page('Category:'+ category)
		filename = 'wiki_test'
		f = csv.writer(open(filename+'.csv', "w"))
		f.writerow(["id", "title", "content"])
		idcount = 0
		mylist = []
		parseArticles(self, cat.categorymembers)

		for a, b, c in mylist:
			f.writerow([a, b,c])

if __name__ == '__main__':
    c = wikiCrawler()
"""
ref:
1. https://pypi.org/project/Wikipedia-API/
2. https://github.com/martin-majlis/Wikipedia-API
"""
