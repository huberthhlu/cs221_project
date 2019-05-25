# -*- coding: utf-8 -*-

import csv
import pandas as pd

hubert = 'hubert_hand_label.csv'
shaoyuan = 'shaoyuan_hand_label.csv'

hubert = pd.read_csv('hubert_hand_label.csv')
shaoyuan = pd.read_csv('shaoyuan_hand_label.csv')
hubert.dropna(how="all", inplace=True)
shaoyuan.dropna(how="all", inplace=True)

merged = hubert.merge(shaoyuan, on='sentences')
merged['label_hubert'] = merged.label_hubert.astype(float)
merged['label_hubert'] = merged.label_hubert.astype(int)
merged['label_shaoyuan'] = merged.label_shaoyuan.astype(float)
merged['label_shaoyuan'] = merged.label_shaoyuan.astype(int)
merged.to_csv("merge.csv",index=False)


newfilename = 'difference.csv'
f = csv.writer(open(newfilename, "w"))
f.writerow(["sentences", "label_hubert", "label_shaoyuan"])


filename = 'merge.csv'
with open(filename, 'r') as csvfile:
	# creating a csv reader object
	csvreader = csv.reader(csvfile)
	# extracting field names through first row
	fields = next(csvreader)

	n = 0
	hubert_unknown = 0
	shaoyuan_unknown = 0
	different = 0

	for row in csvreader:
		sentence = row[0]
		hubert_label = row[1]
		shaoyuan_label = row[2]
		print("H: {} ; SY: {}".format(hubert_label,shaoyuan_label))
		n += 1
		if hubert_label == '2': 
			hubert_unknown += 1
		if shaoyuan_label == '2': 
			shaoyuan_unknown += 1
		if hubert_label != shaoyuan_label: 
			different += 1
			f.writerow([sentence, hubert_label, shaoyuan_label])


# statistic:
print('Shao-Yuan unknown rate = {}'.format(shaoyuan_unknown/float(n)))
print('Hubert unknown rate = {}'.format(hubert_unknown/float(n)))
print('Difference rate between Shao-Yuan and Hubert answer = {}'.format(different/float(n)))

		
