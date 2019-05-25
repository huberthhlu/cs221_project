# -*- coding: utf-8 -*-

import csv

filename = 'hubert_hand_label.csv'
newfilename = 'shaoyuan_hand_label.csv'

f = csv.writer(open(newfilename, "w"))
f.writerow(["sentences", "label_shaoyuan"])

with open(filename, 'r') as csvfile:
	# creating a csv reader object
	csvreader = csv.reader(csvfile)
	# extracting field names through first row
	fields = next(csvreader)
	for row in csvreader:
		sentence = row[0]
		f.writerow([sentence])

print("=====================")
print("DONE!")
