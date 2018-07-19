"""
read csv file to panel data
"""
import os
import csv
import pandas as pd

path = "./data/zz500/"
def readData(path):
	panel_data = {}
	for file in os.listdir(path):
		data = pd.read_csv(path + file, index_col=0)
		panel_data[file[:-4]] = data
	panel_data = pd.Panel(panel_data).transpose('minor', 'items', 'major')
	#print (panel_data.items)
	#print (panel_data.major_axis)
	#print (panel_data.minor_axis)
	#data = panel_data.major_xs("000006")
	#print (data.CLOSE[0])
	return panel_data

#readData(path)
"""
for symbol in panel_data.major_axis:
	print (symbol)
print (panel_data.major_axis)
print (panel_data.minor_axis)
print (panel_data.items)
#panel_data = cleanData(panel_data)
#date_list = panel_data.major_axis.astype(str)
"""