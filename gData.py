"""
get data
"""
import os
import re
import csv
import time
import pandas as pd
# import tushare as ts

def atoi(text):
    return int(text) if text.isdigit() else text

def naturalKeys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def getData():
	data = pd.read_csv('./data/zz500.csv', encoding='cp1252')
	symbol_list = data['code'].tolist()
	symbol_list.sort()
	#symbol_list.sort(key=naturalKeys)
	
	panel_data = {}
	for symbol in symbol_list:
		history_data = ts.get_h_data(str(symbol).zfill(6), "2009-01-01", "2018-06-30")
		history_data.to_csv('./data/symbol/'+str(symbol).zfill(6)+'.csv')
		time.sleep(120)
		#history_data = pd.read_csv('./data/symbol/'+str(symbol).zfill(6)+'.csv', encoding='cp1252')
		#history_data.set_index('date', inplace=True)
		#panel_data[str(symbol).zfill(6)] = history_data	

	panel_data = pd.Panel(panel_data).transpose('minor', 'major', 'items')
	#print (panel_data.major_axis)
	#print (panel_data.minor_axis)
	#print (panel_data.items)
	return panel_data

def getDataUpdate(date):
	data = []
	path = "./alpha191/date/origin/"
	file = "alpha191_" + date + '.csv'
	data = pd.read_csv(path + file, index_col=0)
	return data

def getAlphaData():
	panel_data = {}
	path = "./alpha191/date/origin/"
	for file in os.listdir(path):
		data = pd.read_csv(path + file, index_col=0)
		panel_data[file[9:-4]] = data
	panel_data = pd.Panel(panel_data).transpose('minor', 'items', 'major')
	#print (panel_data.items)
	#print (panel_data.major_axis)
	#print (panel_data.minor_axis)
	return panel_data

#getAlphaData()
"""
panel_data = getData()
date_list = panel_data.major_axis.astype(str)
counter=0
for date in panel_data.major_axis:
	print (panel_data.major_xs(date))
	#panel_data.major_xs(date).to_csv('./data/date/'+date_list[counter]+'.csv')
	counter += 1
"""