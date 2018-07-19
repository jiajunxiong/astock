"""
process data
"""

import csv
import math
from gData import *
import numpy as np
import pandas as pd
from datetime import datetime

def readData():
    """
    read all data from file 
    """
    panel_data = getAlphaData()
    #panel_data = panel_data.transpose('minor', 'major', 'items')
    return panel_data
    #return pd.read_csv("data.csv")
    
def readDataUpdate(date):
    """
    read today data from file
    """
    data = getDataUpdate(date)
    return data
    
def cleanData(data):
    """
    data wrapper
    """
    # delete too much missing data rows
    #data = data.dropna(1)
    #data.dropna(thresh=4, inplace=True)
    #data = data.dropna(subset=['factor_1']) 
    
    # fillna
    #data['factor_1'] = data.groupby('industry').transform(lambda x: x.fillna(x.mean()))
    #data['factor_1'] = data.groupby('industry').transform(lambda x: x.fillna(x.median()))
    #data['factor_2'] = data.groupby('industry').transform(lambda x: x.fillna(x.mean()))
    #data['factor_2'] = data.groupby('industry').transform(lambda x: x.fillna(x.median()))
    return data

def updateData(data):
    
    return data
    
def normalizeByIndustry(data):
    """
    data normalization by industry
    """
    data["factor"] = (data.groupby('industry')['factor'].transform(lambda x: x/x.sum()))
    return data

def normalizationStandard(data):
    """
    data normalization
    """
    # standard
    data = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    return data

def normalizationZscore(data):
    """
    data normalization
    """
    # Z-score
    data = data.apply(lambda x: (x - np.average(x)) / np.std(x))
    return data

def equalWeight(panel_data):
    """
    calculate all trading days equal weight
    """
    path = './alpha191/date/normal_ew/'
    for date in panel_data.major_axis:
        data = panel_data.major_xs(date)
        data = normalizationStandard(data)
        data.loc['Row_sum'] = data.apply(lambda x: x.sum())
        data['Col_sum'] = data.apply(lambda x: x.sum(), axis=1)
        data.to_csv(path+date+'.csv')
    
def equalWeightUpdate(date, data):
    """
    calculate one trading day equal weight
    """
    path = './alpha191/date/normal_ew/'
    data = normalizationStandard(data)
    data.loc['Row_sum'] = data.apply(lambda x: x.sum())
    data['Col_sum'] = data.apply(lambda x: x.sum(), axis=1)
    file = date+'.csv'
    data.to_csv(path + file)
    
def selfWeight(panel_data):
    path = './alpha191/date/normal_sw/'
    for date in panel_data.major_axis:
        data = panel_data.major_xs(date)
        data = normalizationStandard(data)
        data.loc['Row_sum'] = data.apply(lambda x: x.sum())
        data['Col_sum'] = data.apply(lambda x: x.sum(), axis=1)
        
        data.loc['Row_sum'] = data.loc['Row_sum']/data.iat[-1,-1]
        sum = 0.0
        for i in range(len(data.Col_sum)-1):
            for j in range(191):
                if not (math.isnan(data.iat[i,j]) or math.isnan(data.iat[-1,j])):
                    sum += data.iat[i,j] * data.iat[-1,j]
            data.Col_sum[i] = sum * 100.0
            sum = 0.0
        data.to_csv(path+date+'.csv')
        
def selfWeightUpdate(date, data):
    """
    calculate one trading day equal weight
    """
    path = './alpha191/date/normal_sw/'
    data = normalizationStandard(data)
    data.loc['Row_sum'] = data.apply(lambda x: x.sum())
    data['Col_sum'] = data.apply(lambda x: x.sum(), axis=1)
    
    data.loc['Row_sum'] = data.loc['Row_sum']/data.iat[-1,-1]
    sum = 0.0
    for i in range(len(data.Col_sum)-1):
        for j in range(191):
            if not (math.isnan(data.iat[i,j]) or math.isnan(data.iat[-1,j])):
                sum += data.iat[i,j] * data.iat[-1,j]
        data.Col_sum[i] = sum * 100.0
        sum = 0.0
    file = date+'.csv'
    data.to_csv(path + file)

def rank(path_nor, path_rank):
    """
    data rank
    """
    for file in os.listdir(path_nor):
        data = pd.read_csv(path_nor + file, index_col=0)
        data.sort_values("Col_sum",inplace=True)
        data.to_csv(path_rank + file)
        
def rankUpdate(date, path_nor, path_rank):
    """
    data rank
    """
    file = date + ".csv"
    data = pd.read_csv(path_nor + file, index_col=0)
    data.sort_values("Col_sum",inplace=True)
    data.to_csv(path_rank + file)

"""
# get origin from jointquant
panel_data = readData()

equalWeight(panel_data)
path_nor = "./alpha191/date/normal_ew/"
path_rank = "./alpha191/date/rank_ew/"
rank(path_nor, path_rank)

selfWeight(panel_data)
path_nor = "./alpha191/date/normal_sw/"
path_rank = "./alpha191/date/rank_sw/"
rank(path_nor, path_rank)
"""


# update date data
date = "2018-07-17"
data = readDataUpdate(date)

equalWeightUpdate(date, data)
path_nor = "./alpha191/date/normal_ew/"
path_rank = "./alpha191/date/rank_ew/"
rankUpdate(date, path_nor, path_rank)

"""
selfWeightUpdate(date, data)
path_nor = "./alpha191/date/normal_sw/"
path_rank = "./alpha191/date/rank_sw/"
rankUpdate(date, path_nor, path_rank)
"""