"""
show back-testing result
"""
import os
import csv
import math
from rData import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

stock_num = 5
holding_period = 2
capital = 10000

# equal capital or equal lot
# equal_capital_flag = True 
# equal_lot_flag = True

def transactionMTMDailyPnl(symbol, date_counter, lot, panel_data):
    
    transaction_pnl = 0.0
    
    today_close = panel_data.major_xs(symbol).CLOSE[date_counter] 
    today_open = panel_data.major_xs(symbol).OPEN[date_counter]
    
    if not (math.isnan(today_close) or math.isnan(today_open)):
        transaction_pnl = (today_close - today_open) * lot
    return transaction_pnl
    
def priorperiodMTMDailyPnl_1(symbol, date_counter, panel_data):

    priorperiod_pnl = 0.0
    lot = 0
    
    today_open = panel_data.major_xs(symbol).OPEN[date_counter]
    today_close = panel_data.major_xs(symbol).CLOSE[date_counter] 
    yesterday_close = panel_data.major_xs(symbol).CLOSE[date_counter-1]
    
    if not (math.isnan(today_close) or math.isnan(yesterday_close) or math.isnan(today_open)):
        lot = capital / today_open
        priorperiod_pnl = (today_close - yesterday_close) * lot
    return priorperiod_pnl, lot
    
def priorperiodMTMDailyPnl_2(symbol, date_counter, lot, panel_data):

    priorperiod_pnl = 0.0
    
    today_open = panel_data.major_xs(symbol).OPEN[date_counter]
    today_close = panel_data.major_xs(symbol).CLOSE[date_counter] 
    yesterday_close = panel_data.major_xs(symbol).CLOSE[date_counter-1]
    
    if not (math.isnan(today_close) or math.isnan(yesterday_close) or math.isnan(today_open)):
        priorperiod_pnl = (today_close - yesterday_close) * lot
    return priorperiod_pnl

def getReturn():
    pnls = []
    list_symbol = []
    list_lot = []
    buy_sell_lot = [[]] * int(holding_period-1)
    buy_sell_list = [[]] * int(holding_period)
    date_counter = 0
    panel_data = readData("./data/zz500/")
    path_rank = "./alpha191/date/rank_ew/"
    for file in os.listdir(path_rank):
        # read symbol.csv file and calculate mtmDailyPnl append to pnl
        print(buy_sell_list)
        print(buy_sell_lot)
        priorperiod_pnl = 0.0
        list_lot = []
        for i in range(holding_period-1):
            if (i==0):
                for symbol in buy_sell_list[0]:
                    pnl, lot = priorperiodMTMDailyPnl_1(symbol, date_counter, panel_data)
                    list_lot.append(lot)
                    priorperiod_pnl += pnl
                buy_sell_lot = [list_lot] + buy_sell_lot
            else:
                for symbol in buy_sell_list[i]:
                    index = buy_sell_list[i].index(symbol)
                    pnl = priorperiodMTMDailyPnl_2(symbol, date_counter, buy_sell_lot[i][index], panel_data)
                    priorperiod_pnl += pnl
        
        transaction_pnl = 0.0
        for symbol in buy_sell_list[holding_period-1]:
            index = buy_sell_list[holding_period-1].index(symbol)
            if buy_sell_lot[holding_period-1]:
                pnl = transactionMTMDailyPnl(symbol, date_counter, buy_sell_lot[holding_period-1][index], panel_data)
            else:
                pnl = 0.0
            transaction_pnl += pnl
        data = pd.read_csv(path_rank + file, index_col=0)
        list_symbol = data.tail(10).index.tolist()
        # calculate sw 
        index = list_symbol.index("Row_sum")
        del list_symbol[index:]
        list_symbol = [x[:6] for x in list_symbol]
        list_symbol = list_symbol[-1*stock_num:] # tomorrow to buy
        buy_sell_list = [list_symbol] + buy_sell_list
        
        pnls.append(transaction_pnl + priorperiod_pnl)
        date_counter += 1
    return pnls
    
pnls = getReturn()
pnls = [round(x, 2) for x in pnls]
cum_pnls = np.cumsum(pnls)
#data = pd.read_csv("./data/000905.csv", index_col=0)
plt.plot(cum_pnls)
#plt.plot(data.CLOSE)
plt.show()
