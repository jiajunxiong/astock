"""
alpha 191 factors
"""

from math import log
import scipy.stats as stats
from datetime import datetime

"""
latest updated data insert to front
"""
VOLUME = [1,2,5,7,5,4,8]
CLOSE = [5,6,8,4,5,6,7]
OPEN = [4,8,4,5,4,4,5]
HIGH = [6,9,9,6,6,7,8]
LOW = [4,5,4,4,4,3,4]
VWAP = [4,5,4,4,4,3,4]

def getListOpen(OPEN, n):
    return OPEN[:n]

def getOpen(OPEN,n):
    return OPEN[n]

def getListHigh(HIGH, n):
    return HIGH[:n]

def getHigh(HIGH,n):
    return HIGH[n]

def getListLow(LOW, n):
    return LOW[:n]

def getLow(LOW,n):
    return LOW[n]

def getVwap(VWAP, n):
    return VWAP[n]

def getListClose(CLOSE, n):
    return CLOSE[:n]

def getClose(CLOSE,n):
    return CLOSE[n]
    
def getListVolume(VOLUME, n):
    return VOLUME[:n]

def getVolume(VOLUME, n):
    return VOLUME[n]

def getDelay(A, n):
    return A[n]

def getListAdd(A, B):
    """
    list A + list B
    """
    return list(map(lambda x: x[0]+x[1], zip(A, B)))

def getSum(A, n):
    """
    sum of last n days
    """
    if A.empty():
        return 0
    else:
        return sum(A[:n])

def getListSub(A, B):
    return list(map(lambda x: x[0]-x[1], zip(A, B)))
    
def getListDiv(A, B):
    return list(map(lambda x: x[0]/x[1], zip(A, B)))

def getListLog(A):
	return [log(x, 10) for x in A]

def getListRank(A):
    A.sort()
    return A
    
def getListDelta(A, n):
    # TODO
    if len(A) == 2:
        return A[1]-A[0]
    else:
        return [A[m]-A[m-n] for m in range(1,len(A))]

def getListCorr(A, B, n):
    value = stats.pearsonr(A,B)
    return value[0]

def updateFactors(data):
    return data

alpha_1 = (-1 * getListCorr(getListRank(getListDelta(getListLog(getListVolume(VOLUME,7)), 1)), getListRank(getListDiv(getListSub(getListClose(CLOSE,6), getListOpen(OPEN,6)), getListOpen(OPEN,6))), 6))
alpha_2 = (-1 * getListDelta(getListDiv(getListSub(getListSub(getListClose(CLOSE, 2), getListLow(LOW,2)), getListSub(getListHigh(HIGH,2), getListClose(CLOSE,2))), getListSub(getListHigh(HIGH,2), getListLow(LOW,2))), 1))

alpha_3=0
for i in range(6):
    alpha_3 += (0 if getClose(CLOSE,i)==getDelay(CLOSE,i+1) else getClose(CLOSE,i)-(min(getLow(LOW,i),getDelay(CLOSE,i+1)) if getClose(CLOSE,i)>getDelay(CLOSE,i+1) else max(getHigh(HIGH,i),getDelay(CLOSE,i+1))))
alpha_13 = (((getHigh(HIGH,0) * getLow(LOW,0))**0.5) - getVwap(VWAP,0))
alpha_14 = getClose(CLOSE,0)-getDelay(CLOSE,5)
alpha_15 = getOpen(OPEN,0)/getDelay(CLOSE,1)-1
#alpha_17 = getListRank((VWAP - MAX(VWAP, 15)))^DELTA(CLOSE, 5) #TODO
alpha_18 = getClose(CLOSE,0)/getDelay(CLOSE,5) 
alpha_19 = ((getClose(CLOSE,0)-getDelay(CLOSE,5))/getDelay(CLOSE,5) if getClose(CLOSE,0)<getDelay(CLOSE,5) else ((0 if getClose(CLOSE,0)==getDelay(CLOSE,5) else (getClose(CLOSE,0)-getDelay(CLOSE,5))/getClose(CLOSE,0))))
alpha_20 = (getClose(CLOSE,0)-getDelay(CLOSE,6))/getDelay(CLOSE,6)*100
print ("alpha_1: ", alpha_1)
print ("alpha_2: ", alpha_2)
print ("alpha_3: ", alpha_3)
print ("alpha_13: ", alpha_13)
print ("alpha_14: ", alpha_14)
print ("alpha_15: ", alpha_15)
print ("alpha_18: ", alpha_18)
print ("alpha_19: ", alpha_19)
print ("alpha_20: ", alpha_20)
