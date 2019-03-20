
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy
import matplotlib.pyplot as plt
plt.rcParams['font.serif'] = ['KaiTi']     #用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
import seaborn as sns
sns.set_style({"font.sans-serif":['KaiTi', 'Arial']},{"axes.unicode_minus":False})

#%% 数据读取
read_path = "/Users/ilio/Desktop/"
data = pd.read_excel(read_path + 'TradeData.xlsx')
open_ = pd.DataFrame(data['open'])
close_ = pd.DataFrame(data['close'])
high_ = pd.DataFrame(data['high'])
low_ =pd.DataFrame(data['low'])
volumn_ = pd.DataFrame(data['vol'])
timelist = data.index

#%% Indicator generator

#ADX generator
TR_1 = pd.DataFrame(data['high']-data['low'])
TR_2 = pd.DataFrame(abs(data['high'] - data['close'].shift(1)))
TR_3 = pd.DataFrame(abs(data['low'] - data['close'].shift(1)))
TR_t = pd.concat([TR_1,TR_2,TR_3],axis=1).max(axis =1)
TR = pd.DataFrame(TR_t)
TR = TR.iloc[1:,:]

HD = pd.DataFrame(data['high']-data['high'].shift(1))
LD = pd.DataFrame(data['low']-data['low'].shift(1))

LD_flag = deepcopy(LD)
LD_flag[LD_flag > 0] = 1
LD_flag[LD_flag <= 0] = 0

HD_flag = deepcopy(HD)
HD_flag[HD_flag > 0] = 1
HD_flag[HD_flag <= 0] = 0

HL_flag = np.array(HD)-np.array(LD)
HL_flag[HL_flag > 0] = 1
HL_flag[HL_flag <=0] = 0

LH_flag = deepcopy(HL_flag)
LH_flag = -(LH_flag - 1)

def ma(temp,n):
    leng = len(temp)
    ma_ = deepcopy(temp)
    ma_.ix[:]=np.nan
    for t in range(n-1,leng):
        te = temp.iloc[t-n+1:t,0]
        ma_temp = te.mean()
        ma_.ix[t,:] = ma_temp
    return ma_

def DMM(n):
    DMM_flag = LD_flag * LH_flag
    DMM_temp = DMM_flag * LD
    DMM_1 = DMM_temp.iloc[1:,:]
    dmm = ma(DMM_1,n)
    return dmm

def DMP(n):
    DMP_flag = HD_flag * HL_flag
    DMP_temp = DMP_flag * HD 
    DMP_1 = DMP_temp.iloc[1:,:]
    dmp = ma(DMP_1,n)
    return dmp

def PDI(n):
    pdi = np.array(DMP(n))* 100 / np.array(TR)
    pdi = pd.DataFrame(pdi).iloc[n-1:,:]
    pdi = pdi.reset_index(drop=True)
    return pdi
def MDI(n):
    mdi = np.array(DMM(n))* 100 / np.array(TR)
    mdi = pd.DataFrame(mdi).iloc[n-1:,:]
    mdi = mdi.reset_index(drop=True)
    return mdi

def ADX(m,n):  
    temp = abs((PDI(n) - MDI(n))/(PDI(n) + MDI(n)))*100
    adx = ma(temp,m)
    adx = adx.iloc[m:,:]
    adx = adx.reset_index(drop=True)
    return adx
#RSI generator
def EMA(S,n):
    ema = deepcopy(S)
    length = len(S.index)
    for t in range(1,length):
        ema.ix[t,:] = 2*S.ix[t,:] / (n + 1) + (n - 1)*ema.ix[t - 1,:] /(n + 1) 
    return ema
def RSI(n, type_):
    if type_ == 1:
        S = close_
    else:
        S = open_
    U = S - S.shift(1)
    D = -U
    U[U < 0] = 0    
    D[D < 0] = 0
    U = U.iloc[1:,:]
    D = D.iloc[1:,:]
    rsi = 100 - 100/(1+ EMA(U,n) / EMA(D,n))
    return rsi

#Stratage

#trade fee : 2 pt

#%stratage 1
# long condition: ADX > 15 and MA5 > MA30 
# short condition: ADX > 15 and MA5 < MA30 

#%stratage 2 
# long condition: ADX < 15 and RSI < 5
# short condition: ADX < 15 and RSI >95

#% rule:
# trade signal >= 20 days

ADX14 = ADX(14,14)
ADX14_MA5 = ma(ADX14,5)
ADX14_MA30 = ma(ADX14,30)
RSI2 =RSI(2,1).iloc[-len(ADX14):,:]

S = close_.iloc[-len(ADX14):,:]
index_ = S.index

ADX14.index = [index_]
ADX14_MA5.index = [index_]
ADX14_MA30.index = [index_]
RSI2.index = [index_]

ADX14.to_excel('ADX14.xlsx')
RSI2.to_excel('RSI2.xlsx')


dailyPnL = deepcopy(ADX14)
dailyPnL.ix[:] = 0
fl = deepcopy(dailyPnL)
trans = deepcopy(dailyPnL)

fl[(ADX14 > 15)& (ADX14_MA5 > ADX14_MA30)] = 1
fl[(ADX14 > 15)& (ADX14_MA5 < ADX14_MA30)] = -1
fl[(ADX14 < 15)& (RSI2 < 5)] = 2
fl[(ADX14 < 15)& (RSI2 > 95)] = -2
#填充
flip = deepcopy(fl)
full = 0
for t in range(1,len(flip)):
    if flip.ix[t,0] == 0:
        if flip.ix[t-1,0] != 0:
            full += 1
            if full <= 20:
                flip.ix[t,0] = flip.ix[t-1,0]
            else:
                full = 0;
                continue
    else:
        full = 0;
#策略天数限制     
count = 0
for t in range(1,len(flip)):
    if (flip.ix[t,0] == flip.ix[t-1,0])&(flip.ix[t,0] != 0) :
        count += 1
        if count > 20:
            flip.ix[t,0] = 0
    else:
        count = 0;
flip.index = [index_]

#策略变化
trans[(flip != flip.shift(1))&(flip * flip.shift(1) != 0)] = 1

#交易时间点
flip_1 = deepcopy(dailyPnL)

flip_1[flip > 0] = 1
flip_1[flip < 0] = -1

flip_2 = deepcopy(flip_1)

flip_1 = flip * flip.shift(1)


        
tradept = pd.DataFrame(index = index_, columns = ['0'])
tradept.ix[:] = 0
for t in range(1,len(flip)):
    if (flip.ix[t,0] != flip.ix[t-1,0])&(flip.ix[t,0] != 0):
        tradept.ix[t,0] = 1


delta = S.shift(-1) - S 


#日收益率
DailyPnL = pd.DataFrame(np.array(delta) * np.array(flip_2) - np.array(tradept)*2)
DailyPnL.index = [index_]

long_flip = deepcopy(flip)
long_flip[flip > 0] = 1
long_flip[flip <= 0] = 0

short_flip = deepcopy(flip)
short_flip[flip < 0] = 1
short_flip[flip >= 0] = 0
#累计收益

def CUM(temp):
    cum = deepcopy(temp)
    
    for t in range(1,len(cum.index)):
        cum.ix[t] = cum.ix[t-1] + cum.ix[t]
    return cum
CumPnL = CUM(DailyPnL)

DailyPnL.to_excel('DailyPnL.xlsx')
CumPnL.to_excel('CumPnL.xlsx')
flip_2.to_excel('flip.xlsx')
fl.to_excel('tradesignal.xlsx')



def MX(dataframe):
    cout=[];
    for i in range(len(dataframe)):
        l=[];
        for j in range(len(dataframe.columns)):
            l.append(((dataframe+1).cumprod(axis=1).ix[i,j]-(dataframe+1).cumprod(axis=1).ix[i,:j].max())/(dataframe+1).cumprod(axis=1).ix[i,:j].max())
        cout.append(np.nanmin(np.array(l)));
    return cout[0]


def cumr(indicator):
    cum = deepcopy(indicator.shift(1)+1)
    cum.ix[0] = 1
    for t in range(1,len(cum.index)):
        cum.ix[t] *= cum.ix[t-1]
    return cum

total_return_ = pd.DataFrame(np.array(DailyPnL)/np.array(S)).T
long_return_ = pd.DataFrame(np.array(total_return_) * np.array(long_flip.T))
short_return_ = pd.DataFrame(np.array(total_return_) * np.array(short_flip.T))

total_drawdown = MX(total_return_)
long_drawdown = MX(long_return_)
short_drawdown = MX(short_return_)

total_c_return = cumr(total_return_.T)
long_c_return = cumr(long_return_.T)
short_c_return = cumr(short_return_.T)


