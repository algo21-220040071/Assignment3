# -*- coding: utf-8 -*-
"""
@author: Liran CHEN 220040071
"""
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import model_selection
import tushare as ts

#get the stock data from tushare
#use your own tushare token
TOKEN = ''
ts.set_token(TOKEN)
pro = ts.pro_api()
#get the daily price of stock 600309.SH
#daily = pd.read_excel('./data.xlsx')
daily = pro.daily(ts_code = '600309.SH', start_date = '20190101',
                  end_date = '20210430', 
                  fields = 'trade_date,open,high,low,close')


#使用前150天的数据，预测当天的涨跌
dayfeature=150
featurenum=5*dayfeature
x=np.zeros((daily.shape[0]-dayfeature,featurenum+1))
y=np.zeros((daily.shape[0]-dayfeature))

for i in range(0,daily.shape[0]-dayfeature):
    x[i,0:featurenum]=np.array(daily[i:i+dayfeature] \
          [['trade_date','open','high','low','close']]).reshape((1,featurenum))
    x[i,featurenum]=daily.ix[i+dayfeature]['open']
 
for i in range(0,daily.shape[0]-dayfeature):
    if daily.ix[i+dayfeature]['close']>=daily.ix[i+dayfeature]['open']:
        y[i]=1
    else:
        y[i]=0          
 
#SVM进行5次预测，输出每一次预测的准确度
clf=svm.SVC(kernel='rbf')
result = []
for i in range(5):
    x_train, x_test, y_train, y_test = \
                model_selection.train_test_split(x, y, test_size = 0.2)
    clf.fit(x_train, y_train)
    result.append(np.mean(y_test == clf.predict(x_test)))
print("svm classifier accuacy:")
print(result)
