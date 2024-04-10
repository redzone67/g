import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import yfinance as yf

discounts = ['DLTR','FIVE','DG','BIG','COST','TJX','TGT','BURL','PSMT','ROST','WMT','KR','^GSPC']


####################年次データ
cols = 2  # 1行あたりのサブプロットの数
num = dataX.shape[1]-1
rows = num // cols + (num % cols > 0)  # 必要な行数

plt.figure(figsize=(12, rows * 4))

data = []
k = 0
for s in discounts[:-1]:  # 各固有ベクトルに対して
    plt.subplot(rows, cols, k + 1)  # サブプロットの位置を指定

    dd = yf.Ticker(s)
    df = dd.get_financials().loc[['TotalRevenue','GrossProfit'],:].T.sort_index()
    df.index = [s.strftime('%Y%m') for s in list(df.index)]
    df['Rev growth']=df['TotalRevenue'].diff()/df['TotalRevenue'].shift(1)*100
    df['GrossMargin']=df['GrossProfit']/df['TotalRevenue']*100
    data += [df]
    plt.plot(df['GrossMargin'])

    plt.xlabel('date')
    plt.title(s)

    k += 1
    
plt.tight_layout()
plt.show()


####################四半期データ
cols = 2  # 1行あたりのサブプロットの数
num = dataX.shape[1]-1
rows = num // cols + (num % cols > 0)  # 必要な行数

plt.figure(figsize=(12, rows * 4))

dataq = []
k = 0
for s in discounts[:-1]:  # 各固有ベクトルに対して
    plt.subplot(rows, cols, k + 1)  # サブプロットの位置を指定

    dd = yf.Ticker(s)
    df = dd.get_income_stmt(freq='quarterly').loc[['TotalRevenue','GrossProfit'],:].T.sort_index()
    df.index = [s.strftime('%Y%m') for s in list(df.index)]
    df['Rev growth']=df['TotalRevenue'].diff()/df['TotalRevenue'].shift(1)*100
    df['GrossMargin']=df['GrossProfit']/df['TotalRevenue']*100
    df.columns = ['TotalRevenue_q','GrossProfit_q','Rev growth_q','GrossMargin_q']
    dataq += [df]
    plt.plot(df['GrossMargin_q'])

    plt.xlabel('date')
    plt.title(s)

    k += 1
    
plt.tight_layout()
plt.show


####################　統合
cols = 2  # 1行あたりのサブプロットの数
num = dataX.shape[1]-1
rows = num // cols + (num % cols > 0)  # 必要な行数

plt.figure(figsize=(12, rows * 4))


k = 0
for s in discounts[:-1]:  # 各固有ベクトルに対して
    plt.subplot(rows, cols, k + 1)  # サブプロットの位置を指定

    df = pd.concat([data[k],dataq[k]],axis=1,join='outer').sort_index()
    plt.plot(df.loc[:,['GrossMargin','GrossMargin_q']])

    plt.xlabel('date')
    plt.title(s)

    k += 1
    
plt.tight_layout()
plt.show()