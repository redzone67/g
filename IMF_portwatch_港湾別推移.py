import pandas as pd
import japanize_matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.simplefilter('ignore')

# ダウンロードしたファイルのディレクトリ
url = "/Users/XXXX/Downloads/Daily_Port_Activity_Data_and_Trade_Estimates.csv"

# DataFrameに格納
df = pd.read_csv(url)

# カラムの内容
'''
 'date',
 'year',
 'month',
 'day',
 'portid',
 'portname',
 'country',
 'ISO3',
 'portcalls_container',
 'portcalls_dry_bulk',
 'portcalls_general_cargo',
 'portcalls_roro',
 'portcalls_tanker',
 'portcalls_cargo',
 'portcalls',
 'import_container',
 'import_dry_bulk',
 'import_general_cargo',
 'import_roro',
 'import_tanker',
 'import_cargo',
 'import',
 'export_container',
 'export_dry_bulk',
 'export_general_cargo',
 'export_roro',
 'export_tanker',
 'export_cargo',
 'export',
 'ObjectId'
'''
# コンテナと一般カーゴの合計を生成（オプション）
df['container & gen_cargo']=df['import_container']+df['import_general_cargo']

# プロットしたいデータを選択
target = 'container & gen_cargo'

dfx = pd.pivot_table(df[df['country']=='United States'],index='date',columns='portname',values=target,aggfunc='sum')
dfx.index = [datetime.strptime(s[:10],"%Y/%m/%d") for s in list(dfx.index)]
dfz = dfx.resample('ME').sum()

# プロット構成
ncols = 2
nrows = math.ceil(len(dfz.columns) / ncols)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 2 * nrows))

#fig.suptitle(target, fontsize=16)

# Flattenして使いやすくする（77個）
axes = axes.flatten()

for i, col in enumerate(dfz.columns):

    x = pd.DataFrame(dfz[col].values,index=dfz.index,columns=[col])
    x['Month']=[str(s)[5:7] for s in x.index]
    x['Year']=[str(s)[:4] for s in x.index]

    x = pd.pivot_table(x,index='Month',columns='Year',values=col)

    ###### 2023年以降の推移を月次で比較
    x = x.loc[:,['2023','2024','2025']]
    x_max = x.max().max()
    x_min = x.min().min()

    axes[i].plot(x.index, x['2023'], label='2023', color='gray', linewidth=1.0)
    axes[i].plot(x.index, x['2024'], label='2024', color='blue', linewidth=1.0)
    axes[i].plot(x.index, x['2025'], label='2025', color='red', linewidth=3.0)
    axes[i].set_title(col)
    axes[i].grid(True)
    axes[i].set_ylim(x_min,x_max*1.5)

# 残ったサブプロットは非表示
for j in range(len(dfz.columns), len(axes)):
    axes[j].axis('off')

# --- 重要 --- 上下の間隔を調整する
plt.subplots_adjust(hspace=0.6)  # ← ここの値を大きくすると間隔が広がる
print(target)
plt.show()