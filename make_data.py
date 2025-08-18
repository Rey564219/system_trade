import numpy as np
import pandas as pd
import talib
import requests
from sklearn.linear_model import LogisticRegression
from datetime import datetime, timedelta
from sklearn.impute import SimpleImputer
import talib
import requests
from datetime import datetime, timedelta, timezone, time as dt_time
from scipy import signal
from pykalman import KalmanFilter
from textblob import TextBlob
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
import os

SYMBOL = 'BTC'
TO_SYMBOL='USD'

def make_train_data(pair,n_features=30):
    folder_path = pair  # フォルダのパスを指定

    # データを格納するリスト
    dataframes = []

    # フォルダ内のCSVファイルを読み込む
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            # ファイルを読み込む
            df = pd.read_csv(file_path, 
                            header=None,  # ヘッダーなしの場合
                            names=["date", "time", "open", "high", "low", "close", "volumeto"])  # カラム名を指定
            # 日付と時間を結合してdatetime型に変換
            df['close_time'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y.%m.%d %H:%M')
            # 必要ない列（dateとtime）を削除
            df = df.drop(columns=["date", "time"])
            # リストに追加
            dataframes.append(df)
        print(filename)

    # 複数のデータフレームを1つに結合

    price_data = pd.concat(dataframes, ignore_index=True)
    
    price_data['close_time'] = price_data['close_time'].ffill()
    # インデックスを設定
    price_data.set_index('close_time', inplace=True)

    price_data = price_data.replace([np.inf, -np.inf], np.nan)  # InfをNaNに置き換え
    
    price_data['close'] = price_data['close'].ffill()

    return price_data

    # 必要ならばCSVに保存

pair = SYMBOL + TO_SYMBOL
price_data=make_train_data(pair)
file_name = f"{pair}.csv"  # 適切なファイル名を作成
print(price_data)
price_data.to_csv(file_name, index=False)  # ファイルを保存
print(pair,'finish')

