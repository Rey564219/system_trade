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
            # ファイルを読み込む（まずは文字列で読み、後で型変換）
            df = pd.read_csv(file_path,
                             header=None,
                             names=["date", "time", "open", "high", "low", "close", "volumeto"],
                             dtype=str,
                             na_values=["", "NA", "NaN"])
            # date/time を文字列化し欠損を埋める
            df["date"] = df["date"].astype(str).str.strip()
            # time が欠損している場合は "00:00" を入れる（適宜調整）
            df["time"] = df["time"].fillna("00:00").astype(str).str.strip()

            # 日付と時間を結合して datetime 型に変換（失敗した行は NaT になる）
            datetime_str = df["date"] + " " + df["time"]
            df["close_time"] = pd.to_datetime(datetime_str, errors="coerce", infer_datetime_format=True)

            # 必要に応じて NaT を含む行をドロップ（もしくは前方埋めなど）
            df = df.dropna(subset=["close_time"]).reset_index(drop=True)

            # 数値列を適切な型に変換
            for col in ["open", "high", "low", "close", "volumeto"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # 不要列を削除してインデックスに日時をセット
            df = df.drop(columns=["date", "time"])
            dataframes.append(df)
            print(filename)

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
file_name = f"/data/{pair}.csv"  # 適切なファイル名を作成
print(price_data)
price_data.to_csv(file_name, index=False)  # ファイルを保存
print(pair,'finish')

