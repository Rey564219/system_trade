
from turtle import pd
import plotly.graph_objects as go

SYMBOL = 'BTC'
TO_SYMBOL = 'USD'
pair = SYMBOL + TO_SYMBOL
interval = 60
dataframes = []

dataframes = []
# ファイルを読み込む
df = pd.read_csv(pair+".csv", 
                header=None,  # ヘッダーなしの場合
                names=["date", "time", "open", "high", "low", "close", "volumeto"])  # カラム名を指定
# 日付と時間を結合してdatetime型に変換
df['close_time'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y.%m.%d %H:%M')
# 必要ない列（dateとtime）を削除
df = df.drop(columns=["date", "time"])
# リストに追加
dataframes.append(df)

price_data = pd.concat(dataframes, ignore_index=True)
# リストに追加
dataframes.append(df)
df = pd.concat(dataframes, ignore_index=True)

df_five = pd.DataFrame()
rule = "5T"
df_five["Open"] = df["open"].resample(rule).first()
df_five["Close"] = df["close"].resample(rule).last()
df_five["High"] = df["high"].resample(rule).max()
df_five["Low"] = df["low"].resample(rule).min()

# 複数のデータフレームを1つに結合

print(df.head())

df_five.to_csv(pair, index=False)