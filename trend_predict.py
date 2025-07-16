import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import requests, time
from textblob import TextBlob
from datetime import datetime, timedelta, timezone, time as dt_time
import tensorflow as tf
from tensorflow.keras import mixed_precision
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import schedule  # Add this import for scheduling tasks
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.preprocessing import StandardScaler 
from sklearn.utils.class_weight import compute_sample_weight
import talib
from pykalman import KalmanFilter
from scipy import signal
from sklearn.model_selection import TimeSeriesSplit
from collections import deque
import json
import schedule
import os
import MetaTrader5 as mt5
try:
    import backtrader as bt
except ImportError:
    print("警告: backtraderが利用できません")
import time
import os
import MetaTrader5 as mt5
import datetime
import backtrader as bt
import matplotlib.pyplot as plt

SYMBOL = "USD"
TO_SYMBOL = "JPY"

DISCORD_WEBHOOK_URL = 'https://discord.com/api/webhooks/1330753855540957226/JVHZYJr9br6aeHOLMc5AKvgqzgwTEO1BNhzNQKg7R3e23cWlR2ojmLanEY3Kyvytazl6'
def send_discord_message(message: str):
    data = {"content": message}
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=data)
        if response.status_code != 204:
            print(f"Discord送信失敗: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Discord送信エラー: {e}")
def mt5_initialize(login, password, server):
    if not mt5.initialize(login=login, password=password, server=server):
        print("MT5接続失敗", mt5.last_error())
        return False
    print("MT5接続成功")
    return True
# --- 成行注文関数 ---
def mt5_market_order(symbol, order_type, lot, price):
    if order_type == "BUY":
        order_type_mt5 = mt5.ORDER_TYPE_BUY
    elif order_type == "SELL":
        order_type_mt5 = mt5.ORDER_TYPE_SELL
    else:
        raise ValueError("order_type must be 'BUY' or 'SELL'")
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type_mt5,
        "price": price,
        "deviation": 10,
        "magic": 234000,
        "comment": "Python market order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"成行注文失敗: {result.retcode}, {result.comment}")
    else:
        print(f"成行注文成功: {order_type} {price} lot={lot}")
    return result

# --- TP/SL同時エントリーの指値注文関数 ---
def mt5_pending_order(symbol, order_type, lot, price, sl, tp):
    if order_type == "BUY":
        order_type_mt5 = mt5.ORDER_TYPE_BUY
    elif order_type == "SELL":
        order_type_mt5 = mt5.ORDER_TYPE_SELL
    else:
        raise ValueError("order_type must be 'BUY' or 'SELL'")
    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": lot,
        "type": order_type_mt5,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": 234000,
        "comment": "Python pending order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"指値注文失敗: {result.retcode}, {result.comment}")
    else:
        print(f"指値注文成功: {order_type} {price} lot={lot} TP={tp} SL={sl}")
    return result

# --- 注文キャンセル関数 ---
def mt5_cancel_order(ticket):
    request = {
        "action": mt5.TRADE_ACTION_REMOVE,
        "order": int(ticket),
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"注文キャンセル失敗: {result.retcode}, {result.comment}")
    else:
        print(f"注文キャンセル成功: ticket={ticket}")
    return result



def get_economic_calendar():
    try:
        # 例: 無料のForexFactoryのAPIを使用
        url = "https://financialmodelingprep.com/api/v3/economic_calendar"
        params = {
            "from": datetime.now().strftime("%Y-%m-%d"),
            "to": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
            "apikey": "fAGWDRuOnl6QTnXw6cIm"  # 適切なAPIキーに置き換えてください
        }
        response = requests.get(url, params=params)
        events = response.json()
        for event in events:
            if 'date' in event:
                event['date'] = pd.to_datetime(event['date']).date()
        return events
    except Exception as e:
        print(f"Error fetching economic calendar: {e}")
        return []


def get_news_sentiment(symbol):
    try:
    # 例: NewsAPIを使用
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey=413c97dda27d4c73ae174d952391cb17"
        response = requests.get(url)
        articles = response.json().get('articles', [])

        sentiments = []
        for article in articles[:5]:  # 最新の5記事を分析
            title = article.get('title', '')
            description = article.get('description', '')
            if title is None:
                title = ''
            if description is None:
                description = ''
            blob = TextBlob(title + " " + description)
            sentiments.append(blob.sentiment.polarity)
        return sum(sentiments) / len(sentiments) if sentiments else 0
    except Exception as e:
        return 0

def calc_rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ema_up = up.ewm(span=window).mean()
    ema_down = down.ewm(span=window).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

def generate_all_feature_diffs(
    df: pd.DataFrame,
    exclude_cols: list = None,
    diff_period: int = 5,
    shift_steps: int = 1
) -> pd.DataFrame:
    """
    DataFrame内のすべての特徴量に対して、指定された期間の差分（Diff）を
    指定されたステップ分シフトさせて生成する。

    Parameters:
        df (pd.DataFrame): 特徴量を含む元のデータ
        exclude_cols (list): 差分計算の対象から除外するカラム名のリスト
        diff_period (int): 差分を取る期間（デフォルト: 5分）
        shift_steps (int): 差分特徴量を未来にずらすステップ数（デフォルト: 1）

    Returns:
        pd.DataFrame: 元のDataFrameに差分特徴量を追加したもの
    """
    df = df.copy()
    exclude_cols = exclude_cols or []
    target_cols = [col for col in df.columns if col not in exclude_cols]

    for col in target_cols:
        diff_col_name = f'diff_{col}_p{diff_period}_s{shift_steps}'
        df[diff_col_name] = (df[col] - df[col].shift(diff_period)).shift(shift_steps)

    return df
# --- 特徴量作成（デノイジング追加） ---
def make_features(df):
    # デノイジング用に移動平均を追加
    df['ma_short'] = df['close'].rolling(window=5).mean()
    df['ma_mid'] = df['close'].rolling(window=15).mean()
    df['ma_long'] = df['close'].rolling(window=50).mean()
    
    # 移動平均の傾き
    df['ma_short_slope'] = df['ma_short'].diff()
    df['ma_mid_slope'] = df['ma_mid'].diff()
    df['ma_long_slope'] = df['ma_long'].diff()
    
    # 移動平均クロス
    df['ma_cross_short_mid'] = df['ma_short'] - df['ma_mid']
    df['ma_cross_short_long'] = df['ma_short'] - df['ma_long']
    df['ma_cross_mid_long'] = df['ma_mid'] - df['ma_long']
    
    # ATR (Average True Range)
    df['prev_close'] = df['close'].shift(1)
    
    df['tr'] = np.maximum.reduce([
        df['high'] - df['low'],
        abs(df['high'] - df['prev_close']),
        abs(df['low'] - df['prev_close'])
    ])
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    # 追加特徴量
    df['return_1'] = df['close'].pct_change(1)
    df['return_3'] = df['close'].pct_change(6)
    df['volatility_5'] = df['close'].rolling(window=5).std()
    df['volatility_15'] = df['close'].rolling(window=15).std()
    
    # ADX, SMA_5, SMA_20を必ず計算
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['SMA_5'] = talib.SMA(df['close'], timeperiod=5)
    df['SMA_20'] = talib.SMA(df['close'], timeperiod=15)
    df['SMA_100'] = talib.SMA(df['close'], timeperiod=100)
    
    # DMI指標を追加
    df['PLUS_DI'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    df['MINUS_DI'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    
    return df
def is_bullish_candles(df, n=2):
    for i in range(1, n + 1):
        if df.iloc[-i]['close'] <= df.iloc[-i]['open']:
            return False
    return True
def is_ma_crossed(df, threshold=0.001):  # 0.001 = 0.1%
    """
    EMAクロス後、乖離率がthreshold以上ならTrue
    """
    latest = df.iloc[-1]

    # クロス済み（短期 > 長期）かつ 乖離率が一定以上
    if latest['ema_fast'] > latest['ema_slow']:
        spread_ratio = (latest['ema_fast'] - latest['ema_slow']) / latest['ema_slow']
        return spread_ratio >= threshold
    return False

def is_bb_breakout(df):
    latest = df.iloc[-1]
    return latest['close'] > latest['bb_upper']
def is_entry_trigger(df):
    """
    エントリートリガーを判定する関数
    """
    try:
        return (
            is_bullish_candles(df, n=2) or
            is_ma_crossed(df) or
            is_bb_breakout(df)
        )
    except Exception:
        # エラーが発生した場合はFalseを返す
        return False
def generate_technical_signal(df, adx_period=14, sma_short=5, sma_mid=20, adx_threshold=20):
    """
    テクニカル指標に基づくシグナル生成
    """
    signals = []
    
    # 必要な指標が存在するか確認
    required_columns = ['ADX', f'SMA_{sma_short}', f'SMA_{sma_mid}', 'SMA_100', 'close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"警告: 必要な列が不足しています: {missing_columns}")
        # 不足している列を簡易的に計算
        if 'ADX' not in df.columns:
            df['ADX'] = 25.0  # デフォルト値
        if f'SMA_{sma_short}' not in df.columns:
            df[f'SMA_{sma_short}'] = df['close'].rolling(sma_short).mean()
        if f'SMA_{sma_mid}' not in df.columns:
            df[f'SMA_{sma_mid}'] = df['close'].rolling(sma_mid).mean()
        if 'SMA_100' not in df.columns:
            df['SMA_100'] = df['close'].rolling(100).mean()
    
    # NaN値を削除
    df = df.dropna()
    
    if len(df) < 100:
        print(f"警告: データが不足しています ({len(df)}行)")
        return ["HOLD"] * len(df)
    
    try:
        adx = df['ADX'].values
        sma_s = df[f'SMA_{sma_short}'].values
        sma_m = df[f'SMA_{sma_mid}'].values
        sma_l = df['SMA_100'].values
        close = df['close'].values
        
        # === EMA（移動平均） ===
        try:
            df['ema_fast'] = talib.EMA(df['close'], timeperiod=5)
            df['ema_slow'] = talib.EMA(df['close'], timeperiod=20)
        except:
            # TALibが使えない場合は通常の移動平均を使用
            df['ema_fast'] = df['close'].rolling(5).mean()
            df['ema_slow'] = df['close'].rolling(20).mean()

        # === ボリンジャーバンド ===
        try:
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
                df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
        except:
            # TALibが使えない場合は手動計算
            rolling_mean = df['close'].rolling(20).mean()
            rolling_std = df['close'].rolling(20).std()
            df['bb_upper'] = rolling_mean + (rolling_std * 2)
            df['bb_middle'] = rolling_mean
            df['bb_lower'] = rolling_mean - (rolling_std * 2)
        
        # シグナル生成（条件を緩和）
        for i in range(100, len(df)):  # 最初の100行はスキップ
            try:
                # 基本的な条件
                adx_now = adx[i] if not np.isnan(adx[i]) else 25.0
                sma_order_up = sma_s[i] > sma_m[i]
                sma_order_down = sma_s[i] < sma_m[i]
                
                # 簡易トリガー判定（条件を緩和）
                close_above_sma_long = close[i] > sma_l[i]
                close_below_sma_long = close[i] < sma_l[i]
                
                # ADXがある程度の値を持つ場合のみ判定
                if adx_now > adx_threshold:
                    if sma_order_up and close_above_sma_long:
                        signals.append("CALL")
                    elif sma_order_down and close_below_sma_long:
                        signals.append("PUT")
                    else:
                        signals.append("HOLD")
                else:
                    signals.append("HOLD")
                    
            except Exception as e:
                signals.append("HOLD")
        
        # 先頭100個はHOLDで埋める
        signals = ["HOLD"] * 100 + signals
        
        # 長さを調整
        while len(signals) < len(df):
            signals.append("HOLD")
        signals = signals[:len(df)]
        
        # シグナル統計を表示
        call_count = signals.count("CALL")
        put_count = signals.count("PUT")
        hold_count = signals.count("HOLD")
        print(f"[シグナル生成] CALL: {call_count}, PUT: {put_count}, HOLD: {hold_count}")
        
        return signals
        
    except Exception as e:
        print(f"シグナル生成エラー: {e}")
        return ["HOLD"] * len(df)
# def calc_winrate_technical(df,open, close, entry_minutes=10):
#     """
#     テクニカル指標によるシグナルでの勝率計算
#     df: 特徴量付きDataFrame
#     close: 終値Series
#     entry_minutes: エントリー後の判定分数
#     """
#     signals = generate_technical_signal(df)
#     entries = []
#     results = []
#     close = close.reset_index(drop=True)
#     for i, sig in enumerate(signals):
#         if sig in ("CALL", "PUT"):
#             if i+entry_minutes+1 < len(close) and i+1 < len(close):
#                 entry_price = close[i+1]
#                 exit_price = close[i+1+entry_minutes]
#                 win = (exit_price > entry_price + 0.02) if sig == "CALL" else (exit_price < entry_price - 0.02)
#                 entries.append(i)
#                 results.append(win)
#     winrate = sum(results) / len(results) if results else None
#     print(f"[テクニカル] {entry_minutes}分 勝率: {winrate:.2%}" if winrate is not None else "[テクニカル] 勝率: データ不足")
#     return winrate, entries, results, signals

def calc_winrate_technical(
    df, open_, close, high=None, low=None,
    entry_minutes=10, tp_pips=5, sl_pips=3, spread=0.02, symbol="USD/JPY",
    lot=1000, start_balance=50000, leverage=3
):
    signals = generate_technical_signal(df)
    entries = []
    results = []
    pips_unit = 0.01 if "JPY" in symbol else 0.0001
    tp_value = tp_pips * pips_unit
    sl_value = sl_pips * pips_unit

    open_ = open_.reset_index(drop=True)
    close = close.reset_index(drop=True)
    if high is None:
        high = df["high"].reset_index(drop=True)
    else:
        high = high.reset_index(drop=True)
    if low is None:
        low = df["low"].reset_index(drop=True)
    else:
        low = low.reset_index(drop=True)
    adx = df['ADX'].values

    balance = start_balance
    balance_curve = [balance]

    for i, sig in enumerate(signals):
        if sig in ("CALL", "PUT"):
            if i + entry_minutes + 1 < len(close) and i + 1 < len(open_):
                entry_price = close[i + 1] + spread if sig == "CALL" else close[i + 1] - spread
                win = False
                if sig == "CALL" and adx[i] > 20:
                    tp_line = entry_price + tp_value
                    sl_line = entry_price - sl_value
                    for t in range(entry_minutes):
                        hi = high[i + 1 + t]
                        lo = low[i + 1 + t]
                        # 先にTP到達
                        if hi >= tp_line and lo <= tp_line:
                            gain = lot * tp_value * leverage
                            balance += gain
                            win = True
                            break
                        # 先にSL到達
                        elif hi >= sl_line and lo <= sl_line:
                            loss = lot * sl_value * leverage
                            balance -= loss
                            win = False
                            break
                    else:
                        # どちらも未到達→entry_minutes後のcloseで決済
                        exit_price = close[i + 1 + entry_minutes]
                        profit = (exit_price - entry_price - spread) * lot * leverage
                        balance += profit
                        win = profit > 0
                    balance_curve.append(balance)
                elif sig == "PUT" and adx[i] > 20:
                    tp_line = entry_price - tp_value
                    sl_line = entry_price + sl_value
                    for t in range(entry_minutes):
                        hi = high[i + 1 + t]
                        lo = low[i + 1 + t]
                        if hi >= tp_line and lo <= tp_line:
                            gain = lot * tp_value * leverage
                            balance += gain
                            win = True
                            break
                        elif hi >= sl_line and lo <= sl_line:
                            loss = lot * sl_value * leverage
                            balance -= loss
                            win = False
                            break
                    else:
                        exit_price = close[i + 1 + entry_minutes]
                        profit = (entry_price - exit_price - spread) * lot * leverage
                        balance += profit
                        win = profit > 0
                    balance_curve.append(balance)

                entries.append(i)
                results.append(win)

                if balance <= 0:
                    print("資金が尽きました。")
                    break

    winrate = sum(results) / len(results) if results else None
    print(f"[テクニカル] {entry_minutes}分 TP/SL勝率: {winrate:.2%} ({len(results)}回)" if winrate is not None else "[テクニカル] 勝率: データ不足")
    print(f"[最終資金] {balance:.2f}円 / ドローダウン最大: {max(balance_curve) - min(balance_curve):.2f}円")
    return winrate, entries, results, signals

def calc_winrate_technical2(
    df, open_, close, high=None, low=None, tp_pips=5, sl_pips=3, spread=0.02, symbol="USD/JPY",
    lot=1000, start_balance=50000, leverage=3
):
    signals = generate_technical_signal(df)
    entries = []
    results = []
    pips_unit = 0.01 if "JPY" in symbol else 0.0001
    tp_value = tp_pips * pips_unit
    sl_value = sl_pips * pips_unit

    open_ = open_.reset_index(drop=True)
    close = close.reset_index(drop=True)
    if high is None:
        high = df["high"].reset_index(drop=True)
    else:
        high = high.reset_index(drop=True)
    if low is None:
        low = df["low"].reset_index(drop=True)
    else:
        low = low.reset_index(drop=True)
    adx = df['ADX'].values

    balance = start_balance
    balance_curve = [balance]

    for i, sig in enumerate(signals):
        if sig in ("CALL", "PUT") and i + 1 < len(open_):
            entry_price = close[i + 1] + spread if sig == "CALL" else close[i + 1] - spread
            t = 0
            hit = None
            if sig == "CALL":
                tp_line = entry_price + tp_value
                sl_line = entry_price - sl_value
                while i + 1 + t < len(df):
                    hi = high[i + 1 + t]
                    lo = low[i + 1 + t]
                    if hi >= tp_line and lo <= tp_line:
                        balance += lot * tp_value * leverage
                        hit = True
                        break
                    elif hi >= sl_line and lo <= sl_line:
                        balance -= lot * sl_value * leverage
                        hit = False
                        break
                    t += 1
                if hit is None:
                    exit_price = close[min(i + 1 + t, len(close) - 1)]
                    profit = (exit_price - entry_price - spread) * lot * leverage
                    balance += profit
                    hit = profit > 0
            elif sig == "PUT":
                tp_line = entry_price - tp_value
                sl_line = entry_price + sl_value
                while i + 1 + t < len(df):
                    hi = high[i + 1 + t]
                    lo = low[i + 1 + t]
                    if hi >= tp_line and lo <= tp_line:
                        balance += lot * tp_value * leverage
                        hit = True
                        break
                    elif hi >= sl_line and lo <= sl_line:
                        balance -= lot * sl_value * leverage
                        hit = False
                        break
                    t += 1
                if hit is None:
                    exit_price = close[min(i + 1 + t, len(close) - 1)]
                    profit = (entry_price - exit_price - spread) * lot * leverage
                    balance += profit
                    hit = profit > 0

            entries.append(i)
            results.append(hit)
            balance_curve.append(balance)
            if balance <= 0:
                print("資金が尽きました。")
                break

    winrate = sum(results) / len(results) if results else None
    print(f"[テクニカル] TP/SL追跡勝率: {winrate:.2%} ({len(results)}回)" if winrate is not None else "[テクニカル] 勝率: データ不足")
    print(f"[最終資金] {balance:.2f}円 / ドローダウン最大: {max(balance_curve) - min(balance_curve):.2f}円")
    return winrate, entries, results, signals

# --- 必要に応じてML予測と組み合わせる例 ---
def make_labels(df, tp_pips=10, sl_pips=5, entry_minutes=10, symbol="USD/JPY"):
    """
    一定pips動いたか（TP/SL）に基づいて CALL / PUT のラベルを生成する。
    クロス円なら1pips = 0.01、それ以外は0.0001。
    - label_up: CALLシグナルの結果
    - label_down: PUTシグナルの結果
    ラベル: 1=TP到達, -1=SL到達, 0=未到達
    """
    # 通貨ペアに応じた1pipsあたりの値幅を決定
    pips_unit = 0.01 if "JPY" in symbol else 0.0001
    tp_value = tp_pips * pips_unit
    sl_value = sl_pips * pips_unit

    # signals = generate_technical_signal(df)  # 既存シグナル関数を使用
    label_up, label_down = [], []

    open_ = df["open"].reset_index(drop=True)
    high = df["high"].reset_index(drop=True)
    low = df["low"].reset_index(drop=True)

    for i in range(len(df)):
        if i + entry_minutes + 1 >= len(df):
            label_up.append(0)
            label_down.append(0)
            continue

        entry_price = open_[i + 1]
        high_seq = high[i + 1 : i + 1 + entry_minutes]
        low_seq = low[i + 1 : i + 1 + entry_minutes]
        # 上昇ラベル
        tp_hit = (high_seq >= entry_price + tp_value).any()
        sl_hit = (low_seq <= entry_price - sl_value).any()
        if tp_hit and (not sl_hit or high_seq.idxmax() < low_seq.idxmin()):
            label_up.append(1)
        elif sl_hit:
            label_up.append(0)
        else:
            label_up.append(0)

        # 下降ラベル
        tp_hit = (low_seq <= entry_price - tp_value).any()
        sl_hit = (high_seq >= entry_price + sl_value).any()
        if tp_hit and (not sl_hit or low_seq.idxmin() < high_seq.idxmax()):
            label_down.append(1)
        elif sl_hit:
            label_down.append(0)
        else:
            label_down.append(0)

    df["label_up"] = label_up
    df["label_down"] = label_down
    return df

def apply_moving_average(data, window=5):
    return data.rolling(window=window).mean()

def apply_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)

    # NaN を補間してからフィルタ適用
    data = np.nan_to_num(data, nan=np.nanmean(data))
    return signal.filtfilt(b, a, data, method="pad")

def resample_data(data, rule='5T'):
    return data.resample(rule).last()
def apply_kalman_filter(data):
    
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaN or inf values before Kalman filtering.")

    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    
    try:
        filtered_data, _ = kf.em(data).smooth(data)
    except Exception as e:
        print("Kalmanフィルタ処理中にエラー:", e)
        return np.full_like(data, np.nan)  # エラー発生時は NaN を返す

    if np.any(np.isnan(filtered_data)) or np.any(np.isinf(filtered_data)):
        raise ValueError("Kalman filtering resulted in NaN or inf values.")

def preprocess_data(price_data):
    price_data['close_ma'] = apply_moving_average(price_data['close'])
    price_data['close_lowpass'] = apply_lowpass_filter(price_data['close'].values, cutoff=0.1, fs=1.0)
    # NaNを線形補間
    price_data['close'].interpolate(method='linear', inplace=True)
    price_data['close_ma'].interpolate(method='linear', inplace=True)
    price_data['close_lowpass'] = np.nan_to_num(price_data['close_lowpass'], nan=np.nanmean(price_data['close_lowpass']))

    price_data['close_kalman'] = apply_kalman_filter(price_data['close'].values)

    return price_data

def calculate_crypto_vix(close, window=30):
    returns = np.log(close / close.shift(1))
    hist_vol = returns.rolling(window=window).std() * np.sqrt(365)  # 年率換算
    return hist_vol * 100  # パーセンテージに変換
def add_log_transformed_features(df):
    """
    数値カラムを自動で対数変換し、元のデータと結合する関数
    負の値は絶対値を取って対数変換し、符号フラグを追加する

    Parameters:
    df (pd.DataFrame): 元のデータフレーム

    Returns:
    pd.DataFrame: 元のデータに対数変換カラムと符号フラグを追加したデータフレーム
    """

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    log_features = {}

    for col in numeric_cols:
        log_col = f'log_{col}'

        # 負の値がある場合は絶対値を取って対数変換
        log_features[log_col] = np.log1p(np.abs(df[col]))

    log_df = pd.DataFrame(log_features, index=df.index)
    df_log_transformed = pd.concat([df, log_df], axis=1)

    return df_log_transformed
def make_train_data(folder_path, n_features=20):
    file_path = pair + ".csv"
    folder_path = './'
    # データを格納するリスト
    dataframes = []
    for filename in os.listdir(folder_path):
        if filename.endswith(pair + ".csv"):
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
    df = pd.DataFrame(price_data)
    X = df.apply(pd.to_numeric, errors='coerce')
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    price_data = X.copy()

# --- 欠損値処理 ---
    price_data = make_features(price_data)
    drop_keywords = ['trend_direction', 'trend_duration', 'trend_duration_bucket', 'trend_confidence']
    drop_cols = [col for col in price_data.columns if any(key in col for key in drop_keywords)]
    X = price_data.drop(columns=drop_cols, errors='ignore')
    X = X.ffill().ffill()
    # --- 特徴量選択 ---
    features = ['atr', 'close', 'open', 'high', 'low',
        'ADX','SMA_5', 'SMA_20','PLUS_DI', 'MINUS_DI'
    ]
    if 'return_1min' in X.columns:
        X = X.drop(['return_1min'], axis=1)
    drop_keywords = ['trend_direction', 'trend_duration', 'trend_duration_bucket', 'trend_confidence','return_1min']
    drop_cols = [col for col in price_data.columns if any(key in col for key in drop_keywords)]
    X = price_data.drop(columns=drop_cols, errors='ignore')
    X = X.ffill().ffill()
    if "close" not in features:
        features.append("close")
    if "open" not in features:
        features.append("open")
    if "high" not in features:
        features.append("high")
    if "low" not in features:
        features.append("low")
    if "ADX" not in features:
        features.append("ADX")

    X = X[features]
    pca = None  # 必要ならPCAを適用
    print('finish make_train_data')
    # スケーラーはNoneで返す
    return X, None, None,None, X.columns.tolist(),  price_data['open'],price_data['close'], price_data['high'], price_data['low'], None, features, None, None, pca, None

def fetch_data(SYMBOL, TO_SYMBOL, col, scaler, X_train, y_train, features, pca, n_features=20):
    price = []
    params = {"fsym": SYMBOL, "tsym": TO_SYMBOL, "limit": 2000}
    while True:
        try:
            response = requests.get("https://min-api.cryptocompare.com/data/histominute", params, timeout=10)
            data = response.json()
            if data.get("Response") != "Success" or not data.get("Data"):
                time.sleep(10)
                continue
            break
        except Exception:
            time.sleep(10)

    for i in data["Data"]:
        price.append({
            "close_time": i["time"],
            "open": i["open"],
            "high": i["high"],
            "low": i["low"],
            "close": i["close"],
            "volumeto": i["volumeto"]
        })
    price_data = pd.DataFrame(price)
    price_data['close_time'] = pd.to_datetime(price_data['close_time'], unit='s')
    price_data.set_index('close_time', inplace=True)
    for coln in ['open', 'high', 'low', 'close', 'volumeto']:
        price_data[coln] = price_data[coln].astype(float)

    # --- 特徴量生成 ---
    price_data = make_features(price_data)
    price_data['rsi'] = calc_rsi(price_data['close'], window=14)
    open_price = price_data['open']
    high = price_data['high']
    low = price_data['low']
    close = price_data['close']
    volume = price_data['volumeto']
    hilo = (price_data['high'] + price_data['low']) / 2

    price_data = preprocess_data(price_data)
    price_data['close_kalman'] = pd.to_numeric(price_data['close_kalman'], errors='coerce')
    # インデックスをdatetime64[ns]型に変換
    price_data.index = pd.to_datetime(price_data.index) 


    # Add technical indicators (as before)
    price_data['ADX'] = talib.ADX(high, low, close, timeperiod=14)
    price_data['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
    price_data['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
    price2 = price_data[['open', 'high', 'low', 'close','volumeto']]

    # --- 欠損値処理 ---
    drop_keywords = ['trend_direction', 'trend_duration', 'trend_duration_bucket', 'trend_confidence','return_1min']
    drop_cols = [col for col in price_data.columns if any(key in col for key in drop_keywords)]
    X = price_data.drop(columns=drop_cols, errors='ignore')
    X = X.ffill().ffill()
    # featuresリストでX_test_selectedを再構成

    return X, None, None,  price_data['open'], price_data['close'], price_data['high'], price_data['low'], None, None, None
pair = SYMBOL + TO_SYMBOL
# make_train_dataを使って学習データ作成
X_train, X_trainb, y_train, y_trainb, col, open_price, close, high, low, label, select, _, _, pca, scaler = make_train_data(pair, n_features=20)

# fetch_dataを使ってテストデータ作成（SYMBOL, TO_SYMBOLは仮の値でOK）
X_test, y_test, feature_names, open_test, close_test, high_test, low_test, label_test, _, _ = fetch_data(SYMBOL, TO_SYMBOL, None, None, None, None, None, None, n_features=20)
df = X_train.copy()

# X_testのインデックスがDatetimeIndexの場合
X_test = X_test[X_test.index.weekday < 5]
# --- パイプライン ---
print('finish make_features')

print('finish make_labels')
df['rsi'] = calc_rsi(df['close'], window=14)
X = df.dropna()

features = [
    'atr', 'close', 'open', 'high', 'low',
    'ADX','SMA_5', 'SMA_20','PLUS_DI', 'MINUS_DI'
]
features = select
X = X[features]
def get_pips_scale(symbol):
    """
    通貨ペアに応じて1pipsあたりの数値を返す。
    例）USD/JPY → 0.01, EUR/USD → 0.0001
    """
    if 'JPY' in symbol:
        return 0.01
    else:
        return 0.0001
pips = get_pips_scale(TO_SYMBOL)

# --- 勝率計算ロジック追加 ---
print("=== 元の実装 ===")
calc_winrate_technical(X_test, open_test, close_test, high_test, low_test)
calc_winrate_technical2(X_test, open_test, close_test, high_test, low_test)
calc_winrate_technical(df, df['open'], df['close'], df['high'], df['low'])
calc_winrate_technical2(df, df['open'], df['close'], df['high'], df['low'])

# Backtraderテストは後で実行されるように分離
def run_backtrader_tests():
    """
    Backtraderテストを実行
    """
    print("\n=== Backtrader実装 ===")
    # テスト用データフレームの準備
    test_df = df.copy()
    if 'open' not in test_df.columns:
        test_df['open'] = df['open']
    if 'close' not in test_df.columns:
        test_df['close'] = df['close']
    if 'high' not in test_df.columns:
        test_df['high'] = df['high']
    if 'low' not in test_df.columns:
        test_df['low'] = df['low']

    # Backtraderテスト
    result1 = calc_winrate_technical_backtrader(test_df, tp_pips=5, sl_pips=3, entry_minutes=10, unlimited_tracking=False)
    result2 = calc_winrate_technical2_backtrader(test_df, tp_pips=5, sl_pips=3, unlimited_tracking=True)
    return result1, result2

def run_realtime_signals(SYMBOL, TO_SYMBOL, interval_minutes=1, n_features=20,tp_pips=6, sl_pips=3, spread=0.02, bet_ratio=0.01, min_bet=1000, start_balance=50000):
    pair = SYMBOL + TO_SYMBOL
    entry_minutes = 10
    leverage = 3
    spread = 0.02
    min_bet = 1000
    bet_ratio = 0.01
    start_balance = 50000

    orders_df = pd.DataFrame(columns=[
        "type", "entry_idx", "entry_time", "entry_price", "bet", "lot", "ticket", "status", "tp", "sl", "exit_idx", "exit_price", "pending_ticket"
    ])
    balance = start_balance
    mt5_initialize(login=12345678, password="yourpassword", server="Yourbroker-Server")

    pending_entry = None

    def fetch_and_predict():
        nonlocal balance, orders_df, pending_entry

        # データ取得
        X_test, _, _, open_test, close_test, high_test, low_test, _, _, _ = fetch_data(
            SYMBOL, TO_SYMBOL, None, None, None, None, None, None, n_features=n_features
        )
        if X_test.empty:
            print("No new data available.")
            return

        latest_idx = len(X_test) - 1
        signals = generate_technical_signal(X_test)
        adx = X_test['ADX'].values if 'ADX' in X_test.columns else np.zeros(len(X_test))
        bet = max(int(balance * bet_ratio), min_bet)
        lot = bet / 100000

        # 1. 前回のエントリー予約を約定させる
        if pending_entry is not None:
            entry_signal, entry_idx = pending_entry
            if latest_idx >= entry_idx:
                entry_price = open_test.iloc[entry_idx + 1]
                entry_time = X_test.index[entry_idx + 1]
                spreaded_price = entry_price + spread if entry_signal == "CALL" else entry_price - spread
                order_type = "BUY" if entry_signal == "CALL" else "SELL"
                # --- 1. 成行エントリー ---
                price = mt5.symbol_info_tick(SYMBOL+TO_SYMBOL).ask if order_type == "BUY" else mt5.symbol_info_tick(SYMBOL+TO_SYMBOL).bid
                result = mt5_market_order(SYMBOL+TO_SYMBOL, order_type, lot, price)
                ticket = result.order if hasattr(result, "order") else None

                # --- 2. TP/SL同時指値注文 ---
                pips_unit = 0.01 if "JPY" in TO_SYMBOL else 0.0001
                tp_pips = tp_pips * pips_unit
                sl_pips = sl_pips * pips_unit
                if order_type == "BUY":
                    tp = spreaded_price + tp_pips
                    sl = spreaded_price - sl_pips
                else:
                    tp = spreaded_price - tp_pips
                    sl = spreaded_price + sl_pips
                pending_result = mt5_pending_order(SYMBOL+TO_SYMBOL, order_type, lot, spreaded_price, sl, tp)
                pending_ticket = pending_result.order if hasattr(pending_result, "order") else None

                orders_df.loc[len(orders_df)] = {
                    "type": order_type,
                    "entry_idx": entry_idx + 1,
                    "entry_time": entry_time,
                    "entry_price": spreaded_price,
                    "bet": bet,
                    "lot": lot,
                    "ticket": ticket,
                    "status": "open",
                    "tp": tp,
                    "sl": sl,
                    "exit_idx": None,
                    "exit_price": None,
                    "pending_ticket": pending_ticket
                }
                pending_entry = None

        # 2. 新規エントリー予約（現足でシグナルが出たら）
        if signals[latest_idx] in ("CALL", "PUT") and adx[latest_idx] > 20:
            pending_entry = (signals[latest_idx], latest_idx)

        # 3. エグジット判定
        exit_indices = []
        for idx, row in orders_df.iterrows():
            bars_held = latest_idx - row["entry_idx"]
            if row["status"] != "open":
                continue
            entry_idx = row["entry_idx"]
            entry_price = row["entry_price"]
            tp = row["tp"]
            sl = row["sl"]
            pending_ticket = row["pending_ticket"]
            high_seq = high_test.iloc[entry_idx: entry_idx + entry_minutes]
            low_seq = low_test.iloc[entry_idx: entry_idx + entry_minutes]
            close_seq = close_test.iloc[entry_idx: entry_idx + entry_minutes]

            tp_hit = sl_hit = False
            tp_idx = sl_idx = None

            if row["type"] == "BUY":
                tp_hit = (high_seq >= tp).any()
                sl_hit = (low_seq <= sl).any()
                tp_idx = high_seq[high_seq >= tp].index.min() if tp_hit else None
                sl_idx = low_seq[low_seq <= sl].index.min() if sl_hit else None
            elif row["type"] == "SELL":
                tp_hit = (low_seq <= tp).any()
                sl_hit = (high_seq >= sl).any()
                tp_idx = low_seq[low_seq <= tp].index.min() if tp_hit else None
                sl_idx = high_seq[high_seq >= sl].index.min() if sl_hit else None

            # TP/SLどちらか先に到達した方で決済
            if tp_hit and (not sl_hit or tp_idx <= sl_idx):
                exit_idx = tp_idx
                exit_price = tp
                # 利確指値注文は約定したのでキャンセル不要
            elif sl_hit and (not tp_hit or sl_idx < tp_idx):
                exit_idx = sl_idx
                exit_price = sl
                # 損切指値注文は約定したのでキャンセル不要
            elif bars_held >= entry_minutes:
                exit_idx = entry_idx + entry_minutes
                exit_price = close_test.iloc[exit_idx]
                # 10分経過で未約定→指値注文キャンセル
                if pending_ticket is not None:
                    mt5_cancel_order(pending_ticket)
            else:
                continue

            # MT5成行決済
            if row["type"] == "BUY":
                price = mt5.symbol_info_tick(SYMBOL+TO_SYMBOL).bid
                mt5_market_order(SYMBOL+TO_SYMBOL, "SELL", row["lot"], price)
                print(f"MT5 BUY決済: {price} lot={row['lot']} (想定価格: {exit_price})")
            elif row["type"] == "SELL":
                price = mt5.symbol_info_tick(SYMBOL+TO_SYMBOL).ask
                mt5_market_order(SYMBOL+TO_SYMBOL, "BUY", row["lot"], price)
                print(f"MT5 SELL決済: {price} lot={row['lot']} (想定価格: {exit_price})")

            orders_df.at[idx, "status"] = "closed"
            orders_df.at[idx, "exit_idx"] = exit_idx
            orders_df.at[idx, "exit_price"] = exit_price
            exit_indices.append(idx)

        # 決済済みエントリーをDataFrameから削除
        orders_df.drop(exit_indices, inplace=True)
        orders_df.reset_index(drop=True, inplace=True)
        orders_df = orders_df[orders_df["status"] == "open"].reset_index(drop=True)

    last_run_minute = None
    while True:
        now = datetime.datetime.now()
        # 土日ならスキップ
        if now.weekday() in [5, 6]:
            time.sleep(300)
            continue
        # 分の頭+1秒（例: 10:00:01, 10:01:01, ...）で実行
        if now.second == 1 and now.minute % interval_minutes == 0:
            if last_run_minute != now.minute:
                fetch_and_predict()
                last_run_minute = now.minute
        time.sleep(0.5)
    # ...（以降は同じ）...
# --- 実行例 ---
if __name__ == "__main__":
    print("=== 通常実行 ===")
    # 元の関数での実行
    pass  # 既存の処理はそのまま
    
    print("\n=== Backtraderテスト実行 ===")
    try:
        # Backtraderテストを実行
        result1, result2 = run_backtrader_tests()
        
        print("\n=== 結果比較 ===")
        if result1 and result2:
            print(f"時間制限あり: 利益 {result1['profit']:.2f}円, 収益率 {result1['return_pct']:.2f}%, 勝率 {result1['winrate']:.2%}")
            print(f"時間制限なし: 利益 {result2['profit']:.2f}円, 収益率 {result2['return_pct']:.2f}%, 勝率 {result2['winrate']:.2%}")
    except Exception as e:
        print(f"Backtraderテストエラー: {e}")
    
    # Run real-time signal generation
    # run_realtime_signals(SYMBOL, TO_SYMBOL)  # コメントアウト

# =====================================
# Backtrader実装
# =====================================

def create_backtrader_data(df):
    """
    DataFrameからBacktraderのデータフィードを作成
    """
    # 必要な列が存在することを確認
    required_columns = ['open', 'high', 'low', 'close']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"必要な列 '{col}' がDataFrameに存在しません")
    
    # 時刻インデックスが必要な場合は、ダミーで作成
    bt_data = df.copy()
    if 'timestamp' not in bt_data.columns:
        bt_data['timestamp'] = pd.date_range(start='2024-01-01', periods=len(bt_data), freq='1min')
    
    # DataFrameの列をBacktraderが期待する形式に変換
    bt_data = bt_data[['timestamp', 'open', 'high', 'low', 'close']].copy()
    bt_data.columns = ['datetime', 'open', 'high', 'low', 'close']
    bt_data['volume'] = 1000  # ダミーボリューム
    bt_data['openinterest'] = 0  # ダミーオープンインタレスト
    bt_data.set_index('datetime', inplace=True)
    
    # NaN値を処理
    bt_data = bt_data.dropna()
    
    print(f"Backtraderデータフィード作成完了: {len(bt_data)}行")
    return bt_data

# Backtrader用のストラテジー
class TrendPredictStrategy(bt.Strategy):
    params = (
        ('tp_pips', 5),
        ('sl_pips', 3),
        ('spread', 0.02),
        ('lot_size', 1000),
        ('leverage', 3),
        ('symbol', 'USD/JPY'),
        ('entry_minutes', 10),
        ('unlimited_tracking', False),
        ('signals', []),  # シグナルをパラメータとして追加
    )
    
    def __init__(self):
        self.signals = list(self.params.signals)  # シグナルを取得
        self.order = None
        self.entry_bar = None
        self.entry_type = None
        self.entry_price = None
        self.pips_unit = 0.01 if "JPY" in self.params.symbol else 0.0001
        self.tp_value = self.params.tp_pips * self.pips_unit
        self.sl_value = self.params.sl_pips * self.pips_unit
        self.trades_count = 0
        self.winning_trades = 0
        self.debug_mode = True  # デバッグモードを有効化
        
    def next(self):
        current_bar = len(self.data) - 1
        
        # デバッグ情報を表示
        if self.debug_mode and current_bar % 100 == 0:
            signal = self.signals[current_bar] if current_bar < len(self.signals) else "UNKNOWN"
            print(f"Bar {current_bar}: Signal={signal}, Position={bool(self.position)}, Close={self.data.close[0]:.5f}")
        
        # まだポジションがない場合、シグナルをチェック
        if not self.position and current_bar < len(self.signals):
            signal = self.signals[current_bar]
            
            if signal == "CALL":
                # 買いエントリー
                self.entry_price = self.data.close[0] + self.params.spread
                size = self.params.lot_size / self.entry_price
                
                self.order = self.buy(size=size)
                self.entry_bar = current_bar
                self.entry_type = "CALL"
                if self.debug_mode:
                    print(f"CALL エントリー: Bar={current_bar}, Price={self.entry_price:.5f}, Size={size:.2f}")
                        
            elif signal == "PUT":
                # 売りエントリー
                self.entry_price = self.data.close[0] - self.params.spread
                size = self.params.lot_size / self.entry_price
                
                self.order = self.sell(size=size)
                self.entry_bar = current_bar
                self.entry_type = "PUT"
                if self.debug_mode:
                    print(f"PUT エントリー: Bar={current_bar}, Price={self.entry_price:.5f}, Size={size:.2f}")
        
        # ポジションがある場合、TP/SLをチェック
        if self.position:
            current_high = self.data.high[0]
            current_low = self.data.low[0]
            
            if self.entry_type == "CALL":
                tp_price = self.entry_price + self.tp_value
                sl_price = self.entry_price - self.sl_value
                
                # TP到達チェック
                if current_high >= tp_price:
                    self.close()
                    if self.debug_mode:
                        print(f'TP HIT: {tp_price:.5f} at bar {current_bar}')
                    self.reset_position()
                    return
                
                # SL到達チェック
                if current_low <= sl_price:
                    self.close()
                    if self.debug_mode:
                        print(f'SL HIT: {sl_price:.5f} at bar {current_bar}')
                    self.reset_position()
                    return
                    
            elif self.entry_type == "PUT":
                tp_price = self.entry_price - self.tp_value
                sl_price = self.entry_price + self.sl_value
                
                # TP到達チェック
                if current_low <= tp_price:
                    self.close()
                    if self.debug_mode:
                        print(f'TP HIT: {tp_price:.5f} at bar {current_bar}')
                    self.reset_position()
                    return
                
                # SL到達チェック
                if current_high >= sl_price:
                    self.close()
                    if self.debug_mode:
                        print(f'SL HIT: {sl_price:.5f} at bar {current_bar}')
                    self.reset_position()
                    return
            
            # 無制限追跡でない場合、entry_minutes後に強制決済
            if not self.params.unlimited_tracking and self.entry_bar is not None:
                if current_bar >= self.entry_bar + self.params.entry_minutes:
                    self.close()
                    if self.debug_mode:
                        print(f'TIME EXIT at bar {current_bar}')
                    self.reset_position()
    
    def reset_position(self):
        self.order = None
        self.entry_bar = None
        self.entry_type = None
        self.entry_price = None
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                if self.debug_mode:
                    print(f'BUY EXECUTED: Price: {order.executed.price:.5f}, Size: {order.executed.size:.2f}')
            else:
                if self.debug_mode:
                    print(f'SELL EXECUTED: Price: {order.executed.price:.5f}, Size: {order.executed.size:.2f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if self.debug_mode:
                print('Order Canceled/Margin/Rejected')
    
    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades_count += 1
            if trade.pnl > 0:
                self.winning_trades += 1
            if self.debug_mode:
                print(f'TRADE CLOSED: Profit: {trade.pnl:.2f}, Commission: {trade.commission:.2f}')
    
    def log(self, txt, dt=None):
        if self.debug_mode:
            dt = dt or self.datas[0].datetime.datetime(0)
            print(f'{dt}: {txt}')

def calc_winrate_technical_backtrader(
    df, tp_pips=5, sl_pips=3, spread=0.02, symbol="USD/JPY",
    lot_size=1000, start_balance=50000, leverage=3, entry_minutes=10,
    unlimited_tracking=False
):
    """
    Backtraderを使用してテクニカル指標に基づくバックテストを実行
    """
    # シグナル生成
    signals = generate_technical_signal(df)
    
    # シグナル統計を表示
    call_count = signals.count("CALL")
    put_count = signals.count("PUT") 
    hold_count = signals.count("HOLD")
    print(f"[シグナル統計] CALL: {call_count}, PUT: {put_count}, HOLD: {hold_count}")
    
    # Backtraderデータフィード作成
    bt_data = create_backtrader_data(df)
    
    # Cerebro（バックテストエンジン）セットアップ
    cerebro = bt.Cerebro()
    
    # データフィードを追加
    data = bt.feeds.PandasData(dataname=bt_data)
    cerebro.adddata(data)
    
    # 戦略を追加（シグナルを事前に渡す）
    cerebro.addstrategy(TrendPredictStrategy,
                       tp_pips=tp_pips,
                       sl_pips=sl_pips,
                       spread=spread,
                       lot_size=lot_size,
                       leverage=leverage,
                       symbol=symbol,
                       entry_minutes=entry_minutes,
                       unlimited_tracking=unlimited_tracking,
                       signals=signals)  # シグナルを直接渡す
    
    # 初期資金設定
    cerebro.broker.setcash(start_balance)
    
    # 手数料設定
    cerebro.broker.setcommission(commission=0.001)
    
    # 戦略実行前の資金
    initial_value = cerebro.broker.getvalue()
    
    # バックテスト実行
    result = cerebro.run()
    
    # 結果の取得
    final_value = cerebro.broker.getvalue()
    profit = final_value - initial_value
    
    # 戦略インスタンスから結果を取得
    strategy_instance = result[0]
    
    # 勝率計算
    winrate = strategy_instance.winning_trades / strategy_instance.trades_count if strategy_instance.trades_count > 0 else 0
    
    print(f"[Backtrader Advanced テクニカル] 初期資金: {initial_value:.2f}円")
    print(f"[Backtrader Advanced テクニカル] 最終資金: {final_value:.2f}円")
    print(f"[Backtrader Advanced テクニカル] 利益: {profit:.2f}円")
    print(f"[Backtrader Advanced テクニカル] 収益率: {(profit/initial_value)*100:.2f}%")
    print(f"[Backtrader Advanced テクニカル] 勝率: {winrate:.2%} ({strategy_instance.winning_trades}/{strategy_instance.trades_count})")
    
    return {
        'initial_value': initial_value,
        'final_value': final_value,
        'profit': profit,
        'return_pct': (profit/initial_value)*100,
        'winrate': winrate,
        'total_trades': strategy_instance.trades_count,
        'winning_trades': strategy_instance.winning_trades,
        'cerebro': cerebro,
        'signals': signals
    }

def calc_winrate_technical2_backtrader(
    df, tp_pips=5, sl_pips=3, spread=0.02, symbol="USD/JPY",
    lot_size=1000, start_balance=50000, leverage=3
):
    """
    Backtraderを使用してテクニカル指標に基づくバックテストを実行（無制限追跡版）
    """
    return calc_winrate_technical_backtrader(
        df, tp_pips=tp_pips, sl_pips=sl_pips, spread=spread, symbol=symbol,
        lot_size=lot_size, start_balance=start_balance, leverage=leverage,
        entry_minutes=9999, unlimited_tracking=True
    )

# 元の関数のBacktraderラッパー関数
def calc_winrate_technical_bt_wrapper(
    df, open_, close, high=None, low=None,
    entry_minutes=10, tp_pips=5, sl_pips=3, spread=0.02, symbol="USD/JPY",
    lot=1000, start_balance=50000, leverage=3
):
    """
    元のcalc_winrate_technical関数の代替となるBacktraderラッパー
    """
    # DataFrameの準備
    test_df = df.copy()
    if 'open' not in test_df.columns:
        test_df['open'] = open_
    if 'close' not in test_df.columns:
        test_df['close'] = close
    if 'high' not in test_df.columns:
        test_df['high'] = high if high is not None else df.get('high', close)
    if 'low' not in test_df.columns:
        test_df['low'] = low if low is not None else df.get('low', close)
    
    # Backtraderテストを実行
    result = calc_winrate_technical_backtrader(
        test_df, tp_pips=tp_pips, sl_pips=sl_pips, spread=spread, 
        symbol=symbol, lot_size=lot, start_balance=start_balance, 
        leverage=leverage, entry_minutes=entry_minutes, unlimited_tracking=False
    )
    
    # 元の関数の戻り値形式に合わせる
    signals = result['signals']
    winrate = result['winrate']
    entries = [i for i, sig in enumerate(signals) if sig in ["CALL", "PUT"]]
    results = [result['profit'] > 0] * result['total_trades']  # 簡易版の結果
    
    return winrate, entries, results, signals

def calc_winrate_technical2_bt_wrapper(
    df, open_, close, high=None, low=None, tp_pips=5, sl_pips=3, spread=0.02, symbol="USD/JPY",
    lot=1000, start_balance=50000, leverage=3
):
    """
    元のcalc_winrate_technical2関数の代替となるBacktraderラッパー
    """
    # DataFrameの準備
    test_df = df.copy()
    if 'open' not in test_df.columns:
        test_df['open'] = open_
    if 'close' not in test_df.columns:
        test_df['close'] = close
    if 'high' not in test_df.columns:
        test_df['high'] = high if high is not None else df.get('high', close)
    if 'low' not in test_df.columns:
        test_df['low'] = low if low is not None else df.get('low', close)
    
    # Backtraderテスト（無制限追跡）を実行
    result = calc_winrate_technical2_backtrader(
        test_df, tp_pips=tp_pips, sl_pips=sl_pips, spread=spread, 
        symbol=symbol, lot_size=lot, start_balance=start_balance, 
        leverage=leverage
    )
    
    # 元の関数の戻り値形式に合わせる
    signals = result['signals']
    winrate = result['winrate']
    entries = [i for i, sig in enumerate(signals) if sig in ["CALL", "PUT"]]
    results = [result['profit'] > 0] * result['total_trades']  # 簡易版の結果
    
    return winrate, entries, results, signals

# 元の関数呼び出しを新しい関数に置き換える
def replace_original_functions():
    """
    元の関数を新しいBacktrader関数に置き換える例
    """
    # 元の関数を新しい関数に置き換える
    global calc_winrate_technical, calc_winrate_technical2
    calc_winrate_technical = calc_winrate_technical_bt_wrapper
    calc_winrate_technical2 = calc_winrate_technical2_bt_wrapper
    
    print("元の関数をBacktrader実装に置き換えました。")

# =====================================
# Backtrader実装終了
# =====================================


