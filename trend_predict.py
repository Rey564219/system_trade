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
import time
import os
import MetaTrader5 as mt5
import datetime
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



class EntryManager:
    def __init__(self, name, start_amount=800, unit=100, min_bet=500, payout=1.8,
                 entry_delay=3, interval_minutes=2, state_file=None, log_file=None):
        self.name = name
        self.start_amount = start_amount
        self.unit = unit
        self.min_bet = min_bet
        self.payout = payout
        self.interval_minutes = interval_minutes
        self.delay_ticks = entry_delay // interval_minutes
        self.entries = deque()

        # ファイル設定
        self.state_file = state_file or f"{name}_state.json"
        self.log_file = log_file or f"{name}_log.csv"

        # 現在のベット額を読み込み or 初期化
        self.current_bet = self.load_state()

        # ログファイルがなければヘッダ作成
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("tick,direction,result,entry_price,exit_price,bet,next_bet\n")

    def create_entry(self, tick, price, direction):
        self.entries.append({
            "tick": tick,
            "entry_price": price,
            "direction": direction,
            "bet": self.current_bet
        })
        return self.current_bet

    def update_and_get_next_bet(self, current_tick, latest_idx, price_series, label=""):
        while self.entries:
            entry = self.entries[0]
            entry_tick = entry["tick"]
            if current_tick >= entry_tick + self.delay_ticks:
                offset = current_tick - entry_tick
                target_index = latest_idx - offset + self.delay_ticks

                if target_index < 0 or target_index >= len(price_series):
                    print(f"[{label}] 判定スキップ: target_index={target_index} が無効")
                    self.entries.popleft()
                    continue

                exit_price = price_series.iloc[target_index]
                entry_price = entry["entry_price"]
                direction = entry["direction"]
                win = (exit_price > entry_price) if direction == "CALL" else (exit_price < entry_price)

                # 金額更新
                if win:
                    self.current_bet = max(self.current_bet - self.unit, self.min_bet)
                else:
                    self.current_bet += self.unit
                if self.current_bet < self.min_bet:
                    self.current_bet = self.start_amount

                # 結果処理
                result = "WIN" if win else "LOSE"
                log_line = f"{current_tick},{direction},{result},{entry_price},{exit_price},{entry['bet']},{self.current_bet}\n"
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(log_line)

                # 状態保存
                self.save_state()

                # 通知
                msg = f"[{label}] {direction} 判定: {'✅勝ち' if win else '❌負け'} → 次のベット額: {self.current_bet}円"
                print(msg)
                send_discord_message(msg)

                self.entries.popleft()
            else:
                break
        return self.current_bet

    def save_state(self):
        data = {"current_bet": self.current_bet}
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    def load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f).get("current_bet", self.start_amount)
            except:
                pass
        return self.start_amount

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
def select_important_features(X, y_up, y_down, n_features=20):
    """
    上昇・下降ラベル両方に対応した特徴量選択。
    それぞれのラベルでLightGBMの重要度を計算し、上位n_featuresを合成して返す。
    """
    # 上昇ラベルで重要度
    model_up = lgb.LGBMClassifier(random_state=42)
    model_up.fit(X, y_up)
    imp_up = pd.DataFrame({'feature': X.columns, 'importance': model_up.feature_importances_})

    # 下降ラベルで重要度
    model_down = lgb.LGBMClassifier(random_state=42)
    model_down.fit(X, y_down)
    imp_down = pd.DataFrame({'feature': X.columns, 'importance': model_down.feature_importances_})

    # 重要度を合算（平均でも最大でもOK。ここでは平均）
    imp = imp_up.set_index('feature').add(imp_down.set_index('feature'), fill_value=0) / 2
    imp = imp.sort_values('importance', ascending=False)

    selected_features = imp.head(n_features).index.tolist()
    print(f"選択された特徴量: {selected_features}")
    return selected_features

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
    return df

def generate_technical_signal(df, adx_period=14, sma_short=5, sma_mid=20, adx_threshold=20):
    signals = []
    adx = df['ADX'].values
    sma_s = df[f'SMA_{sma_short}'].values
    sma_m = df[f'SMA_{sma_mid}'].values
    # bb_width = (df['BBANDS_uppearband'] - df['BBANDS_lowerband']).values
    # bb_mean = pd.Series(bb_width).rolling(20).mean().values
    # rsi = df['RSI'].values
    # macd_hist = df['MACD_macdhist'].values
    # atr = df['ATR'].values
    # atr_mean = pd.Series(atr).rolling(20).mean().values

    for i in range(3, len(df)):
        adx_now = adx[i]
        adx_prev = adx[i-3]
        # adx_up = adx_now > adx_prev and adx_now >= adx_threshold
        # sma_s_up = sma_s[i] > sma_s[i-3]
        # sma_m_up = sma_m[i] > sma_m[i-3]
        sma_order_up = sma_s[i] > sma_m[i]
        sma_order_down = sma_s[i] < sma_m[i]
        # --- 追加フィルタ ---
        # bb_ok = bb_width[i] > bb_mean[i] * 0.68 if not np.isnan(bb_mean[i]) else False
        # rsi_ok = 32 < rsi[i] < 68
        # macd_ok_up = macd_hist[i] > 0
        # macd_ok_down = macd_hist[i] < 0
        # atr_ok = atr[i] > atr_mean[i] * 0.68 if not np.isnan(atr_mean[i]) else False
        # --- シグナル判定 ---
        # if adx_up and sma_s_up and sma_m_up and sma_order_up and bb_ok and rsi_ok and macd_ok_up and atr_ok:
        #     signals.append("CALL")
        # elif adx_up and not sma_s_up and not sma_m_up and sma_order_down and bb_ok and rsi_ok and macd_ok_down and atr_ok:
        #     signals.append("PUT")
        # else:
        #     signals.append("HOLD")
        # if adx_up and sma_s_up and sma_m_up and sma_order_up:
        #     signals.append("CALL")
        # elif adx_up and not sma_s_up and not sma_m_up and sma_order_down:
        #     signals.append("PUT")
        # else:
        #     signals.append("HOLD")
        if adx_now > 20 and sma_order_up:
            signals.append("CALL")
        elif adx_now > 20 and sma_order_down:
            signals.append("PUT")
        else:
            signals.append("HOLD")
    # 先頭3つはHOLDで埋める
    signals = ["HOLD"] * 3 + signals
    return signals
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
    entry_minutes=10, tp_pips=6, sl_pips=3, spread=0.02, symbol="USD/JPY",
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
                entry_price = open_[i + 1] + spread if sig == "CALL" else open_[i + 1] - spread
                high_seq = high[i + 1 : i + 1 + entry_minutes]
                low_seq = low[i + 1 : i + 1 + entry_minutes]

                win = False

                if sig == "CALL" and adx[i] > 20:
                    tp_hit = (high_seq >= entry_price + tp_value).any()
                    sl_hit = (low_seq <= entry_price - sl_value).any()
                    if tp_hit and (not sl_hit or high_seq.idxmax() < low_seq.idxmin()):
                        gain = lot * tp_value * leverage
                        balance += gain
                        win = True
                    elif sl_hit:
                        loss = lot * sl_value * leverage
                        balance -= loss
                        win = False
                    else:
                        # TP/SLどちらも未到達→entry_minutes後のcloseで決済
                        exit_price = close[i + 1 + entry_minutes]
                        profit = (exit_price - entry_price - spread) * lot * leverage
                        balance += profit
                        win = profit > 0
                    balance_curve.append(balance)
                elif sig == "PUT" and adx[i] > 20:
                    tp_hit = (low_seq <= entry_price - tp_value).any()
                    sl_hit = (high_seq >= entry_price + sl_value).any()
                    if tp_hit and (not sl_hit or low_seq.idxmin() < high_seq.idxmax()):
                        gain = lot * tp_value * leverage
                        balance += gain
                        win = True
                    elif sl_hit:
                        loss = lot * sl_value * leverage
                        balance -= loss
                        win = False
                    else:
                        # TP/SLどちらも未到達→entry_minutes後のcloseで決済
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

# --- モデル訓練 ---
def train_models(X, y_up, y_down):
    """
    上昇エントリー用・下降エントリー用でモデルを分けて学習する
    X: 特徴量 (DataFrame)
    y_up: 上昇エントリー用ラベル (label_up, Series)
    y_down: 下降エントリー用ラベル (label_down, Series)
    戻り値: (model_up, model_down)
    """
    # signalsはXと同じ長さでSeriesとして生成
    signals = pd.Series(generate_technical_signal(X), index=X.index)
    mask_up = signals == "CALL"
    mask_down = signals == "PUT"

    X_up = X
    y_up_filtered = y_up
    tss = TimeSeriesSplit(n_splits=5)

    for train_idx, valid_idx in tss.split(X_up, y_up_filtered):
        X_train_up, X_valid_up = X_up.iloc[train_idx], X_up.iloc[valid_idx]
        y_train_up, y_valid_up = y_up_filtered.iloc[train_idx], y_up_filtered.iloc[valid_idx]
        break
    X_train_up = X_train_up[mask_up]
    y_train_up = y_train_up[mask_up]
    # for train_idx, valid_idx in tss.split(X, y_up):
    #     X_train_up, X_valid_up = X.iloc[train_idx], X.iloc[valid_idx]
    #     y_train_up, y_valid_up = y_up.iloc[train_idx], y_up.iloc[valid_idx]
    #     break
    model_up = lgb.LGBMClassifier(class_weight='balanced', random_state=42)
    model_up.fit(X_train_up, y_train_up)
    acc_up = accuracy_score(y_valid_up, model_up.predict(X_valid_up))
    pre_up = precision_score(y_valid_up, model_up.predict(X_valid_up), labels=[1], average='macro')
    print(f"[上昇エントリー] Validation accuracy: {acc_up:.2%}")
    print(f"[上昇エントリー] Validation precision: {pre_up:.2%}")
    X_down = X
    y_down_filtered = y_up
    tss = TimeSeriesSplit(n_splits=5)

    for train_idx, valid_idx in tss.split(X_down, y_down_filtered):
        X_train_down, X_valid_down = X_down.iloc[train_idx], X_down.iloc[valid_idx]
        y_train_down, y_valid_down = y_down_filtered.iloc[train_idx], y_down_filtered.iloc[valid_idx]
        break
    X_train_down = X_train_down[mask_down]
    y_train_down = y_train_down[mask_down]
    # for train_idx, valid_idx in tss.split(X, y_down):
    #     X_train_down, X_valid_down = X.iloc[train_idx], X.iloc[valid_idx]
    #     y_train_down, y_valid_down = y_down.iloc[train_idx], y_down.iloc[valid_idx]
    #     break
    model_down = lgb.LGBMClassifier(class_weight='balanced', random_state=42)
    model_down.fit(X_train_down, y_train_down)
    acc_down = accuracy_score(y_valid_down, model_down.predict(X_valid_down))
    pre_down = precision_score(y_valid_down, model_down.predict(X_valid_down), labels=[1], average='macro')
    print(f"[下降エントリー] Validation accuracy: {acc_down:.2%}")
    print(f"[下降エントリー] Validation precision: {pre_down:.2%}")

    return model_up, model_down


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
    file_path = folder_path + ".csv"
    df = pd.read_csv(file_path)
    X = df.apply(pd.to_numeric, errors='coerce')
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.loc[:, X.isnull().mean() < 0.9]
    price_data = X.copy()

    # --- 特徴量生成 ---
    price_data = (price_data)
    price_data['rsi'] = calc_rsi(price_data['close'], window=14)
    price_data = make_features(price_data)


# --- 欠損値処理 ---
    drop_keywords = ['trend_direction', 'trend_duration', 'trend_duration_bucket', 'trend_confidence']
    drop_cols = [col for col in price_data.columns if any(key in col for key in drop_keywords)]
    X = price_data.drop(columns=drop_cols, errors='ignore')
    X = X.fillna(method='ffill').fillna(method='ffill')
    # --- 特徴量選択 ---
    features = [
        'ma_short', 'ma_mid', 'ma_long',
        'ma_short_slope', 'ma_mid_slope', 'ma_long_slope',
        'ma_cross_short_mid', 'ma_cross_short_long', 'ma_cross_mid_long',
        'atr', 'rsi', 'return_1', 'return_3', 'volatility_5', 'volatility_15', 'close', 'open', 'high', 'low',
        'ADX','SMA_5', 'SMA_20'
    ]
    X = make_labels(X)
    y_up = X['label_up'].copy()
    y_down = X['label_down'].copy()
    X = X.drop(['label_up','label_down','return_1min'], axis=1)
    drop_keywords = ['trend_direction', 'trend_duration', 'trend_duration_bucket', 'trend_confidence','return_1min']
    drop_cols = [col for col in price_data.columns if any(key in col for key in drop_keywords)]
    X = price_data.drop(columns=drop_cols, errors='ignore')
    X = X.fillna(method='ffill').fillna(method='ffill')
    select = select_important_features(X, y_up, y_down, n_features=n_features)
    if "close" not in select:
        select.append("close")
    if "open" not in select:
        select.append("open")
    if "high" not in select:
        select.append("high")
    if "low" not in select:
        select.append("low")
    if "SMA_5" not in select:
        select.append("SMA_5")
    if "SMA_20" not in select:
        select.append("SMA_20")
    if "ADX" not in select:
        select.append("ADX")

    X = X[select]
    pca = None  # 必要ならPCAを適用
    print('finish make_train_data')
    # スケーラーはNoneで返す
    return X, X, None,None, X.columns.tolist(),  price_data['open'],price_data['close'], price_data['high'], price_data['low'], None, select, None, None, pca, None

def fetch_data(SYMBOL, TO_SYMBOL, col, scaler, X_train, y_train, select, pca, n_features=20):
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
    price_data['BBANDS_upperband'], price_data['BBANDS_middleband'], price_data['BBANDS_lowerband'] = talib.BBANDS(close, timeperiod=15, nbdevup=2, nbdevdn=2, matype=0)
    price_data['BBANDS_middleband'] -= hilo
    price_data['BBANDS_lowerband'] -= hilo
    price_data['DEMA'] = talib.DEMA(close, timeperiod=30) - hilo
    price_data['EMA'] = talib.EMA(close, timeperiod=30) - hilo
    price_data['EMA20'] = talib.EMA(close, timeperiod=20) - hilo
    price_data['EMA50'] = talib.EMA(close, timeperiod=50) - hilo
    price_data['HT_TRENDLINE'] = (talib.HT_TRENDLINE(close) - hilo) / close
    price_data['KAMA'] = talib.KAMA(close, timeperiod=30) - hilo
    price_data['MA'] = talib.MA(close, timeperiod=30, matype=0) - hilo
    price_data['MIDPOINT'] = talib.MIDPOINT(close, timeperiod=14) - hilo
    price_data['SMA'] = talib.SMA(close, timeperiod=30) - hilo
    price_data['T3'] = talib.T3(close, timeperiod=5, vfactor=0) - hilo
    price_data['TEMA'] = talib.TEMA(close, timeperiod=30) - hilo
    price_data['TRIMA'] = talib.TRIMA(close, timeperiod=30) - hilo
    price_data['WMA'] = talib.WMA(close, timeperiod=30) - hilo
    price_data['ADX'] = talib.ADX(high, low, close, timeperiod=14)
    price_data['ADXR'] = talib.ADXR(high, low, close, timeperiod=14)
    price_data['APO'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
    price_data['AROON_aroondowc'], price_data['AROON_aroonup'] = talib.AROON(high, low, timeperiod=14)
    price_data['AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)
    price_data['BOP'] = talib.BOP(open_price, high, low, close)
    price_data['CCI'] = talib.CCI(high, low, close, timeperiod=14)
    price_data['DX'] = talib.DX(high, low, close, timeperiod=14)
    price_data['MACD_macd'], price_data['MACD_macdsignal'], price_data['MACD_macdhist'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    price_data['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
    price_data['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
    price_data['MINUS_DM'] = talib.MINUS_DM(high, low, timeperiod=14)
    price_data['MOM'] = talib.MOM(close, timeperiod=10)
    price_data['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
    price_data['PLUS_DM'] = talib.PLUS_DM(high, low, timeperiod=14)
    price_data['RSI'] = talib.RSI(close, timeperiod=14)
    price_data['STOCH_slowk'], price_data['STOCH_slowd'] = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    price_data['STOCHF_fastk'], price_data['STOCHF_fastd'] = talib.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
    price_data['STOCHRSI_fastk'], price_data['STOCHRSI_fastd'] = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    price_data['TRIX'] = talib.TRIX(close, timeperiod=10)
    price_data['ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    price_data['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)
    price_data['ADOSC'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    price_data['OBV'] = talib.OBV(close, volume)
    price_data['ATR'] = talib.ATR(high, low, close, timeperiod=14)
    price_data['NATR'] = talib.NATR(high, low, close, timeperiod=14)
    price_data['TRANGE'] = talib.TRANGE(high, low, close)
    price_data['HT_DCPERIOD'] = talib.HT_DCPERIOD(close)
    price_data['HT_DCPHASE'] = talib.HT_DCPHASE(close)
    price_data['HT_PHASOR_inphase'], price_data['HT_PHASOR_quadrature'] = talib.HT_PHASOR(close)
    price_data['HT_SINE_sine'], price_data['HT_SINE_leadsine'] = talib.HT_SINE(close)
    price_data['HT_TRENDMODE'] = talib.HT_TRENDMODE(close)
    price_data['BETA'] = talib.BETA(high, low, timeperiod=5)
    price_data['CORREL'] = talib.CORREL(high, low, timeperiod=30)
    price_data['LINEARREG'] = talib.LINEARREG(close, timeperiod=14) - close
    price_data['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(close, timeperiod=14)
    price_data['LINEARREG_INTERCEPT'] = talib.LINEARREG_INTERCEPT(close, timeperiod=14) - close
    price_data['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(close, timeperiod=14)
    price_data['STDDEV'] = talib.STDDEV(close, timeperiod=5, nbdev=1)
    price_data['STDDEV'] = price_data['STDDEV'] / close
    price_data["CMO"] = talib.CMO(close, timeperiod=14)
    price_data["PLUS_DI"] = talib.PLUS_DI(high, low, close, timeperiod=14)
    price_data["MINUS_DI"] = talib.MINUS_DI(high, low, close, timeperiod=14)
    price_data["ROC"] = talib.ROC(close, timeperiod=14)
    price_data["CDLLADDERBOTTOM"] = talib.CDLLADDERBOTTOM(open_price, high, low, close)
    price_data["AVGPRICE"] = talib.AVGPRICE(open_price, high, low, close)
    price_data["HT_TRENDLINE"] = talib.HT_TRENDLINE(close)
    price_data["SAREXT"] = talib.SAREXT(high, low)
    price_data["APO"] = talib.APO(close)
    price_data['close_ma_diff'] = close - price_data['close_ma']
    price_data['close_lowpass_diff'] = close - price_data['close_lowpass']
    price_data['close_kalman_diff'] = close - price_data['close_kalman']
    price_data['Crypto_VIX'] = calculate_crypto_vix(close)

    price2 = price_data[['open', 'high', 'low', 'close','volumeto']]

    price2 = add_log_transformed_features(price2)
    
    open_price = price2['log_open']
    high = price2['log_high']
    low = price2['log_low']
    close = price2['log_close']
    volume = price2['log_volumeto']
    hilo = (high + low) / 2


    price2 = price2.replace([np.inf, -np.inf], np.nan)  # InfをNaNに置き換え
    
    price2['log_close'] = price2['log_close'].bfill()


    # 各段階でのデータチェック
    print("NaN count before preprocessing:", price2['log_close'].isna().sum())
    print("Inf count before preprocessing:", np.isinf(price2['log_close']).sum())
    

    price2['close_ma'] = apply_moving_average(price2['log_close'])
    price2['close_lowpass'] = apply_lowpass_filter(price2['log_close'].values, cutoff=0.1, fs=1.0)
    price2['log_close'] = price2['log_close'].interpolate(method='linear')
    price2['close_ma'] = price2['close_ma'].interpolate(method='linear')
    price2['close_lowpass'] = np.nan_to_num(price2['close_lowpass'], nan=np.nanmean(price2['close_lowpass']))

    price2['close_kalman'] = apply_kalman_filter(price2['log_close'].values)
    # インデックスをdatetime64[ns]型に変換
    price2.index = pd.to_datetime(price2.index)
        
    # Add technical indicators (as before)
    price2['BBANDS_uppeatrand'], price2['BBANDS_middleband'], price2['BBANDS_loweatrand'] = talib.BBANDS(close, timeperiod=15, nbdevup=2, nbdevdn=2, matype=0)
    price2['BBANDS_middleband'] -= hilo
    price2['BBANDS_loweatrand'] -= hilo
    price2['DEMA'] = talib.DEMA(close, timeperiod=30) - hilo
    price2['EMA'] = talib.EMA(close, timeperiod=30) - hilo
    price2['EMA20'] = talib.EMA(close, timeperiod=20) - hilo
    price2['EMA50'] = talib.EMA(close, timeperiod=50) - hilo
    price2['HT_TRENDLINE'] = (talib.HT_TRENDLINE(close) - hilo) / close
    price2['KAMA'] = talib.KAMA(close, timeperiod=30) - hilo
    price2['MA'] = talib.MA(close, timeperiod=30, matype=0) - hilo
    price2['MIDPOINT'] = talib.MIDPOINT(close, timeperiod=14) - hilo
    price2['SMA'] = talib.SMA(close, timeperiod=30) - hilo
    price2['T3'] = talib.T3(close, timeperiod=5, vfactor=0) - hilo
    price2['TEMA'] = talib.TEMA(close, timeperiod=30) - hilo
    price2['TRIMA'] = talib.TRIMA(close, timeperiod=30) - hilo
    price2['WMA'] = talib.WMA(close, timeperiod=30) - hilo
    price2['ADX'] = talib.ADX(high, low, close, timeperiod=14)
    price2['ADXR'] = talib.ADXR(high, low, close, timeperiod=14)
    price2['APO'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
    price2['AROON_aroondowc'], price2['AROON_aroonup'] = talib.AROON(high, low, timeperiod=14)
    price2['AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)
    price2['BOP'] = talib.BOP(open_price, high, low, close)
    price2['CCI'] = talib.CCI(high, low, close, timeperiod=14)
    price2['DX'] = talib.DX(high, low, close, timeperiod=14)
    price2['MACD_macd'], price2['MACD_macdsignal'], price2['MACD_macdhist'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    price2['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
    price2['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
    price2['MINUS_DM'] = talib.MINUS_DM(high, low, timeperiod=14)
    price2['MOM'] = talib.MOM(close, timeperiod=10)
    price2['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
    price2['PLUS_DM'] = talib.PLUS_DM(high, low, timeperiod=14)
    price2['RSI'] = talib.RSI(close, timeperiod=14)
    price2['STOCH_slowk'], price2['STOCH_slowd'] = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    price2['STOCHF_fastk'], price2['STOCHF_fastd'] = talib.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
    price2['STOCHRSI_fastk'], price2['STOCHRSI_fastd'] = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    price2['TRIX'] = talib.TRIX(close, timeperiod=10)
    price2['ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    price2['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)
    price2['ADOSC'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    price2['OBV'] = talib.OBV(close, volume)
    price2['ATR'] = talib.ATR(high, low, close, timeperiod=14)
    price2['NATR'] = talib.NATR(high, low, close, timeperiod=14)
    price2['TRANGE'] = talib.TRANGE(high, low, close)
    price2['HT_DCPERIOD'] = talib.HT_DCPERIOD(close)
    price2['HT_DCPHASE'] = talib.HT_DCPHASE(close)
    price2['HT_PHASOR_inphase'], price2['HT_PHASOR_quadrature'] = talib.HT_PHASOR(close)
    price2['HT_SINE_sine'], price2['HT_SINE_leadsine'] = talib.HT_SINE(close)
    price2['HT_TRENDMODE'] = talib.HT_TRENDMODE(close)
    price2['BETA'] = talib.BETA(high, low, timeperiod=5)
    price2['CORREL'] = talib.CORREL(high, low, timeperiod=30)
    price2['LINEARREG'] = talib.LINEARREG(close, timeperiod=14) - close
    price2['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(close, timeperiod=14)
    price2['LINEARREG_INTERCEPT'] = talib.LINEARREG_INTERCEPT(close, timeperiod=14) - close
    price2['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(close, timeperiod=14)
    price2['STDDEV'] = talib.STDDEV(close, timeperiod=5, nbdev=1)
    price2['STDDEV'] = price2['STDDEV'] / close
    price2["CMO"] = talib.CMO(close, timeperiod=14)
    price2["PLUS_DI"] = talib.PLUS_DI(high, low, close, timeperiod=14)
    price2["MINUS_DI"] = talib.MINUS_DI(high, low, close, timeperiod=14)
    price2["ROC"] = talib.ROC(close, timeperiod=14)
    price2["CDLLADDERBOTTOM"] = talib.CDLLADDERBOTTOM(open_price, high, low, close)
    price2["AVGPRICE"] = talib.AVGPRICE(open_price, high, low, close)
    price2["HT_TRENDLINE"] = talib.HT_TRENDLINE(close)
    price2["SAREXT"] = talib.SAREXT(high, low)
    price2["APO"] = talib.APO(close)

    price2['close_ma_diff'] = close - price2['close_ma']
    price2['close_lowpass_diff'] = close - price2['close_lowpass']
    price2['close_kalman_diff'] = close - price2['close_kalman']
    
    # 1分ごとのリターン（価格変動率）を計算
    price2['return_1min'] = price2['log_close'].pct_change(6)
    # ボラティリティの計算（標準偏差で計算）
    volatility = price2['return_1min'].rolling(window=60).mean().fillna(0)
    threshold = volatility.mean()

    price2 = pd.DataFrame(price2)
    price2['ATR'] = talib.ATR(high, low, close, timeperiod=14)
    price2['BB_width'] = (price2['BBANDS_uppeatrand'] - price2['BBANDS_loweatrand']) / price2['BBANDS_middleband']
    price2['Crypto_VIX'] = calculate_crypto_vix(close)
    colss = [c for c in price2.columns if "log" not in c]
    for cols in colss:
        log_col = f'log_{cols}'

        # 負の値がある場合は絶対値を取って対数変換
        price2[log_col] = price2[cols]
        if cols in price2.columns:
            price2.drop(columns=[cols], inplace=True)

    print(price2)
    price_data = pd.concat([price_data,price2],axis=1)
    # --- 欠損値処理 ---
    drop_keywords = ['trend_direction', 'trend_duration', 'trend_duration_bucket', 'trend_confidence','return_1min']
    drop_cols = [col for col in price_data.columns if any(key in col for key in drop_keywords)]
    X = price_data.drop(columns=drop_cols, errors='ignore')
    X = X.fillna(method='ffill').fillna(method='ffill')
    # featuresリストでX_test_selectedを再構成
    features = [
        'ma_short', 'ma_mid', 'ma_long',
        'ma_short_slope', 'ma_mid_slope', 'ma_long_slope',
        'ma_cross_short_mid', 'ma_cross_short_long', 'ma_cross_mid_long',
        'atr', 'rsi', 'return_1', 'return_3', 'volatility_5', 'volatility_15', 'close', 'open', 'high', 'low',
        'ADX','SMA_5', 'SMA_20'
    ]
    # --- 特徴量選択 ---
    # X_test_selected = X[features]
    X_test_selected = X[select]
    # --- PCA ---
    if pca is not None:
        X_test_selected = pca.transform(X_test_selected)
        X_test_selected = pd.DataFrame(X_test_selected)
    print('finish fetch_data')
    # features = select

    return X_test_selected, None, features,  price_data['open'], price_data['close'], price_data['high'], price_data['low'], None, None, None
pair = SYMBOL + TO_SYMBOL
# make_train_dataを使って学習データ作成
X_train, X_trainb, y_train, y_trainb, col, open_price, close, high, low, label, select, _, _, pca, scaler = make_train_data(pair, n_features=20)

# fetch_dataを使ってテストデータ作成（SYMBOL, TO_SYMBOLは仮の値でOK）
X_test, y_test, feature_names, open_test, close_test, high_test, low_test, label_test, _, _ = fetch_data(SYMBOL, TO_SYMBOL, col, scaler, X_trainb, y_trainb, select, pca, n_features=20)
df = X_train.copy()

# X_testのインデックスがDatetimeIndexの場合
X_test = X_test[X_test.index.weekday < 5]
# --- パイプライン ---
print('finish make_features')
df = make_labels(df)
y_up = df['label_up']
y_down = df['label_down']
print('finish make_labels')
df['rsi'] = calc_rsi(df['close'], window=14)
X = df.dropna()
y_up = y_up.loc[X.index]
y_down = y_down.loc[X.index]
features = [
    'ma_short', 'ma_mid', 'ma_long',
    'ma_short_slope', 'ma_mid_slope', 'ma_long_slope',
    'ma_cross_short_mid', 'ma_cross_short_long', 'ma_cross_mid_long',
    'atr', 'rsi', 'return_1', 'return_3', 'volatility_5', 'volatility_15', 'close', 'open', 'high', 'low',
    'ADX','SMA_5', 'SMA_20'
]
# features = select
X = X[select]
model_up, model_down = train_models(X, y_up, y_down)
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

def calc_winrate(
    X_test, open_test, high_test, low_test,
    model_up, model_down,
    tp_pips=6, sl_pips=3,
    spread=0.02,  # ← スプレッド（片道）
    entry_minutes=10, lot=1000, start_balance=50000, leverage=3
):
    preds_up = model_up.predict(X_test)
    preds_down = model_down.predict(X_test)
    tp_pips = tp_pips * pips  # TPの価格変動幅
    sl_pips = sl_pips * pips  # SLの価格変動幅
    open_test = open_test.reset_index(drop=True)
    high_test = high_test.reset_index(drop=True)
    low_test = low_test.reset_index(drop=True)
    adx = talib.ADX(high_test, low_test, open_test, timeperiod=14)
    balance = start_balance
    results_up, results_down = [], []
    balance_curve = [balance]

    for i in range(len(preds_up)):
        if i + entry_minutes + 1 >= len(open_test):
            continue

        raw_entry = open_test[i + 1]
        high_seq = high_test[i + 1: i + 1 + entry_minutes]
        low_seq = low_test[i + 1: i + 1 + entry_minutes]

        # ロング（買い）
        if preds_up[i] == 1 and preds_down[i] != 1 and adx[i] > 20:
            entry_price = raw_entry + spread  # askで買う
            tp_price = entry_price + tp_pips
            sl_price = entry_price - sl_pips

            tp_hit = (high_seq >= tp_price).any()
            sl_hit = (low_seq <= sl_price).any()

            if tp_hit and (not sl_hit or high_seq.idxmax() < low_seq.idxmin()):
                gain = lot * tp_pips * leverage
                balance += gain
                results_up.append(True)
            elif sl_hit:
                loss = lot * sl_pips * leverage
                balance -= loss
                results_up.append(False)
            else:
                # TP/SLどちらも未到達→entry_minutes後のcloseで決済
                exit_price = open_test[i + 1 + entry_minutes]
                profit = (exit_price - entry_price - spread) * lot * leverage
                balance += profit
                results_up.append(profit > 0)
            balance_curve.append(balance)

        # ショート（売り）
        elif preds_down[i] == 1 and preds_up[i] != 1 and adx[i] > 20:
            entry_price = raw_entry - spread  # bidで売る
            tp_price = entry_price - tp_pips
            sl_price = entry_price + sl_pips

            tp_hit = (low_seq <= tp_price).any()
            sl_hit = (high_seq >= sl_price).any()

            if tp_hit and (not sl_hit or low_seq.idxmin() < high_seq.idxmax()):
                gain = lot * tp_pips * leverage
                balance += gain
                results_down.append(True)
            elif sl_hit:
                loss = lot * sl_pips * leverage
                balance -= loss
                results_down.append(False)
            else:
                # TP/SLどちらも未到達→entry_minutes後のcloseで決済
                exit_price = open_test[i + 1 + entry_minutes]
                profit = (entry_price - exit_price - spread) * lot * leverage
                balance += profit
                results_down.append(profit > 0)
            balance_curve.append(balance)

        if balance <= 0:
            print("資金が尽きました。")
            break

    winrate_up = sum(results_up) / len(results_up) if results_up else None
    winrate_down = sum(results_down) / len(results_down) if results_down else None

    print(f"[ロング勝率] {winrate_up:.2%} ({len(results_up)}回)")
    print(f"[ショート勝率] {winrate_down:.2%} ({len(results_down)}回)")
    print(f"[最終資金] {balance:.2f}円 / ドローダウン最大: {max(balance_curve) - min(balance_curve):.2f}円")

    return balance, results_up, results_down

calc_winrate(X_test, open_test, high_test,low_test, model_up, model_down)
calc_winrate_technical(X_test, open_test, close_test, entry_minutes=5)
calc_winrate_technical(X_test, open_test, close_test)

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
            SYMBOL, TO_SYMBOL, col, None, X_trainb, y_trainb, select, pca, n_features=n_features
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
    # Run real-time signal generation
    run_realtime_signals(SYMBOL, TO_SYMBOL)


