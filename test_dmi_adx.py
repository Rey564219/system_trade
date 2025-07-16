"""
DMI_ADX.py のBacktraderテスト用サンプルファイル
"""
import pandas as pd
import numpy as np
import sys
import os

SYMBOL = 'USD'
TO_SYMBOL = 'JPY'
# 現在のディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from DMI_ADX import (
    calc_winrate_technical_backtrader_advanced,
    calc_winrate_technical_bt_wrapper,
    calc_winrate_technical2_bt_wrapper,
    replace_original_functions,
    create_backtrader_data,
    generate_technical_signal,
    make_features,
    make_train_data,
    fetch_data
)

def create_sample_data(n_points=1000):
    """
    テスト用のサンプルデータを生成
    """
    np.random.seed(42)
    
    # 基本的な価格データを生成
    base_price = 150.0
    price_changes = np.random.normal(0, 0.01, n_points)
    prices = [base_price]
    
    for change in price_changes:
        new_price = prices[-1] + change
        prices.append(max(new_price, 140.0))  # 最低価格を設定
    
    prices = np.array(prices[1:])  # 最初の値を除去
    
    # OHLC データを生成
    opens = prices
    closes = prices + np.random.normal(0, 0.005, n_points)
    highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, 0.003, n_points))
    lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, 0.003, n_points))
    
    # DataFrameを作成
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': np.random.randint(1000, 10000, n_points)
    })
    
    return df

def test_backtrader_implementation():
    """
    Backtrader実装のテスト
    """
    print("=== Backtrader実装テスト開始 ===")
    
    # サンプルデータ生成
    pair = SYMBOL + TO_SYMBOL
            # make_train_dataを使って学習データ作成
    df, X_trainb, y_train, y_trainb, col, open_price, close, high, low, label, select, _, _, pca, scaler = make_train_data(pair, n_features=20)

    # fetch_dataを使ってテストデータ作成（SYMBOL, TO_SYMBOLは仮の値でOK）
    X, y_test, feature_names, open_test, close_test, high_test, low_test, label_test, _, _ = fetch_data(SYMBOL, TO_SYMBOL, col, scaler, X_trainb, y_trainb, select, pca, n_features=20)
    

    # 特徴量を追加
    try:
        df = make_features(df)
        X = make_features(X)
        print("特徴量生成完了")
    except Exception as e:
        print(f"特徴量生成エラー: {e}")
        # 必要な特徴量を手動で追加
        df['ADX'] = 25.0  # 固定値
        df['PLUS_DI'] = 20.0
        df['MINUS_DI'] = 15.0
        df['SMA_5'] = df['close'].rolling(5).mean()
        df['SMA_20'] = df['close'].rolling(20).mean()
        df['SMA_100'] = df['close'].rolling(100).mean()
        df['atr'] = (df['high'] - df['low']).rolling(14).mean()
        X['ADX'] = 25.0  # 固定値
        X['PLUS_DI'] = 20.0
        X['MINUS_DI'] = 15.0
        X['SMA_5'] = X['close'].rolling(5).mean()
        X['SMA_20'] = X['close'].rolling(20).mean() 
        X['SMA_100'] = X['close'].rolling(100).mean()
        X['atr'] = (X['high'] - X['low']).rolling(14).mean()
        # EMAとボリンジャーバンドを手動で追加
        df['ema_fast'] = df['close'].ewm(span=5).mean()
        df['ema_slow'] = df['close'].ewm(span=20).mean()
        X['ema_fast'] = X['close'].ewm(span=5).mean()
        X['ema_slow'] = X['close'].ewm(span=20).mean
        # 簡易ボリンジャーバンド
        bb_period = 20
        bb_std = 2
        bb_ma = df['close'].rolling(bb_period).mean()
        bb_stddev = df['close'].rolling(bb_period).std()
        df['bb_upper'] = bb_ma + (bb_stddev * bb_std)
        df['bb_middle'] = bb_ma
        df['bb_lower'] = bb_ma - (bb_stddev * bb_std)
        
        bb_ma = X['close'].rolling(bb_period).mean()
        bb_stddev = X['close'].rolling(bb_period).std() 
        X['bb_upper'] = bb_ma + (bb_stddev * bb_std)
        X['bb_middle'] = bb_ma
        X['bb_lower'] = bb_ma - (bb_stddev * bb_std)
        df = df.dropna()
        X = X.dropna()
    
    print(f"データサイズ: {len(df)}")
    print(f"データ列: {df.columns.tolist()}")
    
    # Backtraderテスト1: 時間制限あり
    print("\n--- テスト1: 時間制限あり (10分) ---")
    try:
        result1 = calc_winrate_technical_backtrader_advanced(
            df, tp_pips=5, sl_pips=3, entry_minutes=10, 
            unlimited_tracking=False, start_balance=50000
        )
        result1_test = calc_winrate_technical_backtrader_advanced(
            X, tp_pips=5, sl_pips=3, entry_minutes=10, 
            unlimited_tracking=False, start_balance=50000
        )
        print(f"結果1: {result1}")
        print(f"結果1_test: {result1_test}")
    except Exception as e:
        print(f"テスト1エラー: {e}")
    
    # Backtraderテスト2: 時間制限なし
    print("\n--- テスト2: 時間制限なし (無制限追跡) ---")
    try:
        result2 = calc_winrate_technical_backtrader_advanced(
            df, tp_pips=5, sl_pips=3, entry_minutes=10, 
            unlimited_tracking=True, start_balance=50000
        )
        result2_test = calc_winrate_technical_backtrader_advanced(
            X, tp_pips=5, sl_pips=3, entry_minutes=10, 
            unlimited_tracking=True, start_balance=50000
        )
        print(f"結果2: {result2}")
        print(f"結果2_test: {result2_test}")
    except Exception as e:
        print(f"テスト2エラー: {e}")
    
    # ラッパー関数のテスト
    print("\n--- テスト3: ラッパー関数 ---")
    try:
        winrate, entries, results, signals = calc_winrate_technical_bt_wrapper(
            df, df['open'], df['close'], df['high'], df['low']
        )
        print(f"勝率: {winrate:.4f}, エントリー数: {len(entries)}, 結果: {results}")
        winrate_test, entries_test, results_test, signals_test = calc_winrate_technical_bt_wrapper(
            X, X['open'], X['close'], X['high'], X['low']
        )
        print(f"勝率_test: {winrate_test:.4f}, エントリー数_test: {len(entries_test)}, 結果_test: {results_test}")
    except Exception as e:
        print(f"テスト3エラー: {e}")
    
    print("\n=== テスト完了 ===")

if __name__ == "__main__":
    test_backtrader_implementation()
