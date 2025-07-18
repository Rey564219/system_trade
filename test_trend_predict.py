"""
trend_predict.py のBacktraderテスト用サンプルファイル
"""
import pandas as pd
import numpy as np
import sys
import os

SYMBOL = 'USD' 
TO_SYMBOL = 'JPY'
# 現在のディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trend_predict import (
    calc_winrate_technical_backtrader,
    calc_winrate_technical2_backtrader,
    calc_winrate_technical_bt_wrapper,
    calc_winrate_technical2_bt_wrapper,
    make_train_data,
    fetch_data
)
# 必要なライブラリのインポートを試行
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("警告: TALibが利用できません。一部の機能が制限されます。")

try:
    import backtrader as bt
    BACKTRADER_AVAILABLE = True
except ImportError:
    BACKTRADER_AVAILABLE = False
    print("警告: Backtraderが利用できません。")

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

def add_basic_features(df):
    """
    基本的な特徴量を追加
    """
    # 移動平均
    df['SMA_5'] = df['close'].rolling(5).mean()
    df['SMA_20'] = df['close'].rolling(20).mean()
    df['SMA_100'] = df['close'].rolling(100).mean()
    
    # ATR
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum.reduce([
        df['high'] - df['low'],
        abs(df['high'] - df['prev_close']),
        abs(df['low'] - df['prev_close'])
    ])
    df['atr'] = df['tr'].rolling(14).mean()
    
    if TALIB_AVAILABLE:
        # TALibを使用した指標
        df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['PLUS_DI'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        df['MINUS_DI'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    else:
        # 簡易的なADX計算
        df['ADX'] = 25.0  # 固定値
        df['PLUS_DI'] = 20.0
        df['MINUS_DI'] = 15.0
    
    # NaNを除去
    df = df.dropna()
    
    return df

def test_trend_predict():
    """
    trend_predict.pyの機能をテスト
    """
    print("=== Trend Predict Backtraderテスト ===")
    
    # 実際のデータを使用
    try:
        pair = SYMBOL + TO_SYMBOL
        print(f"データ取得中: {pair}")
        
        # make_train_dataを使ってデータを取得
        X_train, X_trainb, y_train, y_trainb, col, open_price, close, high, low, label, select, _, _, pca, scaler = make_train_data(pair, n_features=20)
        
        # fetch_dataを使ってテストデータを取得
        X_test, y_test, feature_names, open_test, close_test, high_test, low_test, label_test, _, _ = fetch_data(SYMBOL, TO_SYMBOL, col, scaler, X_trainb, y_trainb, select, pca, n_features=20)
        
        # 訓練データを使用
        df = X_train.copy()
        
        print(f"データサイズ: {len(df)}")
        print(f"データ列: {df.columns.tolist()}")
        
        # 必要な列があることを確認
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            print("エラー: 必要な価格データ列が不足しています")
            return
        
        # テストデータも使用
        df_test = X_test.copy()
        print(f"テストデータサイズ: {len(df_test)}")
        
        if BACKTRADER_AVAILABLE and TALIB_AVAILABLE:
            try:
                print("\n--- テスト1: 時間制限あり (10分) ---")
                result1 = calc_winrate_technical_backtrader(
                    df, tp_pips=5, sl_pips=3, entry_minutes=10, 
                    unlimited_tracking=False, start_balance=50000
                )
                print(f"結果1: 利益={result1['profit']:.2f}, 勝率={result1['winrate']:.2%}")
                
                print("\n--- テスト1_test: 時間制限あり (10分) ---")
                result1_test = calc_winrate_technical_backtrader(
                    df_test, tp_pips=5, sl_pips=3, entry_minutes=10, 
                    unlimited_tracking=False, start_balance=50000
                )
                print(f"結果1_test: 利益={result1_test['profit']:.2f}, 勝率={result1_test['winrate']:.2%}")
                
                print("\n--- テスト2: 時間制限なし ---")
                result2 = calc_winrate_technical2_backtrader(
                    df, tp_pips=5, sl_pips=3, start_balance=50000
                )
                print(f"結果2: 利益={result2['profit']:.2f}, 勝率={result2['winrate']:.2%}")
                
                print("\n--- テスト3: ラッパー関数 ---")
                try:
                    winrate, entries, results, signals = calc_winrate_technical_bt_wrapper(
                        df, df['open'], df['close'], df['high'], df['low']
                    )
                    print(f"勝率: {winrate:.4f}, エントリー数: {len(entries)}, 結果: {len(results)}")
                except Exception as e:
                    print(f"ラッパー関数テストエラー: {e}")
                
            except Exception as e:
                print(f"Backtraderテストエラー: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("必要なライブラリが不足しているため、テストをスキップします。")
            
    except Exception as e:
        print(f"データ取得エラー: {e}")
        import traceback
        traceback.print_exc()
        
        # フォールバック: サンプルデータを使用
        print("\nサンプルデータでテストを続行...")
        df = create_sample_data(2000)
        df = add_basic_features(df)
        
        if BACKTRADER_AVAILABLE:
            try:
                print("\n--- サンプルデータテスト ---")
                result = calc_winrate_technical_backtrader(
                    df, tp_pips=5, sl_pips=3, entry_minutes=10, 
                    unlimited_tracking=False, start_balance=50000
                )
                print(f"サンプルデータ結果: 利益={result['profit']:.2f}, 勝率={result['winrate']:.2%}")
                
            except Exception as e:
                print(f"サンプルデータテストエラー: {e}")
    
    print("\n=== テスト完了 ===")

if __name__ == "__main__":
    test_trend_predict()
