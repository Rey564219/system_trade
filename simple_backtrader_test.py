"""
簡単なBacktraderテスト（外部依存関係なし）
"""
import pandas as pd
import numpy as np
import backtrader as bt
import matplotlib.pyplot as plt

# 必要な関数を個別に実装
def generate_technical_signal_simple(df, adx_period=14, sma_short=5, sma_mid=20, adx_threshold=20):
    """
    簡単なテクニカル指標に基づくシグナル生成
    """
    signals = []
    
    # 簡単なADX計算（実際のADXより簡易版）
    high_low = df['high'] - df['low']
    close_prev = df['close'].shift(1)
    high_close_prev = np.abs(df['high'] - close_prev)
    low_close_prev = np.abs(df['low'] - close_prev)
    
    tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
    atr = tr.rolling(adx_period).mean()
    
    # 簡単なDI計算
    dm_plus = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), 
                       np.maximum(df['high'] - df['high'].shift(1), 0), 0)
    dm_minus = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), 
                        np.maximum(df['low'].shift(1) - df['low'], 0), 0)
    
    di_plus = 100 * (dm_plus / atr).rolling(adx_period).mean()
    di_minus = 100 * (dm_minus / atr).rolling(adx_period).mean()
    
    # 簡単なADX計算
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(adx_period).mean()
    
    # SMA計算
    sma_s = df['close'].rolling(sma_short).mean()
    sma_m = df['close'].rolling(sma_mid).mean()
    
    # シグナル生成
    for i in range(len(df)):
        if i < adx_period:
            signals.append("HOLD")
            continue
            
        adx_val = adx.iloc[i] if not pd.isna(adx.iloc[i]) else 0
        di_plus_val = di_plus.iloc[i] if not pd.isna(di_plus.iloc[i]) else 0
        di_minus_val = di_minus.iloc[i] if not pd.isna(di_minus.iloc[i]) else 0
        
        if adx_val > adx_threshold:
            if di_plus_val > di_minus_val and sma_s.iloc[i] > sma_m.iloc[i]:
                signals.append("CALL")
            elif di_minus_val > di_plus_val and sma_s.iloc[i] < sma_m.iloc[i]:
                signals.append("PUT")
            else:
                signals.append("HOLD")
        else:
            signals.append("HOLD")
    
    return signals

def create_backtrader_data(df):
    """
    DataFrameからBacktraderのデータフィードを作成
    """
    # 時刻インデックスが必要な場合は、ダミーで作成
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1min')
    
    # DataFrameの列をBacktraderが期待する形式に変換
    bt_data = df[['timestamp', 'open', 'high', 'low', 'close']].copy()
    bt_data.columns = ['datetime', 'open', 'high', 'low', 'close']
    bt_data['volume'] = 1000  # ダミーボリューム
    bt_data['openinterest'] = 0  # ダミーオープンインタレスト
    bt_data.set_index('datetime', inplace=True)
    
    return bt_data

class SimpleStrategy(bt.Strategy):
    params = (
        ('tp_pips', 5),
        ('sl_pips', 3),
        ('spread', 0.02),
        ('lot_size', 1000),
        ('leverage', 3),
        ('symbol', 'USD/JPY'),
        ('entry_minutes', 10),
    )
    
    def __init__(self):
        self.signals = []
        self.current_signal_index = 0
        self.order = None
        self.entry_bar = None
        self.entry_type = None
        self.pips_unit = 0.01 if "JPY" in self.params.symbol else 0.0001
        self.tp_value = self.params.tp_pips * self.pips_unit
        self.sl_value = self.params.sl_pips * self.pips_unit
        
    def set_signals(self, signals):
        self.signals = signals
        
    def next(self):
        current_bar = len(self.data) - 1
        
        # まだポジションがない場合、シグナルをチェック
        if not self.position and current_bar < len(self.signals):
            signal = self.signals[current_bar]
            
            if signal == "CALL":
                entry_price = self.data.close[0] + self.params.spread
                size = self.params.lot_size / entry_price
                
                self.order = self.buy(size=size)
                self.entry_bar = current_bar
                self.entry_type = "CALL"
                        
            elif signal == "PUT":
                entry_price = self.data.close[0] - self.params.spread
                size = self.params.lot_size / entry_price
                
                self.order = self.sell(size=size)
                self.entry_bar = current_bar
                self.entry_type = "PUT"
        
        # entry_minutes後に強制決済
        if self.order is not None and self.entry_bar is not None:
            if current_bar >= self.entry_bar + self.params.entry_minutes:
                if self.position:
                    self.close()
                self.order = None
                self.entry_bar = None
                self.entry_type = None
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f'BUY EXECUTED: Price: {order.executed.price:.5f}, Size: {order.executed.size:.2f}')
            else:
                print(f'SELL EXECUTED: Price: {order.executed.price:.5f}, Size: {order.executed.size:.2f}')
    
    def notify_trade(self, trade):
        if trade.isclosed:
            print(f'TRADE CLOSED: Profit: {trade.pnl:.2f}')

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
        prices.append(max(new_price, 140.0))
    
    prices = np.array(prices[1:])
    
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

def run_backtrader_test():
    """
    Backtrader実装のテスト
    """
    print("=== 簡単なBacktraderテスト ===")
    
    # サンプルデータ生成
    df = create_sample_data(500)
    print(f"データサイズ: {len(df)}")
    
    # シグナル生成
    signals = generate_technical_signal_simple(df)
    print(f"シグナル数: {len(signals)}")
    print(f"CALL: {signals.count('CALL')}, PUT: {signals.count('PUT')}, HOLD: {signals.count('HOLD')}")
    
    # Backtraderデータフィード作成
    bt_data = create_backtrader_data(df)
    
    # Cerebro（バックテストエンジン）セットアップ
    cerebro = bt.Cerebro()
    
    # データフィードを追加
    data = bt.feeds.PandasData(dataname=bt_data)
    cerebro.adddata(data)
    
    # 戦略を追加
    cerebro.addstrategy(SimpleStrategy,
                       tp_pips=5,
                       sl_pips=3,
                       spread=0.02,
                       lot_size=1000,
                       leverage=3,
                       symbol='USD/JPY',
                       entry_minutes=10)
    
    # 初期資金設定
    cerebro.broker.setcash(50000)
    
    # 手数料設定
    cerebro.broker.setcommission(commission=0.001)
    
    # 戦略実行前の資金
    initial_value = cerebro.broker.getvalue()
    print(f"初期資金: {initial_value:.2f}円")
    
    # バックテスト実行
    result = cerebro.run()
    
    # 結果の取得
    final_value = cerebro.broker.getvalue()
    profit = final_value - initial_value
    
    # 戦略インスタンスにシグナルを設定
    strategy_instance = result[0]
    strategy_instance.set_signals(signals)
    
    print(f"最終資金: {final_value:.2f}円")
    print(f"利益: {profit:.2f}円")
    print(f"収益率: {(profit/initial_value)*100:.2f}%")
    
    return {
        'initial_value': initial_value,
        'final_value': final_value,
        'profit': profit,
        'return_pct': (profit/initial_value)*100,
        'signals': signals
    }

if __name__ == "__main__":
    result = run_backtrader_test()
    print("\n=== テスト完了 ===")
    print(f"結果: {result}")
