input ENUM_TIMEFRAMES TimeFrame = PERIOD_M5; // 使用時間足
input int MaPeriod = 20;
input int nCounter = 3;                      // 逆行本数
input double leverage = 3.0;
int trendDir = 0;
int reverseCount = 0;
bool isEntryReady = false;
double slPips = 0, tpPips = 0;
datetime lastTradeTime = 0;
int OnInit()
{
   return INIT_SUCCEEDED;
}
void OnTick()
{
   if(!IsNewBar(TimeFrame)) return;
   int dir = DetectTrend();
   if(dir == 0) return;

   double haOpen, haClose, haHigh, haLow;
   GetHeikinAshi(1, haOpen, haClose, haHigh, haLow);

   bool haReversal = (dir == 1 && haClose < haOpen) || (dir == -1 && haClose > haOpen);

   if(dir != trendDir) {
      trendDir = dir;
      reverseCount = 0;
      isEntryReady = false;
   }

   if(haReversal)
      reverseCount++;
   else {
      if(reverseCount >= nCounter) {
         isEntryReady = true;
         // 損切と利確のpips計算
         double shadow = (dir == 1) ? (haLow - haOpen) : (haHigh - haOpen);
         slPips = NormalizeDouble(MathAbs(shadow), _Digits);
         tpPips = NormalizeDouble(slPips * 1.5, _Digits);
      }
      reverseCount = 0;
   }

   if(isEntryReady && TimeCurrent() - lastTradeTime > 60) {
      double lots = CalculateLots(slPips);
      EnterTrade((dir == 1) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL, slPips, tpPips);
      lastTradeTime = TimeCurrent();
      isEntryReady = false;
   }
}

void GetHeikinAshi(int index, double &haOpen, double &haClose, double &haHigh, double &haLow)
{
   double prevHaOpen, prevHaClose;
   if(index+1 < Bars(_Symbol, TimeFrame)) {
      GetHeikinAshi(index+1, prevHaOpen, prevHaClose, haHigh, haLow);
   } else {
      prevHaOpen = iOpen(_Symbol, TimeFrame, index+1);
      prevHaClose = iClose(_Symbol, TimeFrame, index+1);
   }

   double open  = iOpen(_Symbol, TimeFrame, index);
   double close = iClose(_Symbol, TimeFrame, index);
   double high  = iHigh(_Symbol, TimeFrame, index);
   double low   = iLow(_Symbol, TimeFrame, index);

   haClose = (open + high + low + close) / 4.0;
   haOpen  = (prevHaOpen + prevHaClose) / 2.0;
   haHigh  = MathMax(high, MathMax(haOpen, haClose));
   haLow   = MathMin(low, MathMin(haOpen, haClose));
}
int DetectTrend()
{
   double ma1 = iMA(_Symbol, TimeFrame, MaPeriod, 0, MODE_SMA, PRICE_CLOSE, 1);
   double ma2 = iMA(_Symbol, TimeFrame, MaPeriod, 0, MODE_SMA, PRICE_CLOSE, 2);
   double haClose, haOpen, haHigh, haLow;
   GetHeikinAshi(1, haOpen, haClose, haHigh, haLow);

   if(ma1 > ma2 && haClose > ma1)
      return 1; // 上昇トレンド
   else if(ma1 < ma2 && haClose < ma1)
      return -1; // 下降トレンド
   else
      return 0;
}
datetime lastBarTime = 0;
bool IsNewBar(ENUM_TIMEFRAMES tf)
{
   datetime currentBarTime = iTime(_Symbol, tf, 0);
   if(currentBarTime != lastBarTime)
   {
      lastBarTime = currentBarTime;
      return true;
   }
   return false;
}


// 注文発注
void EnterTrade(int type, double sl_pips, double tp_pips)
{
   double price = (type == ORDER_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) :
                                              SymbolInfoDouble(_Symbol, SYMBOL_BID);

   double sl = (type == ORDER_TYPE_BUY) ? price - sl_pips : price + sl_pips;
   double tp = (type == ORDER_TYPE_BUY) ? price + tp_pips : price - tp_pips;

   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   request.action   = TRADE_ACTION_DEAL;
   request.symbol   = _Symbol;
   request.volume   = lots;
   request.type     = type;
   request.price    = price;
   request.sl       = sl;
   request.tp       = tp;
   request.deviation = 10;
   request.magic    = 123456;

   OrderSend(request, result);
}

// ロット数算出（レバレッジ計算含む）
double CalculateLots(double sl_pips)
{
    double balance        = AccountInfoDouble(ACCOUNT_BALANCE);
    double riskPercent    = 1.0;  // リスク割合 (%)
    double riskAmount     = balance * riskPercent / 100.0;

    double tickValue      = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double lotStep        = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    double minLot         = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double maxLot         = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

    // リスクに基づいた理論ロット数
    double rawLot         = (riskAmount * leverage) / (sl_pips * tickValue);

    // 最小ロットを保証
    double finalLot       = MathMax(minLot, MathMin(rawLot, maxLot));

    // ロットの刻みに合わせて丸める
    finalLot = NormalizeDouble(finalLot / lotStep, 0) * lotStep;

    return finalLot;
}

