input ENUM_TIMEFRAMES TimeFrame = PERIOD_M5;
input int MaPeriod = 36;
input int nCounter = 3;
input double leverage = 3.0;

int trendDir = 0;
int reverseCount = 0;
bool isEntryReady = false;
double slPips = 0, tpPips = 0;
datetime lastTradeTime = 0;
input int atrPeriod      = 14;
input double rrRatio     = 1.8;
int magicNumber = 1954305;
int maHandle;
double maBuffer[];

//-------------------------------------------
int OnInit()
{
   maHandle = iMA(_Symbol, TimeFrame, MaPeriod, 0, MODE_SMA, PRICE_CLOSE);
   if (maHandle == INVALID_HANDLE)
   {
      Print("MAハンドル作成失敗");
      return INIT_FAILED;
   }
   return INIT_SUCCEEDED;
}

//-------------------------------------------
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

//-------------------------------------------
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

//-------------------------------------------
int DetectTrend()
{
   if(CopyBuffer(maHandle, 0, 1, 2, maBuffer) != 2)
   {
      Print("MA取得失敗");
      return 0;
   }

   double ma1 = maBuffer[0];
   double ma2 = maBuffer[1];

   double haClose, haOpen, haHigh, haLow;
   GetHeikinAshi(1, haOpen, haClose, haHigh, haLow);

   if(ma1 > ma2 && haClose > ma1)
      return 1;
   else if(ma1 < ma2 && haClose < ma1)
      return -1;
   else
      return 0;
}

//-------------------------------------------
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

//-------------------------------------------
void EnterTrade(int type, double sl_pips, double tp_pips)
{
   double lot = 0.1;  // 固定ロットまたは外部で計算して渡す
   double price = (type == ORDER_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) :
                                             SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double sl = (type == ORDER_TYPE_BUY) ? price - sl_pips * point : price + sl_pips * point;
   double tp = (type == ORDER_TYPE_BUY) ? price + tp_pips * point : price - tp_pips * point;
   double deviation = 10;  // 許容スリッページ（ポイント）

   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = lot;
   request.type = type;
   request.price = price;
   request.sl = NormalizeDouble(sl, _Digits);
   request.tp = NormalizeDouble(tp, _Digits);
   request.deviation = deviation;
   request.magic = 123456;
   request.type_filling = ORDER_FILLING_IOC;  // 成行注文
   request.comment = "EnterTrade";

   if(!OrderSend(request, result))
   {
      Print("注文送信失敗。エラー: ", GetLastError());
   }
   else
   {
      if(result.retcode == TRADE_RETCODE_DONE)
         Print("注文成功: チケット#", result.order);
      else
         Print("注文エラー: ", result.retcode);
   }
}


//-------------------------------------------
double CalculateLots(double sl_pips)
{
   double balance     = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskPercent = 1.0;
   double riskAmount  = balance * riskPercent / 100.0;

   double tickValue   = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double lotStep     = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minLot      = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot      = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

   double rawLot      = (riskAmount * leverage) / (sl_pips * tickValue);
   double finalLot    = MathMax(minLot, MathMin(rawLot, maxLot));

   finalLot = NormalizeDouble(finalLot / lotStep, 0) * lotStep;
   return finalLot;
}
