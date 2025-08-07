input ENUM_TIMEFRAMES TimeFrame = PERIOD_M5;
input int MaPeriod = 40; // 40に変更
input int nCounter = 2;
input double leverage = 10.0;
input double RiskPercent = 1.0; // リスクパーセントを追加

int trendDir = 0;
int reverseCount = 0;
bool isEntryReady = false;
double slPips = 0, tpPips = 0;
datetime lastTradeTime = 0;
input int atrPeriod      = 14;
input double rrRatio     = 1.5; // 2:1に変更

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
// ...existing code...

// 反転期間中の最安値・最高値を記録する変数を追加
double reversalMin = 0.0;
double reversalMax = 0.0;

//-------------------------------------------
void OnTick()
{
   if(!IsNewBar(TimeFrame)) return;

   int dir = DetectTrend();
   if(dir == 0) return;

   double haOpen, haClose, haHigh, haLow;
   GetHeikinAshi(1, haOpen, haClose, haHigh, haLow);

   // トレンドと反対の平均足かどうかを判断
   bool haReversal = (dir == 1 && haClose < haOpen) || (dir == -1 && haClose > haOpen);

   if(dir != trendDir) {
      trendDir = dir;
      reverseCount = 0;
      isEntryReady = false;
      reversalMin = 0.0;
      reversalMax = 0.0;
   }

   if(haReversal) {
      reverseCount++;
      double low = iLow(_Symbol, TimeFrame, 1);
      double high = iHigh(_Symbol, TimeFrame, 1);
      if(reverseCount == 1) {
         reversalMin = low;
         reversalMax = high;
      } else {
         if(low < reversalMin) reversalMin = low;
         if(high > reversalMax) reversalMax = high;
      }
   } else {
      // 反対の平均足がnCounter回続いた後、元のトレンドと同じ平均足に反転した場合
      if(reverseCount >= nCounter) {
         isEntryReady = true;

         double currentPrice = (dir == 1) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : 
                                            SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
         int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

         // 5桁ブローカー対応
         if(digits == 5 || digits == 3) point *= 10;

         if(dir == 1) { // 買いエントリー
            slPips = MathAbs(currentPrice - reversalMin) / point;
         } else { // 売りエントリー
            slPips = MathAbs(reversalMax - currentPrice) / point;
         }

         tpPips = slPips * rrRatio; // 2:1の比率

         // デバッグ情報出力
         Print("SL/TP Debug: reversalMin=", reversalMin, " reversalMax=", reversalMax, " currentPrice=", currentPrice);
         Print("SL/TP Debug: point=", point, " digits=", digits, " slPips=", slPips, " tpPips=", tpPips);
      }
      reverseCount = 0;
      reversalMin = 0.0;
      reversalMax = 0.0;
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
   if(CopyBuffer(maHandle, 0, 1, 5, maBuffer) != 5)
   {
      Print("MA取得失敗");
      return 0;
   }
   int adxHandle;
   double adxBuffer[1];
   adxHandle = iADX(_Symbol, TimeFrame, 14);
   if(CopyBuffer(adxHandle, 0, 1, 1, adxBuffer) != 1)
   {
      Print("ADX値取得失敗");
      return 0;
   }
   double adx = adxBuffer[0];
   double ma1 = maBuffer[0];
   double ma2 = maBuffer[4];

   double haClose, haOpen, haHigh, haLow;
   GetHeikinAshi(1, haOpen, haClose, haHigh, haLow);

   if(ma1 > ma2 && adx > 20)
      return 1;
   else if(ma1 < ma2 && adx > 20)
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
   double lot = CalculateLots(sl_pips);
   double price = (type == ORDER_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) :
                                             SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   
   // 5桁ブローカー対応
   if(digits == 5 || digits == 3) point *= 10;
   
   double sl = (type == ORDER_TYPE_BUY) ? price - sl_pips * point : price + sl_pips * point;
   double tp = (type == ORDER_TYPE_BUY) ? price + tp_pips * point : price - tp_pips * point;
   double deviation = 10;
   
   // デバッグ情報を出力
   Print("EnterTrade Debug: price=", price, " sl_pips=", sl_pips, " tp_pips=", tp_pips, " point=", point);
   Print("EnterTrade Debug: sl=", sl, " tp=", tp, " type=", (type == ORDER_TYPE_BUY ? "BUY" : "SELL"));

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
   request.magic = magicNumber; // magicNumberを使用
   request.type_filling = ORDER_FILLING_IOC;
   request.comment = "HeikinAshi Entry";

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
   double balance    = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = balance * RiskPercent / 100.0;
   double tickValue  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double lotStep    = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minLot     = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot     = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

   if(sl_pips <= 0 || sl_pips < 1.0) {
      Print("SLが小さすぎます。最小ロットを使用します。sl_pips=", sl_pips);
      return minLot;
   }

   double rawLot = riskAmount / (sl_pips * tickValue);
   double finalLot = MathMax(minLot, MathMin(rawLot, maxLot));

   // ロット数をブローカーのロットステップに合わせて丸める
   finalLot = MathFloor(finalLot / lotStep) * lotStep;
   finalLot = NormalizeDouble(finalLot, (int)MathLog10(1.0/lotStep));

   // デバッグ情報
   Print("CalculateLots Debug: balance=", balance, " riskAmount=", riskAmount, " sl_pips=", sl_pips);
   Print("CalculateLots Debug: tickValue=", tickValue, " rawLot=", rawLot, " finalLot=", finalLot);
   Print("CalculateLots Debug: lotStep=", lotStep, " minLot=", minLot, " maxLot=", maxLot);

   return finalLot;
}