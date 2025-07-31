#property strict
#include <Trade\Trade.mqh>

input double RiskPercent   = 1.0;
input double leverage      = 10.0;
input int    ZigZagDepth   = 12;
input int    ZigZagDeviation = 5;
input int    ZigZagBackstep = 3;
input int    ma_period     = 20;
input int    atr_period    = 14;
input double atr_threshold = 0.0002;  // ATRがこれ未満ならエントリーしない
input int    magicNumber   = 1325325;
input double sl_margin     = 10;

CTrade trade;

int zz_handle_5m;
int ma_handle_5m;
int atr_handle_5m;

double zz_buf_5m[];
double ma_buf_5m[];
double atr_buf_5m[];

int OnInit()
{
   zz_handle_5m = iCustom(_Symbol, PERIOD_M5, "Examples\\ZigZag", ZigZagDepth, ZigZagDeviation, ZigZagBackstep);
   ma_handle_5m = iMA(_Symbol, PERIOD_M5, ma_period, 0, MODE_EMA, PRICE_CLOSE);
   atr_handle_5m = iATR(_Symbol, PERIOD_M5, atr_period);

   if(zz_handle_5m == INVALID_HANDLE || ma_handle_5m == INVALID_HANDLE || atr_handle_5m == INVALID_HANDLE)
   {
      Print("インジケーター初期化失敗");
      return INIT_FAILED;
   }

   ArraySetAsSeries(zz_buf_5m, true);
   ArraySetAsSeries(ma_buf_5m, true);
   ArraySetAsSeries(atr_buf_5m, true);

   return INIT_SUCCEEDED;
}

void OnTick()
{
   if(!CopyBuffer(zz_handle_5m, 0, 0, 100, zz_buf_5m)) return;
   if(!CopyBuffer(ma_handle_5m, 0, 0, 11, ma_buf_5m)) return;
   if(!CopyBuffer(atr_handle_5m, 0, 0, 2, atr_buf_5m)) return;

   double atr = atr_buf_5m[0];
   if(atr == 0 || atr < atr_threshold)
      return; // ATRが閾値未満 → エントリー見送り

   double high1, low1, high2, low2;
   if(!GetZigZagHighLow(zz_buf_5m, high1, low1, high2, low2)) return;

   double ma_now = ma_buf_5m[0];
   double ma_past = ma_buf_5m[5];
   if(ma_now == 0 || ma_past == 0) return;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   bool ma_up = ma_now > ma_past;
   bool ma_down = ma_now < ma_past;

   double zz_now = zz_buf_5m[1];
   if(zz_now == 0.0) return;

   // エントリー判定
   if(!PositionSelect(_Symbol)) {
      double high, low;
      if(ShouldBuy(zz_buf_5m, high, low) && ma_up && ask > high)
      {
         if(!IsLastCandleBullish()) return;

         double sl = low - sl_margin * _Point;
         double lots = CalculateLots((ask - sl) / _Point);
         EnterTrade(ORDER_TYPE_BUY, sl, 0, lots);
      }
      else if(ShouldSell(zz_buf_5m, high, low) && ma_down && bid < low)
      {
         if(!IsLastCandleBearish()) return;

         double sl = high + sl_margin * _Point;
         double lots = CalculateLots((sl - bid) / _Point);
         EnterTrade(ORDER_TYPE_SELL, sl, 0, lots);
      }
   }
   else
   {
      ulong ticket = PositionGetTicket(0);
      int type = (int)PositionGetInteger(POSITION_TYPE);

      if(type == POSITION_TYPE_BUY && !ma_up)
         trade.PositionClose(ticket);
      else if(type == POSITION_TYPE_SELL && !ma_down)
         trade.PositionClose(ticket);
   }
}

bool GetZigZagHighLow(const double &zz_buf[], double &high1, double &low1, double &high2, double &low2)
{
   int found_high = 0, found_low = 0;
   high1 = high2 = low1 = low2 = -1;

   for(int i = 1; i < 100; i++)
   {
      double val = zz_buf[i];
      if(val == 0.0) continue;

      double price = iClose(_Symbol, PERIOD_M5, i);

      if(val > price)
      {
         if(found_high == 0) { high1 = val; found_high++; }
         else if(found_high == 1) { high2 = val; found_high++; }
      }
      else
      {
         if(found_low == 0) { low1 = val; found_low++; }
         else if(found_low == 1) { low2 = val; found_low++; }
      }

      if(found_high >= 2 && found_low >= 2)
         return true;
   }

   return false;
}

bool ShouldBuy(const double &zz_buf[], double &lastHigh, double &lastLow)
{
   int count = 0;
   lastHigh = 0;
   lastLow = 0;

   for(int i = 1; i < 100 && count < 2; i++)
   {
      double val = zz_buf[i];
      if(val == 0) continue;

      double price = iClose(_Symbol, PERIOD_M5, i);
      if(val < price && count == 0) {
         lastLow = val; count++;
      }
      else if(val > price && count == 1) {
         lastHigh = val; count++;
      }
   }

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   return (count == 2 && ask > lastHigh);
}

bool ShouldSell(const double &zz_buf[], double &lastHigh, double &lastLow)
{
   int count = 0;
   lastHigh = 0;
   lastLow = 0;

   for(int i = 1; i < 100 && count < 2; i++)
   {
      double val = zz_buf[i];
      if(val == 0) continue;

      double price = iClose(_Symbol, PERIOD_M5, i);
      if(val > price && count == 0) {
         lastHigh = val; count++;
      }
      else if(val < price && count == 1) {
         lastLow = val; count++;
      }
   }

   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   return (count == 2 && bid < lastLow);
}

bool IsLastCandleBullish()
{
   double open = iOpen(_Symbol, PERIOD_M5, 1);
   double close = iClose(_Symbol, PERIOD_M5, 1);
   return close > open;
}

bool IsLastCandleBearish()
{
   double open = iOpen(_Symbol, PERIOD_M5, 1);
   double close = iClose(_Symbol, PERIOD_M5, 1);
   return close < open;
}

void EnterTrade(int type, double sl, double tp, double lot)
{
   double price = (type == ORDER_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                                           : SymbolInfoDouble(_Symbol, SYMBOL_BID);

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
   request.tp = (tp > 0) ? NormalizeDouble(tp, _Digits) : 0;
   request.magic = magicNumber;
   request.deviation = 10;
   request.type_filling = ORDER_FILLING_IOC;

   if(!OrderSend(request, result) || result.retcode != TRADE_RETCODE_DONE)
      Print("注文失敗: ", result.retcode);
   else
      Print("注文成功: ", result.order);
}

double CalculateLots(double sl_pips)
{
   double balance    = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = balance * RiskPercent / 100.0;
   double tickValue  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double lotStep    = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minLot     = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot     = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

   double rawLot = (riskAmount * leverage) / (sl_pips * tickValue);
   double finalLot = MathMax(minLot, MathMin(rawLot, maxLot));
   finalLot = NormalizeDouble(finalLot / lotStep, 0) * lotStep;
   return finalLot;
}
