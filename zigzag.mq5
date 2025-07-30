#property strict
#include <Trade\Trade.mqh>

input double RiskPercent = 1.0;
input double leverage    = 3.0;
input int ZigZagDepth    = 12;
input int ZigZagDeviation = 5;
input int ZigZagBackstep = 3;
input int ma_period      = 20;
input int magicNumber    = 1325325;
input double sl_margin   = 10;

int zz_handle_5m;
int zz_handle_15m;

int ma_handle_5m;
int ma_handle_15m;

double zz_buf_5m[];
double zz_buf_15m[];

double ma_buf_5m[];
double ma_buf_15m[];

CTrade trade;

//+------------------------------------------------------------------+
int OnInit()
{
   zz_handle_5m = iCustom(_Symbol, PERIOD_M5, "Examples\\ZigZag", ZigZagDepth, ZigZagDeviation, ZigZagBackstep);
   zz_handle_15m = iCustom(_Symbol, PERIOD_M15, "Examples\\ZigZag", ZigZagDepth, ZigZagDeviation, ZigZagBackstep);

   ma_handle_5m = iMA(_Symbol, PERIOD_M5, ma_period, 0, MODE_EMA, PRICE_CLOSE);
   ma_handle_15m = iMA(_Symbol, PERIOD_M15, ma_period, 0, MODE_EMA, PRICE_CLOSE);

   return INIT_SUCCEEDED;
}
//+------------------------------------------------------------------+
void OnTick()
{
   // 5分足 ZigZag取得
   if(!CopyBuffer(zz_handle_5m, 0, 0, 100, zz_buf_5m)) return;
   // 15分足 ZigZag取得
   if(!CopyBuffer(zz_handle_15m, 0, 0, 100, zz_buf_15m)) return;

   // 5分足 EMA取得（11本分）
   if(!CopyBuffer(ma_handle_5m, 0, 0, 11, ma_buf_5m)) return;
   // 15分足 EMA取得（11本分）
   if(!CopyBuffer(ma_handle_15m, 0, 0, 11, ma_buf_15m)) return;

   // 5分足 ZigZag直近2高値・安値取得
   double high1_5m = -1, high2_5m = -1;
   double low1_5m = -1, low2_5m = -1;
   int found_high_5m = 0, found_low_5m = 0;
   for(int i=1; i<100; i++)
   {
      double val = zz_buf_5m[i];
      if(val == 0) continue;
      if(val < iClose(_Symbol, PERIOD_M5, i))
      {
         if(found_low_5m == 0) { low1_5m = val; found_low_5m++; }
         else if(found_low_5m == 1) { low2_5m = val; found_low_5m++; }
      }
      else
      {
         if(found_high_5m == 0) { high1_5m = val; found_high_5m++; }
         else if(found_high_5m == 1) { high2_5m = val; found_high_5m++; }
      }
      if(found_low_5m >= 2 && found_high_5m >= 2) break;
   }
   if(high1_5m == -1 || high2_5m == -1 || low1_5m == -1 || low2_5m == -1) return;

   // ZigZagトレンド判定
   bool isUptrendZZ_5m = (low1_5m > low2_5m) && (high1_5m > high2_5m);
   bool isDowntrendZZ_5m = (low1_5m < low2_5m) && (high1_5m < high2_5m);

   // EMA傾き判定
   double ma_now_5m = ma_buf_5m[0];
   double ma_past_5m = ma_buf_5m[10];

   if(ma_now_5m == 0 || ma_past_5m == 0) return;

   bool ema_up_5m = ma_now_5m > ma_past_5m;
   bool ema_down_5m = ma_now_5m < ma_past_5m;

   // 両方の時間足でトレンド方向が一致しているか
   bool isUptrend = isUptrendZZ_5m && ema_up_5m;
   bool isDowntrend = isDowntrendZZ_5m && ema_down_5m;
   bool isUptrendexit = isUptrendZZ_5m;
   bool isDowntrendexit = isDowntrendZZ_5m;

   double price_ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double price_bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   // ポジション保有時トレンド判定・決済
   if(PositionSelect(_Symbol))
   {
      ulong ticket = PositionGetTicket(0);
      int pos_type = (int)PositionGetInteger(POSITION_TYPE);

      if(pos_type == POSITION_TYPE_BUY)
      {
         if(!isUptrendexit)
         {
            if(!trade.PositionClose(ticket))
               Print("買いポジション決済失敗 エラー:", GetLastError());
            else
               Print("買いポジション決済完了");
            return;
         }
      }
      else if(pos_type == POSITION_TYPE_SELL)
      {
         if(!isDowntrendexit)
         {
            if(!trade.PositionClose(ticket))
               Print("売りポジション決済失敗 エラー:", GetLastError());
            else
               Print("売りポジション決済完了");
            return;
         }
      }
      return; // トレンド継続なら何もしない
   }

   // エントリー判定
   double sl, tp=0, lots;
   double zz_current_5m = zz_buf_5m[1];
   if(zz_current_5m == 0.0) return;

   // 買いエントリー
   if(isUptrend && price_ask > zz_current_5m)
   {
      sl = zz_current_5m - sl_margin * _Point;
      lots = CalculateLots((price_ask - sl) / _Point);
      EnterTrade(ORDER_TYPE_BUY, sl, tp, lots);
   }
   // 売りエントリー
   if(isDowntrend && price_bid < zz_current_5m)
   {
      sl = zz_current_5m + sl_margin * _Point;
      lots = CalculateLots((sl - price_bid) / _Point);
      EnterTrade(ORDER_TYPE_SELL, sl, tp, lots);
   }
}
//+------------------------------------------------------------------+
void EnterTrade(int type, double sl, double tp, double lot)
{
   double price = (type == ORDER_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) :
                                             SymbolInfoDouble(_Symbol, SYMBOL_BID);

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
   if(tp > 0)
      request.tp = NormalizeDouble(tp, _Digits);
   else
      request.tp = 0;
   request.deviation = 10;
   request.magic = magicNumber;
   request.type_filling = ORDER_FILLING_IOC;
   request.comment = "ZigZag+EMA Entry";

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
//+------------------------------------------------------------------+
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
