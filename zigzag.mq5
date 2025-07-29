#property strict
#include <Trade\Trade.mqh>

input double RiskReward = 1.8;       // TP/SL比率
input double RiskPercent = 1.0;      // 資金に対するリスク%
input double leverage = 3.0;       // 証拠金レバレッジ（ブローカーに合わせて変更）
input int ZigZagDepth = 12;
input int ZigZagDeviation = 5;
input int ZigZagBackstep = 3;
input int magicNumber = 1325325;

int zz_handle_15m, zz_handle_5m;
double zz_15m[], zz_5m[];

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   zz_handle_15m = iCustom(_Symbol, PERIOD_M15, "Examples\\ZigZag", ZigZagDepth, ZigZagDeviation, ZigZagBackstep);
   zz_handle_5m  = iCustom(_Symbol, PERIOD_M5,  "Examples\\ZigZag", ZigZagDepth, ZigZagDeviation, ZigZagBackstep);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   if(!CopyBuffer(zz_handle_15m, 0, 0, 100, zz_15m)) return;
   if(!CopyBuffer(zz_handle_5m,  0, 0, 100, zz_5m))  return;

   //--- トレンド判定（15分足ZigZag）
   int peak15 = -1, valley15 = -1;
   for(int i = 1; i < 100; i++)
   {
      if(zz_15m[i] != 0)
      {
         if(valley15 == -1)
            valley15 = i;
         else if(peak15 == -1)
         {
            peak15 = i;
            break;
         }
      }
   }
   if(peak15 == -1 || valley15 == -1) return;

   bool isUptrend = valley15 > peak15;

   //--- エントリーチャンス検出（5分足 ZigZag）
   int lastZZIndex = -1;
   double lastZZVal = 0;
   for(int i = 1; i < 100; i++)
   {
      if(zz_5m[i] != 0)
      {
         lastZZIndex = i;
         lastZZVal = zz_5m[i];
         break;
      }
   }
   if(lastZZIndex == -1) return;

   double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   // ポジションがなければのみエントリー
   if(PositionSelect(_Symbol)) return;

   double sl_pips, tp_pips;

   //--- Buy条件（押し目買い）
   if(isUptrend && price > lastZZVal)
   {
      sl_pips = price - lastZZVal;
      tp_pips = sl_pips * RiskReward;
      double lots = CalculateLots(sl_pips / _Point);
      EnterTrade(ORDER_TYPE_BUY, sl_pips, tp_pips);
   }

   //--- Sell条件（戻り売り）
   if(!isUptrend && price < lastZZVal)
   {
      sl_pips = lastZZVal - price;
      tp_pips = sl_pips * RiskReward;
      double lots = CalculateLots(sl_pips / _Point);
      EnterTrade(ORDER_TYPE_SELL, sl_pips, tp_pips);
   }
}

//+------------------------------------------------------------------+
//| 注文発注関数（ユーザー提供）                                    |
//+------------------------------------------------------------------+
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


//+------------------------------------------------------------------+
//| ロット計算（ユーザー提供＋修正）                                 |
//+------------------------------------------------------------------+
double CalculateLots(double sl_pips)
{
    double balance     = AccountInfoDouble(ACCOUNT_BALANCE);
    double riskAmount  = balance * RiskPercent / 100.0;
    double tickValue   = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double lotStep     = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    double minLot      = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double maxLot      = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

    // ロット計算（tickValueは1ロットあたり）
    double rawLot = (riskAmount * leverage) / (sl_pips * tickValue);

    // ロット制限内に収める
    double finalLot = MathMax(minLot, MathMin(rawLot, maxLot));

    // ロット刻みに丸める
    finalLot = NormalizeDouble(finalLot / lotStep, 0) * lotStep;
    return finalLot;
}
