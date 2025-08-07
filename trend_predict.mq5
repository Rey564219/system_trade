//+------------------------------------------------------------------+
//|                                  TrendBB_EA.mq5                  |
//+------------------------------------------------------------------+
#property strict
input int smaShortPeriod = 5;
input int smaMidPeriod   = 20;
input int smaLongPeriod  = 60;
input int bbPeriod       = 20;
input double bbDeviation = 2.0;
input int atrPeriod      = 14;
input double rrRatio     = 1.8; // 1.5〜2倍の中間
input double leverage    = 3.0;


input double riskPercent = 1.0;   // リスク1%
input ulong magicNumber  = 123456;

double lots;

int OnInit()
{
   return INIT_SUCCEEDED;
}

void OnTick()
{
    if (PositionSelect(Symbol())) return; // ポジション保有中はスキップ

    MqlRates m5[], m15[];
    ArraySetAsSeries(m5, true);
    ArraySetAsSeries(m15, true);

    if (CopyRates(Symbol(), PERIOD_M5, 0, 60, m5) < 3) return;
    if (CopyRates(Symbol(), PERIOD_M15, 0, 60, m15) < 3) return;

    // === 15分足でトレンド判定 ===
    double smaS_m15[], smaM_m15[], smaL_m15[];
    if (!CopyBuffer(iMA(Symbol(), PERIOD_M15, smaShortPeriod, 0, MODE_SMA, PRICE_CLOSE), 0, 0, 3, smaS_m15)) return;
    if (!CopyBuffer(iMA(Symbol(), PERIOD_M15, smaMidPeriod,   0, MODE_SMA, PRICE_CLOSE), 0, 0, 3, smaM_m15)) return;
    if (!CopyBuffer(iMA(Symbol(), PERIOD_M15, smaLongPeriod,  0, MODE_SMA, PRICE_CLOSE), 0, 0, 3, smaL_m15)) return;

    bool upTrend   = smaS_m15[1] > smaM_m15[1] && smaM_m15[1] > smaL_m15[1];
    bool downTrend = smaS_m15[1] < smaM_m15[1] && smaM_m15[1] < smaL_m15[1];

    // === 5分足でトリガー条件 ===
    double smaS_m5[], smaM_m5[];
    if (!CopyBuffer(iMA(Symbol(), PERIOD_M5, smaShortPeriod, 0, MODE_SMA, PRICE_CLOSE), 0, 0, 3, smaS_m5)) return;
    if (!CopyBuffer(iMA(Symbol(), PERIOD_M5, smaMidPeriod,   0, MODE_SMA, PRICE_CLOSE), 0, 0, 3, smaM_m5)) return;

    double bbUpper[], bbLower[];
    if (!CopyBuffer(iBands(Symbol(), PERIOD_M5, bbPeriod, 0, bbDeviation, PRICE_CLOSE), 1, 0, 3, bbUpper)) return;
    if (!CopyBuffer(iBands(Symbol(), PERIOD_M5, bbPeriod, 0, bbDeviation, PRICE_CLOSE), 2, 0, 3, bbLower)) return;

    double atr[];
    if (!CopyBuffer(iATR(Symbol(), PERIOD_M5, atrPeriod), 0, 1, 1, atr)) return;
    double atrValue = atr[0];

    bool bullish2 = m5[2].close > m5[2].open && m5[1].close > m5[1].open;
    bool bearish2 = m5[2].close < m5[2].open && m5[1].close < m5[1].open;

    bool bbUpperTouch = m5[1].high >= bbUpper[1];
    bool bbLowerTouch = m5[1].low  <= bbLower[1];

    bool goldenCross = smaS_m5[2] < smaM_m5[2] && smaS_m5[1] > smaM_m5[1];
    bool deadCross   = smaS_m5[2] > smaM_m5[2] && smaS_m5[1] < smaM_m5[1];

    double slPips = atrValue * 2;
    double tpPips = atrValue * 2 * rrRatio;
    double lots   = CalculateLots(slPips);

    // === 買い条件 ===
    if (upTrend && (bullish2 || bbUpperTouch || goldenCross))
        EnterTrade(ORDER_TYPE_BUY, slPips, tpPips);

    // === 売り条件 ===
    if (downTrend && (bearish2 || bbLowerTouch || deadCross))
        EnterTrade(ORDER_TYPE_SELL, slPips, tpPips);
}

// 注文発注
void EnterTrade(int type, double sl_pips, double tp_pips)
{
   double lot = 0.1;  // 固定ロットまたは外部で計算して渡す
   double price = (type == ORDER_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) :
                                             SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   // --- 修正: pipsをポイントに変換 ---
   double pipValue = ( _Digits == 3 || _Digits == 5 ) ? 10 * point : point;
   double sl = (type == ORDER_TYPE_BUY) ? price - sl_pips * pipValue : price + sl_pips * pipValue;
   double tp = (type == ORDER_TYPE_BUY) ? price + tp_pips * pipValue : price - tp_pips * pipValue;
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

