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

    double slPips = atrValue;
    double tpPips = atrValue * rrRatio;
    double lots   = CalculateLotsByATR(atrValue);

    // === 買い条件 ===
    if (upTrend && (bullish2 || bbUpperTouch || goldenCross))
        EnterTrade(ORDER_TYPE_BUY, slPips, tpPips, lots);

    // === 売り条件 ===
    if (downTrend && (bearish2 || bbLowerTouch || deadCross))
        EnterTrade(ORDER_TYPE_SELL, slPips, tpPips, lots);
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

