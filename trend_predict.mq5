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

double lots;

int OnInit()
{
   return INIT_SUCCEEDED;
}
void OnTick()
{
    // 既にポジションがあればエントリーしない
    if (PositionSelect(Symbol())) return;

    // 必要なローソク足データの取得
    MqlRates price[];
    ArraySetAsSeries(price, true);
    if (CopyRates(Symbol(), PERIOD_CURRENT, 0, 60, price) < 60) return;

    // 各種インジケーター
    double smaShort[], smaMid[], smaLong[], upperBB[], lowerBB[], atr[];
    if (!iMA(Symbol(), PERIOD_CURRENT, smaShortPeriod, 0, MODE_SMA, PRICE_CLOSE).CopyBuffer(0, 0, 3, smaShort)) return;
    if (!iMA(Symbol(), PERIOD_CURRENT, smaMidPeriod,   0, MODE_SMA, PRICE_CLOSE).CopyBuffer(0, 0, 3, smaMid)) return;
    if (!iMA(Symbol(), PERIOD_CURRENT, smaLongPeriod,  0, MODE_SMA, PRICE_CLOSE).CopyBuffer(0, 0, 3, smaLong)) return;

    if (!iBands(Symbol(), PERIOD_CURRENT, bbPeriod, 0, bbDeviation, PRICE_CLOSE).CopyBuffer(1, 0, 3, upperBB)) return;
    if (!iBands(Symbol(), PERIOD_CURRENT, bbPeriod, 0, bbDeviation, PRICE_CLOSE).CopyBuffer(2, 0, 3, lowerBB)) return;

    if (!iATR(Symbol(), PERIOD_CURRENT, atrPeriod).CopyBuffer(0, 1, 1, atr)) return;

    double atr_value = atr[0];

    // ローソク足判定（陽線2本 or 陰線2本）
    bool bullishCandle = price[2].close > price[2].open && price[1].close > price[1].open;
    bool bearishCandle = price[2].close < price[2].open && price[1].close < price[1].open;

    // BBタッチ
    bool bbUpperTouch = price[1].high >= upperBB[1];
    bool bbLowerTouch = price[1].low  <= lowerBB[1];

    // SMAクロス
    bool goldenCross = smaShort[2] < smaMid[2] && smaShort[1] > smaMid[1];
    bool deadCross   = smaShort[2] > smaMid[2] && smaShort[1] < smaMid[1];

    // トレンド判定
    bool upTrend   = smaShort[1] > smaMid[1] && smaMid[1] > smaLong[1];
    bool downTrend = smaShort[1] < smaMid[1] && smaMid[1] < smaLong[1];

    // TP/SLはATRに基づく
    double sl_pips = atr_value;
    double tp_pips = atr_value * rrRatio;

    // ロット計算（ATRに基づく）
    double lots = CalculateLotsByATR(atr_value);

    // === 買いエントリー（OR条件） ===
    if (upTrend && (bullishCandle || bbUpperTouch || goldenCross))
    {
        EnterTrade(ORDER_TYPE_BUY, sl_pips, tp_pips, lots);
    }

    // === 売りエントリー（OR条件） ===
    if (downTrend && (bearishCandle || bbLowerTouch || deadCross))
    {
        EnterTrade(ORDER_TYPE_SELL, sl_pips, tp_pips, lots);
    }
}

double CalculateLotsByATR(double atr_value)
{
    double balance      = AccountInfoDouble(ACCOUNT_BALANCE);
    double riskPercent  = 1.0;  // 資金の1%をリスク
    double riskAmount   = balance * riskPercent / 100.0;

    double tickValue    = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tickSize     = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);

    double sl_value     = atr_value / tickSize * tickValue;

    double lots = riskAmount / sl_value;

    // ロット制限（ブローカー仕様）
    double minLot   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double maxLot   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double lotStep  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

    lots = MathMax(minLot, MathMin(lots, maxLot));
    lots = NormalizeDouble(lots / lotStep, 0) * lotStep;

    return lots;
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
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskPercent = 1.0; // 資金の1%
   double riskAmount = balance * riskPercent / 100.0;
   double lotSize = (riskAmount * leverage) / (sl_pips * SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE));
   lotSize = MathMin(lotSize, SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX));
   lotSize = NormalizeDouble(lotSize, 2);
   return lotSize;
}
