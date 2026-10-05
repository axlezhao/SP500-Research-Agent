# Findings: short-term stock prediction on the S&P 500

*Live run of 2 October 2026. Data from January 2014 to October 2026; out-of-sample period May 2020 to September 2026.*

> **These results predate the October 2026 research overhaul** ([ROADMAP](ROADMAP.md#done)). That run still had three biases since fixed: former members were labelled sector "Unknown" (a hint that they would leave the index), stocks that stopped trading were dropped from the backtest on the signal date, and "5 days ago" counted rows rather than sessions. It also predicted a yes/no label, chose models by AUC, and traded a plain weekly top 20% at a flat cost. The current pipeline uses a ranked target, more signals, cost-aware portfolios, an untouched holdout from 2025 and deflated statistics, so the numbers below will change. Re-run `python run_pipeline.py` and read `reports/research/research_report.md` for current results.

## Summary

**Question.** Can a model built only from free public data (prices, SEC filings and macro series) rank S&P 500 stocks well enough to beat holding the index, after trading costs?

**Answer. No, not in this setup.**
- **Ranking:** the best ranking has a small edge that isn't statistically reliable (rank IC 0.023, t = 1.85).
- **Before costs:** a portfolio of its top 20% earned 17.6% a year, exactly matching the S&P 500.
- **After costs:** trading it every week costs 3.8% a year, so it ends up 3.9 points a year behind.
- **Attribution:** after accounting for market, size, value and momentum exposure, no strategy tested has alpha distinguishable from zero.

The rest of this document covers how that was measured, what else was tried, and what the results do and don't show.

## Method

| Element | Choice |
|---|---|
| Universe | Stocks that were S&P 500 members on each date, reconstructed from Wikipedia's change log: 758 tickers since 2014. Prices exist for 98 of the 255 that left the index. |
| Inputs | 22 features: price and volume signals; SEC fundamentals as first reported (usable the day after filing); lagged VIX and yields; ranks among the day's members |
| Validation | Five expanding-window walk-forward folds, with a gap as long as the prediction horizon between training and test labels. Every result below is out of sample. |
| Models | Logistic regression, random forest, gradient boosting; selected by mean out-of-sample AUC |
| Portfolios | Every horizon: buy the top 20% (long-only), or also short the bottom 20% (long-short). Enter one session after the signal; 10 bps cost per unit of weight traded |
| Benchmarks | S&P 500 (SPY, dividends included) and an equal-weight portfolio of all members, over the same sessions |
| Risk-adjusted return | Sharpe in excess of 3-month T-bills for long portfolios; raw for long-short, which is self-financing |
| Attribution | Regression on market, size (small minus big), value (cheap minus expensive) and momentum (12-1 month) factors built from the same stocks and dates |

## Results: the production setup

The production setup predicts whether a stock beats the median index member over the next 5 trading days. Logistic regression won the model comparison (AUC 0.511, against 0.506 for the other two).

| | Total return | CAGR | Sharpe | Max drawdown |
|---|---|---|---|---|
| Long top 20% | +125% | 13.8% | 0.62 | −26.9% |
| Long-short | −15% | −2.6% | −0.10 | −25.5% |
| Equal-weight members | +141% | 15.0% | 0.78 | −20.6% |
| **S&P 500 (SPY)** | **+178%** | **17.6%** | **0.92** | −23.2% |

- **Ranking:** AUC 0.511, where 0.5 is no skill. Rank IC 0.023, t-stat 1.85; |t| ≥ 2 is the usual bar.
- **Spread:** average 5-day returns rise only gently from the bottom prediction quintile (0.26%) to the top (0.36%).

**Before and after costs**

| | Gross CAGR | Cost drag | Net CAGR | Turnover per week |
|---|---|---|---|---|
| Long top 20% | 17.6% | 3.8% | 13.8% | about a third of the book |
| Long-short | 3.7% | 6.3% | −2.6% | both sides |

**Where the returns come from**

| | Alpha / year (t) | Market β | Size β | Value β | Momentum β | R² |
|---|---|---|---|---|---|---|
| Long top 20% | −2.8% (−1.00) | **1.03** | **+0.45** | **−0.24** | −0.05 | 0.86 |
| Long-short | −1.5% (−0.36) | **0.23** | **−0.31** | **−0.68** | **−0.17** | 0.51 |

*Bold: |t| ≥ 2.*

- **Long-only:** the market, plus a tilt toward smaller stocks and away from value stocks. These exposures explain 86% of its variation, and the remainder isn't significant.
- **Long-short:** above all a bet *against* value stocks (β −0.68, t −10.0). The model favours expensive, recently strong names, a costly tilt in stretches when value stocks recovered.

## Experiments: what else was tried

Each setup used the same model, validation and backtest. Production was chosen by a rule fixed before looking at any backtest: the highest out-of-sample rank-IC t-stat.

| Setup | Portfolio | IC t-stat | Long-only CAGR | S&P 500 CAGR | Long-short CAGR | Long-short alpha (t) |
|---|---|---|---|---|---|---|
| 5-day, up or down | top/bottom 20% | 1.22 | 11.1% | 17.6% | −8.8% | −4.3% (−0.92) |
| **5-day, vs. all members** | top/bottom 20% | **1.85** | 13.8% | 17.6% | −2.6% | −1.5% (−0.36) |
| 5-day, vs. own sector | sector-neutral | 0.86 | 9.0% | 17.6% | −11.5% | **−10.7% (−2.86)** |
| 21-day, vs. all members | top/bottom 20% | 1.27 | 21.2% | 18.5% | +7.4% | +5.9% (1.24) |
| 21-day, vs. own sector | sector-neutral | −0.03 | 20.9% | 18.5% | +1.2% | +0.7% (0.20) |

**1. Predict relative performance, not direction.**
- **Why:** "will it go up?" is mostly a question about the market, which moves every stock together.
- **Result:** asking "will it beat the median member?" raised the rank IC by 40% and cut the long-short loss from −8.8% to −2.6% a year.

**2. A monthly horizon looks better, but not convincingly.**
- **Result:** at 21 days, the long-only portfolio beat the S&P 500 (21.2% against 18.5% a year) and long-short turned positive (+7.4%).
- **But:**
  - that rests on only 76 monthly rebalances;
  - the IC t-stat is lower (1.27);
  - neither alpha is distinguishable from zero (long-short t = 1.24; long-only +1.4% a year, t = 0.45).
- **Production stays at 5 days:** switching after seeing this backtest would be exactly the cherry-picking the selection rule prevents.

**3. Comparing within sectors made it worse.**
- **What it does:** sector-neutral portfolios remove any bet on which sectors do well.
- **Result:** at 5 days, the long-short lost 10.7% a year of alpha, the only significant result in the table, and a negative one.
- **Why:** the little signal the model has seems to come from differences *across* sectors (size and value), which this construction strips out while still paying for the trading.

**4. At a weekly rebalance, costs decide the outcome.**
- **The arithmetic:** the long book replaces about a third of its holdings every week, which at 10 bps costs 3.8% a year. That is larger than any edge the ranking provides.

**5. The strongest single inputs are known effects, and weak here.**
- **Short-term reversal:** last week's return has an IC of −0.015; recent losers tend to bounce relative to recent winners.
- **Momentum:** 12-1 month momentum has an IC of +0.014.
- **Together:** neither reaches a t-stat of 1.1 on its own; the model combines them into something only slightly better.

## Limitations

- **Survivorship bias.** 157 former members have no usable prices, mostly acquired or delisted companies. Missing failures flatter every long strategy and the benchmarks.
- **Multiple testing.** Three models and five setups were compared. With that many tries, a t-stat near 1.9 is within what chance can produce.
- **Simplified costs.** A flat 10 bps per unit traded; no market impact or short-borrow fees, so real long-short costs would be higher.
- **One period.** 2020–2026 included a crash, a rate-hiking cycle and long stretches when the largest stocks led the market, which hurts a small-cap tilt. Another period could look different.
- **Free data.** Yahoo prices are unofficial and occasionally miss corporate actions; SEC XBRL tagging varies between companies.

## Lessons

- **Measurement matters more than the model.** The project's first version reported AUC 0.69 on random-walk data because of a random train/test split. Every honest number since has been close to 0.5.
- **No easy edge.** Public price and fundamental data, used in standard ways, doesn't produce a short-term edge in large US stocks after costs. That's consistent with how competitive this market is.
- **The process is the reusable part.** Point-in-time data, walk-forward validation, realistic execution, a pre-committed selection rule and factor attribution apply to any signal worth testing.

## Reproduce

```bash
python run_pipeline.py
```

The run writes `reports/research/research_report.md` with these tables, regenerated from the latest data, so numbers drift as new data arrives. The setups and the selection rule are defined in `src/sp500_agent/config.py` (`PRODUCTION_SPEC`, `EXPERIMENT_SPECS`).
