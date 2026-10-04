# What 12 years of S&P 500 data says about short-term stock prediction

*Live run of 2 October 2026 · data from 2 January 2014 to 1 October 2026 · out-of-sample period May 2020 to September 2026*

## The question

Can a model built only from free public data (prices, SEC filings and macro series) rank S&P 500 stocks well enough to beat simply holding the index, after trading costs?

**Short answer: no.** The best ranking has a small, statistically weak edge. Before costs, its long portfolio matched the S&P 500 (17.6% a year each); weekly trading costs then put it 3.9 points a year behind. Once market, size, value and momentum exposures are accounted for, no strategy has alpha distinguishable from zero. This document explains how that was measured, what was tried, and why each attempt fell short.

## How it was tested

- **Universe.** Only stocks that were S&P 500 members on each date. Membership is reconstructed from Wikipedia's change log: 758 stocks over the period. Prices exist for 98 of the 255 that left the index. The rest were delisted, so survivorship bias is reduced, not removed.
- **Point-in-time inputs.** Price and volume signals, plus SEC fundamentals as first reported, each usable only from the day after filing. Also lagged macro data (VIX, Treasury yields) and cross-sectional ranks among the day's members. 22 model inputs in total.
- **Walk-forward validation.** Five expanding-window folds. Each trains only on dates before its test block, with a gap as long as the prediction horizon, so no training label overlaps a test label. Every number below is out of sample.
- **Backtest.** Rebalance every horizon. Enter one session after the signal and buy the top 20% of the ranking (long-only); long-short also shorts the bottom 20%. Costs are 10 bps per unit of weight traded. Benchmarks are SPY with dividends and an equal-weight portfolio of all members. Sharpe ratios are in excess of 3-month T-bills, except long-short, which is self-financing.
- **Attribution.** Each strategy's returns regressed on four factors built from the same members and dates: market, size (small minus big), value (cheap minus expensive by book-to-market) and momentum (12-1 month winners minus losers).

## Result for the production setup

Production setup: predict whether a stock beats the median member over the next 5 trading days. Model: logistic regression, chosen over a random forest and gradient boosting by walk-forward AUC.

| | Total return | CAGR | Sharpe | Max drawdown |
|---|---|---|---|---|
| Long top 20% | +125% | 13.8% | 0.62 | −26.9% |
| Long-short | −15% | −2.6% | −0.10 | −25.5% |
| Equal-weight members | +141% | 15.0% | 0.78 | −20.6% |
| **S&P 500 (SPY)** | **+178%** | **17.6%** | **0.92** | −23.2% |

- **Ranking quality:** AUC 0.511 (0.5 = no skill); rank IC 0.023 with a t-stat of 1.85 (|t| ≥ 2 is the usual bar). Returns rise only gently from the bottom to the top prediction quintile, from 0.26% to 0.36% per 5 days.

## What the returns really are

| | Alpha / year (t) | Market β | Size β | Value β | Momentum β | R² |
|---|---|---|---|---|---|---|
| Long top 20% | −2.8% (−1.00) | **1.03** | **+0.45** | **−0.24** | −0.05 | 0.86 |
| Long-short | −1.5% (−0.36) | **0.23** | **−0.31** | **−0.68** | **−0.17** | 0.51 |

*Bold: |t| ≥ 2.*

- **Long-only portfolio:** the market plus a tilt toward smaller stocks and away from value. These exposures explain 86% of its variation; what's left is slightly negative and not significant.
- **Long-short portfolio:** mostly a bet against value stocks (β −0.68, t −9.96). The model's ranking leans toward expensive, recently strong names. That partly explains why it lagged in years when value recovered.

## What was tried

The same model under each research setup, walked forward and backtested identically. The production setup was picked by a rule fixed before looking at backtests: the highest out-of-sample rank-IC t-stat.

| Setup | Portfolio | IC t-stat | Long-only CAGR | S&P 500 CAGR | Long-short CAGR | Long-short alpha (t) |
|---|---|---|---|---|---|---|
| 5-day, up or down | top/bottom 20% | 1.22 | 11.1% | 17.6% | −8.8% | −4.3% (−0.92) |
| **5-day, vs. all members** | top/bottom 20% | **1.85** | 13.8% | 17.6% | −2.6% | −1.5% (−0.36) |
| 5-day, vs. own sector | sector-neutral | 0.86 | 9.0% | 17.6% | −11.5% | **−10.7% (−2.86)** |
| 21-day, vs. all members | top/bottom 20% | 1.27 | 21.2% | 18.5% | +7.4% | +5.9% (1.24) |
| 21-day, vs. own sector | sector-neutral | −0.03 | 20.9% | 18.5% | +1.2% | +0.7% (0.20) |

**1. Predicting relative performance beats predicting direction.** Asking "will it go up?" mostly asks the model to forecast the market, which moves every stock together. Asking "will it beat the median member?" raised the rank IC by 40% and cut the long-short loss from −8.8% to −2.6% a year.

**2. A monthly horizon looks better, but not convincingly.** At 21 days the long-only portfolio beat the S&P 500 (21.2% against 18.5% a year) and long-short turned positive (+7.4%). But that rests on 76 monthly rebalances. The IC t-stat is lower (1.27), and the long-short alpha (t = 1.24) and long-only alpha (+1.4% a year, t = 0.45) are not distinguishable from zero. Switching production to this setup after seeing it win the backtest would be exactly the kind of cherry-picking the selection rule exists to prevent.

**3. Comparing within sectors made things worse.** Sector-neutral portfolios remove the bet on which sectors do well. The 5-day sector-neutral long-short lost 10.7% a year of alpha, which is significant (t = −2.86), and its ranking quality was the weakest. Whatever little signal the model has seems to come from cross-sector differences such as size and value, which the sector-neutral version strips out, while it still pays the trading costs.

**4. Costs explain the whole shortfall.** Before costs, the 5-day long-only portfolio earned 17.6% a year, the same as the S&P 500, and long-short earned +3.7% a year. But the long-only book replaces about a third of its holdings every week, and at 10 bps per unit traded that costs 3.8% a year (6.3% for long-short, which trades both sides). After costs, long-only trails the index by 3.9 points and long-short loses money. A weak signal at a weekly rebalance can't pay for its own trading.

**5. The strongest single inputs are known effects, and both are weak here.** Last week's return has a negative IC (−0.015): recent losers tend to bounce relative to recent winners (short-term reversal). 12-1 month momentum has a positive IC (+0.014). Neither has a t-stat above 1.1 on its own; the model combines them into something only slightly better (0.023, t = 1.85).

## Caveats

- **Survivorship bias.** 157 former members have no usable prices, mostly companies acquired or delisted. Missing failures flatter every long strategy and the benchmarks alike.
- **Multiple testing.** Three models and five setups were compared. With that many tries, a t-stat around 1.9 is what chance alone can produce.
- **Simplified costs.** A flat 10 bps per unit traded, with no market impact or short-borrow fees. Real long-short costs would be higher.
- **One period.** The out-of-sample window (2020–2026) includes a crash, a rate-hiking cycle and long stretches when the largest stocks led the market, which hurts a portfolio tilted toward smaller stocks. Other periods could look different.

## Takeaways

- A careful pipeline matters more than a clever model. The project's first version reported AUC 0.69 on random-walk data because of a random train/test split. Every honest number since has been close to 0.5.
- Public price and fundamental data, used in standard ways, does not produce a short-term edge in large US stocks after costs. That is consistent with the efficient-markets literature and with how competitive this corner of the market is.
- The useful output here is the measurement, not the ranking: walk-forward validation, point-in-time data, realistic execution, and attribution against known factors.

## Reproduce

```bash
python run_pipeline.py          # live data; writes reports/research/research_report.md with these tables
```

Results change as new data arrives. The selection rule and the setups compared are fixed in `src/sp500_agent/config.py` (`PRODUCTION_SPEC`, `EXPERIMENT_SPECS`).
