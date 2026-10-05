"""No look-ahead: features are the same whether or not the future exists, and the bias fixes hold."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_live_world
from sp500_agent.features import (
    add_earnings_features,
    build_research_features,
    forward_returns,
    on_session_calendar,
    standardized_surprises,
)
from sp500_agent.ingest import _companies, delistings

NO_NEWS = pd.DataFrame(columns=["ticker", "date", "title"])
LABEL_PREFIXES = ("future_return", "tradable_return", "target_")


def _eps_and_releases(world):
    rng = np.random.default_rng(5)
    rows, releases = [], []
    for ticker in world.companies["ticker"]:
        for q, end in enumerate(pd.date_range("2018-03-31", periods=14, freq="QE")):
            rows.append({"ticker": ticker, "period_end": end, "eps": 1 + 0.05 * q + rng.normal(0, 0.1), "filed": end + pd.Timedelta(days=35)})
            releases.append({"ticker": ticker, "date": end + pd.Timedelta(days=21), "accepted": end + pd.Timedelta(days=21, hours=16, minutes=5)})
    return pd.DataFrame(rows), pd.DataFrame(releases)


def _build(world, end=None, **extra):
    prices = world.prices if end is None else world.prices[world.prices["date"] <= end]
    macro = world.macro if end is None else world.macro[world.macro["date"] <= end]
    return build_research_features(
        prices, world.companies, NO_NEWS, pit_fundamentals=world.fundamentals, macro=macro, membership=world.membership, **extra,
    )


def test_features_do_not_change_when_the_future_is_removed():
    world = make_live_world()
    eps, releases = _eps_and_releases(world)
    full = _build(world, eps=eps, earnings_dates=releases)
    cut = full["date"].sort_values().unique()[300]
    truncated = _build(world, end=cut, eps=eps, earnings_dates=releases)
    columns = [c for c in truncated.columns if not c.startswith(LABEL_PREFIXES) and c in full.columns and pd.api.types.is_numeric_dtype(truncated[c])]
    assert {"beta_252_z", "residual_momentum", "high_52w", "return_5d_vs_industry", "sue", "ear", "vix_regime", "earnings_yield_z"} <= set(columns)
    key = ["ticker", "date"]
    merged = truncated[key + columns].merge(full[key + columns], on=key, suffixes=("_cut", "_full"))
    assert len(merged) == len(truncated)
    for col in columns:
        a, b = merged[f"{col}_cut"].astype(float), merged[f"{col}_full"].astype(float)
        same = ((a - b).abs() < 1e-9) | (a.isna() & b.isna())
        assert same.all(), f"{col} uses information from after the row's date"


def test_missing_sessions_are_counted_as_sessions():
    dates = pd.bdate_range("2024-01-01", periods=30)
    full = pd.DataFrame({"ticker": "A", "date": dates, "close": 100 * 1.01 ** np.arange(30)})
    other = pd.DataFrame({"ticker": "B", "date": dates, "close": 50.0})
    gappy = full.drop(index=[22, 23])  # two sessions missing for A
    panel = on_session_calendar(pd.concat([gappy, other]))
    a = panel[panel["ticker"] == "A"].reset_index(drop=True)
    assert len(a) == 30 and a["_filled"].sum() == 2
    assert a.loc[22, "close"] == a.loc[21, "close"]  # carried forward
    features = build_research_features(pd.concat([gappy, other]), pd.DataFrame({"ticker": ["A", "B"]}), NO_NEWS)
    a_rows = features[features["ticker"] == "A"].set_index("date")
    assert a_rows.loc[dates[25], "return_5d"] == pytest.approx(1.01**5 - 1)  # five sessions back, not five rows back
    assert not a_rows.index.isin(dates[[22, 23]]).any()  # the inserted sessions are dropped again


def test_stocks_that_stop_trading_keep_their_returns():
    dates = pd.bdate_range("2024-01-01", periods=40)
    alive = pd.DataFrame({"ticker": "LIVE", "date": dates, "close": 100.0})
    stale = pd.DataFrame({"ticker": "STALE", "date": dates[:-2], "close": 100.0})  # two days behind: not delisted
    acquired = pd.DataFrame({"ticker": "GONE", "date": dates[:20], "close": np.r_[np.full(19, 50.0), 60.0]})
    failed = pd.DataFrame({"ticker": "BUST", "date": dates[:20], "close": 10.0})
    prices = pd.concat([alive, stale, acquired, failed], ignore_index=True)
    prices["_filled"] = False
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    out = pd.concat([prices, forward_returns(prices, {"BUST": -0.3})], axis=1).set_index(["ticker", "date"])
    # GONE's last price is session 19; a signal on session 16 enters on 17 and would exit on 22: it runs to the last price.
    assert out.loc[("GONE", dates[16]), "tradable_return_5d"] == pytest.approx(60 / 50 - 1)
    assert out.loc[("GONE", dates[16]), "future_return_5d"] == pytest.approx(60 / 50 - 1)
    assert out.loc[("BUST", dates[16]), "tradable_return_5d"] == pytest.approx(-0.3)  # a failure: the delisting return applies
    assert np.isnan(out.loc[("GONE", dates[19]), "tradable_return_5d"])  # can't buy a stock with no next price
    assert np.isnan(out.loc[("LIVE", dates[37]), "future_return_5d"])  # the end of the data is unknown, not a delisting
    assert np.isnan(out.loc[("STALE", dates[36]), "future_return_5d"])  # a slightly stale feed isn't a delisting either


def test_delistings_separate_failures_mergers_and_index_removals():
    dates = pd.bdate_range("2023-01-01", periods=300)
    frames = [
        pd.DataFrame({"ticker": "LIVE", "date": dates, "close": 10.0}),
        pd.DataFrame({"ticker": "MERGED", "date": dates[:200], "close": 10.0}),
        pd.DataFrame({"ticker": "CRASH", "date": dates[:200], "close": np.r_[np.full(130, 10.0), np.linspace(10, 2, 70)]}),
        pd.DataFrame({"ticker": "DROPPED", "date": dates[:200], "close": 10.0}),  # trimmed after leaving the index
    ]
    membership = pd.DataFrame({"ticker": ["DROPPED"], "start": [pd.NaT], "end": [dates[200] - pd.Timedelta(days=60)]})
    changes = pd.DataFrame({"removed": ["MERGED"], "reason": ["Acquired by LIVE"]})
    table = delistings(pd.concat(frames, ignore_index=True), changes, membership).set_index("ticker")
    assert set(table.index) == {"MERGED", "CRASH"}
    assert table.loc["MERGED", "delisting_return"] == 0.0 and table.loc["CRASH", "delisting_return"] == -0.30


def test_former_members_get_a_real_sector_or_none_never_a_placeholder():
    constituents = pd.DataFrame({"ticker": ["AAA"], "company_name": ["Alpha"], "sector": ["Information Technology"], "sub_industry": ["Software"]})
    changes = pd.DataFrame({"removed": ["OLD", "GONE"], "removed_name": ["Oldco", "Goneco"]})
    ciks = pd.DataFrame({"ticker": ["AAA", "OLD"], "cik": [1, 2]})
    industries = pd.DataFrame({"ticker": ["AAA", "OLD"], "sic": [7372, 2834], "sic_description": ["Software", "Pharma"]})
    companies = _companies(constituents, changes, ciks, ["AAA", "OLD", "GONE"], industries).set_index("ticker")
    assert companies.loc["AAA", "sector"] == "Information Technology"  # GICS kept for current members
    assert companies.loc["OLD", "sector"] == "Health Care"  # from its SIC code
    assert pd.isna(companies.loc["GONE", "sector"])  # unknown stays missing: no "will leave the index" flag
    assert companies.loc["OLD", "industry"] == "SIC 28" and companies.loc["AAA", "industry"] == "SIC 73"
    assert "Unknown" not in set(companies["sector"].dropna())


def test_rank_targets_and_normalised_features(live_features):
    members = live_features[live_features["in_index"]]
    target = members.dropna(subset=["target_rank_market_5d"])
    assert target["target_rank_market_5d"].between(0, 1).all()
    assert np.allclose(target.groupby("date")["target_rank_market_5d"].mean(), 0.5)
    z = members.groupby("date")["momentum_60d_z"].mean().dropna()
    assert z.abs().max() < 1e-6  # centred every day
    assert live_features.loc[~live_features["in_index"], "momentum_60d_z"].isna().all()  # only members are ranked
    # Ranks of volatility-scaled returns: a stock's target doesn't depend on how volatile everything was that week.
    assert set(live_features.filter(like="target_rank").columns) == {"target_rank_market_5d", "target_rank_sector_5d", "target_rank_market_21d", "target_rank_sector_21d"}


def test_earnings_surprise_is_known_only_after_the_release():
    eps = pd.DataFrame({
        "ticker": "A",
        "period_end": pd.date_range("2021-03-31", periods=10, freq="QE"),
        "eps": [1.0, 1.1, 0.9, 1.2, 1.05, 1.2, 1.0, 1.3, 1.1, 2.5],
        "filed": pd.date_range("2021-03-31", periods=10, freq="QE") + pd.Timedelta(days=40),
    })
    releases = pd.DataFrame({"ticker": "A", "date": [pd.Timestamp("2023-07-20")], "accepted": [pd.Timestamp("2023-07-19 16:30")]})
    sue = standardized_surprises(eps, releases).set_index("available")
    last = sue.iloc[-1]
    assert sue.index[-1] == pd.Timestamp("2023-07-20") and last["sue"] > 2  # the big beat, dated by its press release
    assert sue.index[-2] == pd.Timestamp("2023-05-11")  # no release on file: the day after the 10-Q

    dates = pd.bdate_range("2023-07-10", periods=15)
    features = pd.DataFrame({"ticker": "A", "date": dates, "return_1d": 0.01, "market_return_1d": 0.0})
    features.loc[dates == pd.Timestamp("2023-07-20"), "return_1d"] = 0.08
    out = add_earnings_features(features, eps, releases).set_index("date")
    assert out.loc["2023-07-19", "sue"] == pytest.approx(sue["sue"].iloc[-2])  # the day before: last quarter's surprise
    assert out.loc["2023-07-20", "sue"] == pytest.approx(last["sue"])
    assert np.isnan(out.loc["2023-07-20", "ear"])  # the day after the release isn't over yet
    assert out.loc["2023-07-21", "ear"] == pytest.approx(0.01 + 0.08 + 0.01)
    assert out.loc["2023-07-28", "ear"] == pytest.approx(0.10)  # carried forward
