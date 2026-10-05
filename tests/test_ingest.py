"""load_live end to end against fake services: universe -> prices -> SEC -> FRED -> cleaning -> quality."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sp500_agent import ingest
from sp500_agent.ingest import clean_former_members, load_live
from test_sources import CHANGES_HTML, CONSTITUENTS_HTML, _company_facts

FRED_CSV = {
    "DGS3MO": "observation_date,DGS3MO\n2018-01-02,1.4\n2023-06-01,5.2\n",
    "DGS10": "observation_date,DGS10\n2018-01-02,2.4\n2023-06-01,3.7\n",
    "VIXCLS": "observation_date,VIXCLS\n2018-01-02,10.0\n2023-06-01,14.0\n",
}


class FakeWeb:
    """Answers Wikipedia and FRED requests."""

    def __init__(self):
        self.calls = 0

    def get_text(self, url, params=None):
        self.calls += 1
        if "Historical_components" in url:
            return CHANGES_HTML
        if "List_of_S" in url:
            return CONSTITUENTS_HTML
        return FRED_CSV[params["id"]]


class FakeSec:
    def __init__(self):
        self.calls = 0

    def get_json(self, url, params=None):
        self.calls += 1
        if "company_tickers" in url:
            return {"0": {"ticker": "OLD", "cik_str": 4}, "1": {"ticker": "AAA", "cik_str": 1}}
        if "submissions" in url:
            return {"sic": "2834" if "CIK0000000004" in url else "7372", "sicDescription": "x", "filings": {"recent": {
                "form": ["8-K"], "filingDate": ["2022-04-28"], "acceptanceDateTime": ["2022-04-28T16:05:00.000Z"], "items": ["2.02"],
            }}}
        return _company_facts()


def fake_downloader(calls):
    dates = pd.bdate_range("2018-01-01", "2023-06-30")

    def download(symbols, **kwargs):
        calls.append(symbols)
        columns = pd.MultiIndex.from_product([symbols, ["Close", "Adj Close", "Volume", "Stock Splits"]])
        frame = pd.DataFrame(np.nan, index=dates, columns=columns)
        rng = np.random.default_rng(0)
        for symbol in symbols:
            if symbol == "GONE":  # symbol reused by a newer listing: prices start long after it left the index
                window = dates >= "2021-01-01"
            elif symbol == "ANCIENT":
                continue  # no data at all
            else:
                window = np.ones(len(dates), dtype=bool)
            close = 100 * np.cumprod(1 + rng.normal(0, 0.01, window.sum()))
            frame.loc[window, (symbol, "Close")] = close
            frame.loc[window, (symbol, "Adj Close")] = close
            frame.loc[window, (symbol, "Volume")] = 1e6
            frame.loc[window, (symbol, "Stock Splits")] = 0.0
        return frame

    return download


@pytest.fixture
def live(tmp_path, monkeypatch):
    calls = []
    web, sec_client = FakeWeb(), FakeSec()
    kwargs = dict(start="2018-01-01", cache_dir=tmp_path, client=web, sec_client=sec_client, price_downloader=fake_downloader(calls), log=lambda msg: None)
    return SimpleNamespace(load=lambda **extra: load_live(**{**kwargs, **extra}), web=web, sec=sec_client, price_calls=calls)


def test_load_live_builds_a_clean_dataset(live):
    data = live.load()
    assert data.source == "live"
    # Current members plus former members since the start date; GONE's prices belong to a later listing.
    assert set(data.prices["ticker"]) == {"AAA", "BRK.B", "NEW", "OLD"}
    assert set(data.companies["ticker"]) == {"AAA", "BRK.B", "NEW", "OLD"}
    old = data.prices[data.prices["ticker"] == "OLD"]
    assert old["date"].max() <= pd.Timestamp("2022-06-01") + pd.Timedelta(days=ingest.POST_REMOVAL_DAYS)
    assert {"close", "raw_close", "volume"} <= set(data.prices.columns)
    assert set(data.fundamentals["ticker"]) == {"AAA", "BRK.B", "NEW", "OLD"}  # CIKs from Wikipedia, or SEC for OLD
    companies = data.companies.set_index("ticker")
    assert companies.loc["OLD", "sector"] == "Health Care"  # from its SIC code: no "Unknown" flag for former members
    assert companies.loc["AAA", "sector"] == "Tech" and companies.loc["AAA", "industry"] == "SIC 73"
    assert set(data.earnings_dates["date"]) == {pd.Timestamp("2022-04-29")}  # after the close: next day
    assert data.eps is not None and data.delistings is not None
    assert data.macro["term_spread"].notna().any()
    quality = data.quality
    assert quality["survivorship"]["former_members"] == 2  # OLD and GONE (ANCIENT left before the start)
    assert quality["survivorship"]["dropped_no_matching_history"] == 1
    assert quality["survivorship"]["dropped_examples"] == ["GONE"]
    assert quality["fundamentals"]["tickers_with_data"] == 4
    assert quality["membership"]["unresolved_changes"] == 1


def test_load_live_reuses_the_cache(live):
    live.load()
    web_calls, sec_calls, price_calls = live.web.calls, live.sec.calls, len(live.price_calls)
    live.load()
    assert (live.web.calls, live.sec.calls, len(live.price_calls)) == (web_calls, sec_calls, price_calls)
    live.load(refresh=True)
    assert len(live.price_calls) > price_calls


def test_limit_restricts_to_current_members(live):
    data = live.load(limit=2)
    assert set(data.prices["ticker"]) == {"AAA", "BRK.B"}
    assert "survivorship" not in data.quality


def test_missing_sec_contact_skips_fundamentals(live, monkeypatch, tmp_path):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    messages = []
    data = load_live(start="2018-01-01", cache_dir=tmp_path / "nosec", client=live.web, price_downloader=fake_downloader([]), log=messages.append)
    assert data.fundamentals is None and "fundamentals" not in data.quality
    assert any("SEC_USER_AGENT" in m for m in messages)


def test_clean_former_members_keeps_current_members_untouched():
    dates = pd.bdate_range("2020-01-01", periods=300)
    prices = pd.DataFrame({"ticker": np.repeat(["CUR", "EX"], len(dates)), "date": np.tile(dates, 2), "close": 1.0})
    membership = pd.DataFrame({"ticker": ["CUR", "EX"], "start": [pd.NaT, pd.NaT], "end": [pd.NaT, dates[100]]})
    cleaned, report = clean_former_members(prices, membership, {"CUR"}, "2020-01-01")
    assert (cleaned["ticker"] == "CUR").sum() == len(dates)
    assert cleaned.loc[cleaned["ticker"] == "EX", "date"].max() <= dates[100] + pd.Timedelta(days=ingest.POST_REMOVAL_DAYS)
    assert report["dropped_no_matching_history"] == 0 and report["rows_trimmed_after_removal"] > 0
