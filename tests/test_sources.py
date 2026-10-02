"""Connector tests against canned responses: no network."""

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sp500_agent.sources import fred, news, sec, wikipedia, yahoo
from sp500_agent.sources.http import HttpClient

CONSTITUENTS_HTML = """
<table><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th><th>Headquarters Location</th><th>Date added</th><th>CIK</th><th>Founded</th></tr>
<tr><td>AAA</td><td>Alpha</td><td>Tech</td><td>Software</td><td>X</td><td>2000-01-01</td><td>0000000001</td><td>1990</td></tr>
<tr><td>BRK.B</td><td>Berkshire</td><td>Financials</td><td>Insurance</td><td>X</td><td>2010-02-16</td><td>0001067983</td><td>1839</td></tr>
<tr><td>NEW</td><td>Newco</td><td>Tech</td><td>Chips</td><td>X</td><td>2022-06-01</td><td>0000000003</td><td>2001</td></tr>
</table>"""

CHANGES_HTML = """
<table>
<tr><th rowspan="2">Effective Date</th><th colspan="2">Added</th><th colspan="2">Removed</th><th rowspan="2">Reason</th><th rowspan="2">Refs</th></tr>
<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>
<tr><td>June 1, 2022</td><td>NEW</td><td>Newco</td><td>OLD</td><td>Oldco</td><td>Acquired</td><td>[1]</td></tr>
<tr><td>March 3, 2019</td><td></td><td></td><td>GONE</td><td>Goneco</td><td>Bankrupt</td><td>[2]</td></tr>
<tr><td>May 5, 2018</td><td>RENAMED</td><td>Renamed Inc</td><td></td><td></td><td>Spin-off</td><td>[3]</td></tr>
<tr><td>January 2, 2005</td><td>OLD</td><td>Oldco</td><td>ANCIENT</td><td>Ancientco</td><td>Too early</td><td>[4]</td></tr>
</table>"""


# -- Wikipedia --------------------------------------------------------------------------------------
def test_parse_constituents_and_changes():
    constituents = wikipedia.parse_constituents(CONSTITUENTS_HTML)
    assert constituents["ticker"].tolist() == ["AAA", "BRK.B", "NEW"]
    assert constituents.loc[1, "cik"] == 1067983 and constituents.loc[0, "sector"] == "Tech"
    changes = wikipedia.parse_changes(CHANGES_HTML)
    assert changes["date"].is_monotonic_decreasing
    assert changes.loc[0, "added"] == "NEW" and changes.loc[0, "removed"] == "OLD"
    assert pd.isna(changes.loc[1, "added"]) and changes.loc[1, "removed"] == "GONE"


def test_membership_walks_changes_backwards():
    constituents = wikipedia.parse_constituents(CONSTITUENTS_HTML)
    changes = wikipedia.parse_changes(CHANGES_HTML)
    membership, report = wikipedia.build_membership(constituents, changes, "2015-01-01")
    by_ticker = {r.ticker: (r.start, r.end) for r in membership.itertuples()}
    assert by_ticker["NEW"][0] == pd.Timestamp("2022-06-01") and pd.isna(by_ticker["NEW"][1])
    assert pd.isna(by_ticker["OLD"][0]) and by_ticker["OLD"][1] == pd.Timestamp("2022-06-01")
    assert by_ticker["GONE"][1] == pd.Timestamp("2019-03-03")
    assert "ANCIENT" not in by_ticker  # change happened before the start date
    assert report["unresolved_changes"] == 1  # RENAMED was added but isn't in today's list


def test_membership_flags():
    membership = pd.DataFrame({"ticker": ["A", "A", "B"], "start": [pd.NaT, pd.Timestamp("2021-01-01"), pd.Timestamp("2020-06-01")], "end": [pd.Timestamp("2020-01-01"), pd.NaT, pd.NaT]})
    frame = pd.DataFrame({"ticker": ["A", "A", "A", "B", "B", "C"], "date": pd.to_datetime(["2019-12-31", "2020-01-01", "2021-01-01", "2020-05-31", "2020-06-01", "2021-01-01"])})
    assert wikipedia.membership_flags(frame, membership).tolist() == [True, False, True, False, True, False]


# -- Yahoo --------------------------------------------------------------------------------------------
def _yahoo_frame(symbols, dates):
    columns = pd.MultiIndex.from_product([symbols, ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]])
    data = pd.DataFrame(np.nan, index=pd.DatetimeIndex(dates, name="Date"), columns=columns)
    return data


def test_fetch_prices_tidies_and_undoes_splits():
    dates = pd.bdate_range("2020-08-26", periods=5)  # a 4:1 split on the 4th day
    raw = _yahoo_frame(["AAPL", "BRK-B", "DEAD"], dates)
    raw[("AAPL", "Close")] = [125.0, 125.0, 125.0, 129.0, 134.0]
    raw[("AAPL", "Adj Close")] = raw[("AAPL", "Close")] * 0.97
    raw[("AAPL", "Stock Splits")] = [0, 0, 0, 4.0, 0]
    raw[("AAPL", "Volume")] = 1e6
    raw[("BRK-B", "Close")] = 300.0
    raw[("BRK-B", "Adj Close")] = 300.0
    calls = []

    def downloader(symbols, **kwargs):
        calls.append((symbols, kwargs))
        return raw

    prices = yahoo.fetch_prices(["AAPL", "BRK.B", "DEAD"], start="2020-01-01", downloader=downloader)
    assert sorted(prices["ticker"].unique()) == ["AAPL", "BRK.B"]  # DEAD had no data
    assert calls[0][0] == ["AAPL", "BRK-B", "DEAD"] and calls[0][1]["auto_adjust"] is False
    aapl = prices[prices["ticker"] == "AAPL"].set_index("date")
    assert aapl["raw_close"].tolist() == [500.0, 500.0, 500.0, 129.0, 134.0]


def test_unfinished_session_is_dropped_before_the_close():
    prices = pd.DataFrame({"ticker": "A", "date": pd.to_datetime(["2026-10-01", "2026-10-02"]), "close": [1.0, 2.0]})
    morning = datetime(2026, 10, 2, 11, 0)
    evening = datetime(2026, 10, 2, 17, 0)
    assert yahoo.drop_unfinished_session(prices, morning)["date"].max() == pd.Timestamp("2026-10-01")
    assert len(yahoo.drop_unfinished_session(prices, evening)) == 2


# -- SEC ----------------------------------------------------------------------------------------------
def _fact(start, end, val, filed, form="10-Q", fp="Q1"):
    row = {"end": end, "val": val, "filed": filed, "form": form, "fp": fp, "accn": f"acc-{filed}-{end}"}
    if start:
        row["start"] = start
    return row


def _company_facts():
    quarters = [
        ("2022-01-01", "2022-03-31", 10, "2022-04-28"),
        ("2022-04-01", "2022-06-30", 11, "2022-07-28"),
        ("2022-07-01", "2022-09-30", 12, "2022-10-27"),
        # Q4 2022 is only available through the annual 10-K (annual 50 -> Q4 = 17)
        ("2023-01-01", "2023-03-31", 14, "2023-04-27"),
    ]
    revenues = [_fact(s, e, v, f) for s, e, v, f in quarters[:2]]
    # The company switches revenue concept mid-way (as many did after ASC 606).
    contract_revenue = [_fact(s, e, v, f) for s, e, v, f in quarters[2:]]
    contract_revenue.append(_fact("2022-01-01", "2022-12-31", 50, "2023-02-15", form="10-K", fp="FY"))
    # A later filing restates Q1 2022 as 99: the point-in-time value must stay 10.
    revenues.append(_fact("2022-01-01", "2022-03-31", 99, "2023-04-27"))
    net_income = [_fact(s, e, v / 10, f) for s, e, v, f in quarters] + [_fact("2022-01-01", "2022-12-31", 5.0, "2023-02-15", form="10-K", fp="FY")]
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": revenues}},
                "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": contract_revenue}},
                "NetIncomeLoss": {"units": {"USD": net_income}},
                "StockholdersEquity": {"units": {"USD": [_fact(None, "2022-12-31", 100, "2023-02-15", "10-K", "FY")]}},
                "Assets": {"units": {"USD": [_fact(None, "2022-12-31", 300, "2023-02-15", "10-K", "FY")]}},
            },
            "dei": {
                # Two share classes reported separately on the cover page.
                "EntityCommonStockSharesOutstanding": {"units": {"shares": [
                    {"end": "2023-01-31", "val": 6, "filed": "2023-02-15", "form": "10-K", "accn": "k1"},
                    {"end": "2023-01-31", "val": 4, "filed": "2023-02-15", "form": "10-K", "accn": "k1"},
                ]}},
            },
        }
    }


def test_ttm_derives_q4_and_keeps_first_reported_values():
    facts = _company_facts()
    frame = sec._facts_frame(facts, "us-gaap", sec.FLOW_CONCEPTS["revenue"])
    ttm = sec.ttm_from_facts(frame).set_index("end")
    # Q1-Q4 2022 = 10 + 11 + 12 + (50 - 33) = 50, public when the 10-K was filed.
    assert ttm.loc[pd.Timestamp("2022-12-31"), "ttm"] == 50
    assert ttm.loc[pd.Timestamp("2022-12-31"), "available"] == pd.Timestamp("2023-02-15")
    # Q2 2022 - Q1 2023 = 11 + 12 + 17 + 14 = 54 (the restated 99 for Q1 2022 never enters).
    assert ttm.loc[pd.Timestamp("2023-03-31"), "ttm"] == 54


def test_point_in_time_timeline():
    timeline = sec.point_in_time(_company_facts()).set_index("available_date")
    row = timeline.loc[pd.Timestamp("2023-02-15")]
    assert row["revenue_ttm"] == 50 and row["net_income_ttm"] == 5.0
    assert row["shares_outstanding"] == 10  # both share classes
    assert row["equity"] == 100 and row["liabilities"] == 200  # assets minus equity when Liabilities isn't tagged
    # Nothing is known before the first full year of quarters is public.
    assert timeline["revenue_ttm"].first_valid_index() == pd.Timestamp("2023-02-15")


class FakeClient:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def _lookup(self, url, params=None):
        self.calls.append((url, params))
        for key, value in self.routes.items():
            if key in url:
                if isinstance(value, Exception):
                    raise value
                return value
        raise KeyError(url)

    def get_json(self, url, params=None):
        return self._lookup(url, params)

    def get_text(self, url, params=None):
        return self._lookup(url, params)

    def get(self, url, params=None):
        return SimpleNamespace(content=self._lookup(url, params))


def test_fetch_fundamentals_reports_failures():
    client = FakeClient({"CIK0000000001": _company_facts(), "CIK0000000002": RuntimeError("404 Not Found")})
    table, failed = sec.fetch_fundamentals(client, {"AAA": 1, "BBB": 2}, workers=2)
    assert set(table["ticker"]) == {"AAA"}
    assert "BBB" in failed and "404" in failed["BBB"]


def test_sec_client_requires_contact_email(monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(RuntimeError, match="SEC_USER_AGENT"):
        sec.sec_client()
    assert sec.sec_client("Project me@example.com").session.headers["User-Agent"] == "Project me@example.com"


def test_filings_fetcher_builds_links_and_filters_forms():
    submissions = {"filings": {"recent": {
        "form": ["8-K", "10-Q", "4", "10-K"],
        "filingDate": ["2026-07-30", "2026-07-31", "2026-07-29", "2025-10-31"],
        "reportDate": ["2026-07-30", "2026-06-27", "", "2025-09-27"],
        "accessionNumber": ["0001-26-1", "0001-26-2", "0001-26-3", "0001-25-9"],
        "primaryDocument": ["a.htm", "b.htm", "c.xml", "d.htm"],
        "items": ["2.02,9.01", "", "", ""],
    }}}
    fetcher = sec.FilingsFetcher(FakeClient({"submissions/CIK0000320193": submissions}), {"AAPL": 320193})
    filings = fetcher.fetch("AAPL")
    assert filings["form"].tolist() == ["8-K", "10-Q", "10-K"]
    assert filings["url"].iloc[0] == "https://www.sec.gov/Archives/edgar/data/320193/0001261/a.htm"
    assert fetcher.fetch("AAPL", forms=("10-K",))["form"].tolist() == ["10-K"]
    assert len(fetcher.client.calls) == 1  # cached
    with pytest.raises(KeyError):
        fetcher.fetch("MSFT")


# -- FRED -----------------------------------------------------------------------------------------------
def test_fred_series_and_spread():
    csvs = {
        "DGS3MO": "observation_date,DGS3MO\n2024-01-02,5.40\n2024-01-03,.\n2024-01-04,5.38\n",
        "DGS10": "observation_date,DGS10\n2024-01-02,3.95\n2024-01-04,4.00\n",
        "VIXCLS": "observation_date,VIXCLS\n2024-01-02,13.2\n2024-01-03,14.0\n2024-01-04,14.1\n",
    }

    class FredClient:
        def get_text(self, url, params):
            return csvs[params["id"]]

    macro = fred.fetch_macro(FredClient(), start="2024-01-01").set_index("date")
    assert pd.isna(macro.loc["2024-01-03", "tbill_3m"])  # "." means missing
    assert macro.loc["2024-01-04", "term_spread"] == pytest.approx(4.00 - 5.38)


# -- News -----------------------------------------------------------------------------------------------
RSS = b"""<?xml version="1.0"?><rss><channel>
<item><title>Apple beats estimates</title><link>https://x/1</link><pubDate>Fri, 02 Oct 2026 12:00:00 +0000</pubDate><description>Strong quarter</description></item>
<item><title>  </title><link>https://x/2</link></item>
<item><title>Apple faces lawsuit</title><link>https://x/3</link><pubDate>Thu, 01 Oct 2026 09:00:00 +0000</pubDate></item>
</channel></rss>"""


def test_yahoo_rss_parsing():
    parsed = news.parse_yahoo_rss(RSS, "AAPL")
    assert parsed["title"].tolist() == ["Apple beats estimates", "Apple faces lawsuit"]
    assert parsed["date"].iloc[0] == pd.Timestamp("2026-10-02 12:00", tz="UTC")


def test_news_fetcher_prefers_finnhub_with_a_key_and_caches():
    finnhub_items = [{"headline": "Deal announced", "datetime": 1790000000, "source": "Reuters", "url": "u", "summary": ""}, {"headline": ""}]
    client = FakeClient({"finnhub.io": finnhub_items})
    fetcher = news.NewsFetcher(client=client, finnhub_key="test-key")
    assert fetcher.source == "Finnhub"
    first = fetcher.fetch("BRK.B")
    assert first["title"].tolist() == ["Deal announced"]
    assert client.calls[0][1]["symbol"] == "BRK-B" and client.calls[0][1]["token"] == "test-key"
    fetcher.fetch("BRK.B")
    assert len(client.calls) == 1

    rss_fetcher = news.NewsFetcher(client=FakeClient({"feeds.finance.yahoo.com": RSS}), finnhub_key="")
    assert rss_fetcher.source == "Yahoo Finance RSS" and len(rss_fetcher.fetch("AAPL")) == 2


# -- HTTP -----------------------------------------------------------------------------------------------
def test_http_client_retries_rate_limits(monkeypatch):
    monkeypatch.setattr("sp500_agent.sources.http.time.sleep", lambda seconds: None)
    responses = [SimpleNamespace(status_code=429, raise_for_status=lambda: None), SimpleNamespace(status_code=200, raise_for_status=lambda: None, text="ok")]

    class Session:
        headers = {}

        def get(self, url, params=None, timeout=None):
            return responses.pop(0)

    client = HttpClient(user_agent="ua me@x.com", max_per_second=0, session=Session())
    assert client.get_text("https://example.test") == "ok"
    assert Session.headers["User-Agent"] == "ua me@x.com"
