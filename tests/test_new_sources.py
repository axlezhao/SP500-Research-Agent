"""SEC submissions and extra XBRL fields, Kenneth French factors and insider trades, against canned data."""

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from sp500_agent.sources import french, insiders, sec
from test_sources import FakeClient, _fact


def test_year_to_date_cash_flows_become_quarters():
    # Cash flow from operations is reported year to date: 3, 6, 9 months, then the full year.
    facts = {"facts": {"us-gaap": {"NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
        _fact("2022-01-01", "2022-03-31", 10, "2022-04-28"),
        _fact("2022-01-01", "2022-06-30", 25, "2022-07-28"),
        _fact("2022-01-01", "2022-09-30", 45, "2022-10-27"),
        _fact("2022-01-01", "2022-12-31", 70, "2023-02-15", form="10-K"),
        _fact("2023-01-01", "2023-03-31", 12, "2023-04-27"),
    ]}}}}}
    quarters, _ = sec.quarterly_values(sec._facts_frame(facts, "us-gaap", sec.FLOW_CONCEPTS["operating_cash_flow"]))
    assert quarters["val"].tolist() == [10, 15, 20, 25, 12]
    assert quarters.loc[1, "filed"] == pd.Timestamp("2022-07-28")
    ttm = sec.ttm_from_facts(sec._facts_frame(facts, "us-gaap", sec.FLOW_CONCEPTS["operating_cash_flow"])).set_index("end")
    assert ttm.loc[pd.Timestamp("2023-03-31"), "ttm"] == 15 + 20 + 25 + 12


def test_quarterly_eps_and_new_fundamentals():
    eps_rows = [_fact(f"{y}-{m:02d}-01", end, v, filed) for y, m, end, v, filed in [
        (2022, 1, "2022-03-31", 1.0, "2022-04-28"), (2022, 4, "2022-06-30", 1.1, "2022-07-28"),
        (2022, 7, "2022-09-30", 1.2, "2022-10-27"), (2023, 1, "2023-03-31", 1.4, "2023-04-27"),
    ]] + [_fact("2022-01-01", "2022-12-31", 4.6, "2023-02-15", form="10-K")]
    facts = {"facts": {"us-gaap": {
        "EarningsPerShareDiluted": {"units": {"USD/shares": eps_rows}},
        "Revenues": {"units": {"USD": [_fact("2022-01-01", "2022-12-31", 100, "2023-02-15", form="10-K")]}},
        "CostOfRevenue": {"units": {"USD": [_fact("2022-01-01", "2022-12-31", 60, "2023-02-15", form="10-K")]}},
        "Assets": {"units": {"USD": [_fact(None, "2022-12-31", 500, "2023-02-15", "10-K")]}},
    }}}
    eps = sec.quarterly_eps(facts)
    assert eps["eps"].round(6).tolist() == [1.0, 1.1, 1.2, 1.3, 1.4]  # Q4 = 4.6 - 3.3
    timeline = sec.point_in_time(facts).set_index("available_date")
    row = timeline.loc[pd.Timestamp("2023-02-15")]
    assert row["gross_profit_ttm"] == 40 and row["assets"] == 500  # revenue minus cost when GrossProfit isn't tagged


def test_submissions_give_industry_and_after_close_release_dates():
    recent = {
        "form": ["8-K", "8-K", "10-Q", "4"],
        "filingDate": ["2024-05-02", "2024-03-01", "2024-05-03", "2024-05-02"],
        "acceptanceDateTime": ["2024-05-02T16:30:12.000Z", "2024-03-01T08:00:00.000Z", "2024-05-03T12:00:00.000Z", "2024-05-02T18:00:00.000Z"],
        "items": ["2.02,9.01", "5.02", "", ""],
    }
    older = {"form": ["8-K"], "filingDate": ["2023-02-02"], "acceptanceDateTime": ["2023-02-02T07:30:00.000Z"], "items": ["2.02"]}
    ancient = {"form": ["8-K"], "filingDate": ["2010-02-02"], "acceptanceDateTime": ["2010-02-02T07:30:00.000Z"], "items": ["2.02"]}
    client = FakeClient({
        "CIK0000000007.json": {"sic": "3571", "sicDescription": "Electronic Computers", "filings": {"recent": recent, "files": [
            {"name": "CIK0000000007-submissions-001.json", "filingFrom": "2020-01-01", "filingTo": "2023-12-31"},
            {"name": "CIK0000000007-submissions-002.json", "filingFrom": "2005-01-01", "filingTo": "2012-12-31"},
        ]}},
        "submissions-001": older,
        "submissions-002": ancient,
    })
    industries, releases, failed = sec.fetch_company_events(client, {"AAA": 7}, since="2014-01-01")
    assert not failed and industries.iloc[0]["sic"] == 3571
    assert sec.sic_sector(industries.iloc[0]["sic"]) == "Information Technology"
    # 4:30 pm release: first tradable session is the next day. The 7:30 am one counts the same day.
    assert releases["date"].tolist() == [pd.Timestamp("2023-02-02"), pd.Timestamp("2024-05-03")]
    assert not any("submissions-002" in url for url, _ in client.calls)  # pages older than `since` aren't fetched


@pytest.mark.parametrize("code,sector", [(2834, "Health Care"), (6798, "Real Estate"), (6022, "Financials"), (4911, "Utilities"), (1311, "Energy"), (3714, "Consumer Discretionary"), (2080, "Consumer Staples"), (3724, "Industrials"), (None, None)])
def test_sic_codes_map_to_sectors(code, sector):
    assert sec.sic_sector(code) == sector


FRENCH_CSV = """This file was created by using the 202608 CRSP database.
The Tbill return is the simple daily rate.

,Mkt-RF,SMB,HML,RMW,CMA,RF
20240102,   1.00,    0.10,   -0.20,   0.00,    0.00,    0.02
20240103,   2.00,    0.00,    0.00,   0.10,    0.10,    0.02
20240104,  -1.00,    0.00,    0.00,   0.00,    0.00,    0.02
20240105,   0.50,    0.00,    0.00,   0.00,    0.00,    0.02

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


def test_french_factors_parse_and_compound_over_holding_windows():
    factors = french.parse_daily_csv(FRENCH_CSV, french.FILES["F-F_Research_Data_5_Factors_2x3_daily"])
    assert factors.columns.tolist() == ["date", "MKT", "SMB", "HML", "RMW", "CMA", "RF"]
    assert factors.loc[0, "MKT"] == pytest.approx(0.01) and len(factors) == 4
    # Signal on Jan 2, enter one session later (Jan 3 close), hold two sessions: Jan 4 and Jan 5 returns.
    period = french.period_factor_returns(factors, [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04")], holding_days=2, lag=1)
    assert period.iloc[0]["MKT"] == pytest.approx(0.99 * 1.005 - 1)
    assert period.iloc[1].isna().all()  # window runs past the data


def _insider_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("SUBMISSION.tsv", "ACCESSION_NUMBER\tFILING_DATE\tISSUERCIK\tDOCUMENT_TYPE\nA1\t31-MAR-2023\t7\t4\nA2\t02-APR-2023\t7\t4\nA3\t03-APR-2023\t99\t4\n")
        archive.writestr("NONDERIV_TRANS.tsv", "ACCESSION_NUMBER\tTRANS_CODE\tTRANS_SHARES\tTRANS_PRICEPERSHARE\tTRANS_ACQUIRED_DISP_CD\nA1\tP\t100\t10\tA\nA2\tS\t50\t12\tD\nA2\tM\t10\t1\tA\nA3\tP\t5\t5\tA\n")
    return buffer.getvalue()


def test_insider_trades_keep_open_market_buys_and_sells_of_known_companies():
    trades = insiders.parse_dataset(_insider_zip(), {7: "AAA"})
    assert trades["code"].tolist() == ["P", "S"]  # option exercise (M) and the unknown issuer dropped
    assert trades["filed"].tolist() == [pd.Timestamp("2023-03-31"), pd.Timestamp("2023-04-02")]
    assert trades["value"].tolist() == [1000.0, 600.0]
    assert insiders.quarters_since("2024-02-10", today=pd.Timestamp("2024-11-20")) == [(2024, 1), (2024, 2), (2024, 3)]


def test_insider_purchases_become_a_trailing_feature():
    from sp500_agent.features import add_insider_features

    dates = pd.bdate_range("2023-03-27", periods=80)
    features = pd.DataFrame({"ticker": "AAA", "date": dates, "market_cap": 1e6})
    out = add_insider_features(features, insiders.parse_dataset(_insider_zip(), {7: "AAA"})).set_index("date")
    assert out.loc["2023-03-31", "insider_purchases_90d"] == 0  # filed that day: usable from the next session
    assert out.loc["2023-04-03", "insider_purchases_90d"] == 1
    assert out.loc["2023-04-04", "insider_net_value_90d"] == pytest.approx((1000 - 600) / 1e6)
    assert out["insider_purchases_90d"].iloc[-1] == 0  # rolled out of the window
