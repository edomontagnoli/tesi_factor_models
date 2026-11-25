"""
Factor regressions for the empirical chapter of the thesis.

Portfolio: 50% MSCI ACWI, 12% MSCI World Value, 20% MSCI World Sector Neutral Quality,
10% MSCI World Small Cap Value Weighted and 8% MSCI ACWI Small Cap (net total return
indices in USD), rebalanced to the target weights at the end of every month.

Factors: JKP theme portfolios (world, value-weighted). Risk-free rate: 1-month T-bill
from the Kenneth French data library. Models: CAPM, FF3, Carhart, FF5 and FF6
(FF5 + momentum), estimated by OLS with Newey-West standard errors.

Data (see README.md):
    Data/[world]_[mkt]_[monthly]_[vw].csv         JKP, included
    Data/[world]_[all_themes]_[monthly]_[vw].csv  JKP, included
    Data/F-F_Research_Data_Factors.csv             Kenneth French, included
    Data/bloomberg_index_data_edit.csv             Bloomberg, NOT included

The Bloomberg index levels are licensed data and cannot be redistributed. To run the
script, download them from a Bloomberg terminal and save them in the file above.

Usage:
    pip install -r requirements.txt
    python factor_regressions.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# paths relative to this file, so that the script runs from any folder
DATA_DIR = Path(__file__).resolve().parent / "Data"
MKT_FILE = DATA_DIR / "[world]_[mkt]_[monthly]_[vw].csv"
THEMES_FILE = DATA_DIR / "[world]_[all_themes]_[monthly]_[vw].csv"
RF_FILE = DATA_DIR / "F-F_Research_Data_Factors.csv"
BLOOMBERG_FILE = DATA_DIR / "bloomberg_index_data_edit.csv"

# Bloomberg column name: (short name, portfolio weight)
# Bloomberg tickers: NDUEACWF, M1WO000V, M1WONQ, M1WOSVW, M1WDSC Index
INDICES = {
    "MSCI ACWI NET TOTAL RETURN INDEX": ("ACWI", 0.50),
    "MSCI WOLRD VALUE NET TOTAL RETURN": ("World Value", 0.12),  # name as in the export
    "MSCI WORLD SECTOR NEUTRAL QUALITY": ("World Quality", 0.20),
    "MSCI WORLD SMALL CAP VALUE WEIGHT NET TOTAL RETURN USD INDEX": ("World SC Value", 0.10),
    "MSCI ACWI SMALL CAP NET TOTAL": ("ACWI Small Cap", 0.08),
}
WEIGHTS = pd.Series(dict(INDICES.values()))

# JKP themes
THEMES = {"size": "SMB", "value": "HML", "momentum": "WML",
          "profitability": "RMW", "investment": "CMA"}

MODELS = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "Carhart": ["Mkt-RF", "SMB", "HML", "WML"],
    "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    "FF6": ["Mkt-RF", "SMB", "HML", "WML", "RMW", "CMA"],
}
FACTORS = MODELS["FF6"]


# --------------------------------------------------------------------------- data

def read_jkp(path):
    # JKP files in long format, one row per factor and month
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.to_period("M")
    return df.pivot(index="date", columns="name", values="ret")


def load_factors():
    mkt = read_jkp(MKT_FILE).squeeze("columns").rename("Mkt-RF")  # already an excess return
    themes = read_jkp(THEMES_FILE)[list(THEMES)].rename(columns=THEMES)
    return pd.concat([mkt, themes], axis=1, join="inner")


def load_rf():
    # Fama-French file: a few lines of text, the monthly table (in %), then the annual one
    with open(RF_FILE, encoding="utf-8", errors="replace") as f:
        start = next(i for i, line in enumerate(f) if "Mkt-RF" in line)
    ff = pd.read_csv(RF_FILE, skiprows=start, index_col=0, dtype=str)
    ff.columns = ff.columns.str.strip()
    ff.index = ff.index.str.strip()
    ff = ff[ff.index.str.fullmatch(r"\d{6}", na=False)]  # keep the yyyymm rows only
    rf = pd.to_numeric(ff["RF"]) / 100
    rf.index = pd.to_datetime(ff.index, format="%Y%m").to_period("M")
    return rf.rename("RF")


def check_data_files():
    missing = [p for p in (MKT_FILE, THEMES_FILE, RF_FILE) if not p.exists()]
    if missing:
        sys.exit("Missing data files:\n  " + "\n  ".join(map(str, missing)))
    if not BLOOMBERG_FILE.exists():
        sys.exit(
            f"Bloomberg data not found: {BLOOMBERG_FILE}\n"
            "The MSCI index levels are licensed Bloomberg data and are not included in\n"
            "this repository. Download the Net Total Return USD levels of NDUEACWF,\n"
            "M1WO000V, M1WONQ, M1WOSVW and M1WDSC from a Bloomberg terminal and save them\n"
            "as a CSV or Excel file with the date in the first column and one column per index,\n"
            "named as in INDICES at the top of this script (see README.md).")


def load_index_returns():
    # Bloomberg export: dates in the first column, one column of index levels per index.
    # The export can be an Excel workbook even if the extension is .csv, so Excel is tried first.
    try:
        raw = pd.read_excel(BLOOMBERG_FILE, index_col=0)
    except ValueError:  # not an Excel file: read it as text
        raw = pd.read_csv(BLOOMBERG_FILE, index_col=0, sep=None, engine="python")
    raw.columns = raw.columns.str.strip()

    # rows whose first cell is not a date are dropped
    raw.index = pd.to_datetime(raw.index, format="mixed", errors="coerce")
    prices = raw.loc[raw.index.notna(), list(INDICES)].apply(pd.to_numeric, errors="coerce")

    monthly = prices.groupby(prices.index.to_period("M")).last()  # month-end levels
    returns = monthly.pct_change(fill_method=None).dropna()  # common sample of the five indices
    returns.columns = WEIGHTS.index
    return returns


# --------------------------------------------------------------------- regressions

def newey_west_fit(y, X):
    lags = int(4 * (len(y) / 100) ** (2 / 9))  # Newey and West (1994) rule: 5 lags for T = 324
    return sm.OLS(y, sm.add_constant(X)).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True}, use_t=True)


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def regression_table(results):
    rows = ["Alpha (monthly, %)", "t", "p", "Alpha (annualized, %)"]
    for f in FACTORS:
        rows += [f, f"t {f}"]
    rows += ["Adj. R2", "N"]

    table = pd.DataFrame("", index=rows, columns=list(results))
    for name, res in results.items():
        alpha = res.params["const"] * 100
        table.loc["Alpha (monthly, %)", name] = f"{alpha:.4f}{stars(res.pvalues['const'])}"
        table.loc["t", name] = f"({res.tvalues['const']:.2f})"
        table.loc["p", name] = f"{res.pvalues['const']:.4f}"
        table.loc["Alpha (annualized, %)", name] = f"{alpha * 12:.2f}"
        for f in MODELS[name]:
            table.loc[f, name] = f"{res.params[f]:.4f}{stars(res.pvalues[f])}"
            table.loc[f"t {f}", name] = f"({res.tvalues[f]:.2f})"
        table.loc["Adj. R2", name] = f"{res.rsquared_adj:.4f}"
        table.loc["N", name] = str(int(res.nobs))
    return table


def loadings_by_index(index_excess, X, portfolio_fit):
    fits = {name: newey_west_fit(index_excess[name], X) for name in index_excess}
    loadings = pd.DataFrame({name: r.params for name, r in fits.items()}).T
    tstats = pd.DataFrame({name: r.tvalues for name, r in fits.items()}).T

    # with constant weights the portfolio loadings are the weighted sum of the index loadings
    contributions = loadings.mul(WEIGHTS, axis=0)
    gap = (contributions.sum() - portfolio_fit.params).abs().max()

    loadings.loc["Portfolio"] = portfolio_fit.params
    tstats.loc["Portfolio"] = portfolio_fit.tvalues
    contributions.loc["Portfolio (sum)"] = contributions.sum()
    return (loadings.drop(columns="const"), tstats.drop(columns="const"),
            contributions.drop(columns="const"), gap)


def vif_table(data):
    vif = {}
    for name in ("FF5", "FF6"):
        X = sm.add_constant(data[MODELS[name]])
        vif[name] = pd.Series([variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])],
                              index=MODELS[name])
    return pd.DataFrame(vif)


def attribution(data, res, factors):
    # average excess return = alpha + sum(beta x average factor premium) in % per year
    f = data[factors]
    beta = res.params[factors]
    premium = f.mean() * 1200
    t_mean = f.mean() / f.std() * np.sqrt(len(f))  # assumes independent observations
    contribution = beta * premium
    alpha = res.params["const"] * 1200
    total = data["Excess"].mean() * 1200
    assert abs(alpha + contribution.sum() - total) < 1e-8

    table = pd.DataFrame({"Loading": beta, "Mean premium": premium,
                          "t (mean)": t_mean, "Contribution": contribution})
    table.loc["Non-market factors (sum)", "Contribution"] = contribution.drop("Mkt-RF").sum()
    table.loc["Alpha", "Contribution"] = alpha
    table.loc["Total excess return", "Contribution"] = total
    return table


# ---------------------------------------------------------------------------- main

def main():
    pd.set_option("display.width", 200)
    check_data_files()

    factors = load_factors().join(load_rf(), how="inner")
    index_returns = load_index_returns()
    portfolio = (index_returns @ WEIGHTS).rename("Portfolio")

    data = factors.join(portfolio, how="inner").dropna()
    data["Excess"] = data["Portfolio"] - data["RF"]
    index_excess = index_returns.loc[data.index].sub(data["RF"], axis=0)

    print(f"Factors:    {factors.index[0]} to {factors.index[-1]} ({len(factors)} months)")
    print(f"Indices:    {index_returns.index[0]} to {index_returns.index[-1]} ({len(index_returns)} months)")
    print(f"Regression: {data.index[0]} to {data.index[-1]} ({len(data)} months)")

    results = {name: newey_west_fit(data["Excess"], data[cols]) for name, cols in MODELS.items()}
    print("\nRegression results (Newey-West t-statistics in parentheses)")
    print(regression_table(results).to_string())

    for name in ("FF6", "FF5"):
        loadings, tstats, contributions, gap = loadings_by_index(
            index_excess, data[MODELS[name]], results[name])
        print(f"\nLoadings by index, {name} (max gap from weighted sum: {gap:.1e})")
        print(loadings.round(4).to_string())
        print(f"\nNewey-West t-statistics by index, {name}")
        print(tstats.round(2).to_string())
        print(f"\nWeight x loading, {name}")
        print(contributions.round(4).to_string())

    print("\nCorrelation between factors")
    print(data[FACTORS].corr().round(2).to_string())
    print("\nVariance inflation factors")
    print(vif_table(data).loc[FACTORS].round(2).to_string(na_rep=""))

    for name in ("FF5", "FF6"):
        print(f"\nReturn attribution, {name} (% per year)")
        print(attribution(data, results[name], MODELS[name]).round(2).to_string(na_rep=""))


if __name__ == "__main__":
    main()
