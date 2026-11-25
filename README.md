# Factor Models: Empirical Analysis

Code for the empirical chapter of my thesis. The script decomposes the return and risk
of a global equity portfolio with factor tilts, estimating five models by OLS with
Newey-West standard errors: CAPM, Fama-French three-factor (FF3), Carhart four-factor,
Fama-French five-factor (FF5) and a six-factor model (FF5 + momentum).

Sample: January 1999 - December 2025 (324 monthly observations).

## Portfolio

| Index | Bloomberg ticker | Weight |
|---|---|---|
| MSCI ACWI | NDUEACWF Index | 50% |
| MSCI World Sector Neutral Quality | M1WONQ Index | 20% |
| MSCI World Value | M1WO000V Index | 12% |
| MSCI World Small Cap Value Weighted | M1WOSVW Index | 10% |
| MSCI ACWI Small Cap | M1WDSC Index | 8% |

Net Total Return indices in USD, rebalanced to the target weights at the end of every month.

## Data

All files are in `Data/`.

| File | Source | Included |
|---|---|---|
| `[world]_[mkt]_[monthly]_[vw].csv` | JKP factors, world market excess return, value-weighted ([jkpfactors.com](https://jkpfactors.com)) | Yes |
| `[world]_[all_themes]_[monthly]_[vw].csv` | JKP theme portfolios, world, value-weighted ([jkpfactors.com](https://jkpfactors.com)) | Yes |
| `F-F_Research_Data_Factors.csv` | Kenneth R. French Data Library, 1-month T-bill rate ([link](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)) | Yes |
| `bloomberg_index_data_edit.csv` | Bloomberg, MSCI index levels | **No** |

The MSCI index data are licensed by Bloomberg and cannot be redistributed. To reproduce
the results, download the index levels from a Bloomberg terminal and save them (CSV or Excel format)
as `Data/bloomberg_index_data_edit.csv`, with:

- the date in the first column;
- one column of index levels per index, with the column names listed in `INDICES`
  at the top of `factor_regressions.py`.

### Transformations

- MSCI index levels: last observation of each month, converted to simple monthly returns.
- Portfolio return: weighted sum of the index returns with fixed weights.
- Risk-free rate: monthly RF from the French file, converted from percent to decimal.
- JKP factors: the market excess return and the size, value, momentum, profitability and
  investment themes (renamed SMB, HML, WML, RMW, CMA), used as they are.
- Excess returns: portfolio and index returns minus RF.
- All series are aligned by calendar month and only the months common to all of them
  are kept.

The JKP themes are analogues of the original Fama-French factors, not the same series.

## How to run

```
pip install -r requirements.txt
python factor_regressions.py
```

The script prints the regression table, the loadings by index, the factor correlations
and variance inflation factors, and the return attribution.

## References

- Jensen, T. I., Kelly, B., and Pedersen, L. H. (2023). Is There a Replication Crisis in
  Finance? *Journal of Finance*, 78(5), 2465-2518.
- Fama, E. F., and French, K. R. (1993, 2015). Kenneth R. French Data Library.
