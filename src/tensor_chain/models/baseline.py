"""Baseline: robust Mahalanobis distance per transaction.

Fixes the original notebook version:
- uses transaction features only (no timeStamp/blockNumber, which are collinear);
- uses gas_used when available (the "gas" field is the limit, not consumption);
- estimates mean/covariance with MCD (robust to the anomalies themselves);
- thresholds by the χ² quantile with k degrees of freedom, instead of "3 × mean";
- fits one model per segment (transaction kind × no ETH / dust / with ETH). Plain
  transfers (gas = 21000) and contract calls are distinct populations, and ~75% of
  mainnet calls carry no value, which would leave log(value) with an isolated spike. A
  single Gaussian would flag almost every call that sends ETH.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.covariance import MinCovDet

from tensor_chain.data.schema import WEI_PER_ETH

# Regularization (in standardized units): heavily repeated discrete values, such as
# gas = 21000, make the covariance of the MCD subset singular.
_RIDGE = 1e-2
# MCD's default (~50% of the points) locks onto the largest cluster: on mainnet the gas price
# is bimodal (base fee only vs. tip) and bots repeat identical gas/price. 0.9 assumes up to
# 10% contamination; on real data it cut the rate from 36% to ~1.4% for calls without ETH.
SUPPORT_FRACTION = 0.9
MIN_GROUP_SIZE = 50
KINDS = ("transfer", "call", "creation")
DUST_ETH = 1e-9  # 1 gwei


def transaction_kind(df: pd.DataFrame) -> pd.Series:
    """Transaction segment, e.g. "call · no ETH".

    "dust" (< 1 gwei) is split from "with ETH": payments of a few wei (common among bots)
    form their own population, ~12% of mainnet calls that carry value.
    """
    kind = np.select(
        [df["is_contract_creation"].to_numpy(), df["is_contract_call"].to_numpy()],
        [KINDS[2], KINDS[1]],
        KINDS[0],
    )
    value_eth = df["value_eth"].to_numpy()
    value = np.select([value_eth == 0, value_eth < DUST_ETH], ["no ETH", "dust"], "with ETH")
    return pd.Series(np.char.add(np.char.add(kind, " · "), value), index=df.index, name="kind")


def transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    gas = df["gas_used"].astype("float64").fillna(df["gas_limit"].astype("float64"))
    return pd.DataFrame(
        {
            # In wei: values range from 1 wei ("dust", common among bots) to thousands of
            # ETH, and log1p over ETH would not compress anything below 1 ETH.
            "log_value_wei": np.log10(1 + df["value_eth"].to_numpy() * WEI_PER_ETH),
            "log_gas": np.log(gas.to_numpy()),
            "log_gas_price_gwei": np.log(df["gas_price_gwei"].to_numpy() + 1e-3),
        },
        index=df.index,
    )


def _squared_distances(
    feats: pd.DataFrame, max_fit_samples: int, random_state: int
) -> tuple[np.ndarray, int]:
    # max > min rather than std > 0: the std of a constant column can come out ~1e-15.
    feats = feats.loc[:, feats.max() > feats.min()]
    if feats.shape[1] == 0:
        return np.zeros(len(feats)), 0
    x = feats.to_numpy()
    x = (x - x.mean(axis=0)) / x.std(axis=0)

    rng = np.random.default_rng(random_state)
    fit_rows = rng.choice(len(x), size=min(len(x), max_fit_samples), replace=False)
    mcd = MinCovDet(support_fraction=SUPPORT_FRACTION, random_state=random_state).fit(x[fit_rows])
    precision = np.linalg.inv(mcd.covariance_ + _RIDGE * np.eye(x.shape[1]))
    centered = x - mcd.location_
    return np.einsum("ij,jk,ik->i", centered, precision, centered), x.shape[1]


def mahalanobis_anomalies(
    df: pd.DataFrame,
    alpha: float = 1e-3,
    *,
    max_fit_samples: int = 50_000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Returns df with kind, score (distance), p_value and is_anomaly (p_value < alpha).

    Segments with fewer than MIN_GROUP_SIZE transactions get no score (NaN).
    """
    kinds = transaction_kind(df)
    feats = transaction_features(df)
    score = pd.Series(np.nan, index=df.index)
    p_value = pd.Series(np.nan, index=df.index)
    thresholds: dict[str, float] = {}
    for kind, rows in kinds.groupby(kinds).groups.items():
        if len(rows) < MIN_GROUP_SIZE:
            continue
        seg_feats = feats.loc[rows]
        if kind.startswith(KINDS[0]) and df.loc[rows, "gas_used"].isna().any():
            # Without gas_used (blocks mode), a transfer's gas is the limit chosen by the
            # wallet (21000, 31500, 500k...), not consumption.
            seg_feats = seg_feats.drop(columns="log_gas")
        d2, k = _squared_distances(seg_feats, max_fit_samples, random_state)
        if k == 0:
            continue
        score.loc[rows] = np.sqrt(d2)
        p_value.loc[rows] = chi2.sf(d2, df=k)
        thresholds[kind] = float(np.sqrt(chi2.isf(alpha, df=k)))

    scored = df.assign(
        kind=kinds, score=score, p_value=p_value, is_anomaly=(p_value < alpha).to_numpy()
    )
    scored.attrs["thresholds"] = thresholds
    return scored
