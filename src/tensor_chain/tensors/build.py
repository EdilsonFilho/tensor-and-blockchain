"""Construction of the tensors used by the models.

- 𝒳 flow      (from × to × time), sparse: CP, completion, CMTF
- 𝒴 features  (address × time × feature), dense: Tucker
- N network   (time × network metric): side matrix for CMTF

𝒳 and 𝒴 share the same address index, so their factors can be coupled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import sparse

OTHER = "__other__"
FREQS = ("10min", "1h", "6h", "1D")
FLOW_METRICS = ("count", "value", "fee")
FEATURES = (
    "n_sent",
    "n_received",
    "eth_sent",
    "eth_received",
    "fee_paid",
    "gas_price_gwei",
    "contract_call_ratio",
    "n_counterparties",
    "error_rate",
)
_RATIO_FEATURES = {"contract_call_ratio", "error_rate"}  # the others use a log1p scale
_AMOUNT_FEATURES = {"eth_sent", "eth_received", "fee_paid"}
# ETH amounts are converted to gwei before log1p: log1p(x) ≈ x for x < 1, and most
# transactions move less than 1 ETH. In gwei the log scale is useful and non-negative.
GWEI_PER_ETH = 1e9


@dataclass
class FlowTensor:
    data: sparse.COO
    addresses: list[str]
    times: pd.DatetimeIndex
    metric: str

    @property
    def density(self) -> float:
        return self.data.nnz / np.prod(self.data.shape)


@dataclass
class FeatureTensor:
    data: np.ndarray
    addresses: list[str]
    times: pd.DatetimeIndex
    features: list[str]


def top_addresses(df: pd.DataFrame, k: int) -> list[str]:
    """The k addresses that appear most often as sender or receiver."""
    counts = pd.concat([df["from"], df["to"].dropna()]).astype(str).value_counts()
    # Deterministic tie-break by address.
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [address for address, _ in ranked[:k]]


def time_index(df: pd.DataFrame, freq: str) -> pd.DatetimeIndex:
    bins = df["timestamp"].dt.floor(freq)
    return pd.date_range(bins.min(), bins.max(), freq=freq)


def _time_codes(df: pd.DataFrame, times: pd.DatetimeIndex, freq: str) -> np.ndarray:
    return ((df["timestamp"].dt.floor(freq) - times[0]) // pd.Timedelta(freq)).to_numpy("int64")


def build_flow_tensor(
    df: pd.DataFrame,
    addresses: list[str],
    freq: str = "1h",
    metric: str = "count",
    *,
    include_other: bool = True,
    log: bool = True,
) -> FlowTensor:
    if metric not in FLOW_METRICS:
        raise ValueError(f"invalid metric: {metric!r}")
    index = list(addresses) + ([OTHER] if include_other else [])
    times = time_index(df, freq)
    df = df[df["to"].notna()]  # contract creations have no receiver
    codes = {a: i for i, a in enumerate(addresses)}
    fallback = len(addresses) if include_other else -1
    src = df["from"].map(codes).fillna(fallback).to_numpy("int64")
    dst = df["to"].map(codes).fillna(fallback).to_numpy("int64")
    keep = (src >= 0) & (dst >= 0)
    if include_other:
        keep &= ~((src == fallback) & (dst == fallback))  # other→other is not informative

    weights = {
        "count": np.ones(len(df)),
        "value": df["value_eth"].to_numpy("float64") * GWEI_PER_ETH,
        "fee": df["fee_eth"].to_numpy("float64") * GWEI_PER_ETH,
    }[metric]
    cells = pd.DataFrame(
        {
            "i": src[keep],
            "j": dst[keep],
            "t": _time_codes(df, times, freq)[keep],
            "w": weights[keep],
        }
    )
    cells = cells.groupby(["i", "j", "t"], as_index=False)["w"].sum()
    cells = cells[cells["w"] > 0]
    values = np.log1p(cells["w"].to_numpy()) if log else cells["w"].to_numpy()
    data = sparse.COO(
        cells[["i", "j", "t"]].to_numpy().T, values, shape=(len(index), len(index), len(times))
    )
    return FlowTensor(data, index, times, metric)


def build_feature_tensor(df: pd.DataFrame, addresses: list[str], freq: str = "1h") -> FeatureTensor:
    times = time_index(df, freq)
    df = df.assign(t=_time_codes(df, times, freq))
    addr_set = set(addresses)

    sent = (
        df[df["from"].isin(addr_set)]
        .groupby(["from", "t"])
        .agg(
            n_sent=("hash", "size"),
            eth_sent=("value_eth", "sum"),
            fee_paid=("fee_eth", "sum"),
            gas_price_gwei=("gas_price_gwei", "mean"),
            contract_call_ratio=("is_contract_call", "mean"),
            error_rate=("is_error", "mean"),
        )
    )
    sent.index.names = ["address", "t"]
    received = (
        df[df["to"].isin(addr_set)]
        .groupby(["to", "t"])
        .agg(n_received=("hash", "size"), eth_received=("value_eth", "sum"))
    )
    received.index.names = ["address", "t"]
    pairs = pd.concat(
        [
            df[["from", "t", "to"]].set_axis(["address", "t", "cp"], axis=1),
            df[["to", "t", "from"]].set_axis(["address", "t", "cp"], axis=1),
        ]
    ).dropna()
    pairs = pairs[pairs["address"].isin(addr_set)].drop_duplicates()
    counterparties = pairs.groupby(["address", "t"]).size().rename("n_counterparties")

    features = [f for f in FEATURES if f != "error_rate" or df["is_error"].notna().any()]
    full = pd.MultiIndex.from_product([addresses, range(len(times))], names=["address", "t"])
    table = (
        pd.concat([sent, received, counterparties], axis=1)
        .reindex(full)[features]
        .astype("float64")
        .fillna(0.0)
    )
    for col in features:
        if col not in _RATIO_FEATURES:
            scale = GWEI_PER_ETH if col in _AMOUNT_FEATURES else 1.0
            table[col] = np.log1p(table[col] * scale)
    data = table.to_numpy().reshape(len(addresses), len(times), len(features))
    return FeatureTensor(data, list(addresses), times, features)


def build_network_matrix(
    txs: pd.DataFrame, blocks: pd.DataFrame, times: pd.DatetimeIndex, freq: str
) -> pd.DataFrame:
    """Aggregated network metrics per time window."""
    tx = (
        txs.assign(t=txs["timestamp"].dt.floor(freq))
        .groupby("t")
        .agg(tx_count=("hash", "size"), median_gas_price_gwei=("gas_price_gwei", "median"))
    )
    if len(blocks):
        bl = (
            blocks.assign(t=blocks["timestamp"].dt.floor(freq))
            .groupby("t")
            .agg(
                base_fee_gwei=("base_fee_gwei", "mean"),
                gas_used=("gas_used", "sum"),
                gas_limit=("gas_limit", "sum"),
            )
        )
        bl["gas_utilization"] = bl.pop("gas_used") / bl.pop("gas_limit")
        tx = tx.join(bl, how="outer")
    return tx.reindex(times)
