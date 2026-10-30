from __future__ import annotations

import numpy as np
import pytest

from tensor_chain.data.schema import to_frame
from tensor_chain.models.baseline import mahalanobis_anomalies
from tensor_chain.tensors.build import (
    OTHER,
    build_feature_tensor,
    build_flow_tensor,
    build_network_matrix,
    top_addresses,
)

from .conftest import A, B, C


def test_top_addresses_ranks_by_activity(txs):
    # A: 4 aparições; B: 3; C e D: 2 cada (desempate alfabético).
    assert top_addresses(txs, 3) == [A, B, C]


def test_flow_tensor_counts(txs):
    flow = build_flow_tensor(txs, [A, B], "1h", "count", log=False)
    assert flow.addresses == [A, B, OTHER]
    assert flow.data.shape == (3, 3, 2)
    dense = flow.data.todense()
    assert dense[0, 1, 0] == 2  # A→B duas vezes na primeira hora
    assert dense[0, 2, 0] == 1  # A→C cai em "outros"
    assert dense[2, 0, 1] == 1  # D→A na segunda hora
    assert dense.sum() == 4  # criação de contrato e outros→outros (C→D) ficam de fora


def test_flow_tensor_value_without_other(txs):
    flow = build_flow_tensor(txs, [A, B], "1h", "value", include_other=False)
    assert flow.data.shape == (2, 2, 2)
    assert flow.data.todense()[0, 1, 0] == pytest.approx(np.log1p(1.5e9))


def test_feature_tensor(txs):
    feats = build_feature_tensor(txs, [A, B], "1h")
    assert "error_rate" not in feats.features  # modo blocos não tem status
    assert feats.data.shape == (2, 2, len(feats.features))
    sent = feats.data[0, 0, feats.features.index("n_sent")]
    assert sent == pytest.approx(np.log1p(3))
    cps = feats.data[0, 0, feats.features.index("n_counterparties")]
    assert cps == pytest.approx(np.log1p(2))  # B e C


def test_network_matrix(txs, blocks):
    from tensor_chain.data.schema import blocks_frame, rows_from_block

    blk = blocks_frame([rows_from_block(b)[1] for b in blocks])
    flow = build_flow_tensor(txs, [A], "1h")
    net = build_network_matrix(txs, blk, flow.times, "1h")
    assert net["tx_count"].tolist() == [5, 1]
    assert net["gas_utilization"].tolist() == [0.5, 0.5]


def _tx_row(i: int, rng: np.random.Generator, *, call: bool) -> dict:
    return {
        "hash": f"0x{i}",
        "block_number": i,
        "timestamp": 1_700_000_000 + i,
        "from": A,
        "to": B,
        # Como na mainnet: ~75% das chamadas de contrato não enviam ETH.
        "value_eth": 0.0 if call and rng.random() < 0.75 else float(rng.lognormal(-2, 0.5)),
        "gas_limit": int(rng.lognormal(12, 0.5)) if call else 21_000,
        "gas_used": None,
        "gas_price_gwei": float(rng.lognormal(3, 0.2)),
        "max_priority_fee_gwei": None,
        "tx_type": 2,
        "is_contract_call": call,
        "is_contract_creation": False,
        "is_error": None,
        "method_id": "0xa9059cbb" if call else None,
    }


def test_baseline_fits_each_transaction_kind():
    # Transferências (gas sempre 21000) e chamadas de contrato (quase sempre sem valor)
    # misturadas: um modelo único marcaria como anômalas as chamadas que enviam ETH.
    rng = np.random.default_rng(1)
    rows = [_tx_row(i, rng, call=i % 2 == 0) for i in range(3_000)]
    rows.append({**rows[1], "hash": "0xbad", "value_eth": 5_000.0, "gas_price_gwei": 2_000.0})
    rows.append(
        {
            **rows[0],
            "hash": "0xcreate",
            "is_contract_call": False,
            "is_contract_creation": True,
            "to": None,
        }
    )
    df = to_frame(rows)

    scored = mahalanobis_anomalies(df, alpha=1e-4).set_index("hash")
    assert scored.loc["0xbad", "is_anomaly"]
    assert scored.groupby("kind")["is_anomaly"].mean().max() < 0.01
    assert set(scored.attrs["thresholds"]) == {
        "transferência · com ETH",
        "chamada · sem ETH",
        "chamada · com ETH",
    }
    # Tipo com poucas transações fica sem score.
    assert np.isnan(scored.loc["0xcreate", "score"])
    assert not scored.loc["0xcreate", "is_anomaly"]
