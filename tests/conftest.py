from __future__ import annotations

import pandas as pd
import pytest

from tensor_chain.data.schema import rows_from_block, to_frame

GWEI = 10**9
ETH = 10**18


def make_tx(
    i: int,
    frm: str,
    to: str | None,
    value_eth: float = 0.0,
    *,
    gas: int = 21_000,
    gas_price_gwei: int = 20,
    data: str = "0x",
) -> dict:
    return {
        "hash": f"0x{i:064x}",
        "from": frm,
        "to": to,
        "value": hex(int(value_eth * ETH)),
        "gas": hex(gas),
        "gasPrice": hex(gas_price_gwei * GWEI),
        "maxPriorityFeePerGas": hex(GWEI),
        "type": "0x2",
        "input": data,
    }


def make_block(number: int, timestamp: int, txs: list[dict]) -> dict:
    return {
        "number": hex(number),
        "timestamp": hex(timestamp),
        "baseFeePerGas": hex(19 * GWEI),
        "gasUsed": hex(15_000_000),
        "gasLimit": hex(30_000_000),
        "transactions": txs,
    }


A, B, C, D = (f"0x{c * 40}" for c in "abcd")


@pytest.fixture
def blocks() -> list[dict]:
    """Três blocos em duas horas distintas."""
    t0 = 1_700_000_000 - 1_700_000_000 % 3600
    return [
        make_block(
            100,
            t0 + 10,
            [
                make_tx(1, A, B, 1.0),
                make_tx(2, A, C, 2.0),
                make_tx(3, B, None, gas=500_000, data="0x6080"),
            ],
        ),
        make_block(
            101,
            t0 + 22,
            [make_tx(4, A, B, 0.5), make_tx(5, C, D, 0.0, gas=90_000, data="0xa9059cbb0000")],
        ),
        make_block(102, t0 + 3600 + 5, [make_tx(6, D, A, 3.0)]),
    ]


@pytest.fixture
def txs(blocks) -> pd.DataFrame:
    rows = [row for block in blocks for row in rows_from_block(block)[0]]
    return to_frame(rows)
