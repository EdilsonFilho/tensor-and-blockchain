from __future__ import annotations

import pytest

from tensor_chain.data import store
from tensor_chain.data.collect import collect_addresses, collect_blocks
from tensor_chain.data.schema import row_from_txlist, to_frame

from .conftest import A, B


def test_block_rows_are_normalized(txs):
    first = txs.iloc[0]
    assert first["value_eth"] == 1.0
    assert first["gas_price_gwei"] == 20
    assert first["fee_estimated"]
    assert first["fee_eth"] == pytest.approx(21_000 * 20 / 1e9)
    creation = txs[txs["is_contract_creation"]]
    assert len(creation) == 1 and creation["to"].isna().all()
    call = txs[txs["is_contract_call"]]
    assert call["method_id"].tolist() == ["0xa9059cbb"]


def test_txlist_row_uses_gas_used():
    row = row_from_txlist(
        {
            "hash": "0x1",
            "blockNumber": "10",
            "timeStamp": "1700000000",
            "from": A.upper(),
            "to": "",
            "value": "0",
            "gas": "100000",
            "gasUsed": "60000",
            "gasPrice": str(10 * 10**9),
            "isError": "1",
            "methodId": "0x",
        }
    )
    df = to_frame([row])
    assert df.loc[0, "from"] == A
    assert df.loc[0, "is_contract_creation"]
    assert df.loc[0, "is_error"]
    assert not df.loc[0, "fee_estimated"]
    assert df.loc[0, "fee_eth"] == pytest.approx(60_000 * 10 / 1e9)


class FakeClient:
    def __init__(self, blocks=None, txlists=None):
        self.blocks = {int(b["number"], 16): b for b in blocks or []}
        self.txlists = txlists or {}
        self.requested = []

    def block(self, number):
        self.requested.append(number)
        return self.blocks[number]

    def txlist(self, address, start, end):
        self.requested.append(address)
        return self.txlists[address]


def test_collect_blocks_resumes(tmp_path, blocks):
    path = store.create_dataset(
        tmp_path, "d", {"mode": "blocks", "start_block": 100, "end_block": 102, "step": 1}
    )
    client = FakeClient(blocks)
    assert collect_blocks(client, path, [100, 101], chunk_size=1) == 2
    assert collect_blocks(client, path, [100, 101, 102], chunk_size=1) == 1
    assert client.requested == [100, 101, 102]

    txs = store.load_transactions(path)
    assert len(txs) == 6 and txs["hash"].is_unique
    assert store.load_blocks(path)["block_number"].tolist() == [100, 101, 102]


def test_collect_addresses_dedupes(tmp_path):
    shared = {
        "hash": "0x1",
        "blockNumber": "1",
        "timeStamp": "1700000000",
        "from": A,
        "to": B,
        "value": "1",
        "gas": "21000",
        "gasUsed": "21000",
        "gasPrice": "1",
        "isError": "0",
    }
    client = FakeClient(txlists={A: [shared], B: [shared]})
    path = store.create_dataset(tmp_path, "s", {"mode": "addresses", "addresses": [A, B]})
    collect_addresses(client, path, [A, B], 0, 10)
    assert len(store.load_transactions(path)) == 1


def test_dataset_params_must_match(tmp_path):
    store.create_dataset(tmp_path, "d", {"mode": "blocks", "start_block": 1, "end_block": 2})
    with pytest.raises(ValueError, match="end_block"):
        store.create_dataset(tmp_path, "d", {"mode": "blocks", "start_block": 1, "end_block": 3})


def test_rejects_unsafe_names(tmp_path):
    with pytest.raises(ValueError):
        store.dataset_path(tmp_path, "../fora")
