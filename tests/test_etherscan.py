from __future__ import annotations

import pytest

from tensor_chain.data.etherscan import EtherscanClient, EtherscanError


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append(params)
        return FakeResponse(self.payloads.pop(0))


def client(payloads, **kwargs) -> tuple[EtherscanClient, FakeSession]:
    session = FakeSession(payloads)
    return EtherscanClient("key", session=session, sleep=lambda _: None, **kwargs), session


def test_requires_api_key():
    with pytest.raises(EtherscanError):
        EtherscanClient(None)


def test_proxy_call_sends_chain_and_key():
    c, session = client([{"jsonrpc": "2.0", "id": 1, "result": "0x10"}])
    assert c.latest_block() == 16
    assert session.calls[0]["chainid"] == 1
    assert session.calls[0]["apikey"] == "key"


def test_retries_on_rate_limit():
    limited = {
        "status": "0",
        "message": "NOTOK",
        "result": "Max calls per sec rate limit reached (3/sec)",
    }
    c, session = client([limited, {"jsonrpc": "2.0", "id": 1, "result": "0x1"}])
    assert c.latest_block() == 1
    assert len(session.calls) == 2


def test_gives_up_after_max_retries():
    limited = {"status": "0", "message": "NOTOK", "result": "rate limit reached"}
    c, _ = client([limited] * 2, max_retries=2)
    with pytest.raises(EtherscanError, match="2 tentativas"):
        c.latest_block()


def test_api_error_is_raised_without_retry():
    c, session = client([{"status": "0", "message": "NOTOK", "result": "Invalid API Key"}])
    with pytest.raises(EtherscanError, match="Invalid API Key"):
        c.latest_block()
    assert len(session.calls) == 1


def test_txlist_empty_result():
    c, _ = client([{"status": "0", "message": "No transactions found", "result": []}])
    assert c.txlist("0xabc", 0, 10) == []


def _pages(*pages):
    return [
        {"status": "1", "message": "OK", "result": [{"hash": h, "blockNumber": b} for h, b in p]}
        for p in pages
    ]


def test_txlist_does_not_trust_page_size():
    # A API devolveu menos do que o pedido (como a V2, que corta em 1000) e ainda há dados.
    c, session = client(
        _pages(
            [("a", "1"), ("b", "2"), ("c", "5")],
            [("c", "5"), ("d", "6")],  # recomeça no bloco 5 e repete "c"
            [("d", "6")],  # nada novo: fim
        )
    )
    rows = c.txlist("0xabc", 0, 10)
    assert [r["hash"] for r in rows] == ["a", "b", "c", "d"]
    assert [call["startblock"] for call in session.calls] == [0, 5, 6]


def test_txlist_rejects_block_larger_than_page():
    c, _ = client(_pages([("a", "1"), ("b", "3")], [("c", "3"), ("d", "3")]))
    with pytest.raises(EtherscanError, match="bloco 3"):
        c.txlist("0xabc", 0, 10)
