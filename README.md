# Tensor Chain: anomalies in Ethereum transactions with tensors

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

This project looks for transactions and addresses with anomalous behavior on **Ethereum mainnet**. It models transactions as tensors and applies tensor decompositions. Data comes from the [Etherscan API V2](https://docs.etherscan.io/).

The original version was a notebook written on a Sunday afternoon (2023). This version is a Python package with a graphical interface (Streamlit) and a command line.

## Architecture

![Architecture](docs/images/architecture.svg)

## Installation

Requires Python 3.10 or later.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # and fill in ETHERSCAN_API_KEY
```

Create a free key at <https://etherscan.io/myapikey>. The free plan allows 3 calls per second and 100k calls per day.

## Usage

### Graphical interface

```sh
tensor-chain ui
```

The interface has three pages:

- **Data:** collection and dataset summary.
- **Tensor:** builds the tensors and lets you inspect slices.
- **Anomalies (baseline):** table with links to Etherscan, charts and CSV export.

### Command line

```sh
tensor-chain latest-block
tensor-chain fetch-blocks --start 21000000 --count 300            # entire network
tensor-chain fetch-addresses 0xabc... 0xdef... --name seeds       # seed addresses
tensor-chain datasets
tensor-chain tensor blocks-21000000-21000299-s1 --top-k 200 --freq 1h
tensor-chain baseline blocks-21000000-21000299-s1 --alpha 1e-3 --out anomalies.csv
```

If a collection is interrupted, run the same command again. It resumes where it stopped.

## Collection modes

| Mode | Endpoint | Coverage | Limitations |
|---|---|---|---|
| **Blocks** | `proxy/eth_getBlockByNumber` | Every transaction in a block range | 1 call per block (1 day ≈ 7,200 blocks ≈ 40 min). The endpoint has no `gasUsed` or status: gas is the declared **limit** and the fee is estimated. |
| **Addresses** | `account/txlist` | Full history of the seed addresses | All fields, but only the transactions that involve the seeds. The API returns 1,000 transactions per call; pagination is automatic. |

Data is stored in `data/datasets/<name>/` as Parquet parts.

## Modeling

The three tensors share the same address index (the top-K most active addresses, plus an optional "other" bucket):

- **𝒳 (from × to × time)**, sparse. Each cell holds the transaction count, the value transferred or the fee paid, in `log1p` (amounts in gwei).
- **𝒴 (address × time × feature)**, dense. The features are: sent, received, value sent and received, fee paid, mean gas price, share of contract calls, number of counterparties and error rate.
- **N (time × network)**: number of transactions, median gas price, base fee and block gas utilization.

### Baseline

The baseline computes the robust Mahalanobis distance (MCD) of each transaction from `log(value in wei)`, `log(gas)` and `log(gas price)`. There is one model per segment: transaction kind (transfer, contract call, contract creation) × value (no ETH, dust below 1 gwei, with ETH). The split is needed because these segments are distinct populations on mainnet: plain transfers always use 21,000 gas, ~75% of contract calls send no ETH, and MEV bots pay a few wei. A transaction is anomalous when its χ² p-value is below α.

### Next steps

| Technique | Tensor | Use |
|---|---|---|
| CP | 𝒳 | Recurring transaction patterns (sender group × receiver group × time profile) |
| Tucker | 𝒴 | Latent factors per mode; the core shows how those factors interact |
| Tensor completion | Masked 𝒳 | Observed interactions the model considered unlikely |
| Coupled (CMTF) | 𝒳 + attributes + N | Addresses whose flow diverges from their attributes and from the network context |

## Development

```sh
pytest
ruff check src tests && ruff format --check src tests
```

The tests do not access the network. The Etherscan client is tested with a fake HTTP session.

## License

[MIT](LICENSE) © 2023-2026 Edilson Filho
