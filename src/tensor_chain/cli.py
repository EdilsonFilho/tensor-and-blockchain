"""Linha de comando: `tensor-chain --help`."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

from tensor_chain.config import get_settings
from tensor_chain.data import store
from tensor_chain.data.collect import block_numbers, collect_addresses, collect_blocks
from tensor_chain.data.etherscan import EtherscanClient
from tensor_chain.models.baseline import mahalanobis_anomalies
from tensor_chain.tensors.build import (
    FLOW_METRICS,
    build_feature_tensor,
    build_flow_tensor,
    top_addresses,
)

app = typer.Typer(help="Anomalias em transações Ethereum com tensores.", no_args_is_help=True)


def _client() -> EtherscanClient:
    settings = get_settings()
    return EtherscanClient(settings.api_key, rate_limit=settings.rate_limit)


def _progress(done: int, total: int) -> None:
    typer.echo(f"\r{done}/{total}", nl=False)
    if done == total:
        typer.echo()


@app.command("latest-block")
def latest_block() -> None:
    """Número do bloco mais recente da mainnet."""
    typer.echo(_client().latest_block())


@app.command("fetch-blocks")
def fetch_blocks(
    start: int = typer.Option(..., help="Primeiro bloco"),
    count: int = typer.Option(..., help="Quantidade de blocos"),
    step: int = typer.Option(1, help="Amostrar 1 a cada N blocos"),
    name: str | None = typer.Option(None, help="Nome do dataset"),
) -> None:
    """Coleta blocos inteiros (rede toda). Retoma se interrompido."""
    numbers = block_numbers(start, count, step)
    name = name or f"blocks-{numbers[0]}-{numbers[-1]}-s{step}"
    meta = {"mode": "blocks", "start_block": numbers[0], "end_block": numbers[-1], "step": step}
    path = store.create_dataset(get_settings().data_dir, name, meta)
    fetched = collect_blocks(_client(), path, numbers, progress=_progress)
    typer.echo(f"{name}: {fetched} blocos baixados")


@app.command("fetch-addresses")
def fetch_addresses(
    addresses: list[str] = typer.Argument(..., help="Endereços-semente"),
    start: int = typer.Option(0, help="Bloco inicial"),
    end: int = typer.Option(99_999_999, help="Bloco final"),
    name: str = typer.Option(..., help="Nome do dataset"),
) -> None:
    """Coleta o histórico completo de endereços (txlist). Retoma se interrompido."""
    addresses = [a.lower() for a in addresses]
    meta = {"mode": "addresses", "start_block": start, "end_block": end, "addresses": addresses}
    path = store.create_dataset(get_settings().data_dir, name, meta)
    fetched = collect_addresses(_client(), path, addresses, start, end, progress=_progress)
    typer.echo(f"{name}: {fetched} endereços baixados")


@app.command()
def datasets() -> None:
    """Lista os datasets coletados."""
    for name in store.list_datasets(get_settings().data_dir):
        typer.echo(name)


@app.command()
def tensor(
    name: str,
    top_k: int = typer.Option(200, help="Endereços mais ativos"),
    freq: str = typer.Option("1h", help="Janela de tempo (10min, 1h, 6h, 1D)"),
    metric: str = typer.Option("count", help=f"Métrica do fluxo: {', '.join(FLOW_METRICS)}"),
) -> None:
    """Monta os tensores 𝒳 e 𝒴 de um dataset e mostra suas dimensões."""
    txs = store.load_transactions(store.dataset_path(get_settings().data_dir, name))
    addresses = top_addresses(txs, top_k)
    flow = build_flow_tensor(txs, addresses, freq, metric)
    feats = build_feature_tensor(txs, addresses, freq)
    typer.echo(f"𝒳 fluxo    {flow.data.shape}  nnz={flow.data.nnz}  densidade={flow.density:.2e}")
    typer.echo(f"𝒴 features {feats.data.shape}  features={', '.join(feats.features)}")


@app.command()
def baseline(
    name: str,
    alpha: float = typer.Option(1e-3, help="Nível de significância (p-valor χ²)"),
    out: Path | None = typer.Option(None, help="CSV de saída com as anomalias"),
) -> None:
    """Baseline: Mahalanobis robusto por transação."""
    txs = store.load_transactions(store.dataset_path(get_settings().data_dir, name))
    scored = mahalanobis_anomalies(txs, alpha)
    anomalies = scored[scored["is_anomaly"]].sort_values("score", ascending=False)
    typer.echo(f"{len(anomalies)} anomalias em {len(scored)} transações")
    if out:
        anomalies.to_csv(out, index=False)
        typer.echo(f"salvo em {out}")


@app.command()
def ui() -> None:
    """Abre a interface gráfica (Streamlit) no navegador."""
    script = Path(__file__).parent / "ui" / "app.py"
    raise typer.Exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(script)]))
