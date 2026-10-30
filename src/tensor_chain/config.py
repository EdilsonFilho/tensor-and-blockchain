"""Settings read from the environment (and from a .env file, if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    data_dir: Path
    rate_limit: float  # calls per second


def get_settings() -> Settings:
    return Settings(
        api_key=os.getenv("ETHERSCAN_API_KEY") or None,
        data_dir=Path(os.getenv("TENSOR_CHAIN_DATA_DIR", "data")),
        rate_limit=float(os.getenv("ETHERSCAN_RATE_LIMIT", "3")),
    )
