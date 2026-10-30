"""Baseline: distância de Mahalanobis robusta por transação.

Corrige a versão original do notebook:
- usa só features da transação (sem timeStamp/blockNumber, que são colineares);
- usa gas_used quando disponível (o campo "gas" é o limite, não o consumo);
- estima média/covariância com MCD (robusto às próprias anomalias);
- limiar pelo quantil χ² com k graus de liberdade, em vez de "3 × média";
- ajusta um modelo por segmento (tipo de transação × sem ETH / poeira / com ETH). Transferências
  simples (gas = 21000) e chamadas de contrato são populações distintas; e ~75% das
  chamadas na mainnet têm valor 0, o que deixaria log(valor) com um pico isolado. Um
  único Gaussiano marcaria como anômalas quase todas as chamadas que enviam ETH.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.covariance import MinCovDet

from tensor_chain.data.schema import WEI_PER_ETH

# Regularização (em unidades padronizadas): valores discretos muito repetidos, como
# gas = 21000, deixam a covariância do subconjunto MCD singular.
_RIDGE = 1e-2
# O padrão do MCD (~50% dos pontos) se fecha no maior aglomerado: na mainnet o gas price é
# bimodal (só base fee vs. gorjeta) e bots repetem gas/preço idênticos. 0.9 assume até 10%
# de contaminação; nos dados reais reduziu a taxa de 36% para ~1,4% nas chamadas sem ETH.
SUPPORT_FRACTION = 0.9
MIN_GROUP_SIZE = 50
KINDS = ("transferência", "chamada", "criação")
DUST_ETH = 1e-9  # 1 gwei


def transaction_kind(df: pd.DataFrame) -> pd.Series:
    """Segmento da transação, por exemplo "chamada · sem ETH".

    "poeira" (< 1 gwei) é separada de "com ETH": pagamentos de poucos wei (comuns em bots)
    formam uma população própria, ~12% das chamadas com valor na mainnet.
    """
    kind = np.select(
        [df["is_contract_creation"].to_numpy(), df["is_contract_call"].to_numpy()],
        [KINDS[2], KINDS[1]],
        KINDS[0],
    )
    value_eth = df["value_eth"].to_numpy()
    value = np.select([value_eth == 0, value_eth < DUST_ETH], ["sem ETH", "poeira"], "com ETH")
    return pd.Series(np.char.add(np.char.add(kind, " · "), value), index=df.index, name="kind")


def transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    gas = df["gas_used"].astype("float64").fillna(df["gas_limit"].astype("float64"))
    return pd.DataFrame(
        {
            # Em wei: valores vão de 1 wei ("poeira", comum em bots) a milhares de ETH, e
            # log1p sobre ETH não comprimiria nada abaixo de 1 ETH.
            "log_value_wei": np.log10(1 + df["value_eth"].to_numpy() * WEI_PER_ETH),
            "log_gas": np.log(gas.to_numpy()),
            "log_gas_price_gwei": np.log(df["gas_price_gwei"].to_numpy() + 1e-3),
        },
        index=df.index,
    )


def _squared_distances(
    feats: pd.DataFrame, max_fit_samples: int, random_state: int
) -> tuple[np.ndarray, int]:
    # max > min, e não std > 0: o std de uma coluna constante pode sair ~1e-15.
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
    """Retorna df com kind, score (distância), p_value e is_anomaly (p_value < alpha).

    Tipos com menos de MIN_GROUP_SIZE transações ficam sem score (NaN).
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
            # Sem gas_used (modo blocos), o gas de uma transferência é o limite escolhido
            # pela carteira (21000, 31500, 500k...), não consumo.
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
