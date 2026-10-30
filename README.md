# Tensor Chain: anomalias em transações Ethereum com tensores

Este projeto procura transações e endereços com comportamento anômalo na **Ethereum mainnet**. Ele modela as transações como tensores e aplica decomposições tensoriais. Os dados vêm da [API V2 da Etherscan](https://docs.etherscan.io/).

A versão original era um notebook escrito num domingo à tarde (2023). Esta versão é um pacote Python com interface gráfica (Streamlit) e linha de comando.

## Instalação

Requer Python 3.10 ou superior.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # e preencha ETHERSCAN_API_KEY
```

Crie uma chave gratuita em <https://etherscan.io/myapikey>. O plano gratuito permite 3 chamadas por segundo e 100 mil chamadas por dia.

## Uso

### Interface gráfica

```sh
tensor-chain ui
```

A interface tem três telas:

- **Dados:** coleta e resumo do dataset.
- **Tensor:** monta os tensores e permite inspecionar fatias.
- **Anomalias (baseline):** tabela com links para o Etherscan, gráficos e exportação em CSV.

### Linha de comando

```sh
tensor-chain latest-block
tensor-chain fetch-blocks --start 21000000 --count 300            # rede toda
tensor-chain fetch-addresses 0xabc... 0xdef... --name sementes    # endereços-semente
tensor-chain datasets
tensor-chain tensor blocks-21000000-21000299-s1 --top-k 200 --freq 1h
tensor-chain baseline blocks-21000000-21000299-s1 --alpha 1e-3 --out anomalias.csv
```

Se a coleta for interrompida, rode o mesmo comando de novo. Ela retoma de onde parou.

## Modos de coleta

| Modo | Endpoint | Cobertura | Limitações |
|---|---|---|---|
| **Blocos** | `proxy/eth_getBlockByNumber` | Todas as transações de um intervalo de blocos | 1 chamada por bloco (1 dia ≈ 7.200 blocos ≈ 40 min). O endpoint não traz `gasUsed` nem o status: o gas usado é o **limite** declarado e a taxa é estimada. |
| **Endereços** | `account/txlist` | Histórico completo dos endereços-semente | Traz todos os campos, mas só enxerga as transações que envolvem as sementes. |

Os dados ficam em `data/datasets/<nome>/`, em partes Parquet.

## Modelagem

Os três tensores usam o mesmo índice de endereços (os top-K mais ativos, mais um bucket opcional "outros"):

- **𝒳 (from × to × tempo)**, esparso. Cada célula guarda a contagem de transações, o valor transferido ou a taxa paga, em `log1p` (valores em gwei).
- **𝒴 (endereço × tempo × feature)**, denso. As features são: enviadas, recebidas, valor enviado e recebido, taxa paga, gas price médio, fração de chamadas a contratos, nº de contrapartes e taxa de erro.
- **N (tempo × rede)**: nº de transações, gas price mediano, base fee e utilização de gas dos blocos.

### Baseline

O baseline calcula a distância de Mahalanobis robusta (MCD) de cada transação a partir de `log(valor)`, `log(gas)` e `log(gas price)`. Há um modelo para cada tipo de transação: transferência, chamada de contrato ou criação de contrato. A separação é necessária porque transferências simples (gas = 21000) e chamadas de contrato formam populações distintas. Uma transação é anômala quando o p-valor χ² fica abaixo de α.

### Próximos passos

| Técnica | Tensor | Uso |
|---|---|---|
| CP | 𝒳 | Padrões recorrentes de transações (grupo de remetentes × grupo de destinatários × perfil temporal) |
| Tucker | 𝒴 | Fatores latentes por modo; o core mostra como esses fatores interagem |
| Tensor completion | 𝒳 com máscara | Interações observadas que o modelo considerava improváveis |
| Coupled (CMTF) | 𝒳 + atributos + N | Endereços cujo fluxo diverge dos atributos e do contexto da rede |

## Desenvolvimento

```sh
pytest
ruff check src tests && ruff format --check src tests
```

Os testes não acessam a rede. O cliente da Etherscan é testado com uma sessão HTTP falsa.
