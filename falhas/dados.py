"""Leitura do dataset AI4I 2020 e criação das features."""

import math
from pathlib import Path

import pandas as pd

ARQUIVO_PADRAO = Path(__file__).resolve().parent.parent / "dados" / "ai4i2020.csv"

COLUNAS = {
    "Type": "tipo",
    "Air temperature [K]": "temp_ar_k",
    "Process temperature [K]": "temp_processo_k",
    "Rotational speed [rpm]": "rotacao_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "desgaste_min",
    "Machine failure": "falha",
}

# TWF, HDF, PWF, OSF e RNF dizem QUAL falha aconteceu: usá-las seria entregar a resposta
# ao modelo (vazamento de dados). UDI e Product ID são só identificadores.
COLUNAS_DESCARTADAS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]

CATEGORICAS = ["tipo"]
NUMERICAS = [
    "temp_ar_k",
    "temp_processo_k",
    "rotacao_rpm",
    "torque_nm",
    "desgaste_min",
    "delta_temp_k",
    "potencia_w",
    "esforco_ferramenta",
]
FEATURES = CATEGORICAS + NUMERICAS
ALVO = "falha"


def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features inspiradas nos mecanismos físicos de falha da máquina.

    - delta_temp_k: dissipação de calor ruim quando processo e ambiente ficam próximos
    - potencia_w: potência mecânica = torque x velocidade angular (falha com potência muito baixa ou alta)
    - esforco_ferramenta: desgaste x torque (sobrecarga da ferramenta)
    """
    df = df.copy()
    df["delta_temp_k"] = df["temp_processo_k"] - df["temp_ar_k"]
    df["potencia_w"] = df["torque_nm"] * df["rotacao_rpm"] * 2 * math.pi / 60
    df["esforco_ferramenta"] = df["desgaste_min"] * df["torque_nm"]
    return df


def carregar(caminho: Path | str = ARQUIVO_PADRAO) -> pd.DataFrame:
    df = pd.read_csv(caminho).drop(columns=COLUNAS_DESCARTADAS).rename(columns=COLUNAS)
    return criar_features(df)[FEATURES + [ALVO]]
