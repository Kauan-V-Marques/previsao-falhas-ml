import math

import pandas as pd
import pytest

from falhas import dados


def test_dataset_tem_10_mil_leituras_e_3_4_por_cento_de_falhas(tabela):
    assert len(tabela) == 10_000
    assert tabela[dados.ALVO].mean() == pytest.approx(0.0339)


def test_colunas_que_entregam_a_resposta_nao_entram(tabela):
    vazamento = {"TWF", "HDF", "PWF", "OSF", "RNF", "UDI", "Product ID"}

    assert vazamento.isdisjoint(tabela.columns)
    assert list(tabela.columns) == dados.FEATURES + [dados.ALVO]


def test_features_fisicas_calculadas_corretamente():
    leitura = pd.DataFrame(
        [
            {
                "temp_ar_k": 298.1,
                "temp_processo_k": 308.6,
                "rotacao_rpm": 1551,
                "torque_nm": 42.8,
                "desgaste_min": 20,
            }
        ]
    )

    resultado = dados.criar_features(leitura).iloc[0]

    assert resultado["delta_temp_k"] == pytest.approx(10.5)
    assert resultado["potencia_w"] == pytest.approx(42.8 * 1551 * 2 * math.pi / 60)
    assert resultado["esforco_ferramenta"] == pytest.approx(856.0)


def test_criar_features_nao_altera_a_tabela_original():
    leitura = pd.DataFrame(
        [
            {
                "temp_ar_k": 300.0,
                "temp_processo_k": 310.0,
                "rotacao_rpm": 1500,
                "torque_nm": 40.0,
                "desgaste_min": 0,
            }
        ]
    )

    dados.criar_features(leitura)

    assert "potencia_w" not in leitura.columns


def test_sem_valores_ausentes(tabela):
    assert not tabela.isna().any().any()
