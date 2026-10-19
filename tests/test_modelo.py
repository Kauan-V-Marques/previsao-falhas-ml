import numpy as np
import pytest

from falhas import modelo


def test_limiar_maximiza_o_f2_em_um_caso_conhecido():
    y = np.array([0, 0, 0, 0, 1, 1])
    probabilidades = np.array([0.1, 0.2, 0.3, 0.6, 0.55, 0.9])

    # Com limiar 0.55 os dois positivos são pegos (recall 1) com 1 falso alarme
    assert modelo.melhor_limiar(y, probabilidades) == pytest.approx(0.55)


def test_metricas_e_matriz_de_confusao():
    y = np.array([0, 0, 1, 1])
    probabilidades = np.array([0.2, 0.7, 0.4, 0.9])

    resultado = modelo.metricas(y, probabilidades, limiar=0.5)

    assert resultado["matriz_confusao"] == {"vp": 1, "fp": 1, "fn": 1, "vn": 1}
    assert resultado["precisao"] == 0.5
    assert resultado["recall"] == 0.5


@pytest.mark.parametrize(
    ("probabilidade", "esperado"), [(0.05, "baixo"), (0.25, "medio"), (0.4, "alto"), (0.9, "alto")]
)
def test_nivel_de_risco(probabilidade, esperado):
    assert modelo.nivel_de_risco(probabilidade, limiar=0.4) == esperado


def test_treino_compara_os_tres_modelos_e_escolhe_pela_pr_auc(resultado_rapido):
    comparacao = resultado_rapido.comparacao

    assert set(comparacao) == {"regressao_logistica", "random_forest", "gradient_boosting"}
    assert resultado_rapido.nome == max(comparacao, key=lambda n: comparacao[n]["pr_auc"])
    assert 0 < resultado_rapido.limiar < 1


def test_modelo_e_muito_melhor_que_o_acaso(resultado_rapido):
    # PR-AUC de um classificador aleatório é a própria prevalência (~3,4%)
    assert resultado_rapido.teste["pr_auc"] > 10 * resultado_rapido.prevalencia
    assert resultado_rapido.teste["roc_auc"] > 0.9
    assert resultado_rapido.teste["recall"] > 0.6
