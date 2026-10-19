import pytest

from falhas.api import create_app

NORMAL = {
    "tipo": "M",
    "temp_ar_k": 298.1,
    "temp_processo_k": 308.6,
    "rotacao_rpm": 1551,
    "torque_nm": 42.8,
    "desgaste_min": 0,
}
# Ferramenta gasta com torque alto: perfil típico de sobrecarga (OSF)
SOBRECARGA = {
    "tipo": "L",
    "temp_ar_k": 300.5,
    "temp_processo_k": 310.2,
    "rotacao_rpm": 1300,
    "torque_nm": 68.0,
    "desgaste_min": 215,
}


@pytest.fixture()
def client(caminho_modelo):
    return create_app(caminho_modelo).test_client()


def test_leitura_normal_tem_risco_baixo(client):
    resposta = client.post("/prever", json=NORMAL)

    assert resposta.status_code == 200
    corpo = resposta.get_json()
    assert corpo["falha_prevista"] is False
    assert corpo["risco"] == "baixo"


def test_sobrecarga_e_prevista_como_falha(client):
    corpo = client.post("/prever", json=SOBRECARGA).get_json()

    assert corpo["falha_prevista"] is True
    assert corpo["risco"] == "alto"


def test_aceita_varias_leituras_de_uma_vez(client):
    corpo = client.post("/prever", json=[NORMAL, SOBRECARGA]).get_json()

    assert [r["falha_prevista"] for r in corpo] == [False, True]


def test_leitura_invalida_retorna_400_com_os_campos(client):
    resposta = client.post("/prever", json={**NORMAL, "tipo": "X", "torque_nm": -5})

    campos = {d["campo"] for d in resposta.get_json()["detalhes"]}
    assert resposta.status_code == 400
    assert {"tipo", "torque_nm"} <= campos


def test_informacoes_do_modelo(client):
    corpo = client.get("/modelo").get_json()

    assert corpo["modelo"] in {"regressao_logistica", "random_forest", "gradient_boosting"}
    assert 0 < corpo["limiar"] < 1
    assert "pr_auc" in corpo["metricas_teste"]
