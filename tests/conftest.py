import joblib
import pytest
import sklearn

from falhas import dados, modelo


@pytest.fixture(scope="session")
def tabela():
    return dados.carregar()


@pytest.fixture(scope="session")
def resultado_rapido(tabela):
    """Treino com uma amostra estratificada para os testes rodarem em poucos segundos."""
    amostra = tabela.groupby(dados.ALVO, group_keys=False).sample(frac=0.35, random_state=1)
    return modelo.treinar(amostra, dobras=3)


@pytest.fixture(scope="session")
def caminho_modelo(resultado_rapido, tmp_path_factory):
    caminho = tmp_path_factory.mktemp("modelo") / "modelo.joblib"
    joblib.dump(
        {
            "pipeline": resultado_rapido.modelo,
            "limiar": resultado_rapido.limiar,
            "nome": resultado_rapido.nome,
            "features": dados.FEATURES,
            "sklearn": sklearn.__version__,
            "treinado_em": "teste",
            "teste": resultado_rapido.teste,
        },
        caminho,
    )
    return caminho
