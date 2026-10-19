"""Treina o modelo e gera o relatório: python -m falhas.treinar"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import sklearn  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import PrecisionRecallDisplay  # noqa: E402

from falhas import dados, modelo  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent


def salvar_graficos(resultado: modelo.Resultado, pasta: Path) -> None:
    pasta.mkdir(parents=True, exist_ok=True)
    prob = resultado.modelo.predict_proba(resultado.x_teste)[:, 1]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    PrecisionRecallDisplay.from_predictions(resultado.y_teste, prob, ax=ax, name=resultado.nome)
    ax.axhline(resultado.prevalencia, linestyle="--", color="gray", label="aleatório (prevalência)")
    ax.set_title("Curva precisão-recall no conjunto de teste")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(pasta / "curva_precisao_recall.png", dpi=120)
    plt.close(fig)

    cm = resultado.teste["matriz_confusao"]
    matriz = [[cm["vn"], cm["fp"]], [cm["fn"], cm["vp"]]]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matriz, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                matriz[i][j],
                ha="center",
                va="center",
                fontsize=16,
                color="white" if matriz[i][j] > 500 else "black",
            )
    ax.set_xticks([0, 1], ["prevê OK", "prevê falha"])
    ax.set_yticks([0, 1], ["real OK", "real falha"])
    ax.set_title(f"Matriz de confusão (limiar {resultado.limiar:.2f})")
    fig.tight_layout()
    fig.savefig(pasta / "matriz_confusao.png", dpi=120)
    plt.close(fig)

    importancia = permutation_importance(
        resultado.modelo,
        resultado.x_teste,
        resultado.y_teste,
        scoring="average_precision",
        n_repeats=10,
        random_state=modelo.SEMENTE,
        n_jobs=-1,
    )
    ordem = importancia.importances_mean.argsort()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(
        [dados.FEATURES[i] for i in ordem],
        importancia.importances_mean[ordem],
        xerr=importancia.importances_std[ordem],
        color="#f26b1d",
    )
    ax.set_title("Importância por permutação (queda na PR-AUC)")
    fig.tight_layout()
    fig.savefig(pasta / "importancia_features.png", dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", default=str(dados.ARQUIVO_PADRAO))
    parser.add_argument("--modelo", default=str(RAIZ / "modelos" / "modelo.joblib"))
    parser.add_argument("--relatorio", default=str(RAIZ / "relatorio"))
    parser.add_argument("--sem-graficos", action="store_true")
    args = parser.parse_args(argv)

    resultado = modelo.treinar(dados.carregar(args.dados))

    caminho_modelo = Path(args.modelo)
    caminho_modelo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": resultado.modelo,
            "limiar": resultado.limiar,
            "nome": resultado.nome,
            "features": dados.FEATURES,
            "sklearn": sklearn.__version__,
            "treinado_em": datetime.now(UTC).isoformat(timespec="seconds"),
            "teste": resultado.teste,
        },
        caminho_modelo,
    )

    pasta = Path(args.relatorio)
    pasta.mkdir(parents=True, exist_ok=True)
    relatorio = {
        "modelo_escolhido": resultado.nome,
        "limiar": resultado.limiar,
        "prevalencia_falhas": resultado.prevalencia,
        "validacao_cruzada_treino": resultado.comparacao,
        "teste": resultado.teste,
    }
    (pasta / "metricas.json").write_text(json.dumps(relatorio, indent=2, ensure_ascii=False) + "\n", "utf-8")
    if not args.sem_graficos:
        salvar_graficos(resultado, pasta)

    print(json.dumps(relatorio, indent=2, ensure_ascii=False))
    print(f"\nModelo salvo em {caminho_modelo}")


if __name__ == "__main__":
    main()
