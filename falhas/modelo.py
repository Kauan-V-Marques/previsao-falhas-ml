"""Comparação de modelos, escolha do limiar de decisão e avaliação final."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from falhas.dados import ALVO, CATEGORICAS, FEATURES, NUMERICAS

SEMENTE = 42

# Deixar passar uma falha (parada não planejada) custa mais que um alarme falso
# (uma inspeção). O F2 dá ao recall o dobro do peso da precisão.
BETA = 2.0


def _preprocessamento(escalar: bool) -> ColumnTransformer:
    numericas = StandardScaler() if escalar else "passthrough"
    return ColumnTransformer(
        [
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
            ("numericas", numericas, NUMERICAS),
        ]
    )


def candidatos() -> dict[str, Pipeline]:
    return {
        "regressao_logistica": Pipeline(
            [
                ("prep", _preprocessamento(escalar=True)),
                ("modelo", LogisticRegression(class_weight="balanced", max_iter=2000)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("prep", _preprocessamento(escalar=False)),
                (
                    "modelo",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=SEMENTE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("prep", _preprocessamento(escalar=False)),
                (
                    "modelo",
                    HistGradientBoostingClassifier(
                        learning_rate=0.08, max_iter=300, class_weight="balanced", random_state=SEMENTE
                    ),
                ),
            ]
        ),
    }


def melhor_limiar(y: np.ndarray, probabilidades: np.ndarray, beta: float = BETA) -> float:
    """Limiar que maximiza o F-beta nas previsões fora da amostra (validação cruzada)."""
    precisao, recall, limiares = precision_recall_curve(y, probabilidades)
    precisao, recall = precisao[:-1], recall[:-1]
    fbeta = (1 + beta**2) * precisao * recall / np.maximum(beta**2 * precisao + recall, 1e-12)
    return float(limiares[int(np.argmax(fbeta))])


def metricas(y: np.ndarray, probabilidades: np.ndarray, limiar: float) -> dict:
    previsto = (probabilidades >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, previsto, labels=[0, 1]).ravel()
    return {
        "roc_auc": round(float(roc_auc_score(y, probabilidades)), 4),
        "pr_auc": round(float(average_precision_score(y, probabilidades)), 4),
        "precisao": round(float(precision_score(y, previsto, zero_division=0)), 4),
        "recall": round(float(recall_score(y, previsto)), 4),
        "f1": round(float(f1_score(y, previsto)), 4),
        "f2": round(float(fbeta_score(y, previsto, beta=BETA)), 4),
        "matriz_confusao": {"vp": int(tp), "fp": int(fp), "fn": int(fn), "vn": int(tn)},
    }


@dataclass
class Resultado:
    modelo: Pipeline
    nome: str
    limiar: float
    comparacao: dict[str, dict]
    teste: dict
    prevalencia: float
    x_teste: pd.DataFrame = field(repr=False)
    y_teste: pd.Series = field(repr=False)


def treinar(df: pd.DataFrame, dobras: int = 5) -> Resultado:
    x, y = df[FEATURES], df[ALVO]
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=SEMENTE
    )
    validacao = StratifiedKFold(n_splits=dobras, shuffle=True, random_state=SEMENTE)

    # 1) Compara os candidatos só com o treino (o teste fica guardado até o fim)
    comparacao, fora_da_amostra = {}, {}
    for nome, pipeline in candidatos().items():
        prob = cross_val_predict(pipeline, x_treino, y_treino, cv=validacao, method="predict_proba")[:, 1]
        fora_da_amostra[nome] = prob
        comparacao[nome] = {
            "pr_auc": round(float(average_precision_score(y_treino, prob)), 4),
            "roc_auc": round(float(roc_auc_score(y_treino, prob)), 4),
        }

    # 2) Escolhe o melhor pela área sob a curva precisão-recall (adequada a classes desbalanceadas)
    nome = max(comparacao, key=lambda n: comparacao[n]["pr_auc"])
    limiar = melhor_limiar(y_treino.to_numpy(), fora_da_amostra[nome])

    # 3) Treina no treino inteiro e avalia uma única vez no teste
    modelo = clone(candidatos()[nome]).fit(x_treino, y_treino)
    prob_teste = modelo.predict_proba(x_teste)[:, 1]

    return Resultado(
        modelo=modelo,
        nome=nome,
        limiar=round(limiar, 4),
        comparacao=comparacao,
        teste=metricas(y_teste.to_numpy(), prob_teste, limiar),
        prevalencia=round(float(y.mean()), 4),
        x_teste=x_teste,
        y_teste=y_teste,
    )


def nivel_de_risco(probabilidade: float, limiar: float) -> str:
    if probabilidade >= limiar:
        return "alto"
    if probabilidade >= limiar / 2:
        return "medio"
    return "baixo"
