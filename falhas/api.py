"""API que usa o modelo treinado: flask --app falhas.api run"""

import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from falhas.dados import FEATURES, criar_features
from falhas.modelo import nivel_de_risco

MODELO_PADRAO = Path(__file__).resolve().parent.parent / "modelos" / "modelo.joblib"


class Leitura(BaseModel):
    """Leitura dos sensores da máquina (mesmas unidades do dataset)."""

    model_config = ConfigDict(extra="forbid")

    tipo: Literal["L", "M", "H"] = Field(description="Qualidade do produto: L (baixa), M (média), H (alta)")
    temp_ar_k: float = Field(ge=250, le=350)
    temp_processo_k: float = Field(ge=250, le=400)
    rotacao_rpm: float = Field(gt=0, le=5000)
    torque_nm: float = Field(ge=0, le=150)
    desgaste_min: float = Field(ge=0, le=400)


def create_app(caminho_modelo: str | Path | None = None) -> Flask:
    app = Flask(__name__)
    pacote = joblib.load(caminho_modelo or os.environ.get("MODELO", MODELO_PADRAO))
    pipeline, limiar = pacote["pipeline"], pacote["limiar"]

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/modelo")
    def info_modelo():
        return {
            "modelo": pacote["nome"],
            "limiar": limiar,
            "treinado_em": pacote["treinado_em"],
            "sklearn": pacote["sklearn"],
            "metricas_teste": pacote["teste"],
        }

    @app.post("/prever")
    def prever():
        corpo = request.get_json(force=True, silent=True)
        leituras = corpo if isinstance(corpo, list) else [corpo or {}]
        try:
            validadas = [Leitura.model_validate(item) for item in leituras]
        except ValidationError as erro:
            detalhes = [
                {"campo": ".".join(map(str, e["loc"])), "erro": e["msg"]}
                for e in erro.errors(include_url=False)
            ]
            return jsonify({"erro": "Leitura inválida.", "detalhes": detalhes}), 400

        tabela = criar_features(pd.DataFrame([v.model_dump() for v in validadas]))[FEATURES]
        probabilidades = pipeline.predict_proba(tabela)[:, 1]
        resultados = [
            {
                "probabilidade_falha": round(float(p), 4),
                "falha_prevista": bool(p >= limiar),
                "risco": nivel_de_risco(float(p), limiar),
            }
            for p in probabilidades
        ]
        return jsonify(resultados if isinstance(corpo, list) else resultados[0])

    return app
