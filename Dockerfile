FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app
COPY pyproject.toml ./
COPY falhas ./falhas
COPY dados ./dados

# Instala e já treina o modelo durante o build: a imagem sobe pronta para responder
RUN pip install --no-cache-dir -e ".[server]" \
    && python -m falhas.treinar --sem-graficos --relatorio /tmp/relatorio \
    && useradd --create-home app

USER app
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "falhas.api:create_app()"]
