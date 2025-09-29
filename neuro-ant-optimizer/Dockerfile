FROM python:3.11-slim

ENV APP_HOME=/opt/neuro-ant-optimizer \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR ${APP_HOME}

RUN apt-get update \ 
    && apt-get install -y --no-install-recommends build-essential git curl \ 
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md CHANGELOG.md LICENSE ./
COPY src ./src
COPY backtest ./backtest
COPY examples ./examples
COPY data ./data
COPY tests ./tests
COPY conda ./conda
COPY mkdocs.yml ./mkdocs.yml

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .[dev]

CMD ["python", "-m", "neuro_ant_optimizer.backtest.backtest", "--help"]
