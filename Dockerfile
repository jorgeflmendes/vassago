FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir uv==0.11.26
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --no-dev
COPY configs ./configs
RUN useradd --create-home researcher && chown -R researcher:researcher /app
USER researcher
CMD ["uv", "run", "vassago", "experiment", "run", "--config", "configs/experiment/smoke.yaml"]
