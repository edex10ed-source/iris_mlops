FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project


COPY . .

RUN uv sync --locked --no-dev

RUN uv run python -c "import sys, uvicorn; print('Python:', sys.executable); print('Uvicorn:', uvicorn.__file__)"

ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

RUN uv run preparar-datos && uv run entrenar

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

CMD ["sh", "-c", "exec uv run --no-sync python -m uvicorn iris_mlops.api:app --host 0.0.0.0 --port ${PORT:-8000}"]