FROM ghcr.io/astral-sh/uv:python3.12-alpine AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --no-install-project

COPY . .
RUN uv sync --frozen --no-default-groups

CMD ["/app/.venv/bin/streamlit", "run", "meta/web.py"]
