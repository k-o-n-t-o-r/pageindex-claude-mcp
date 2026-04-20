FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY vendor/PageIndex/pageindex/ vendor/PageIndex/pageindex/
COPY claude_backend.py mcp_server.py ./

ENV PAGEINDEX_STORE_PATH=/data

VOLUME /data
EXPOSE 8000

ENTRYPOINT ["uv", "run", "python", "mcp_server.py"]
CMD ["--transport", "sse", "--host", "0.0.0.0", "--port", "8000"]
