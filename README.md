# PageIndex Claude MCP


MCP server for [PageIndex](https://github.com/VectifyAI/PageIndex) with a Claude LLM backend.

<br>

> [!NOTE]                                                                     
> PageIndex normally requires OpenAI. This server patches it at runtime to use Claude instead (via the Claude Agent SDK). The upstream PageIndex library is vendored as a git submodule under `vendor/PageIndex/`.


## Quick start

**Requirements**: Python 3.11+, [uv](https://docs.astral.sh/uv/), and a `CLAUDE_CODE_OAUTH_TOKEN` (see `.env.example`).

```bash
git clone --recurse-submodules https://github.com/k-o-n-t-o-r/pageindex-claude-mcp.git
cd pageindex-claude-mcp
uv sync
```

Add to **Claude Code**:

```bash
claude mcp add pageindex-claude-mcp -- uv --directory /absolute/path/to/pageindex-claude-mcp run python mcp_server.py
```

Config for **Claude Desktop**:

```json
{
  "mcpServers": {
    "pageindex-claude-mcp": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/pageindex-claude-mcp", "run", "python", "mcp_server.py"]
    }
  }
}
```

See `.env.example` for all available settings.

## Configuration

Environment variables (also listed in `.env.example`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `PAGEINDEX_STORE_PATH` | `~/.pageindex-store` | Document store location |
| `CLAUDE_MODEL` | `sonnet` | Model (sonnet/opus/haiku or full ID) |
| `CLAUDE_EFFORT` | - | Reasoning effort: `low`, `medium`, `high`, or `max`. Unset uses the SDK default. |
| `CLAUDE_CODE_OAUTH_TOKEN` | - | OAuth token for Claude Agent SDK |
| `PAGEINDEX_LOG_FILE` | `~/.pageindex-store/mcp_server.log` | Log file path |
| `PAGEINDEX_CONCURRENCY` | `4` | Max concurrent indexing jobs |
| `PAGEINDEX_IMPORT_DIR` | - | If set, restrict `add_document`/`add_documents` to paths under this directory (after resolving symlinks). Recommended when running over SSE. |

## Tools

| Tool | Description |
|------|-------------|
| `add_document` | Index a PDF or Markdown file |
| `add_documents` | Batch-index multiple files concurrently |
| `list_documents` | List all indexed documents |
| `search_documents` | Reasoning-based search across the store |
| `get_document_tree` | Full hierarchical structure of a document |
| `get_page_text` | Extract text from specific PDF pages |
| `remove_document` | Remove a document and its index |

## Docker

The Docker image runs the server over SSE, suitable for remote or headless deployments.

```bash
docker build -t pageindex-claude-mcp .
docker run -p 8000:8000 -e CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-... pageindex-claude-mcp
```

The document store is persisted at `/data` inside the container. Mount a volume to keep it across restarts:

```bash
docker run -p 8000:8000 \
  -e CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-... \
  -v pageindex-data:/data \
  pageindex-claude-mcp
```

To use stdio transport instead (e.g. for piping into an MCP client):

```bash
docker run -i --rm -e CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-... pageindex-claude-mcp --transport stdio
```

## Running standalone

```bash
uv run python mcp_server.py                                          # stdio (default)
uv run python mcp_server.py --transport sse --host 0.0.0.0 --port 8000  # SSE
```

## Tests

Install dev dependencies, then run:

```bash
uv sync --group dev
uv run python -m pytest tests/test_claude_backend.py   # unit tests (mocked LLM, fast)
uv run python tests/test_integration.py                # integration tests (real LLM, spends quota)
```

## License

MIT
