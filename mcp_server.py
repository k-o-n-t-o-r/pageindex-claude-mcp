#!/usr/bin/env python3
"""
PageIndex MCP Server -- document knowledge store backed by Claude.

Wraps PageIndex (https://github.com/VectifyAI/PageIndex) as an MCP server,
using Claude as the LLM backend instead of OpenAI.

See .env.example for all configuration environment variables.
"""

import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("mcp_server")


def _setup_logging():
    """Configure file logging. Called at server start, not at import."""
    log_file = os.environ.get(
        "PAGEINDEX_LOG_FILE",
        str(Path.home() / ".pageindex-store" / "mcp_server.log"),
    )
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        filename=log_file,
    )

_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_vendor_pageindex = str(Path(__file__).resolve().parent / "vendor" / "PageIndex")
if _vendor_pageindex not in sys.path:
    sys.path.insert(0, _vendor_pageindex)

import claude_backend
claude_backend.activate()

import pymupdf
from mcp.server.fastmcp import Context, FastMCP

STORE_PATH = Path(os.environ.get("PAGEINDEX_STORE_PATH", Path.home() / ".pageindex-store"))
DOCS_DIR = STORE_PATH / "documents"
STRUCTS_DIR = STORE_PATH / "structures"
MANIFEST_PATH = STORE_PATH / "manifest.json"

SUPPORTED_EXTS = (".pdf", ".md", ".markdown")

_manifest_lock = asyncio.Lock()


def _resolve_import_path(file_path: str) -> Path:
    """Sanitize and resolve a client-supplied import path.

    Expands ~, resolves symlinks, and — if PAGEINDEX_IMPORT_DIR is set —
    rejects any file outside that directory. Raises ValueError with a
    client-safe message on any failure.
    """
    try:
        resolved = Path(os.path.expanduser(file_path)).resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError(f"file not found: {file_path}")

    if not resolved.is_file():
        raise ValueError(f"not a regular file: {file_path}")

    if resolved.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError("only .pdf, .md, and .markdown files are supported.")

    import_dir = os.environ.get("PAGEINDEX_IMPORT_DIR")
    if import_dir:
        try:
            root = Path(os.path.expanduser(import_dir)).resolve(strict=True)
        except (OSError, RuntimeError):
            raise ValueError("PAGEINDEX_IMPORT_DIR is set but does not exist.")
        if not resolved.is_relative_to(root):
            raise ValueError(
                f"path is outside the allowed import directory "
                f"({root}). Set PAGEINDEX_IMPORT_DIR or move the file there."
            )

    return resolved


def _ensure_store():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    STRUCTS_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_PATH.exists():
        MANIFEST_PATH.write_text(json.dumps({"documents": []}, indent=2))


def _load_manifest() -> dict:
    _ensure_store()
    return json.loads(MANIFEST_PATH.read_text())


def _save_manifest(manifest: dict):
    _ensure_store()
    fd, tmp_path = tempfile.mkstemp(dir=str(STORE_PATH), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp_path, str(MANIFEST_PATH))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _find_document(manifest: dict, name_or_id: str) -> dict | None:
    for doc in manifest.get("documents", []):
        if doc["id"] == name_or_id or doc["name"] == name_or_id:
            return doc
    return None


def _count_nodes(structure: list) -> int:
    count = 0
    for node in structure:
        count += 1
        if "nodes" in node:
            count += _count_nodes(node["nodes"])
    return count


def _count_pdf_pages(pdf_path: str) -> int:
    doc = pymupdf.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


async def _index_one(file_path: str, dest_file, ext: str):
    name = os.path.basename(file_path)
    logger.info("[%s] Starting indexing (type=%s)", name, ext)
    t0 = time.time()

    if ext == ".pdf":
        from pageindex import page_index
        structure_data = await asyncio.to_thread(
            page_index, str(dest_file),
            if_add_node_id="yes", if_add_node_summary="yes",
            if_add_doc_description="yes", if_add_node_text="no",
        )
        num_pages = _count_pdf_pages(str(dest_file))
    else:
        from pageindex.page_index_md import md_to_tree
        structure_data = await md_to_tree(
            md_path=str(dest_file),
            if_add_node_summary="yes", summary_token_threshold=200,
            if_add_doc_description="yes", if_add_node_text="no",
            if_add_node_id="yes",
        )
        num_pages = 0

    logger.info("[%s] Indexing complete in %.1fs", name, time.time() - t0)
    return structure_data, num_pages


def _tree_to_slim(structure: list) -> list:
    result = []
    for node in structure:
        entry = {"title": node.get("title", ""), "node_id": node.get("node_id", "")}
        summary = node.get("summary", node.get("prefix_summary", ""))
        if summary:
            entry["summary"] = summary[:300] + "..." if len(summary) > 300 else summary
        if node.get("start_index") is not None:
            entry["pages"] = f"{node['start_index']}-{node.get('end_index', '?')}"
        if "nodes" in node:
            entry["children"] = _tree_to_slim(node["nodes"])
        result.append(entry)
    return result


def _find_node(structure: list, node_id: str) -> dict | None:
    for node in structure:
        if node.get("node_id") == node_id:
            return node
        if "nodes" in node:
            found = _find_node(node["nodes"], node_id)
            if found:
                return found
    return None


def _collect_line_nums(structure: list, result: list):
    for node in structure:
        if "line_num" in node:
            result.append(node["line_num"])
        if "nodes" in node:
            _collect_line_nums(node["nodes"], result)


def _get_node_text_md(doc_name: str, node_id: str, structure: list) -> str:
    all_line_nums = []
    _collect_line_nums(structure, all_line_nums)
    all_line_nums.sort()

    node = _find_node(structure, node_id)
    if not node or "line_num" not in node:
        return node.get("summary", node.get("prefix_summary", "")) if node else ""

    start_line = node["line_num"]
    idx = all_line_nums.index(start_line) if start_line in all_line_nums else -1
    end_line = all_line_nums[idx + 1] if idx >= 0 and idx + 1 < len(all_line_nums) else None

    doc_path = DOCS_DIR / doc_name
    if not doc_path.exists():
        return node.get("summary", node.get("prefix_summary", ""))

    lines = doc_path.read_text(errors="replace").splitlines()
    start = max(0, start_line - 1)
    end = (end_line - 1) if end_line else len(lines)
    section_text = "\n".join(lines[start:end]).strip()
    if len(section_text) > 2000:
        section_text = section_text[:2000] + "\n[... truncated]"
    return section_text


mcp = FastMCP(
    "PageIndex Knowledge Store",
    instructions=(
        "Document knowledge store powered by PageIndex. "
        "Workflow: list_documents -> search_documents -> get_document_tree -> get_page_text"
    ),
)


@mcp.tool()
async def add_document(file_path: str) -> str:
    """Add a PDF or Markdown document to the knowledge store.

    Copies the file, runs PageIndex to build a hierarchical index, and
    registers it in the manifest.

    Args:
        file_path: Absolute path to a .pdf, .md, or .markdown file. If
            PAGEINDEX_IMPORT_DIR is set, the path must resolve inside it.
    """
    try:
        resolved = _resolve_import_path(file_path)
    except ValueError as e:
        return f"Error: {e}"

    ext = resolved.suffix.lower()
    file_path = str(resolved)

    _ensure_store()
    base_name = resolved.name
    doc_id = uuid.uuid4().hex[:12]
    stored_name = f"{doc_id}_{base_name}"

    async with _manifest_lock:
        manifest = _load_manifest()
        for doc in manifest["documents"]:
            if doc["name"] == base_name:
                return f"Error: '{base_name}' already exists (id={doc['id']}). Remove it first."

    dest_file = DOCS_DIR / stored_name
    shutil.copy2(file_path, dest_file)

    try:
        structure_data, num_pages = await _index_one(file_path, dest_file, ext)
    except Exception as e:
        dest_file.unlink(missing_ok=True)
        return f"Error during indexing: {e}"

    struct_file = f"{doc_id}_structure.json"
    (STRUCTS_DIR / struct_file).write_text(
        json.dumps(structure_data, indent=2, ensure_ascii=False))

    num_nodes = _count_nodes(structure_data.get("structure", []))

    async with _manifest_lock:
        manifest = _load_manifest()
        for doc in manifest["documents"]:
            if doc["name"] == base_name:
                dest_file.unlink(missing_ok=True)
                (STRUCTS_DIR / struct_file).unlink(missing_ok=True)
                return f"Error: '{base_name}' was added concurrently (id={doc['id']}). Remove it first."
        manifest["documents"].append({
            "id": doc_id, "name": base_name, "type": ext.lstrip("."),
            "added_at": datetime.now(timezone.utc).isoformat(),
            "structure_file": struct_file, "original_file": stored_name,
            "num_nodes": num_nodes, "num_pages": num_pages,
            "description": structure_data.get("doc_description", ""),
        })
        _save_manifest(manifest)

    return f"Indexed: {base_name} (id={doc_id}, {num_nodes} nodes, {num_pages} pages)"


@mcp.tool()
async def add_documents(file_paths: list[str], ctx: Context = None) -> str:
    """Add multiple documents to the knowledge store concurrently.

    Args:
        file_paths: List of absolute paths to .pdf, .md, or .markdown files.
            If PAGEINDEX_IMPORT_DIR is set, each path must resolve inside it.
    """
    if not file_paths:
        return "Error: file_paths list is empty."

    _ensure_store()
    manifest = _load_manifest()
    existing_names = {doc["name"] for doc in manifest["documents"]}

    work = []
    results = []
    for path in file_paths:
        base_name = os.path.basename(os.path.expanduser(path))
        try:
            resolved = _resolve_import_path(path)
        except ValueError as e:
            results.append(f"[{base_name}] Error: {e}")
            continue
        ext = resolved.suffix.lower()
        base_name = resolved.name
        if base_name in existing_names:
            results.append(f"[{base_name}] Error: already exists")
            continue
        doc_id = uuid.uuid4().hex[:12]
        stored_name = f"{doc_id}_{base_name}"
        dest_file = DOCS_DIR / stored_name
        shutil.copy2(str(resolved), dest_file)
        existing_names.add(base_name)
        work.append((str(resolved), base_name, ext, doc_id, dest_file, stored_name))

    if not work:
        return "\n".join(results)

    try:
        concurrency = int(os.environ.get("PAGEINDEX_CONCURRENCY", "4"))
        concurrency = max(1, min(concurrency, 32))
    except ValueError:
        concurrency = 4
    sem = asyncio.Semaphore(concurrency)
    completed = 0
    total = len(work)

    async def _limited(file_path, dest_file, ext):
        nonlocal completed
        async with sem:
            result = await _index_one(file_path, dest_file, ext)
            completed += 1
            if ctx:
                await ctx.report_progress(completed, total)
            return result

    tasks = [_limited(w[0], w[4], w[2]) for w in work]  # file_path, dest_file, ext
    indexed = await asyncio.gather(*tasks, return_exceptions=True)

    async with _manifest_lock:
        manifest = _load_manifest()
        live_names = {doc["name"] for doc in manifest["documents"]}
        for (path, base_name, ext, doc_id, dest_file, stored_name), outcome in zip(work, indexed):
            if isinstance(outcome, BaseException):
                dest_file.unlink(missing_ok=True)
                results.append(f"[{base_name}] Error: {outcome}")
                continue
            if base_name in live_names:
                dest_file.unlink(missing_ok=True)
                results.append(f"[{base_name}] Error: added concurrently")
                continue
            structure_data, num_pages = outcome
            struct_file = f"{doc_id}_structure.json"
            (STRUCTS_DIR / struct_file).write_text(
                json.dumps(structure_data, indent=2, ensure_ascii=False))
            num_nodes = _count_nodes(structure_data.get("structure", []))
            manifest["documents"].append({
                "id": doc_id, "name": base_name, "type": ext.lstrip("."),
                "added_at": datetime.now(timezone.utc).isoformat(),
                "structure_file": struct_file, "original_file": stored_name,
                "num_nodes": num_nodes, "num_pages": num_pages,
                "description": structure_data.get("doc_description", ""),
            })
            live_names.add(base_name)
            results.append(f"[{base_name}] OK: id={doc_id}, {num_nodes} nodes")
        _save_manifest(manifest)

    succeeded = sum(1 for r in results if "] OK:" in r)
    failed = len(results) - succeeded
    summary = f"Done: {succeeded} indexed, {failed} failed out of {len(file_paths)} files."
    if failed:
        errors = [r for r in results if "Error" in r]
        summary += "\n\nErrors:\n" + "\n".join(errors)
    return summary


@mcp.tool()
def list_documents() -> str:
    """List all documents in the knowledge store."""
    manifest = _load_manifest()
    docs = manifest.get("documents", [])
    if not docs:
        return "The knowledge store is empty. Use add_document to index a file."

    lines = [f"Documents ({len(docs)}):\n"]
    for doc in docs:
        desc = doc.get("description", "")
        desc_line = f"  Description: {desc}\n" if desc else ""
        lines.append(
            f"- [{doc['id']}] {doc['name']} ({doc['type']})\n"
            f"  Nodes: {doc['num_nodes']}, Pages: {doc.get('num_pages', 'N/A')}\n"
            f"{desc_line}"
            f"  Added: {doc['added_at']}"
        )
    return "\n".join(lines)


@mcp.tool()
def get_document_tree(document_name: str) -> str:
    """Get the full hierarchical tree structure of an indexed document.

    Args:
        document_name: Filename (e.g. 'report.pdf') or document ID.
    """
    manifest = _load_manifest()
    doc_entry = _find_document(manifest, document_name)
    if doc_entry is None:
        return f"Error: '{document_name}' not found. Use list_documents to see available documents."

    struct_path = STRUCTS_DIR / doc_entry["structure_file"]
    if not struct_path.exists():
        return f"Error: structure file missing for '{document_name}'."

    return json.dumps(json.loads(struct_path.read_text()), indent=2, ensure_ascii=False)


@mcp.tool()
async def search_documents(query: str) -> str:
    """Search the knowledge store using reasoning-based retrieval.

    Two-step process: first selects relevant documents, then navigates
    their tree structures to find matching sections.

    Args:
        query: Natural language question or search query.
    """
    from claude_backend import _call_llm_async

    manifest = _load_manifest()
    docs = manifest.get("documents", [])
    if not docs:
        return "The knowledge store is empty."

    # Step 1: Document selection
    doc_list = [{"doc_id": d["id"], "doc_name": d["name"],
                 "doc_description": d.get("description", "")} for d in docs]

    doc_select_prompt = (
        "Select documents relevant to this query. "
        f"Query: {query}\n\n"
        f"Documents: {json.dumps(doc_list, indent=2, ensure_ascii=False)}\n\n"
        'Reply as JSON: {"thinking": "...", "answer": ["doc_id1", ...]}\n'
        "Select up to 5. Return only JSON."
    )

    doc_select_response = await _call_llm_async(doc_select_prompt)
    try:
        cleaned = _strip_code_fences(doc_select_response)
        selected = json.loads(cleaned)
        selected_ids = selected.get("answer", [])
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Document selection parse failed: %s", e)
        return "Search failed: could not parse document selection."

    if not selected_ids:
        return f"No documents found relevant to: {query}"

    # Step 2: Tree search within selected documents
    selected_docs = [d for d in docs if d["id"] in selected_ids]
    all_retrieved = []

    for doc in selected_docs:
        struct_path = STRUCTS_DIR / doc["structure_file"]
        if not struct_path.exists():
            continue
        data = json.loads(struct_path.read_text())
        structure = data.get("structure", [])
        if not structure:
            continue

        slim_tree = _tree_to_slim(structure)
        tree_prompt = (
            f"Find nodes relevant to this query in the document tree.\n"
            f"Query: {query}\n\nDocument: {doc['name']}\n"
            f"Tree: {json.dumps(slim_tree, indent=2, ensure_ascii=False)}\n\n"
            'Reply as JSON: {"thinking": "...", "node_list": ["node_id1", ...]}\n'
            "Select the most specific nodes. Return only JSON."
        )

        tree_response = await _call_llm_async(tree_prompt)
        try:
            cleaned = _strip_code_fences(tree_response)
            node_ids = json.loads(cleaned).get("node_list", [])
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Node selection parse failed for %s: %s", doc["name"], e)
            continue

        # Step 3: Retrieve content from selected nodes
        for nid in node_ids:
            node = _find_node(structure, nid)
            if not node:
                continue
            title = node.get("title", "")
            if doc["type"] in ("md", "markdown"):
                text = _get_node_text_md(doc["original_file"], nid, structure)
            else:
                summary = node.get("summary", node.get("prefix_summary", ""))
                start = node.get("start_index", "?")
                end = node.get("end_index", "?")
                text = (f"{summary}\n\n"
                        f'[Use get_page_text("{doc["name"]}", {start}, {end}) for full text]')
            all_retrieved.append({"doc": doc["name"], "section": title,
                                  "node_id": nid, "content": text})

    if not all_retrieved:
        return f"No relevant sections found for: {query}"

    # Step 4: Synthesize answer
    context_parts = [f"[{r['doc']}] {r['section']}:\n{r['content']}"
                     for r in all_retrieved]
    answer_prompt = (
        f"Answer based on these document sections.\n\n"
        f"Question: {query}\n\n"
        f"Sections:\n\n{'---'.join(context_parts)}\n\n"
        "Cite sources. If information is insufficient, say so."
    )
    answer = await _call_llm_async(answer_prompt)
    sources = "\n".join(f"  - [{r['doc']}] {r['section']}" for r in all_retrieved)
    return f"{answer}\n\n---\nSources:\n{sources}"


@mcp.tool()
def get_page_text(document_name: str, start_page: int, end_page: int) -> str:
    """Extract text from specific pages of a PDF in the store.

    Args:
        document_name: Filename or document ID.
        start_page: First page (1-indexed, inclusive).
        end_page: Last page (1-indexed, inclusive).
    """
    manifest = _load_manifest()
    doc_entry = _find_document(manifest, document_name)
    if doc_entry is None:
        return f"Error: '{document_name}' not found."
    if doc_entry["type"] != "pdf":
        return "Error: get_page_text only works with PDF documents."

    doc_path = DOCS_DIR / doc_entry["original_file"]
    if not doc_path.exists():
        return f"Error: file missing for '{document_name}'."

    doc = pymupdf.open(str(doc_path))
    total_pages = len(doc)
    start_page = max(1, start_page)
    end_page = min(total_pages, end_page)

    if end_page < start_page:
        doc.close()
        return f"Error: end_page ({end_page}) is before start_page ({start_page})."

    if start_page > total_pages:
        doc.close()
        return f"Error: start_page {start_page} exceeds document length ({total_pages} pages)."

    pages_text = []
    for page_num in range(start_page - 1, end_page):
        pages_text.append(f"--- Page {page_num + 1} ---\n{doc[page_num].get_text()}")
    doc.close()
    return "\n\n".join(pages_text)


@mcp.tool()
async def remove_document(document_name: str) -> str:
    """Remove a document and its index from the knowledge store.

    Args:
        document_name: Filename or document ID.
    """
    async with _manifest_lock:
        manifest = _load_manifest()
        doc_entry = _find_document(manifest, document_name)
        if doc_entry is None:
            return f"Error: '{document_name}' not found."

        manifest["documents"] = [d for d in manifest["documents"] if d["id"] != doc_entry["id"]]
        _save_manifest(manifest)

        doc_file = DOCS_DIR / doc_entry["original_file"]
        struct_file = STRUCTS_DIR / doc_entry["structure_file"]
        for path in (doc_file, struct_file):
            try:
                path.unlink(missing_ok=True)
            except OSError as e:
                logger.warning("Orphaned file after remove_document: %s (%s)", path, e)

    return f"Removed '{doc_entry['name']}' (id={doc_entry['id']})."


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = "\n".join(text.split("\n")[:-1])
    return text.strip()


class _StdoutShim:
    """Proxy stdout: Python-level writes go to stderr; .buffer keeps the
    original stdout buffer so MCP's stdio transport can still write protocol
    messages. Used under --transport stdio to stop PageIndex's progress
    prints from corrupting JSON-RPC on stdout."""

    def __init__(self, original_stdout, write_sink):
        self._buffer = original_stdout.buffer
        self._sink = write_sink

    @property
    def buffer(self):
        return self._buffer

    def write(self, s):
        return self._sink.write(s)

    def flush(self):
        return self._sink.flush()

    def isatty(self):
        return False

    def writable(self):
        return True

    def readable(self):
        return False

    @property
    def encoding(self):
        return self._sink.encoding

    def fileno(self):
        return self._sink.fileno()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PageIndex MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    _setup_logging()
    _ensure_store()

    if args.transport == "sse":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="sse")
    else:
        sys.stdout = _StdoutShim(sys.stdout, sys.stderr)
        mcp.run(transport="stdio")
