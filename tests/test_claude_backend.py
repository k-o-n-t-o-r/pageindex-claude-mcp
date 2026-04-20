#!/usr/bin/env python3
"""Tests for the Claude backend and MCP server integration."""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "vendor" / "PageIndex"))


def mock_llm_response(prompt_text, max_retries=5):
    if "detect if there is a table of content" in prompt_text:
        if "Table of Contents" in prompt_text or "Chapter" in prompt_text:
            return json.dumps({"thinking": "TOC found.", "toc_detected": "yes"})
        return json.dumps({"thinking": "No TOC.", "toc_detected": "no"})
    if "check if" in prompt_text.lower() and "complete" in prompt_text.lower():
        return json.dumps({"thinking": "Complete.", "completed": "yes"})
    if "detect if there are page numbers" in prompt_text:
        return json.dumps({"thinking": "Yes.", "page_index_given_in_toc": "yes"})
    if "extract the full table of contents" in prompt_text.lower():
        return ("Chapter 1: Introduction : 1\nChapter 2: Methods : 3\n"
                "Chapter 3: Results : 5\nChapter 4: Conclusion : 7")
    if "transform the whole table of content into a JSON" in prompt_text:
        return json.dumps({"table_of_contents": [
            {"structure": "1", "title": "Introduction", "page": 1},
            {"structure": "2", "title": "Methods", "page": 3},
            {"structure": "3", "title": "Results", "page": 5},
            {"structure": "4", "title": "Conclusion", "page": 7},
        ]})
    if "add the physical_index" in prompt_text:
        return json.dumps([
            {"structure": "1", "title": "Introduction", "physical_index": "<physical_index_3>"},
            {"structure": "2", "title": "Methods", "physical_index": "<physical_index_5>"},
            {"structure": "3", "title": "Results", "physical_index": "<physical_index_7>"},
            {"structure": "4", "title": "Conclusion", "physical_index": "<physical_index_9>"},
        ])
    if "fix" in prompt_text.lower() and "index" in prompt_text.lower():
        return json.dumps({"physical_index": "<physical_index_3>"})
    if "generate a description of the partial document" in prompt_text:
        return "Covers key concepts and findings."
    if "generate a one-sentence description for the document" in prompt_text:
        return "A test document with four chapters."
    if "check if the given section appears or starts" in prompt_text:
        return json.dumps({"thinking": "Section found.", "answer": "yes"})
    if "check if the current section starts in the beginning" in prompt_text:
        return json.dumps({"thinking": "Starts at beginning.", "start_begin": "yes"})
    return json.dumps({"completed": "yes"})


async def _async_mock(prompt_text, max_retries=5):
    return mock_llm_response(prompt_text)


def create_test_pdf(path, num_pages=8):
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 200), "Test Research Document", fontsize=24)
    page = doc.new_page()
    page.insert_text((72, 72),
        "Table of Contents\n\n"
        "Chapter 1: Introduction .............. 1\n"
        "Chapter 2: Methods ................... 3\n"
        "Chapter 3: Results ................... 5\n"
        "Chapter 4: Conclusion ................ 7\n", fontsize=12)
    chapters = [
        ("Introduction", "Background and research questions."),
        ("Introduction (cont)", "Related work review."),
        ("Methods", "Experimental methodology."),
        ("Methods (cont)", "Statistical analysis."),
        ("Results", "Main findings."),
        ("Results (cont)", "Sensitivity analysis."),
    ]
    for title, content in chapters[:num_pages - 2]:
        page = doc.new_page()
        page.insert_text((72, 72), title, fontsize=18)
        page.insert_text((72, 120), content, fontsize=12)
        for i in range(5):
            page.insert_text((72, 160 + i * 30),
                f"Lorem ipsum dolor sit amet. Paragraph {i + 1}.", fontsize=11)
    doc.save(path)
    doc.close()


def create_test_markdown(path):
    Path(path).write_text(
        "# Test Document\n\n"
        "## Chapter 1: Getting Started\n\nBasics of the system.\n\n"
        "### 1.1 Installation\n\nInstall steps.\n\n"
        "### 1.2 Configuration\n\nConfig details.\n\n"
        "## Chapter 2: Advanced Topics\n\n"
        "### 2.1 Performance\n\nTuning tips.\n\n"
        "### 2.2 Troubleshooting\n\nCommon issues.\n\n"
        "## Chapter 3: Reference\n\nAPI docs.\n"
    )


def test_monkey_patching():
    import claude_backend
    claude_backend._activated = False
    claude_backend.activate()
    utils = sys.modules["pageindex.utils"]
    pi = sys.modules["pageindex.page_index"]
    assert utils.llm_completion.__name__ == "claude_llm_completion"
    assert pi.llm_completion.__name__ == "claude_llm_completion"
    assert utils.llm_acompletion.__name__ == "claude_llm_acompletion"
    assert utils.count_tokens.__name__ == "claude_count_tokens"


def test_prompt_building():
    from claude_backend import _build_prompt_text
    assert _build_prompt_text("Hello") == "Hello"
    history = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
    ]
    result = _build_prompt_text("Follow-up", history)
    assert "[System instruction]" in result
    assert "Follow-up" in result


def test_token_counting():
    from claude_backend import claude_count_tokens
    assert claude_count_tokens("") == 0
    assert claude_count_tokens(None) == 0
    assert claude_count_tokens("Hello, world!") > 0


def test_mcp_tools_with_mock_pdf():
    import importlib
    import claude_backend

    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = Path(tmpdir) / "store"
        pdf_path = Path(tmpdir) / "test.pdf"
        create_test_pdf(str(pdf_path))
        os.environ["PAGEINDEX_STORE_PATH"] = str(store_path)

        import mcp_server
        importlib.reload(mcp_server)

        result = mcp_server.list_documents()
        assert "empty" in result.lower()

        with patch.object(claude_backend, '_call_llm', side_effect=mock_llm_response), \
             patch.object(claude_backend, '_call_llm_async', new=_async_mock):
            result = asyncio.run(mcp_server.add_document(str(pdf_path)))
            assert "api_key" not in result.lower(), "OpenAI leak detected"
            assert result.startswith("Indexed:"), f"add_document failed: {result}"
            print(f"  add_document: {result[:80]}...")

        # Test downstream tools against the real indexed document
        base_name = "test.pdf"
        result = mcp_server.get_page_text(base_name, 1, 3)
        assert "Page 1" in result and "Page 2" in result
        result = mcp_server.get_document_tree(base_name)
        assert "structure" in json.loads(result)
        result = asyncio.run(mcp_server.remove_document(base_name))
        assert "Removed" in result


def test_error_handling():
    import importlib
    import mcp_server

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["PAGEINDEX_STORE_PATH"] = str(Path(tmpdir) / "store")
        importlib.reload(mcp_server)

        result = asyncio.run(mcp_server.add_document("/tmp/nonexistent.pdf"))
        assert "Error" in result
        bad_file = Path(tmpdir) / "test.txt"
        bad_file.write_text("hello")
        result = asyncio.run(mcp_server.add_document(str(bad_file)))
        assert "Error" in result
        result = mcp_server.get_page_text("nonexistent.pdf", 1, 1)
        assert "Error" in result


def test_mcp_tools_with_mock_markdown():
    import importlib
    import claude_backend

    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = Path(tmpdir) / "store"
        md_path = Path(tmpdir) / "guide.md"
        create_test_markdown(str(md_path))
        os.environ["PAGEINDEX_STORE_PATH"] = str(store_path)

        import mcp_server
        importlib.reload(mcp_server)

        with patch.object(claude_backend, '_call_llm', side_effect=mock_llm_response), \
             patch.object(claude_backend, '_call_llm_async', new=_async_mock):
            result = asyncio.run(mcp_server.add_document(str(md_path)))
            assert result.startswith("Indexed:"), f"add_document (md) failed: {result}"
            print(f"  add_document (md): {result[:80]}...")

        result = mcp_server.list_documents()
        assert "guide.md" in result

        result = mcp_server.get_document_tree("guide.md")
        parsed = json.loads(result)
        assert "structure" in parsed

        result = asyncio.run(mcp_server.remove_document("guide.md"))
        assert "Removed" in result


def test_search_documents():
    import importlib
    import claude_backend

    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = Path(tmpdir) / "store"
        md_path = Path(tmpdir) / "guide.md"
        create_test_markdown(str(md_path))
        os.environ["PAGEINDEX_STORE_PATH"] = str(store_path)

        import mcp_server
        importlib.reload(mcp_server)

        # Index a document first
        with patch.object(claude_backend, '_call_llm', side_effect=mock_llm_response), \
             patch.object(claude_backend, '_call_llm_async', new=_async_mock):
            result = asyncio.run(mcp_server.add_document(str(md_path)))
            assert result.startswith("Indexed:"), f"Setup failed: {result}"

        # Get real doc_id and node_ids from the indexed document
        manifest = mcp_server._load_manifest()
        doc = manifest["documents"][0]
        doc_id = doc["id"]
        struct_data = json.loads(
            (mcp_server.STRUCTS_DIR / doc["structure_file"]).read_text())
        first_node_id = struct_data["structure"][0].get("node_id", "n0")

        # Build a search-aware mock
        async def search_mock(prompt_text, max_retries=5):
            if "Select documents relevant" in prompt_text:
                return json.dumps({"thinking": "Relevant.", "answer": [doc_id]})
            if "Find nodes relevant" in prompt_text:
                return json.dumps({"thinking": "Found.", "node_list": [first_node_id]})
            if "Answer based on" in prompt_text:
                return "The guide covers getting started with the system."
            return mock_llm_response(prompt_text)

        with patch.object(claude_backend, '_call_llm', side_effect=mock_llm_response), \
             patch.object(claude_backend, '_call_llm_async', new=search_mock):
            result = asyncio.run(mcp_server.search_documents("getting started"))
            assert "Sources:" in result, f"Search missing sources: {result[:200]}"
            assert "guide.md" in result
            print(f"  search result: {result[:120]}...")


def test_add_documents_batch():
    import importlib
    import claude_backend

    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = Path(tmpdir) / "store"
        md1 = Path(tmpdir) / "doc_a.md"
        md2 = Path(tmpdir) / "doc_b.md"
        create_test_markdown(str(md1))
        create_test_markdown(str(md2))
        os.environ["PAGEINDEX_STORE_PATH"] = str(store_path)

        import mcp_server
        importlib.reload(mcp_server)

        with patch.object(claude_backend, '_call_llm', side_effect=mock_llm_response), \
             patch.object(claude_backend, '_call_llm_async', new=_async_mock):
            result = asyncio.run(mcp_server.add_documents([str(md1), str(md2)]))
            assert "2 indexed" in result, f"Batch add failed: {result}"
            print(f"  batch add: {result[:80]}...")

        result = mcp_server.list_documents()
        assert "doc_a.md" in result and "doc_b.md" in result

        # Partial failure: one bad path
        with patch.object(claude_backend, '_call_llm', side_effect=mock_llm_response), \
             patch.object(claude_backend, '_call_llm_async', new=_async_mock):
            md3 = Path(tmpdir) / "doc_c.md"
            create_test_markdown(str(md3))
            result = asyncio.run(mcp_server.add_documents(
                [str(md3), "/tmp/nonexistent.md"]))
            assert "1 indexed" in result and "1 failed" in result


def test_duplicate_document_handling():
    import importlib
    import claude_backend

    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = Path(tmpdir) / "store"
        md_path = Path(tmpdir) / "dup.md"
        create_test_markdown(str(md_path))
        os.environ["PAGEINDEX_STORE_PATH"] = str(store_path)

        import mcp_server
        importlib.reload(mcp_server)

        with patch.object(claude_backend, '_call_llm', side_effect=mock_llm_response), \
             patch.object(claude_backend, '_call_llm_async', new=_async_mock):
            result = asyncio.run(mcp_server.add_document(str(md_path)))
            assert result.startswith("Indexed:"), f"First add failed: {result}"

            # Second add with same name should fail
            result = asyncio.run(mcp_server.add_document(str(md_path)))
            assert "already exists" in result, f"Duplicate not caught: {result}"


def test_invalid_page_ranges():
    import importlib
    import claude_backend

    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = Path(tmpdir) / "store"
        pdf_path = Path(tmpdir) / "pages.pdf"
        create_test_pdf(str(pdf_path))
        os.environ["PAGEINDEX_STORE_PATH"] = str(store_path)

        import mcp_server
        importlib.reload(mcp_server)

        with patch.object(claude_backend, '_call_llm', side_effect=mock_llm_response), \
             patch.object(claude_backend, '_call_llm_async', new=_async_mock):
            result = asyncio.run(mcp_server.add_document(str(pdf_path)))
            assert result.startswith("Indexed:"), f"Setup failed: {result}"

        # end < start (after clamping)
        result = mcp_server.get_page_text("pages.pdf", 5, 2)
        assert "Error" in result

        # Both beyond document length
        result = mcp_server.get_page_text("pages.pdf", 100, 200)
        assert "Error" in result

        # Valid range still works
        result = mcp_server.get_page_text("pages.pdf", 1, 3)
        assert "Page 1" in result


def main():
    tests = [
        ("Monkey-Patching", test_monkey_patching),
        ("Prompt Building", test_prompt_building),
        ("Token Counting", test_token_counting),
        ("Error Handling", test_error_handling),
        ("MCP Tools + Mock PDF", test_mcp_tools_with_mock_pdf),
        ("MCP Tools + Mock Markdown", test_mcp_tools_with_mock_markdown),
        ("Search Documents", test_search_documents),
        ("Batch Add Documents", test_add_documents_batch),
        ("Duplicate Document Handling", test_duplicate_document_handling),
        ("Invalid Page Ranges", test_invalid_page_ranges),
    ]

    passed = failed = 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
            passed += 1
            print("  PASS")
        except Exception as e:
            failed += 1
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
