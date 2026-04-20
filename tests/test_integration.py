#!/usr/bin/env python3
"""
Integration tests for the PageIndex MCP server.

Tests the full add -> search -> remove workflow against the Claude Agent SDK.

Usage:
    uv run python tests/test_integration.py
"""

import asyncio
import importlib
import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "vendor" / "PageIndex"))

TEST_DOC_CONTENT = """\
# Solar System Overview

## The Sun

The Sun is the star at the center of our solar system. It is a nearly perfect
sphere of hot plasma with a surface temperature of about 5,778 Kelvin.
The Sun accounts for 99.86% of the total mass of the solar system.

## Inner Planets

### Mercury

Mercury is the smallest planet and closest to the Sun. It has no atmosphere
and temperatures range from -180\u00b0C at night to 430\u00b0C during the day.

### Venus

Venus is the second planet from the Sun. It has a thick atmosphere of carbon
dioxide that traps heat, making it the hottest planet at 465\u00b0C.

### Earth

Earth is the third planet from the Sun and the only known planet to support
life. It has liquid water on its surface and a nitrogen-oxygen atmosphere.

### Mars

Mars is the fourth planet, known as the Red Planet due to iron oxide on its
surface. It has the tallest volcano in the solar system, Olympus Mons.

## Outer Planets

### Jupiter

Jupiter is the largest planet in the solar system. It is a gas giant with a
Great Red Spot, a storm larger than Earth that has raged for centuries.

### Saturn

Saturn is famous for its extensive ring system made of ice and rock particles.
It is the second largest planet and has at least 146 known moons.

### Neptune

Neptune is the farthest planet from the Sun. It has the strongest winds in
the solar system, reaching speeds of 2,100 km/h.
"""

SEARCH_QUERY = "What is the hottest planet and why?"
EXPECTED_KEYWORDS = ["venus", "465", "carbon dioxide", "atmosphere"]


def _reset_modules():
    """Force re-import of claude_backend and mcp_server with fresh state."""
    import claude_backend
    claude_backend._activated = False
    importlib.reload(claude_backend)
    claude_backend.activate()
    import mcp_server
    importlib.reload(mcp_server)
    return mcp_server


def main():
    print(f"\n{'='*60}")
    print("  Backend: Claude Agent SDK")
    print(f"{'='*60}")

    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = Path(tmpdir) / "store"
        doc_path = Path(tmpdir) / "solar_system.md"
        doc_path.write_text(TEST_DOC_CONTENT)

        os.environ["PAGEINDEX_STORE_PATH"] = str(store_path)
        try:
            mcp_server = _reset_modules()

            # -- Step 1: Add document --
            print("\n  [1/4] Adding document...")
            result = asyncio.run(mcp_server.add_document(str(doc_path)))
            print(f"        {result}")
            if "Error" in result:
                print("  FAIL: add_document returned an error")
                return 1
            assert "solar_system.md" in result
            print("        OK")

            # -- Step 2: List documents --
            print("\n  [2/4] Listing documents...")
            result = mcp_server.list_documents()
            print(f"        Found {result.count('solar_system.md')} match(es)")
            assert "solar_system.md" in result
            print("        OK")

            # -- Step 3: Search --
            print(f"\n  [3/4] Searching: \"{SEARCH_QUERY}\"")
            result = asyncio.run(mcp_server.search_documents(SEARCH_QUERY))
            result_lower = result.lower()
            print(f"        Response length: {len(result)} chars")

            if "Search failed" in result or result == "Error":
                print(f"  FAIL: search returned error:\n        {result[:200]}")
                return 1

            matched = [kw for kw in EXPECTED_KEYWORDS if kw in result_lower]
            missed = [kw for kw in EXPECTED_KEYWORDS if kw not in result_lower]
            print(f"        Keywords matched: {matched}")
            if missed:
                print(f"        Keywords missed:  {missed}")

            if len(matched) < 2:
                print("  FAIL: search result doesn't contain enough expected keywords")
                print(f"        Result preview: {result[:300]}")
                return 1
            print("        OK")

            # -- Step 4: Remove document --
            print("\n  [4/4] Removing document...")
            result = asyncio.run(mcp_server.remove_document("solar_system.md"))
            print(f"        {result}")
            assert "Removed" in result

            result = mcp_server.list_documents()
            assert "empty" in result.lower()
            print("        OK")

            print("\n  PASSED")
            return 0

        finally:
            os.environ.pop("PAGEINDEX_STORE_PATH", None)


if __name__ == "__main__":
    sys.exit(main())
