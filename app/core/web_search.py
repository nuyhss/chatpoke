"""
Web search helpers.

Primary: Tavily when configured.
Fallback: DuckDuckGo when the optional dependency is installed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.config import DUCKDUCKGO_MAX_RESULTS, DUCKDUCKGO_REGION, TAVILY_API_KEY


def _search_tavily(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    if not TAVILY_API_KEY:
        return []

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query, max_results=max_results)
        return [
            {
                "title": item.get("title", ""),
                "snippet": item.get("content", ""),
                "link": item.get("url", ""),
                "source": "tavily",
            }
            for item in response.get("results", [])
        ]
    except Exception:
        return []


def _search_duckduckgo(
    query: str,
    max_results: int = 5,
    region: Optional[str] = None,
) -> List[Dict[str, Any]]:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return []

    results: List[Dict[str, Any]] = []
    with DDGS() as ddgs:
        for item in ddgs.text(
            query,
            region=region or DUCKDUCKGO_REGION,
            max_results=max_results,
        ) or []:
            results.append(
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("body", ""),
                    "link": item.get("href", ""),
                    "source": "duckduckgo",
                }
            )
    return results


def search_web(
    query: str,
    max_results: int = 5,
    region: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search web results using Tavily first, then DuckDuckGo fallback."""
    results = _search_tavily(query, max_results=max_results)
    if results:
        return results
    return _search_duckduckgo(
        query,
        max_results=max_results or DUCKDUCKGO_MAX_RESULTS,
        region=region or DUCKDUCKGO_REGION,
    )


def format_search_results(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""

    blocks = []
    for item in results:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        source = item.get("source", "")

        meta_line = f"Source: {source}" if source else ""
        if meta_line:
            blocks.append(f"[{title}]\n{snippet}\n{meta_line}\nLink: {link}")
        else:
            blocks.append(f"[{title}]\n{snippet}\nLink: {link}")

    return "\n\n".join(blocks)
