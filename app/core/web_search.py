"""
Web search helpers.

Default provider: DuckDuckGo.
Optional providers:
- Tavily when an API key is configured
- duckduckgo-search package when installed

The DuckDuckGo path works even without extra Python packages by using the
HTML results page as a lightweight fallback.
"""

from __future__ import annotations

from html import unescape
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

import requests

from app.config import (
    DUCKDUCKGO_MAX_RESULTS,
    DUCKDUCKGO_REGION,
    TAVILY_API_KEY,
    WEB_SEARCH_PROVIDER,
    WEB_SEARCH_TIMEOUT,
)

_VALID_WEB_SEARCH_PROVIDERS = {"auto", "duckduckgo", "tavily"}
_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
}


def get_web_search_provider() -> str:
    provider = (WEB_SEARCH_PROVIDER or "").strip().lower()
    if provider in _VALID_WEB_SEARCH_PROVIDERS:
        return provider
    return "duckduckgo"


def _provider_order(explicit_provider: Optional[str] = None) -> List[str]:
    provider = (explicit_provider or get_web_search_provider()).strip().lower()
    if provider not in _VALID_WEB_SEARCH_PROVIDERS:
        provider = get_web_search_provider()

    if provider == "tavily":
        return ["tavily", "duckduckgo"]
    if provider == "auto":
        return ["tavily", "duckduckgo"] if TAVILY_API_KEY else ["duckduckgo", "tavily"]
    return ["duckduckgo", "tavily"]


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
            if item.get("url")
        ]
    except Exception:
        return []


def _normalize_duckduckgo_link(link: str) -> str:
    normalized = (link or "").strip()
    if not normalized:
        return ""

    if normalized.startswith("//"):
        normalized = f"https:{normalized}"

    parsed = urlparse(normalized)
    if "duckduckgo.com" not in parsed.netloc:
        return normalized

    if parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return unquote(target[0])
    return normalized


class _DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self, max_results: int):
        super().__init__()
        self.max_results = max_results
        self.results: List[Dict[str, str]] = []
        self._in_result_link = False
        self._in_snippet = False
        self._current_title: List[str] = []
        self._current_snippet: List[str] = []
        self._current_link = ""

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]):
        attr_map = dict(attrs)
        classes = (attr_map.get("class") or "").split()

        if tag == "a" and (
            "result__a" in classes
            or "result-link" in classes
            or attr_map.get("data-testid") == "result-title-a"
        ):
            self._flush_pending()
            self._in_result_link = True
            self._current_title = []
            self._current_snippet = []
            self._current_link = _normalize_duckduckgo_link(attr_map.get("href") or "")
            return

        if tag in {"a", "div", "span"} and (
            "result__snippet" in classes or "result-snippet" in classes
        ):
            self._in_snippet = True

    def handle_endtag(self, tag: str):
        if tag == "a" and self._in_result_link:
            self._in_result_link = False
            return

        if tag in {"a", "div", "span"} and self._in_snippet:
            self._in_snippet = False

    def handle_data(self, data: str):
        text = (data or "").strip()
        if not text:
            return

        if self._in_result_link:
            self._current_title.append(text)
        elif self._in_snippet:
            self._current_snippet.append(text)

    def close(self):
        self._flush_pending()
        super().close()

    def _flush_pending(self):
        if len(self.results) >= self.max_results:
            return

        title = unescape(" ".join(self._current_title)).strip()
        snippet = unescape(" ".join(self._current_snippet)).strip()
        if title and self._current_link:
            self.results.append(
                {
                    "title": title,
                    "snippet": snippet,
                    "link": self._current_link,
                    "source": "duckduckgo",
                }
            )
        self._current_title = []
        self._current_snippet = []
        self._current_link = ""


def _search_duckduckgo_package(
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
            link = _normalize_duckduckgo_link(item.get("href", ""))
            if not link:
                continue
            results.append(
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("body", ""),
                    "link": link,
                    "source": "duckduckgo",
                }
            )
    return results


def _search_duckduckgo_html(
    query: str,
    max_results: int = 5,
    region: Optional[str] = None,
) -> List[Dict[str, Any]]:
    response = requests.get(
        "https://html.duckduckgo.com/html/",
        params={
            "q": query,
            "kl": region or DUCKDUCKGO_REGION,
        },
        headers=_DEFAULT_HEADERS,
        timeout=WEB_SEARCH_TIMEOUT,
    )
    response.raise_for_status()

    parser = _DuckDuckGoHTMLParser(max_results=max_results)
    parser.feed(response.text)
    parser.close()
    return parser.results[:max_results]


def _search_duckduckgo(
    query: str,
    max_results: int = 5,
    region: Optional[str] = None,
) -> List[Dict[str, Any]]:
    try:
        results = _search_duckduckgo_package(
            query=query,
            max_results=max_results,
            region=region,
        )
        if results:
            return results
    except Exception:
        pass

    try:
        return _search_duckduckgo_html(
            query=query,
            max_results=max_results,
            region=region,
        )
    except Exception:
        return []


def search_web(
    query: str,
    max_results: int = 5,
    region: Optional[str] = None,
    provider: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search web results using the configured provider order.

    Provider values:
    - duckduckgo
    - tavily
    - auto
    """
    max_results = max(1, max_results or DUCKDUCKGO_MAX_RESULTS)
    effective_region = region or DUCKDUCKGO_REGION

    for current_provider in _provider_order(provider):
        if current_provider == "duckduckgo":
            results = _search_duckduckgo(
                query=query,
                max_results=max_results,
                region=effective_region,
            )
        else:
            results = _search_tavily(query=query, max_results=max_results)

        if results:
            return results

    return []


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
