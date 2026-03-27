"""
In-memory BM25 keyword index for document chunks.

This complements vector search for exact tokens like error codes, commands,
product names, and mixed Korean/English technical terms.
"""

import logging
import math
import re
from collections import Counter
from typing import Iterable, List, Optional, Tuple

from langchain_core.documents import Document

logger = logging.getLogger("tilon.keyword_index")

_TOKEN_RE = re.compile(r"[A-Za-z0-9._-]+|[가-힣]+")
_KOREAN_PARTICLE_SUFFIXES = (
    "으로부터", "에서부터", "에게서는", "한테서는", "으로는", "에게서", "한테서",
    "이라도", "라도", "이라는", "라는", "이라고", "라고", "이랑", "랑", "으로", "에서",
    "에는", "에도", "에게", "한테", "와는", "과는", "와도", "과도", "부터", "까지",
    "처럼", "보다", "마저", "조차", "밖에", "마다", "하고", "이며", "이고", "이다",
    "와", "과", "은", "는", "이", "가", "을", "를", "에", "의", "도", "만", "로",
)


def _expand_korean_token(token: str) -> Iterable[str]:
    lowered = token.lower()
    yield lowered

    if not re.fullmatch(r"[가-힣]+", lowered):
        return

    seen = {lowered}
    candidates = [lowered]

    for _ in range(2):
        next_round = []
        for candidate in candidates:
            for suffix in _KOREAN_PARTICLE_SUFFIXES:
                if not candidate.endswith(suffix):
                    continue
                stem = candidate[:-len(suffix)]
                if len(stem) < 2 or stem in seen:
                    continue
                seen.add(stem)
                next_round.append(stem)
                yield stem
        if not next_round:
            break
        candidates = next_round


def tokenize_text(text: str) -> List[str]:
    """Tokenize Korean/English technical text while preserving codes like E-401."""
    tokens: List[str] = []
    for raw_token in _TOKEN_RE.findall(text or ""):
        tokens.extend(_expand_korean_token(raw_token))
    return tokens


class InMemoryBM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.clear()

    def clear(self) -> None:
        self._documents: List[Document] = []
        self._tokenized_docs: List[List[str]] = []
        self._doc_freq: Counter = Counter()
        self._avg_doc_len = 0.0
        self._total_doc_len = 0

    def rebuild(self, docs: List[Document]) -> None:
        self._documents = list(docs)
        self._tokenized_docs = [tokenize_text(doc.page_content) for doc in self._documents]
        self._doc_freq = Counter()

        self._total_doc_len = 0
        for tokens in self._tokenized_docs:
            self._total_doc_len += len(tokens)
            self._doc_freq.update(set(tokens))

        self._avg_doc_len = (
            self._total_doc_len / len(self._tokenized_docs)
            if self._tokenized_docs
            else 0.0
        )
        logger.info("Keyword index rebuilt with %d chunks", len(self._documents))

    def add_documents(self, docs: List[Document]) -> None:
        """
        Incrementally add docs without rebuilding the entire index.
        This avoids O(N^2) behavior during multi-file uploads.
        """
        if not docs:
            return

        added = 0
        for doc in docs:
            tokens = tokenize_text(doc.page_content)
            self._documents.append(doc)
            self._tokenized_docs.append(tokens)
            self._doc_freq.update(set(tokens))
            self._total_doc_len += len(tokens)
            added += 1

        total_docs = len(self._tokenized_docs)
        self._avg_doc_len = (self._total_doc_len / total_docs) if total_docs else 0.0
        logger.info(
            "Keyword index incrementally updated (+%d chunks, total=%d)",
            added,
            len(self._documents),
        )

    def search(
        self,
        query: str,
        k: int = 4,
        source_filter: Optional[str] = None,
        doc_id_filter: Optional[str] = None,
        source_type_filter: Optional[str] = None,
        owner_id_filter: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        query_tokens = tokenize_text(query)
        if not query_tokens or not self._documents:
            return []

        results: List[Tuple[Document, float]] = []
        total_docs = len(self._documents)

        for doc, doc_tokens in zip(self._documents, self._tokenized_docs):
            if source_filter and doc.metadata.get("source") != source_filter:
                continue
            if doc_id_filter and doc.metadata.get("doc_id") != doc_id_filter:
                continue
            if source_type_filter and doc.metadata.get("source_type") != source_type_filter:
                continue
            if owner_id_filter and doc.metadata.get("source_type") == "upload" and doc.metadata.get("owner_id") != owner_id_filter:
                continue
            if not doc_tokens:
                continue

            score = self._score(query_tokens, doc_tokens, total_docs)
            if score > 0:
                results.append((doc, score))

        results.sort(key=lambda item: item[1], reverse=True)
        return results[:k]

    def _score(self, query_tokens: List[str], doc_tokens: List[str], total_docs: int) -> float:
        term_freq = Counter(doc_tokens)
        doc_len = len(doc_tokens) or 1
        avg_doc_len = self._avg_doc_len or 1.0
        score = 0.0

        for token in query_tokens:
            freq = term_freq.get(token, 0)
            if not freq:
                continue

            doc_freq = self._doc_freq.get(token, 0)
            idf = math.log(1 + ((total_docs - doc_freq + 0.5) / (doc_freq + 0.5)))
            denom = freq + self.k1 * (1 - self.b + self.b * (doc_len / avg_doc_len))
            score += idf * ((freq * (self.k1 + 1)) / denom)

        return score


_keyword_index = InMemoryBM25Index()


def rebuild_keyword_index(docs: List[Document]) -> None:
    _keyword_index.rebuild(docs)


def add_keyword_documents(docs: List[Document]) -> None:
    _keyword_index.add_documents(docs)


def clear_keyword_index() -> None:
    _keyword_index.clear()


def search_keyword_index(
    query: str,
    k: int = 4,
    source_filter: Optional[str] = None,
    doc_id_filter: Optional[str] = None,
    source_type_filter: Optional[str] = None,
    owner_id_filter: Optional[str] = None,
) -> List[Tuple[Document, float]]:
    return _keyword_index.search(
        query=query,
        k=k,
        source_filter=source_filter,
        doc_id_filter=doc_id_filter,
        source_type_filter=source_type_filter,
        owner_id_filter=owner_id_filter,
    )
