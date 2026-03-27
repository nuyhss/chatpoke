"""
Background file watcher — monitors the persistent library folder.

Drop a PDF or image into data/library/ and it gets automatically parsed,
chunked, and stored in ChromaDB. Chat uploads in data/uploads/ are not
watched, so temporary user files do not get re-ingested on restart.
"""

import time
import threading
import logging
from pathlib import Path
from typing import Set, Dict

from app.config import LIBRARY_DIR

logger = logging.getLogger("tilon.watcher")

WATCH_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
POLL_INTERVAL = 5  # seconds


class FileWatcher:
    """Watches the library folder for new files and auto-ingests them."""

    def __init__(self):
        self._known_files: Set[str] = set()
        self._suppressed_files: Dict[str, float] = {}
        self._thread: threading.Thread = None
        self._running = False

    def start(self):
        """Start watching in a background thread."""
        self._running = True
        self._known_files = self._scan_existing()
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info(
            "File watcher started — monitoring %s (%d existing files)",
            LIBRARY_DIR, len(self._known_files),
        )

    def stop(self):
        """Stop the watcher."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("File watcher stopped.")

    def _scan_existing(self) -> Set[str]:
        """Get set of files currently in the library folder."""
        LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
        files = set()
        for ext in WATCH_EXTENSIONS:
            for f in LIBRARY_DIR.glob(f"*{ext}"):
                files.add(str(f))
        return files

    def _watch_loop(self):
        """Poll for new files every POLL_INTERVAL seconds."""
        while self._running:
            try:
                self._prune_suppressed()
                current_files = self._scan_existing()
                new_files = current_files - self._known_files

                for file_path in sorted(new_files):
                    self._ingest_file(Path(file_path))

                self._known_files = current_files

            except Exception as e:
                logger.error("File watcher error: %s", e)

            time.sleep(POLL_INTERVAL)

    def suppress(self, file_path: Path, ttl_seconds: int = 90):
        """Temporarily suppress watcher ingestion for a file handled elsewhere."""
        self._suppressed_files[file_path.name] = time.time() + ttl_seconds

    def _prune_suppressed(self):
        now = time.time()
        expired = [name for name, expires_at in self._suppressed_files.items() if expires_at <= now]
        for name in expired:
            self._suppressed_files.pop(name, None)

    def _ingest_file(self, file_path: Path):
        """Ingest a single new file, skip if already in vectorstore."""
        time.sleep(1)

        try:
            from app.pipeline.ingest import ingest_single_file
            from app.core.vectorstore import get_ingested_sources

            suppressed_until = self._suppressed_files.get(file_path.name)
            if suppressed_until and suppressed_until > time.time():
                logger.debug("Watcher skipping %s — recently handled by upload path", file_path.name)
                return

            # Skip if already ingested (e.g., by /upload endpoint)
            already = get_ingested_sources()
            if file_path.name in already:
                logger.debug("Watcher skipping %s — already ingested", file_path.name)
                return

            logger.info("Auto-ingesting new file: %s", file_path.name)
            result = ingest_single_file(file_path)
            logger.info(
                "Auto-ingest complete: %s → %d chunks",
                file_path.name, result.get("count", 0),
            )
        except Exception as e:
            logger.error("Auto-ingest failed for %s: %s", file_path.name, e)


# Global watcher instance
_watcher = FileWatcher()


def start_watcher():
    _watcher.start()


def stop_watcher():
    _watcher.stop()


def suppress_watcher_for(file_path: Path, ttl_seconds: int = 90):
    _watcher.suppress(file_path, ttl_seconds=ttl_seconds)
