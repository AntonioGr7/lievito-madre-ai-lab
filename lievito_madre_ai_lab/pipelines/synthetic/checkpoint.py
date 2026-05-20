"""Append-only JSONL checkpoint store for resumable LLM pipelines.

LLM-driven generation is the most expensive step in the lab — a 10k-doc
job is many dollars and many minutes of wall-clock — so a crash 90% of the
way through must not throw away the work. Each generator stage owns one
checkpoint file under ``output_dir/_checkpoints/``: every successful LLM
call appends one record keyed by a stable identifier (chunk_id, group_id,
or row_id). On resume, the stage loads the known keys, skips items already
present, and processes only the remainder.

Crash-safety is best-effort, not transactional:

- Writes are append-only and flushed after each batch. The OS can still
  truncate a final partial line on hard kill, but we tolerate that on load
  by skipping any unparseable line.
- One file per stage and one writer per file — no concurrent-writer
  protection needed because all writes happen sequentially after each
  ``asyncio.gather`` batch returns.

Not designed for cross-process or cross-machine resumption — the assumption
is the same script restarts on the same machine.
"""
from __future__ import annotations

import json
from pathlib import Path


class CheckpointStore:
    """Append-only JSONL store keyed by ``key_field``.

    On instantiation the file is scanned (line by line) to populate the
    in-memory ``_done`` set. Subsequent ``is_done(key)`` checks are O(1).
    A corrupt trailing line (from a hard kill mid-write) is silently
    skipped — losing one batch of work is much cheaper than refusing to
    resume.
    """

    def __init__(self, path: str | Path, *, key_field: str):
        self.path = Path(path)
        self.key_field = key_field
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done: set[str] = set()
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        # Corrupt tail line — common after hard kill. Skip silently.
                        continue
                    key = rec.get(key_field)
                    if key is not None:
                        self._done.add(str(key))

    def is_done(self, key: str) -> bool:
        return str(key) in self._done

    def n_done(self) -> int:
        return len(self._done)

    def append_many(self, records: list[dict]) -> None:
        """Append a batch of records, all of which must carry ``key_field``.

        Done as a single ``open(..., 'a')`` so the page cache flushes once
        per batch rather than per line — important on large jobs where the
        checkpoint file grows to tens of thousands of lines.
        """
        if not records:
            return
        with open(self.path, "a") as f:
            for record in records:
                key = record.get(self.key_field)
                if key is None:
                    raise ValueError(
                        f"record missing key_field {self.key_field!r}: {record}"
                    )
                f.write(json.dumps(record) + "\n")
                self._done.add(str(key))
            f.flush()

    def load_all(self) -> list[dict]:
        """Return every record on disk (including those written this run).

        Used by callers at end-of-stage to assemble the consolidated
        result from past + current runs.
        """
        records: list[dict] = []
        if not self.path.exists():
            return records
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def clear(self) -> None:
        """Delete the underlying file and reset state. Used by ``--fresh``."""
        if self.path.exists():
            self.path.unlink()
        self._done.clear()
