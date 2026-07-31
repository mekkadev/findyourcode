"""SQLite persistence: chunks, vectors (sqlite-vec or brute force) and an FTS5 index."""

from __future__ import annotations

import contextlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunker import Chunk

SCHEMA_VERSION = "1"

# sqlite-vec refuses a knn query above this k; beyond it we answer exactly instead.
VEC_MAX_K = 4096
EXACT_SCAN_LIMIT = 20_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    rel TEXT UNIQUE NOT NULL,
    sha TEXT NOT NULL,
    mtime REAL,
    size INTEGER,
    lang TEXT,
    n_chunks INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    rel TEXT NOT NULL,
    lang TEXT,
    kind TEXT,
    symbol TEXT,
    parent TEXT,
    start_line INTEGER,
    end_line INTEGER,
    code TEXT,
    sha TEXT
);
CREATE INDEX IF NOT EXISTS chunks_file ON chunks(file_id);
CREATE INDEX IF NOT EXISTS chunks_rel ON chunks(rel);

CREATE TABLE IF NOT EXISTS emb_cache (
    sig TEXT NOT NULL,
    sha TEXT NOT NULL,
    vec BLOB NOT NULL,
    PRIMARY KEY (sig, sha)
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
    USING fts5(text, tokenize='unicode61 remove_diacritics 2');
"""


@dataclass
class Filters:
    langs: list[str] | None = None
    paths: list[str] | None = None
    kinds: list[str] | None = None

    def sql(self, prefix: str = "") -> tuple[str, list]:
        clauses, params = [], []
        if self.langs:
            clauses.append(f"{prefix}lang IN ({','.join('?' * len(self.langs))})")
            params += [lang.lower() for lang in self.langs]
        if self.kinds:
            clauses.append(f"{prefix}kind IN ({','.join('?' * len(self.kinds))})")
            params += [kind.lower() for kind in self.kinds]
        if self.paths:
            clauses.append("(" + " OR ".join([f"{prefix}rel LIKE ?"] * len(self.paths)) + ")")
            params += [f"%{p}%" for p in self.paths]
        return (" AND ".join(clauses) if clauses else "1=1"), params

    def active(self) -> bool:
        return bool(self.langs or self.paths or self.kinds)


@dataclass
class Row:
    id: int
    rel: str
    lang: str
    kind: str
    symbol: str
    parent: str
    start_line: int
    end_line: int
    code: str


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        try:
            self.db.execute("SELECT count(*) FROM sqlite_master").fetchone()
        except sqlite3.DatabaseError as exc:
            raise SystemExit(
                f"{self.path} is not a readable index ({exc}).\n"
                f"Delete it with `fyc clear` and index again."
            ) from exc
        self.db.execute("PRAGMA auto_vacuum=INCREMENTAL")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(_SCHEMA)
        self._vec_ok = _load_sqlite_vec(self.db)
        self._matrix: tuple[np.ndarray, np.ndarray] | None = None

    # ---- metadata -----------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: object) -> None:
        self.db.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )

    def prepare(self, signature: str, dim: int, force: bool = False) -> None:
        """Bind the index to one embedding model; refuse to mix vector spaces."""
        current = self.get_meta("signature")
        if current and current != signature and not force:
            raise SystemExit(
                f"index was built with '{current}', now using '{signature}'.\n"
                f"Run `fyc index --reindex` to rebuild."
            )
        self.set_meta("signature", signature)
        self.set_meta("dim", dim)
        self.set_meta("schema", SCHEMA_VERSION)
        backend = "vec0" if self._vec_ok else "numpy"
        stored = self.get_meta("backend")
        if stored and stored != backend:
            self.reset_vectors()
        self.set_meta("backend", backend)
        if self._vec_ok:
            try:
                self.db.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks "
                    f"USING vec0(embedding float[{dim}] distance_metric=cosine)"
                )
            except sqlite3.OperationalError:
                self.db.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks "
                    f"USING vec0(embedding float[{dim}])"
                )
        else:
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS vectors "
                "(chunk_id INTEGER PRIMARY KEY, vec BLOB NOT NULL)"
            )
        self.db.commit()

    @property
    def vector_backend(self) -> str:
        return "sqlite-vec" if self._vec_ok else "numpy"

    # ---- writing ------------------------------------------------------

    def file_states(self) -> dict[str, str]:
        return {r["rel"]: r["sha"] for r in self.db.execute("SELECT rel, sha FROM files")}

    def remove_files(self, rels: list[str]) -> None:
        for rel in rels:
            ids = [r["id"] for r in self.db.execute("SELECT id FROM chunks WHERE rel = ?", (rel,))]
            self._drop_chunks(ids)
            self.db.execute("DELETE FROM files WHERE rel = ?", (rel,))
        self._matrix = None

    def add_file(
        self,
        rel: str,
        sha: str,
        mtime: float,
        size: int,
        lang: str,
        chunks: list[Chunk],
        texts: list[str],
        vectors: np.ndarray,
    ) -> None:
        self.remove_files([rel])
        cur = self.db.execute(
            "INSERT INTO files(rel, sha, mtime, size, lang, n_chunks) VALUES (?,?,?,?,?,?)",
            (rel, sha, mtime, size, lang, len(chunks)),
        )
        file_id = cur.lastrowid
        for chunk, text, vector in zip(chunks, texts, vectors, strict=True):
            cur = self.db.execute(
                "INSERT INTO chunks(file_id, rel, lang, kind, symbol, parent,"
                " start_line, end_line, code, sha) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    file_id,
                    chunk.rel,
                    chunk.lang,
                    chunk.kind,
                    chunk.symbol,
                    chunk.parent,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.code,
                    chunk.sha,
                ),
            )
            chunk_id = cur.lastrowid
            blob = np.asarray(vector, dtype=np.float32).tobytes()
            if self._vec_ok:
                self.db.execute(
                    "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)", (chunk_id, blob)
                )
            else:
                self.db.execute(
                    "INSERT INTO vectors(chunk_id, vec) VALUES (?, ?)", (chunk_id, blob)
                )
            self.db.execute("INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)", (chunk_id, text))
        self._matrix = None

    def cached_vectors(self, sig: str, shas: list[str], dim: int) -> dict[str, np.ndarray]:
        """Vectors already computed for this text: live index first, archive second."""
        found: dict[str, np.ndarray] = {}
        table, column = self._vector_table()
        for i in range(0, len(shas), 500):
            batch = shas[i : i + 500]
            marks = ",".join("?" * len(batch))
            queries = [
                (
                    f"SELECT c.sha AS sha, v.{column} AS vec FROM chunks c "
                    f"JOIN {table} v ON v.{'rowid' if self._vec_ok else 'chunk_id'} = c.id "
                    f"WHERE c.sha IN ({marks})",
                    list(batch),
                ),
                (
                    f"SELECT sha, vec FROM emb_cache WHERE sig = ? AND sha IN ({marks})",
                    [sig, *batch],
                ),
            ]
            for sql, params in queries:
                try:
                    rows = self.db.execute(sql, params).fetchall()
                except sqlite3.OperationalError:
                    continue
                for row in rows:
                    found.setdefault(
                        row["sha"], np.frombuffer(row["vec"], dtype=np.float32, count=dim)
                    )
        return found

    def archive_vectors(self, sig: str) -> None:
        """Keep the vectors of the current index reachable across a --reindex."""
        table, column = self._vector_table()
        key = "rowid" if self._vec_ok else "chunk_id"
        with contextlib.suppress(sqlite3.OperationalError):
            self.db.execute(
                f"INSERT OR REPLACE INTO emb_cache(sig, sha, vec) "
                f"SELECT ?, c.sha, v.{column} FROM chunks c JOIN {table} v ON v.{key} = c.id",
                (sig,),
            )

    def _vector_table(self) -> tuple[str, str]:
        return ("vec_chunks", "embedding") if self._vec_ok else ("vectors", "vec")

    def cache_vectors(self, sig: str, shas: list[str], vectors: np.ndarray) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO emb_cache(sig, sha, vec) VALUES (?,?,?)",
            [
                (sig, sha, np.asarray(v, dtype=np.float32).tobytes())
                for sha, v in zip(shas, vectors, strict=True)
            ],
        )

    def prune_cache(self, keep: int = 200_000) -> None:
        """Live vectors are their own cache, so the archive only keeps what left the index."""
        self.db.execute("DELETE FROM emb_cache WHERE sha IN (SELECT sha FROM chunks)")
        self.db.execute(
            "DELETE FROM emb_cache WHERE rowid NOT IN "
            "(SELECT rowid FROM emb_cache ORDER BY rowid DESC LIMIT ?)",
            (keep,),
        )
        self.db.commit()
        self.db.execute("PRAGMA incremental_vacuum")

    def reset_vectors(self) -> None:
        self.db.execute("DROP TABLE IF EXISTS vec_chunks")
        self.db.execute("DROP TABLE IF EXISTS vectors")
        self.db.execute("DELETE FROM chunks_fts")
        self.db.execute("DELETE FROM chunks")
        self.db.execute("DELETE FROM files")
        self._matrix = None

    def commit(self) -> None:
        self.db.commit()

    def close(self) -> None:
        self.db.commit()
        self.db.close()

    def _drop_chunks(self, ids: list[int]) -> None:
        if not ids:
            return
        marks = ",".join("?" * len(ids))
        self.db.execute(f"DELETE FROM chunks_fts WHERE rowid IN ({marks})", ids)
        table = "vec_chunks" if self._vec_ok else "vectors"
        column = "rowid" if self._vec_ok else "chunk_id"
        with contextlib.suppress(sqlite3.OperationalError):
            self.db.execute(f"DELETE FROM {table} WHERE {column} IN ({marks})", ids)
        self.db.execute(f"DELETE FROM chunks WHERE id IN ({marks})", ids)

    # ---- reading ------------------------------------------------------

    def check_backend(self) -> None:
        """An index built with sqlite-vec is unreadable without it, and vice versa."""
        stored = self.get_meta("backend")
        current = "vec0" if self._vec_ok else "numpy"
        if stored and stored != current:
            missing = (
                "sqlite-vec is not available" if stored == "vec0" else "sqlite-vec is now loaded"
            )
            raise SystemExit(
                f"index stores vectors as '{stored}' but {missing}.\n"
                f"Install it (pip install sqlite-vec) or rebuild with `fyc index --reindex`."
            )

    def search_vector(self, query: np.ndarray, k: int, filters: Filters) -> list[tuple[int, float]]:
        self.check_backend()
        blob = np.asarray(query, dtype=np.float32).tobytes()
        allowed = self._allowed_ids(filters)
        if allowed is not None and not allowed:
            return []

        # A filter that selects a small slice is answered exactly: an approximate
        # global top-N may not contain a single row that passes it.
        if allowed is not None and len(allowed) <= EXACT_SCAN_LIMIT:
            return self._exact_top_k(query, k, sorted(allowed))

        if self._vec_ok:
            fetch = min(k if allowed is None else max(k * 8, 200), VEC_MAX_K)
            rows = self.db.execute(
                "SELECT rowid AS id, distance FROM vec_chunks "
                "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (blob, fetch),
            ).fetchall()
            # A zero-length vector has no cosine: sqlite-vec returns NULL, and those rows
            # sort first. Drop them rather than scoring them ahead of real matches.
            hits = [
                (r["id"], 1.0 - float(r["distance"])) for r in rows if r["distance"] is not None
            ]
            if allowed is not None:
                hits = [h for h in hits if h[0] in allowed]
            return hits[:k]

        ids, matrix = self._numpy_matrix()
        if len(ids) == 0:
            return []
        scores = matrix @ np.asarray(query, dtype=np.float32)
        if allowed is not None:
            mask = np.fromiter((i in allowed for i in ids), dtype=bool, count=len(ids))
            scores = np.where(mask, scores, -np.inf)
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(ids[i]), float(scores[i])) for i in top if np.isfinite(scores[i])]

    def _exact_top_k(self, query: np.ndarray, k: int, ids: list[int]) -> list[tuple[int, float]]:
        vector = np.asarray(query, dtype=np.float32)
        scored: list[tuple[int, float]] = []
        for start in range(0, len(ids), 900):
            batch = ids[start : start + 900]
            found = self.vectors_for(batch)
            if not found:
                continue
            keys = list(found)
            matrix = np.vstack([found[cid] for cid in keys])
            scored.extend(zip(keys, (matrix @ vector).tolist(), strict=True))
        scored.sort(key=lambda pair: -pair[1])
        return scored[:k]

    def search_lexical(self, query: str, k: int, filters: Filters) -> list[tuple[int, float]]:
        match = _fts_query(query)
        if not match:
            return []
        where, params = filters.sql("c.")
        sql = (
            "SELECT f.rowid AS id, bm25(chunks_fts) AS score FROM chunks_fts f "
            "JOIN chunks c ON c.id = f.rowid "
            f"WHERE chunks_fts MATCH ? AND {where} "
            "ORDER BY score LIMIT ?"
        )
        try:
            rows = self.db.execute(sql, [match, *params, k]).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(r["id"], -float(r["score"])) for r in rows]

    def vectors_for(self, ids: list[int]) -> dict[int, np.ndarray]:
        if not ids:
            return {}
        dim = int(self.get_meta("dim") or 0)
        marks = ",".join("?" * len(ids))
        if self._vec_ok:
            sql = f"SELECT rowid AS id, embedding AS vec FROM vec_chunks WHERE rowid IN ({marks})"
        else:
            sql = f"SELECT chunk_id AS id, vec FROM vectors WHERE chunk_id IN ({marks})"
        return {
            r["id"]: np.frombuffer(r["vec"], dtype=np.float32, count=dim)
            for r in self.db.execute(sql, ids)
        }

    def rows(self, ids: list[int]) -> dict[int, Row]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self.db.execute(
            f"SELECT id, rel, lang, kind, symbol, parent, start_line, end_line, code "
            f"FROM chunks WHERE id IN ({marks})",
            ids,
        )
        return {r["id"]: _to_row(r) for r in rows}

    def chunk_at(self, location: str) -> Row | None:
        """Resolve `path` or `path:line` to the chunk that covers it."""
        path, _, line_text = location.rpartition(":")
        if path and line_text.isdigit():
            line = int(line_text)
        else:
            path, line = location, None

        pattern = f"%{path.lstrip('./')}"
        if line is None:
            row = self.db.execute(
                "SELECT * FROM chunks WHERE rel LIKE ? ORDER BY start_line LIMIT 1", (pattern,)
            ).fetchone()
        else:
            row = self.db.execute(
                "SELECT * FROM chunks WHERE rel LIKE ? AND start_line <= ? AND end_line >= ? "
                "ORDER BY end_line - start_line LIMIT 1",
                (pattern, line, line),
            ).fetchone()
        return _to_row(row) if row else None

    def stats(self) -> dict:
        def one(sql: str) -> int:
            return self.db.execute(sql).fetchone()[0]

        langs = self.db.execute(
            "SELECT lang, COUNT(*) n FROM chunks GROUP BY lang ORDER BY n DESC LIMIT 12"
        ).fetchall()
        return {
            "files": one("SELECT COUNT(*) FROM files"),
            "chunks": one("SELECT COUNT(*) FROM chunks"),
            "signature": self.get_meta("signature") or "-",
            "backend": self.get_meta("backend") or "-",
            "db_bytes": self.path.stat().st_size if self.path.exists() else 0,
            "langs": [(r["lang"], r["n"]) for r in langs],
        }

    def _allowed_ids(self, filters: Filters) -> set[int] | None:
        if not filters.active():
            return None
        where, params = filters.sql()
        return {r[0] for r in self.db.execute(f"SELECT id FROM chunks WHERE {where}", params)}

    def _numpy_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        if self._matrix is None:
            dim = int(self.get_meta("dim") or 0)
            ids, vectors = [], []
            for row in self.db.execute("SELECT chunk_id, vec FROM vectors ORDER BY chunk_id"):
                ids.append(row["chunk_id"])
                vectors.append(np.frombuffer(row["vec"], dtype=np.float32, count=dim))
            matrix = np.vstack(vectors) if vectors else np.zeros((0, max(dim, 1)), dtype=np.float32)
            self._matrix = (np.asarray(ids, dtype=np.int64), matrix)
        return self._matrix


def _to_row(r: sqlite3.Row) -> Row:
    return Row(
        r["id"],
        r["rel"],
        r["lang"],
        r["kind"],
        r["symbol"],
        r["parent"],
        r["start_line"],
        r["end_line"],
        r["code"],
    )


def _load_sqlite_vec(db: sqlite3.Connection) -> bool:
    try:
        import sqlite_vec

        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        return True
    except Exception:
        return False


_FTS_SAFE = str.maketrans(dict.fromkeys('"*():^-+,.!?/\\[]{}<>=&|~%$#@', " "))


def _fts_query(query: str) -> str:
    terms = [t for t in query.translate(_FTS_SAFE).split() if len(t) > 1]
    return " OR ".join(f'"{t}"' for t in terms[:32])
