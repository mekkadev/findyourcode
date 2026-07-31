"""Measure retrieval quality on a set of query -> expected-file cases.

Turns "the ranking feels better" into recall@k and MRR, so a change to the model,
alpha or chunk size can be judged instead of argued about.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .embeddings import Embedder
from .search import search
from .store import Store


@dataclass
class Case:
    query: str
    expect: list[str]
    note: str = ""


@dataclass
class CaseResult:
    case: Case
    rank: int | None
    top: str = ""


@dataclass
class Report:
    results: list[CaseResult] = field(default_factory=list)

    def recall_at(self, k: int) -> float:
        if not self.results:
            return 0.0
        hit = sum(1 for r in self.results if r.rank is not None and r.rank <= k)
        return hit / len(self.results)

    def mrr(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 / r.rank for r in self.results if r.rank) / len(self.results)

    @property
    def misses(self) -> list[CaseResult]:
        return [r for r in self.results if r.rank is None]


def load_cases(path: Path) -> list[Case]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("cases", [])

    cases = []
    for item in data:
        expect = item.get("expect", [])
        cases.append(
            Case(
                query=item["query"],
                expect=[expect] if isinstance(expect, str) else list(expect),
                note=item.get("note", ""),
            )
        )
    if not cases:
        raise SystemExit("no cases found — expected a JSON list of {query, expect}")
    return cases


def evaluate(
    store: Store,
    embedder: Embedder,
    cfg: Config,
    cases: list[Case],
    limit: int = 10,
    mode: str = "hybrid",
    fusion: str = "",
    reranker=None,
    graph: bool | None = None,
) -> Report:
    report = Report()
    for case in cases:
        hits = search(
            store,
            embedder,
            case.query,
            cfg,
            limit=limit,
            mode=mode,
            fusion=fusion,
            reranker=reranker,
            graph=graph,
        )
        rank = None
        for position, hit in enumerate(hits, 1):
            if any(marker in hit.row.rel for marker in case.expect):
                rank = position
                break
        report.results.append(CaseResult(case=case, rank=rank, top=hits[0].row.rel if hits else ""))
    return report


def levels(limit: int) -> list[int]:
    """Only report a recall@k the run could actually have measured."""
    ks = [k for k in (1, 3, 10) if k <= limit]
    if limit not in ks:
        ks.append(limit)
    return ks


def render_report(report: Report, title: str = "", limit: int = 10) -> str:
    lines = [title] if title else []
    for result in report.results:
        mark = f"#{result.rank}" if result.rank else "miss"
        lines.append(f"  {mark:>5}  {result.case.query[:56]:<56}  {result.top}")
    scores = "  ".join(f"recall@{k} {report.recall_at(k):.2f}" for k in levels(limit))
    lines.append(f"  {scores}  MRR {report.mrr():.3f}  ({len(report.results)} cases)")
    return "\n".join(lines)


def render_sweep(rows: list[tuple[str, Report]], limit: int = 10) -> str:
    ks = levels(limit)
    header = f"  {'setting':<16}" + "".join(f"{'recall@' + str(k):>11}" for k in ks) + f"{'MRR':>8}"
    lines = [header, "  " + "-" * (len(header) - 2)]
    for label, report in rows:
        scores = "".join(f"{report.recall_at(k):>11.2f}" for k in ks)
        lines.append(f"  {label:<16}{scores}{report.mrr():>8.3f}")
    return "\n".join(lines)
