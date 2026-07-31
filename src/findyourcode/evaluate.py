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
        )
        rank = None
        for position, hit in enumerate(hits, 1):
            if any(marker in hit.row.rel for marker in case.expect):
                rank = position
                break
        report.results.append(CaseResult(case=case, rank=rank, top=hits[0].row.rel if hits else ""))
    return report


def render_report(report: Report, title: str = "") -> str:
    lines = [title] if title else []
    for result in report.results:
        mark = f"#{result.rank}" if result.rank else "miss"
        lines.append(f"  {mark:>5}  {result.case.query[:56]:<56}  {result.top}")
    lines.append(
        f"  recall@1 {report.recall_at(1):.2f}  recall@3 {report.recall_at(3):.2f}  "
        f"recall@10 {report.recall_at(10):.2f}  MRR {report.mrr():.3f}"
        f"  ({len(report.results)} cases)"
    )
    return "\n".join(lines)


def render_sweep(rows: list[tuple[str, Report]]) -> str:
    header = f"  {'setting':<16}{'recall@1':>10}{'recall@3':>10}{'recall@10':>11}{'MRR':>8}"
    lines = [header, "  " + "-" * (len(header) - 2)]
    for label, report in rows:
        lines.append(
            f"  {label:<16}{report.recall_at(1):>10.2f}{report.recall_at(3):>10.2f}"
            f"{report.recall_at(10):>11.2f}{report.mrr():>8.3f}"
        )
    return "\n".join(lines)
