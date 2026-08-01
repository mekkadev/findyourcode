#!/usr/bin/env python3
"""Reproduce the numbers in the README.

    python scripts/benchmark.py                 # against this python's stdlib
    python scripts/benchmark.py --corpus ~/repo --cases my_cases.json

Indexes a corpus into a scratch directory, times it, then reports recall@k and
MRR and the fusion sweep. Nothing here is hidden inside the tool: it drives the
same CLI a reader would.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "examples" / "eval_stdlib.json"
MULTIHOP = ROOT / "examples" / "eval_multihop.json"
RUSSIAN = ROOT / "examples" / "eval_stdlib_ru.json"
ONEWORD = ROOT / "examples" / "eval_oneword.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, help="directory to index (default: this stdlib)")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/fyc-benchmark"))
    parser.add_argument("--model", default="", help="embedding model to pass to `fyc index`")
    parser.add_argument("--provider", default="", help="embedding provider (default: configured)")
    parser.add_argument("--keep", action="store_true", help="reuse an existing copy of the corpus")
    args = parser.parse_args()

    corpus = args.corpus or Path(sysconfig.get_paths()["stdlib"])
    if not corpus.is_dir():
        print(f"no corpus at {corpus}", file=sys.stderr)
        return 2
    if not args.cases.is_file():
        print(f"no cases at {args.cases}", file=sys.stderr)
        return 2

    target = args.workdir / corpus.name
    if not (args.keep and target.exists()):
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"copying {corpus} -> {target}")
        shutil.copytree(corpus, target, ignore=shutil.ignore_patterns("__pycache__"))

    globals_ = (["--provider", args.provider] if args.provider else []) + (
        ["--model", args.model] if args.model else []
    )
    index = [*globals_, "index", "--reindex"]
    first = timed(target, index, "first index")
    again = timed(target, [*globals_, "index"], "re-index, nothing changed")

    print()
    run(target, [*globals_, "eval", str(args.cases.resolve())])
    print()
    run(target, [*globals_, "eval", str(args.cases.resolve()), "--sweep"])
    if args.cases == DEFAULT_CASES:
        # The other two sets only mean anything against the stdlib they were written for.
        for label, extra in (
            ("one- and two-word queries", ["eval", str(ONEWORD)]),
            ("the same, one blend for every query", ["eval", str(ONEWORD), "--alpha", "0.75"]),
            ("multi-hop, text only", ["eval", str(MULTIHOP), "--no-graph"]),
            ("multi-hop, with the call graph", ["eval", str(MULTIHOP)]),
            ("the same questions in russian", ["eval", str(RUSSIAN)]),
        ):
            print(f"\n-- {label}")
            run(target, [*globals_, *extra])

    db = target / ".findyourcode" / "index.db"
    print(f"\nindex on disk   {db.stat().st_size / 1e6:.0f} mb")
    print(f"first index     {first:.0f}s")
    print(f"re-index        {again:.1f}s")
    return 0


def timed(cwd: Path, arguments: list[str], label: str) -> float:
    started = time.time()
    run(cwd, arguments)
    elapsed = time.time() - started
    print(f"-- {label}: {elapsed:.1f}s\n")
    return elapsed


def run(cwd: Path, arguments: list[str]) -> None:
    subprocess.run([sys.executable, "-m", "findyourcode", *arguments], cwd=cwd, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
