import json

import pytest

from findyourcode import cli
from findyourcode.config import load_config
from findyourcode.embeddings import get_embedder
from findyourcode.evaluate import Case, CaseResult, Report, evaluate, load_cases, render_report
from findyourcode.indexer import build_index
from findyourcode.store import Store

FILES = {
    "auth/login.py": "def verify_password(login, password):\n    return check(login, password)\n",
    "ui/plot.py": "def draw_axis(canvas, ticks):\n    canvas.line(ticks)\n",
}


def test_metrics_arithmetic():
    report = Report(
        results=[
            CaseResult(Case("a", ["x"]), rank=1),
            CaseResult(Case("b", ["y"]), rank=4),
            CaseResult(Case("c", ["z"]), rank=None),
        ]
    )
    assert report.recall_at(1) == pytest.approx(1 / 3)
    assert report.recall_at(5) == pytest.approx(2 / 3)
    assert report.mrr() == pytest.approx((1 + 0.25) / 3)
    assert [r.case.query for r in report.misses] == ["c"]


def test_empty_report_does_not_divide_by_zero():
    empty = Report()
    assert empty.recall_at(1) == 0.0 and empty.mrr() == 0.0


def test_load_cases_accepts_string_and_list(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps([{"query": "a", "expect": "one.py"}, {"query": "b", "expect": ["x", "y"]}]),
        encoding="utf-8",
    )
    cases = load_cases(path)
    assert cases[0].expect == ["one.py"]
    assert cases[1].expect == ["x", "y"]

    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_cases(empty)


def test_evaluate_ranks_cases(repo):
    root = repo(FILES)
    cfg = load_config(root, provider="hash")
    embedder = get_embedder(cfg.provider)
    store = Store(cfg.db_path)
    build_index(cfg, embedder, store)

    report = evaluate(
        store,
        embedder,
        cfg,
        [
            Case("verify password", ["auth/login.py"]),
            Case("draw axis ticks", ["ui/plot.py"]),
            Case("quantum entanglement", ["nothing/here.py"]),
        ],
    )
    assert [r.rank for r in report.results[:2]] == [1, 1]
    assert report.results[2].rank is None
    assert "recall@1" in render_report(report)
    store.close()


def test_eval_command(repo, tmp_path, capsys):
    root = repo(FILES)
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps([{"query": "verify password", "expect": "auth/login.py"}]), encoding="utf-8"
    )
    cli.main(["-C", str(root), "--provider", "hash", "index", "-q"])
    capsys.readouterr()

    assert cli.main(["-C", str(root), "eval", str(cases)]) == 0
    assert "MRR 1.000" in capsys.readouterr().out

    assert cli.main(["-C", str(root), "eval", str(cases), "--sweep"]) == 0
    assert "recall@1" in capsys.readouterr().out

    # A miss alone is a measurement, not a failure — only a threshold fails the run.
    miss = tmp_path / "miss.json"
    miss.write_text(json.dumps([{"query": "nothing", "expect": "absent.py"}]), encoding="utf-8")
    assert cli.main(["-C", str(root), "eval", str(miss)]) == 0
    capsys.readouterr()
    assert cli.main(["-C", str(root), "eval", str(miss), "--min-mrr", "0.5"]) == 1
    assert "below the required" in capsys.readouterr().err
