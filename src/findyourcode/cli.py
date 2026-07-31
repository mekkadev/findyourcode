"""Command line interface: index, find, similar, eval, status, clear, providers, mcp."""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import sys
import time
from pathlib import Path

from . import __version__
from .config import load_config
from .embeddings import PROVIDERS, get_embedder
from .evaluate import evaluate, load_cases, render_report, render_sweep
from .format import as_json, as_paths, render, symbol_of, use_color
from .indexer import build_index
from .rerank import DEFAULT_MODEL as RERANK_DEFAULT
from .search import build_trace, search, similar_to
from .store import Filters, Store


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    if not Path(args.root).is_dir():
        print(f"'{args.root}' is not a directory", file=sys.stderr)
        return 2
    if getattr(args, "limit", 1) < 1:
        print("--limit must be at least 1", file=sys.stderr)
        return 2
    try:
        return args.handler(args)
    except BrokenPipeError:
        # `| head` closed the pipe; keep the interpreter from complaining on exit.
        with contextlib.suppress(OSError, ValueError):
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    except RuntimeError as exc:  # provider and transport failures
        print(f"error: {exc}", file=sys.stderr)
        return 3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fyc", description="Semantic code search — find code by meaning, not by text."
    )
    parser.add_argument("--version", action="version", version=f"findyourcode {__version__}")
    parser.add_argument("-C", "--root", default=".", help="project root (default: .)")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), help="embedding provider")
    parser.add_argument("--model", help="embedding model name")
    sub = parser.add_subparsers(dest="command")

    index = sub.add_parser("index", help="build or update the index")
    _add_global_options(index)
    index.add_argument("--reindex", action="store_true", help="drop the index and start over")
    index.add_argument("--workers", type=int, help="parser threads")
    index.add_argument("--watch", action="store_true", help="keep the index fresh until Ctrl-C")
    index.add_argument("--interval", type=float, default=2.0, help="seconds between watch passes")
    index.add_argument("-q", "--quiet", action="store_true")
    index.set_defaults(handler=cmd_index)

    find = sub.add_parser("find", help="search the index")
    _add_global_options(find)
    find.add_argument("query", nargs="+")
    _add_result_options(find)
    find.add_argument("--mode", choices=["hybrid", "semantic", "lexical"], default="hybrid")
    find.add_argument("--fusion", choices=["blend", "rrf"], help="how the two rankings are merged")
    find.add_argument("--explain", action="store_true", help="show per-retriever ranks")
    find.add_argument(
        "--trace",
        nargs="?",
        type=int,
        const=0,
        help="print the call path around each hit (optionally how many hops)",
    )
    find.add_argument(
        "--no-graph",
        dest="graph",
        action="store_false",
        default=None,
        help="rank on text alone, without call-graph propagation",
    )
    find.add_argument(
        "--rerank",
        nargs="?",
        const=RERANK_DEFAULT,
        help="rescore the shortlist with a cross-encoder (optionally name the model)",
    )
    find.set_defaults(handler=cmd_find)

    similar = sub.add_parser("similar", help="find code similar to a place in the codebase")
    _add_global_options(similar)
    similar.add_argument("location", help="path or path:line")
    similar.add_argument(
        "--same-file", action="store_true", help="also return chunks from the same file"
    )
    _add_result_options(similar)
    similar.set_defaults(handler=cmd_similar)

    ev = sub.add_parser("eval", help="measure ranking quality on a file of cases")
    _add_global_options(ev)
    ev.add_argument("cases", help='JSON file: [{"query": "...", "expect": "path/part"}]')
    ev.add_argument("-n", "--limit", type=int, default=10)
    ev.add_argument("--mode", choices=["hybrid", "semantic", "lexical"], default="hybrid")
    ev.add_argument("--fusion", choices=["blend", "rrf"], help="how the two rankings are merged")
    ev.add_argument("--alpha", type=float, help="weight of the semantic branch (blend fusion)")
    ev.add_argument("--sweep", action="store_true", help="compare several alpha values and modes")
    ev.add_argument(
        "--no-graph",
        dest="graph",
        action="store_false",
        default=None,
        help="score on text alone, without call-graph propagation",
    )
    ev.add_argument(
        "--rerank", nargs="?", const=RERANK_DEFAULT, help="rescore with a cross-encoder"
    )
    ev.add_argument("--min-mrr", type=float, help="exit non-zero below this MRR (for CI)")
    ev.add_argument("--min-recall", type=float, help="exit non-zero below this recall@1")
    ev.set_defaults(handler=cmd_eval)

    status = sub.add_parser("status", help="show index statistics")
    _add_global_options(status)
    status.set_defaults(handler=cmd_status)

    clear = sub.add_parser("clear", help="delete the index")
    _add_global_options(clear)
    clear.set_defaults(handler=cmd_clear)

    providers = sub.add_parser("providers", help="list embedding providers")
    _add_global_options(providers)
    providers.set_defaults(handler=cmd_providers)

    doctor = sub.add_parser("doctor", help="check the setup and the index")
    _add_global_options(doctor)
    doctor.set_defaults(handler=cmd_doctor)

    mcp = sub.add_parser("mcp", help="serve the index to agents over MCP (stdio)")
    _add_global_options(mcp)
    mcp.set_defaults(handler=cmd_mcp)
    return parser


def _add_global_options(sub: argparse.ArgumentParser) -> None:
    """Accept the global flags after the subcommand too — `fyc index --provider hash` is
    what everyone types. SUPPRESS matters: argparse copies a subparser's defaults over
    the main namespace, so a plain default would erase the value given before the verb."""
    sub.add_argument("-C", "--root", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    sub.add_argument(
        "--provider", choices=sorted(PROVIDERS), default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    sub.add_argument("--model", default=argparse.SUPPRESS, help=argparse.SUPPRESS)


def _add_result_options(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("-n", "--limit", type=int, default=10)
    sub.add_argument("--lang", action="append", help="filter by language (repeatable)")
    sub.add_argument("--path", action="append", help="filter by path substring (repeatable)")
    sub.add_argument("--kind", action="append", help="filter by kind: function, class, method, ...")
    sub.add_argument("-L", "--lines", type=int, default=8, help="snippet lines (0 = full chunk)")
    sub.add_argument(
        "-f",
        "--format",
        choices=["pretty", "paths", "files", "json"],
        default="pretty",
        help="pretty (default), paths (path:line), files, or json",
    )
    sub.add_argument("--json", action="store_true", help="alias for --format json")
    sub.add_argument("--per-file", type=int, help="max results from one file (0 = no limit)")


def cmd_index(args) -> int:
    cfg = load_config(
        args.root,
        provider=args.provider,
        model=args.model,
        workers=getattr(args, "workers", None),
    )
    embedder = _embedder(cfg)
    store = Store(cfg.db_path)
    say = _reporter(quiet=args.quiet)
    say(f"provider {embedder.signature}, vectors via {store.vector_backend}")

    stats = build_index(cfg, embedder, store, reindex=args.reindex, progress=say)
    say(None)

    if stats.errors and not args.quiet:
        for line in stats.errors[:10]:
            print(f"warn: {line}", file=sys.stderr)
    print(_summary(stats))

    if args.watch:
        store = _watch(cfg, embedder, store, args)
    store.close()
    return 0


def _summary(stats) -> str:
    return (
        f"indexed {stats.indexed} files ({stats.chunks} chunks), "
        f"{stats.unchanged} unchanged, {stats.removed} removed, "
        f"{stats.embedded} embedded, {stats.reused} from cache, "
        f"{stats.elapsed:.1f}s"
    )


def _watch(cfg, embedder, store: Store, args) -> Store:
    """Re-scan on a timer, keeping the model in memory — a pass over an unchanged tree is cheap."""
    print(f"watching {cfg.root} every {args.interval:g}s — Ctrl-C to stop", file=sys.stderr)
    try:
        while True:
            time.sleep(max(args.interval, 0.2))
            if not cfg.db_path.exists():
                # someone removed .findyourcode; the open handle writes to a dead inode
                print("index disappeared — rebuilding", file=sys.stderr)
                store.close()
                store = Store(cfg.db_path)
            stats = build_index(cfg, embedder, store)
            if stats.indexed or stats.removed:
                print(_summary(stats))
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
    return store


def cmd_find(args) -> int:
    cfg = load_config(args.root, provider=args.provider, model=args.model)
    if not cfg.db_path.exists():
        print("no index here — run `fyc index` first", file=sys.stderr)
        return 2

    store = Store(cfg.db_path)
    provider, model = _parse_signature(store.get_meta("signature"), cfg)
    if (args.provider and args.provider != provider) or (args.model and args.model != model):
        print(
            f"note: querying with {provider}:{model} — the model the index was built with",
            file=sys.stderr,
        )
    embedder = _embedder(cfg, provider=provider, model=model)

    hits = search(
        store,
        embedder,
        " ".join(args.query),
        cfg,
        limit=args.limit,
        filters=_filters(args),
        mode=args.mode,
        fusion=args.fusion or "",
        per_file=args.per_file,
        reranker=_reranker(args, cfg),
        graph=args.graph,
    )
    traces = _traces(store, hits, embedder, args, cfg)
    store.close()
    return _emit(hits, args, explain=args.explain, traces=traces)


def _traces(store, hits, embedder, args, cfg):
    """--trace follows the edge the query cares about, so the path needs the vector."""
    if getattr(args, "trace", None) is None or not hits:
        return None
    if _output_format(args) in ("paths", "files"):
        return None  # nowhere to print it
    if args.trace:
        cfg.trace_depth = max(1, args.trace)
    vector = embedder.embed_query(" ".join(args.query))
    if not vector.any():
        vector = None
    return {hit.row.id: build_trace(store, hit.row, cfg, vector) for hit in hits}


def cmd_similar(args) -> int:
    cfg = load_config(args.root)
    if not cfg.db_path.exists():
        print("no index here — run `fyc index` first", file=sys.stderr)
        return 2

    store = Store(cfg.db_path)
    anchor, hits = similar_to(
        store,
        args.location,
        cfg,
        limit=args.limit,
        filters=_filters(args),
        same_file=args.same_file,
    )
    store.close()

    if anchor is None:
        print(f"nothing indexed at '{args.location}'", file=sys.stderr)
        return 2
    if _output_format(args) == "pretty":
        label = " ".join(p for p in (anchor.kind, symbol_of(anchor)) if p)
        print(f"like {anchor.rel}:{anchor.start_line}-{anchor.end_line} {label}\n", file=sys.stderr)
    return _emit(hits, args)


def cmd_eval(args) -> int:
    cfg = load_config(args.root, alpha=args.alpha)
    if not cfg.db_path.exists():
        print("no index here — run `fyc index` first", file=sys.stderr)
        return 2

    try:
        cases = load_cases(Path(args.cases))
    except (OSError, ValueError) as exc:
        print(f"cannot read cases from '{args.cases}': {exc}", file=sys.stderr)
        return 2
    store = Store(cfg.db_path)
    provider, model = _parse_signature(store.get_meta("signature"), cfg)
    embedder = _embedder(cfg, provider=provider, model=model)

    if args.sweep and (args.min_mrr is not None or args.min_recall is not None):
        print("--sweep compares settings; a threshold needs a single run", file=sys.stderr)
        return 2

    if args.sweep:
        rows = []
        configured = cfg.alpha
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            cfg.alpha = alpha
            rows.append(
                (
                    f"blend a={alpha:g}",
                    evaluate(store, embedder, cfg, cases, args.limit, fusion="blend"),
                )
            )
        cfg.alpha = configured  # the rows below are not alpha sweeps; restore the real one
        for mode in ("semantic", "lexical"):
            rows.append(
                (
                    mode,
                    evaluate(store, embedder, cfg, cases, args.limit, mode=mode, fusion="blend"),
                )
            )
        rows.append(("rrf", evaluate(store, embedder, cfg, cases, args.limit, fusion="rrf")))
        rows.append(
            (
                "no graph",
                evaluate(store, embedder, cfg, cases, args.limit, fusion="blend", graph=False),
            )
        )
        print(render_sweep(rows, args.limit))
        store.close()
        return 0

    report = evaluate(
        store,
        embedder,
        cfg,
        cases,
        args.limit,
        mode=args.mode,
        fusion=args.fusion or "",
        reranker=_reranker(args, cfg),
        graph=args.graph,
    )
    store.close()
    print(render_report(report, f"{provider}:{model}", args.limit))

    if args.min_mrr is not None and report.mrr() < args.min_mrr:
        print(f"MRR {report.mrr():.3f} is below the required {args.min_mrr}", file=sys.stderr)
        return 1
    if args.min_recall is not None and report.recall_at(1) < args.min_recall:
        print(
            f"recall@1 {report.recall_at(1):.2f} is below the required {args.min_recall}",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_status(args) -> int:
    cfg = load_config(args.root)
    if not cfg.db_path.exists():
        print("no index here — run `fyc index` first", file=sys.stderr)
        return 2
    store = Store(cfg.db_path)
    stats = store.stats()
    store.close()
    print(f"root      {cfg.root}")
    print(f"index     {cfg.db_path} ({stats['db_bytes'] / 1e6:.1f} MB)")
    print(f"model     {stats['signature']}")
    print(f"vectors   {stats['backend']}")
    print(f"files     {stats['files']}")
    print(f"chunks    {stats['chunks']}")
    if stats["calls"]:
        print(f"graph     {stats['symbols']} symbols, {stats['calls']} call sites")
    if stats["langs"]:
        top = ", ".join(f"{lang} {n}" for lang, n in stats["langs"])
        print(f"languages {top}")
    return 0


def cmd_clear(args) -> int:
    cfg = load_config(args.root)
    if cfg.index_dir.exists():
        shutil.rmtree(cfg.index_dir)
        print(f"removed {cfg.index_dir}")
    else:
        print("nothing to remove")
    return 0


def cmd_doctor(args) -> int:
    from .diagnose import render, run

    checks = run(load_config(args.root))
    print(render(checks))
    return 0 if all(check.ok for check in checks) else 1


def cmd_mcp(args) -> int:
    from .mcp_server import serve

    return serve(args.root)


def cmd_providers(args) -> int:
    for name, (_, _, description) in PROVIDERS.items():
        print(f"{name:<8} {description}")
    return 0


def _reporter(quiet: bool):
    """Progress goes to stderr; on a terminal it repaints one line instead of scrolling."""
    live = sys.stderr.isatty()
    state = {"dirty": False}

    def say(message: str | None) -> None:
        if quiet:
            return
        if message is None:
            if state["dirty"]:
                print(file=sys.stderr)
                state["dirty"] = False
            return
        if live:
            print(f"\r\033[K{message}", end="", file=sys.stderr, flush=True)
            state["dirty"] = True
        else:
            print(message, file=sys.stderr)

    return say


def _reranker(args, cfg):
    """--rerank names a model, or takes the configured one; absent means no second pass."""
    model = getattr(args, "rerank", None) or cfg.rerank
    if not model:
        return None
    from .rerank import Reranker

    return Reranker(model)


def _filters(args) -> Filters:
    return Filters(langs=args.lang, paths=args.path, kinds=args.kind)


def _output_format(args) -> str:
    return "json" if args.json else args.format


def _emit(hits, args, explain: bool = False, traces=None) -> int:
    style = _output_format(args)
    if style == "json":
        print(as_json(hits, traces))
    elif style in ("paths", "files"):
        text = as_paths(hits, with_line=style == "paths")
        if text:
            print(text)
    else:
        print(
            render(
                hits,
                snippet_lines=args.lines,
                explain=explain,
                color=use_color(),
                traces=traces,
            )
        )
    return 0 if hits else 1


def _parse_signature(signature: str | None, cfg) -> tuple[str, str]:
    """A stored signature is 'provider:model:dim' — queries must use the same pair."""
    if not signature or signature.count(":") < 2:
        return cfg.provider, cfg.model
    provider, rest = signature.split(":", 1)
    return provider, rest.rsplit(":", 1)[0]


def _embedder(cfg, provider: str | None = None, model: str | None = None):
    try:
        return get_embedder(
            provider or cfg.provider,
            model if model is not None else cfg.model,
            cfg.batch_size,
        )
    except (RuntimeError, ValueError) as exc:  # a provider that cannot load the named model
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    sys.exit(main())
