"""Command line interface: fyc index / find / similar / status / clear / providers."""

from __future__ import annotations

import argparse
import shutil
import sys

from . import __version__
from .config import load_config
from .embeddings import PROVIDERS, get_embedder
from .format import as_json, as_paths, render, symbol_of, use_color
from .indexer import build_index
from .search import search, similar_to
from .store import Filters, Store


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.handler(args)


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
    index.add_argument("--reindex", action="store_true", help="drop the index and start over")
    index.add_argument("--workers", type=int, help="parser threads")
    index.add_argument("-q", "--quiet", action="store_true")
    index.set_defaults(handler=cmd_index)

    find = sub.add_parser("find", help="search the index")
    find.add_argument("query", nargs="+")
    _add_result_options(find)
    find.add_argument("--mode", choices=["hybrid", "semantic", "lexical"], default="hybrid")
    find.add_argument("--fusion", choices=["blend", "rrf"], help="how the two rankings are merged")
    find.add_argument("--explain", action="store_true", help="show per-retriever ranks")
    find.set_defaults(handler=cmd_find)

    similar = sub.add_parser("similar", help="find code similar to a place in the codebase")
    similar.add_argument("location", help="path or path:line")
    similar.add_argument(
        "--same-file", action="store_true", help="also return chunks from the same file"
    )
    _add_result_options(similar)
    similar.set_defaults(handler=cmd_similar)

    status = sub.add_parser("status", help="show index statistics")
    status.set_defaults(handler=cmd_status)

    clear = sub.add_parser("clear", help="delete the index")
    clear.set_defaults(handler=cmd_clear)

    providers = sub.add_parser("providers", help="list embedding providers")
    providers.set_defaults(handler=cmd_providers)
    return parser


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
    store.close()

    if stats.errors and not args.quiet:
        for line in stats.errors[:10]:
            print(f"warn: {line}", file=sys.stderr)
    print(
        f"indexed {stats.indexed} files ({stats.chunks} chunks), "
        f"{stats.unchanged} unchanged, {stats.removed} removed, "
        f"{stats.embedded} embedded, {stats.reused} from cache, "
        f"{stats.elapsed:.1f}s"
    )
    return 0


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
    )
    store.close()
    return _emit(hits, args, explain=args.explain)


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


def _filters(args) -> Filters:
    return Filters(langs=args.lang, paths=args.path, kinds=args.kind)


def _output_format(args) -> str:
    return "json" if args.json else args.format


def _emit(hits, args, explain: bool = False) -> int:
    style = _output_format(args)
    if style == "json":
        print(as_json(hits))
    elif style in ("paths", "files"):
        text = as_paths(hits, with_line=style == "paths")
        if text:
            print(text)
    else:
        print(render(hits, snippet_lines=args.lines, explain=explain, color=use_color()))
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
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    sys.exit(main())
