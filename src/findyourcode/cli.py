"""Command line interface: fyc index / find / status / clear / providers."""

from __future__ import annotations

import argparse
import shutil
import sys

from . import __version__
from .config import load_config
from .embeddings import PROVIDERS, get_embedder
from .format import as_json, render, use_color
from .indexer import build_index
from .search import search
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
    find.add_argument("-n", "--limit", type=int, default=10)
    find.add_argument("--lang", action="append", help="filter by language (repeatable)")
    find.add_argument("--path", action="append", help="filter by path substring (repeatable)")
    find.add_argument("--kind", action="append", help="filter by kind: function, class, method, ...")
    find.add_argument("--mode", choices=["hybrid", "semantic", "lexical"], default="hybrid")
    find.add_argument("--fusion", choices=["blend", "rrf"], help="how the two rankings are merged")
    find.add_argument("-L", "--lines", type=int, default=8, help="snippet lines (0 = full chunk)")
    find.add_argument("--explain", action="store_true", help="show per-retriever ranks")
    find.add_argument("--json", action="store_true")
    find.set_defaults(handler=cmd_find)

    status = sub.add_parser("status", help="show index statistics")
    status.set_defaults(handler=cmd_status)

    clear = sub.add_parser("clear", help="delete the index")
    clear.set_defaults(handler=cmd_clear)

    providers = sub.add_parser("providers", help="list embedding providers")
    providers.set_defaults(handler=cmd_providers)
    return parser


def cmd_index(args) -> int:
    cfg = load_config(
        args.root,
        provider=args.provider,
        model=args.model,
        workers=getattr(args, "workers", None),
    )
    embedder = _embedder(cfg)
    store = Store(cfg.db_path)
    say = (lambda msg: None) if args.quiet else (lambda msg: print(msg, file=sys.stderr))
    say(f"provider {embedder.signature}, vectors via {store.vector_backend}")

    stats = build_index(cfg, embedder, store, reindex=args.reindex, progress=say)
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
        filters=Filters(langs=args.lang, paths=args.path, kinds=args.kind),
        mode=args.mode,
        fusion=args.fusion or "",
    )
    store.close()

    if args.json:
        print(as_json(hits))
    else:
        print(render(hits, snippet_lines=args.lines, explain=args.explain, color=use_color()))
    return 0 if hits else 1


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
