# contributing

## setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[all]"
pytest
```

131 tests, all on the `hash` provider: deterministic, offline, no model download.
keep it that way — a test that needs the network or the 220mb default model does
not belong in the suite. ci runs the same on python 3.10, 3.11 and 3.12, plus a
smoke test over `examples/demo_repo`, which is also how to see it work for real:

```bash
cd examples/demo_repo && fyc index && fyc find "checking the password on sign in"
```

## where things live

`walker` picks files, `chunker` cuts them, `graph` reads the calls out of the
same parse, `enrich` writes what gets embedded, `store` is sqlite, `search` is
the two retrievers, the blend and the graph propagation, `evaluate` is recall and
mrr, `cli` is the surface. a change usually belongs in one of them.

## changes that touch ranking

anything in `search`, `enrich`, `graph` or `chunker` moves the numbers, so
measure it instead of arguing about it:

```bash
fyc eval ../eval_demo.json          # from inside examples/demo_repo
fyc eval examples/eval_stdlib.json --sweep     # against an indexed cpython stdlib
fyc eval examples/eval_multihop.json --no-graph # what the call graph is worth
```

put recall@1, recall@3, recall@10 and mrr before and after in the pull request.
if a change wins on one query shape, show the other — the stdlib eval is 26
meaning queries and 10 identifier queries, and they disagree.

## conventions

- python 3.10+, standard library first. a new runtime dependency needs a reason;
  anything heavy goes in an optional extra, as `fastembed` and `sqlite-vec` do.
- no formatter is enforced. match the file you are editing. one change per pull
  request.
- commit subjects in the imperative, saying what the commit does: "cap how many
  chunks one file may take in a result page".
- the index records the model that built it and refuses to mix vector spaces. if
  you change what goes into a vector, a chunk id or the schema, say so in the
  pull request — users will need `fyc index --reindex`.
- new languages come from `tree-sitter-language-pack`; add the extension and the
  definition node names in `languages.py`, with a test.

## bugs

open an issue with `fyc --version`, the provider and model, the exact command
and what you saw. a small repository that reproduces it is worth more than a
description of one.
