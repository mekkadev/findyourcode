# changelog

notable changes, newest first. semver.

## [0.3.0] - 2026-07-31

### added

- `fyc mcp` — an mcp server over stdio, so an agent searches the same index you
  do: `search_code`, `find_similar` and `index_status`, with both text and
  structured results. no dependency; the json-rpc surface is hand-rolled.
- `fyc doctor` — python, sqlite, sqlite-vec, fts5, grammars, git, providers, and
  whether the index still covers what is on disk.
- packaging for release: `py.typed`, classifiers, keywords, project urls, and a
  tag-triggered pypi workflow using trusted publishing.
- ci now lints (`ruff`), type-checks (`mypy`), measures coverage, builds the
  distribution, runs on macos and windows and python 3.13, and answers an mcp
  handshake in the smoke job.
- contributing, security, changelog, issue and pull request templates,
  dependabot.
- `scripts/benchmark.py`, so the numbers in the readme can be checked in one
  command, on this python's stdlib or on your own repository.
- `scripts/record_demo.py`, which types the readme demo at human speed and runs
  it for real inside a pty — the gif is the tool's own output, not a mock-up.
- an svg header and a social preview card under `docs/`.

- `--rerank`, a cross-encoder second pass over the shortlist. off by default
  because it was measured and it loses: recall@1 drops from 0.83 to 0.78 with
  every reranker tried. it earns recall@10 0.97 against 0.92, so the flag stays,
  with the numbers beside it.
- one opt-in test against the real embedding model, so the path a user actually
  takes is not left entirely to the deterministic stand-in.
- a prior-art section naming the neighbours and what they do better.

### fixed

- `fyc doctor` called an index fresh when a file had been edited: it compared the
  set of paths and threw the hashes away. it also reported every check green on an
  index whose vector backend no search could read.
- the mcp server died on any json line that was not an object, returned whole
  chunks twice — 19 kb for one search — served a stale handle after a reindex,
  accepted an invalid `mode` without complaint, and ignored a configured reranker
  that the cli would have honoured.
- `fyc eval --sweep` returned before the `--min-mrr` gate could run, silently
  disabling a ci threshold; its blend rows inherited `fusion` from config, so under
  `fusion = "rrf"` all five measured rrf; and every report claimed recall@10 no
  matter how small `--limit` was.
- a query with no word characters (`()`, `...`, `&&`, or an empty argument)
  crashed with a traceback: a zero-length vector has no cosine, sqlite-vec
  returns null for it, and those rows sort first. they are dropped now.

### changed

- 94 tests, up from 67, and 91% coverage. the new ones cover the http
  providers, the local provider's e5 prefixes, `render`, rrf fusion, the config
  loader, the numpy vector path, and the indexer's error accounting — all
  places where a mutation used to survive the suite.

## [0.2.0] - 2026-07-31

### added

- `fyc similar path:line` — nearest chunks to a place in the codebase, from
  vectors already in the index.
- `fyc eval` — recall@k and mrr over a file of query/expected-file cases, with
  `--sweep` to compare fusions and alpha values and `--min-mrr` / `--min-recall`
  to fail ci below a threshold.
- `fyc index --watch`, which keeps the index fresh until ctrl-c.
- output formats `-f pretty|paths|files|json`, and `--explain` for per-retriever
  ranks.
- a per-file cap, so one file cannot own a result page.
- rate and eta while an index builds.

### changed

- bm25-only candidates get an exact cosine before ranking, so one incidental
  keyword match no longer outranks the right answer.
- vectors stored once instead of twice, and freed pages reclaimed incrementally.
- default blend measured rather than guessed: 0.75 to the semantic branch.

### fixed

- `--reindex` after a model switch handed the previous model's vectors to the
  new one.
- a query-time `--provider` or `--model` that disagrees with the index now warns
  instead of returning results from the wrong vector space.
- a language with no recognised definition nodes was passed an empty language
  rather than a named guess.
- a closed pipe (`fyc find ... | head`) no longer raises on exit.
- defects from two adversarial review rounds, and two gaps in the suite that
  mutation testing exposed.

## [0.1.0] - 2026-07-31

### added

- semantic code search over tree-sitter chunks: real boundaries — function,
  method, class — with a line-window fallback so no file drops out of the index.
- enrichment before embedding: split identifiers, path, enclosing class,
  docstring and file summary in front of the code.
- hybrid retrieval over one sqlite file — vectors through `sqlite-vec`, bm25
  through `fts5`.
- incremental indexing, by file sha256 and by the hash of the embedded text.
- providers: `local` (onnx, offline, default), `voyage`, `openai`, `hash`.
- `fyc index / find / status / clear / providers`, `.findyourcode.toml` and
  `FYC_*` overrides, and ci on python 3.10-3.12.
