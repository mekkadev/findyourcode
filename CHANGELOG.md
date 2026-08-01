# changelog

notable changes, newest first. semver.

## [0.4.0] - 2026-07-31

### added

- a call graph, extracted while tree-sitter already holds the parse, so it costs
  nothing at index time and 2.7mb on 15k chunks. every call site and every
  definition is stored as a name rather than a resolved edge, so resolution
  happens at query time and can never go stale behind an incremental re-index.
- `fyc find --trace` — the call path around each hit instead of a list of hits:
  what reaches it, and what it reaches, following the branch the query is about.
- ranking now uses those edges. up to five chunks that neither retriever
  returned, but that a strong result calls or that call it, join the page below
  the best direct answer — never above it. on ordinary queries the ranking is
  unchanged (mrr 0.850 either way); on a set of 17 multi-hop questions ten
  results find what text alone needs twenty to find.
- `search_code` takes `trace: true` over mcp, so an agent can follow a flow
  instead of reading four files to reconstruct it.
- `examples/eval_multihop.json` and `examples/eval_stdlib_ru.json`, and
  `docs/BENCHMARKS.md` with the method and the losing configurations.
- `fyc eval --no-graph` and a `no graph` row in `--sweep`, so the claim above is
  one command.
- `fyc status` reports the size of the graph.
- a reach window, which is what makes the graph safe to listen to. a call edge is
  evidence about *which* nearly-relevant chunk to surface, never that an
  irrelevant one is relevant, so the dense index is asked for 400 candidates
  instead of 80: the first 80 are the ranking exactly as before and the rest only
  answers whether a call neighbour is somewhere the query points at all. of
  everything tried — corroboration by several results, direction, the rank of the
  result it came from, the candidate's own cosine — this is the only thing that
  separated a neighbour worth having from one worth ignoring, and it is a rank
  rather than a cosine because the cosine scale shifts per query. with it
  `graph_weight` stops being a trade: 0.65 through 0.95 all leave the ordinary set
  at mrr 0.850 with not one question changing rank, so it now ships at 0.85 and
  the multi-hop set keeps recall@10 0.59 against 0.41 without the graph. costs one
  deeper read, 38ms → 45ms per query.

### measured and not shipped

- a second propagation hop. the premise holds — for 5 of the 17 multi-hop
  questions the answer is two calls away, and two hops do reach it — but the
  candidate pool goes from ~40 to ~250 for the same five slots, so it arrives
  ranked 285th of its own pool. worse, at `graph_weight = 0.8` it hands back the
  whole win the graph buys there, recall@10 0.53 → 0.41.
- imports read into a name → module map, to settle a bare call to an imported
  name. eight of 1265 ambiguous references on the stdlib, zero on npm's own
  source, no eval number moved.

both are in `docs/BENCHMARKS.md` with their tables. neither is in the code.

### fixed

two adversarial review passes over the new code, each finding reproduced before
it was believed:

- the fanout cap counted definitions across all languages, so five javascript
  `handler`s deleted the edges of five python ones — in exactly the mixed-language
  repositories the graph is for.
- elixir writes a definition as a call (`def deliver do`), so every call site
  claimed to define its callee and the whole language ended up with no calls at
  all. a grammar that cannot tell the two apart now keeps only what its chunk is
  named after.
- java, ruby and php hang the receiver on the call node rather than on the callee,
  so `Linecache.getline(p)` lost its qualifier and resolved half to the wrong
  file. reading the qualifier off the tree instead of off the text fixed that and
  made a deeply chained expression linear rather than quadratic — 8000 chained
  calls went from 2.9s to 0.11s.
- a reranker rewrites every score from scratch, ceiling included, so it could put
  a chunk no retriever returned at the top of the page.
- `--path` cost the graph its slots: five neighbours were chosen and only then
  filtered, leaving the eligible ones behind.
- `trace_depth = 0` still expanded one hop; a shared `visited` set hid a helper
  that two branches of a call tree both reach; a chunk whose vector was missing
  was dropped from a trace instead of ranked last.
- `edges_to` gave one popular name the whole row budget, starving the rare name
  defined beside it. each name gets a share now.
- a `defs` table dropped for the wrong shape was recreated empty and never
  refilled, because only `refs` was checked.
- a fresh index was 47% larger than its contents: `PRAGMA auto_vacuum` was issued
  after `journal_mode=WAL`, which silently discards it, so nothing was ever
  reclaimed. every vector was also copied into the archive table it was about to
  make redundant. the stdlib index is 70.6mb where it was 103mb.
- `fyc eval --sweep` left alpha at 1.0 for the rows printed after the alpha
  sweep, which mattered as soon as one of them was not a pure semantic or
  lexical run.

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

- `fyc index --provider voyage` and `fyc index --model ...` — both documented in
  the readme — exited 2 with "unrecognized arguments", because the global flags
  only parsed before the subcommand. either order works now, and a value given
  before the verb is no longer erased by the subparser's defaults.
- a model name the provider cannot load printed a traceback instead of the
  message inside it.
- a filter cost a full brute-force scan of everything it matched: `--lang python`
  on the stdlib index took 0.36s against 0.06s unfiltered. the vector index is
  asked first now and the scan only runs when the approximate answer cannot fill
  the page — same results, 6x faster.
- opening an index ran `CREATE TABLE IF NOT EXISTS` and an auto_vacuum pragma, so
  every `fyc find` took a write lock beside a running `--watch`.
- a single minified line printed in full: one 400 kb line filled the terminal.
- `fyc index --watch` reported successful passes forever after its index was
  deleted underneath it, writing to an unlinked file.
- indexing held up to 512 files in memory at once regardless of their size.
- a file that produced no chunks — an empty file, a file of only whitespace — was
  never recorded, so every later run counted it as new and `fyc doctor` could
  never report the index as fresh.
- `fyc --version` reported 0.2.0 while the distribution was 0.3.0. the version is
  read from the installed metadata now, so the two cannot drift again.
- the sdist carried `tests/` without `conftest.py`, so the suite it shipped could
  not run, and left out examples, scripts and docs entirely. a manifest fixes it,
  and the suite was run from inside the built sdist to prove it.
- readme images used repository-relative paths, which resolve on github and break
  on pypi, where the readme is the landing page.
- `fyc doctor` claimed .gitignore was honoured whenever git was installed, without
  checking whether the root is a repository at all.
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
