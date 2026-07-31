<img src="https://raw.githubusercontent.com/mekkadev/findyourcode/main/docs/header.svg" alt="findyourcode — search a codebase by what the code does, not the words in it" width="640">

<img src="https://raw.githubusercontent.com/mekkadev/findyourcode/main/docs/demo.gif" alt="fyc index, a search by meaning, a call path printed with --trace, and fyc doctor" width="880">

the first query is `reject a request without a valid ticket`. the word `reject`
appears nowhere in that repository. grep cannot do this; an llm reading the whole
repository can, but not in 40ms and not for free.

the second query answers with a path instead of a list — the middleware, what it
calls to verify the ticket, and what that calls in turn. that part is not
embeddings at all.

## install

```bash
pip install -e ".[all]"
```

the default model is 220mb, downloaded once, then everything runs offline. no
server, no docker, no api key. the index is a single `.findyourcode/index.db` next
to your code.

## use

```bash
fyc index                          # once per repo, incremental after that
fyc index --watch                  # keep it fresh while you work

fyc find "why do payments retry"
fyc find "where do we check the auth header" --trace
fyc find "rate limit" --lang python --path src/api
fyc find "worker entrypoint" -f paths | fzf
fyc similar src/auth/session.py:42 # where else does this pattern live
fyc doctor                         # why is it behaving like that
```

`fyc similar` answers a question `find` cannot, from vectors already in the index:

```console
$ fyc similar py311/queue.py:100
like py311/queue.py:97-109 method Queue.empty

 1. py311/asyncio/queues.py:95-97  method Queue.empty      [0.758]
 2. py311/asyncio/queues.py:86-88  method Queue.qsize      [0.732]
 3. py311/sched.py:98-101          method scheduler.empty  [0.711]
```

other commands: `status`, `clear`, `providers`, `eval`, `mcp`. flags worth
knowing: `-n` results, `-L` snippet lines, `--kind function`,
`--mode semantic|lexical`, `--explain`, `--json`, `--no-graph`.

## the call graph

embeddings answer *what looks like this*. they cannot answer *what is one call
away*, and the thing you asked about is often not the thing that implements it.
while tree-sitter has the file open, every call site and every definition is
recorded — no second parse, 2.7mb on a 15k-chunk index — and retrieval gets a
structural signal next to the two textual ones.

`--trace` prints the path rather than the page:

```console
$ fyc find "verify that a ticket has not expired" -n 1 --trace -L 3

 1. web/tickets.ts:8-18  function verifyTicket  [0.834]
    ↑ web/middleware.ts:8  guard
    → web/crypto.ts:15  constantTimeEqual
    → web/tickets.ts:20  rolesFor
     8 export async function verifyTicket(ticket: string): Promise<Principal | null> {
     9   const [login, expiry, mac] = ticket.split(".");
    10   if (!login || Number(expiry) * 1000 < Date.now()) {
    ... 8 more lines
```

`↑` is who reaches it, `→` is what it reaches, and which branch gets followed is
decided by the query rather than by the source order.

names are resolved conservatively, because a wrong edge is worse than a missing
one. only within one language. `linecache.getline` resolves to `linecache.py`
because the qualifier says which of the four `getline` definitions is meant. a
name defined in the caller's own file beats one from elsewhere. a name defined in
more than eight places — `run`, `handle`, `get` — is dropped rather than guessed.

the same edges feed ranking: up to five chunks that neither retriever returned,
but that a strong result calls or that call it, join the page *below* the best
direct answer and never above it. on ordinary queries that leaves the ranking
untouched — mrr 0.850 with the graph and without it. on questions whose answer
lives one call away, ten results find what text alone needs twelve to find.
both claims are one command each, in [benchmarks](https://github.com/mekkadev/findyourcode/blob/main/docs/BENCHMARKS.md).

## agents

`fyc mcp` serves the index over mcp on stdio: `search_code`, `find_similar`,
`index_status`. `search_code` takes `trace: true`, which is worth more to an agent
than to a human — it answers "how does the request get here" in one call instead
of four file reads. this is a retriever to put *next to* an agent's grep, not a
replacement for it: claude code ships no index on purpose, and cursor's own
numbers say grep and semantic together beat either alone.

```bash
claude mcp add findyourcode -- fyc -C /path/to/repo mcp
```

or, for anything that reads a json config:

```json
{ "mcpServers": { "findyourcode": { "command": "fyc", "args": ["-C", "/path/to/repo", "mcp"] } } }
```

it is the same index the cli uses, so `fyc index` (or `--watch`) keeps the agent
current too.

## how it works

`tree-sitter` cuts each file at real boundaries — a function, a method, a class —
instead of every n bytes. a small class stays whole, a large one becomes a header
plus one chunk per method, a huge function becomes overlapping windows. anything
without a grammar falls back to line windows, so no file drops out of the index.

each chunk is rewritten before it is embedded. identifiers are split into words, so
`checkUserCredentials` reads as `check user credentials`. the path, the enclosing
class, the docstring, and the file's own summary when it has one, all go in front
of the code — most models only ever read the first 128–512 tokens:

```text
python class CredentialChecker
about: Validates the login/password pair a client presents at sign-in.
names: credential checker validates login password pair client presents sign init
       users secret bytes check record permission unknown digest hashlib pbkdf2 ...
file: api/session.py (api session)
code:
class CredentialChecker:
    ...
```

that `names:` line is why a query never phrased like the code still lands on it.

retrieval is two searches over one sqlite file — vectors through `sqlite-vec`, bm25
through `fts5` — blended 0.75 to the vector side, then the graph adds what neither
of them could reach. candidates that only bm25 returned get an exact cosine before
ranking, otherwise one incidental keyword match outranks the right answer.

the index is incremental twice over: files by sha256, chunks by the hash of the text
that goes to the model. editing one function re-embeds one function.

## numbers

cpython 3.11 stdlib — 672 files, 15k chunks, one cpu, default model:

```
first index      305s
re-index         2.4s    nothing changed
search            42ms   over 15k chunks
cold start       2.6s    python starting and the model loading
on disk           71mb   of which the call graph is 2.7mb
```

36 queries against the whole stdlib — 26 by meaning, 10 by exact identifier:
recall@1 0.81, recall@10 0.92, mrr 0.850. split them and the reason for two
retrievers shows. on the identifier half the vectors collapse to recall@1 0.50
where bm25 gets 0.90; on the meaning half bm25 is the one that falls behind. a
benchmark of one shape only would have argued convincingly for deleting the
branch that saves the other half.

ask in russian and the same 26 questions still land on english code: recall@10
0.81 against 0.88 for english, where bm25 alone gets 0.27.

the settings that lost are printed next to the settings that won —
[docs/BENCHMARKS.md](https://github.com/mekkadev/findyourcode/blob/main/docs/BENCHMARKS.md) has the fusion sweep, the multi-hop set,
the multilingual control, and the cross-encoder rerank that was measured and
rejected. don't take any of it on faith; it is one command on your own code:

```bash
python scripts/benchmark.py                            # this python's stdlib
python scripts/benchmark.py --corpus ~/work/monorepo --cases my_cases.json
```

## models

`local` runs offline through onnx and is the default. `voyage` (`voyage-code-3`,
best on code) and `openai` need a key. `hash` is a deterministic lexical stand-in
with no dependencies — the test suite runs on it.

```bash
fyc index --model intfloat/multilingual-e5-large
fyc index --provider voyage
```

the index remembers which model built it and refuses to mix vector spaces, so
changing models means `fyc index --reindex` rather than silently wrong results.

## config

`.findyourcode.toml` in the project root, overridden by `FYC_<FIELD>` or a flag.

```toml
[findyourcode]
provider = "local"
max_chunk_lines = 110      # larger chunks, coarser addressing
alpha = 0.75               # weight of the semantic branch
per_file = 2               # so one file cannot own the page
graph_weight = 0.65        # how loudly a call edge argues for a chunk
graph_limit = 5            # how many the graph may add to a page
exclude = ["**/generated/**", "*.pb.go"]
```

files come from `git ls-files -co --exclude-standard`, so `.gitignore` is honoured
for free. outside a git repo it walks the tree with built-in exclusions.

## languages

python, javascript, typescript, tsx, go, rust, java, kotlin, swift, scala, ruby,
php, c#, c, c++, lua, bash, elixir and the rest of `tree-sitter-language-pack`.
unknown extensions still get indexed through the line-window fallback.

## development

```bash
pytest                     # 141 tests on the hash provider — offline, deterministic
FYC_TEST_REAL_MODEL=1 pytest tests/test_real_model.py   # the real model, ~220mb
cd examples/demo_repo && fyc index && fyc find "checking the password on sign in"
```

`scripts/benchmark.py` reproduces the numbers above and `scripts/record_demo.py`
re-records the gif at the top from live output.

`walker` picks files, `chunker` cuts them, `graph` reads the calls out of the same
parse, `enrich` writes what gets embedded, `store` is sqlite, `search` is the two
retrievers, the blend and the propagation, `evaluate` is recall and mrr,
`diagnose` is `doctor`, `mcp_server` is the agent surface, `cli` is yours. see
`contributing.md`.

## prior art

this is a crowded idea and pretending otherwise would be silly. the neighbours
worth knowing, and what they do better:

- [CodeRAG](https://github.com/Neverdecel/CodeRAG) — the closest match. tree-sitter
  chunking, hybrid dense + bm25, incremental, mcp, and its own eval harness with
  more metrics than this one. also ships reranking and a rest api.
- [claude-context](https://github.com/zilliztech/claude-context) — the same idea for
  agents, at a hundred times the mindshare. needs milvus and an embedding provider.
- [chunkhound](https://github.com/chunkhound/chunkhound) — duckdb + hnsw, git-history
  search, multi-hop retrieval. its good path wants an api key or ollama.
- [seagoat](https://github.com/kantord/SeaGOAT) — local-first, chromadb, unions
  vector hits with ripgrep. runs a daemon per repository, no mcp.
- [qmd](https://github.com/tobi/qmd) — same storage stack exactly (fts5 + sqlite-vec
  + rrf + local model + mcp), aimed at notes rather than code.
- [ripgrep](https://github.com/BurntSushi/ripgrep) — wins outright on every query
  where you know the string. that is most queries. this tool is for the rest.

what is actually different here is that retrieval is not only embeddings. the call
graph tree-sitter hands over for free is used twice — once to rank what similarity
cannot reach, once to answer with a path instead of a page. the direction is not
mine: cursor puts a symbol layer in front of its embeddings and aider ranks with
pagerank over tree-sitter and no embeddings at all. what is mine is carrying both
in one sqlite file you can pip install next to your repository.

beyond that: the enrichment step is explicit and documented rather than
incidental, every weight was chosen by measurement with the losing configurations
published, bm25-only candidates get an exact cosine before ranking, and there is
no server, no daemon, no vector database and no api key.

## next

- edges from imports and type references, not only calls
- query expansion for one-word queries
- publish to pypi

## stack

`python` `tree-sitter` `sqlite-vec` `fts5` `onnx` `numpy`

mit
