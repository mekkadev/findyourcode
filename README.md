<img src="https://raw.githubusercontent.com/mekkadev/findyourcode/main/docs/header.svg" alt="findyourcode — search a codebase by what the code does, not the words in it" width="640">

<img src="https://raw.githubusercontent.com/mekkadev/findyourcode/main/docs/demo.gif" alt="fyc index, two searches by meaning, and fyc doctor, on a five-file repository" width="880">

the first query is `reject a request without a valid ticket`. `reject` appears
nowhere in that repository, and the two answers it returns are in different
languages. grep cannot do this. an llm reading the whole repository can, but not
in 40ms and not for free.

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
`--mode semantic|lexical`, `--explain`, `--json`.

## agents

`fyc mcp` serves the index over mcp on stdio: `search_code`, `find_similar`,
`index_status`. this is a retriever to put *next to* an agent's grep, not a
replacement for it — claude code ships no index on purpose, and cursor's own
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
through `fts5` — blended 0.75 to the vector side. that number was measured, not
guessed. candidates that only bm25 returned get an exact cosine before ranking,
otherwise one incidental keyword match outranks the right answer.

the index is incremental twice over: files by sha256, chunks by the hash of the text
that goes to the model. editing one function re-embeds one function.

## numbers

cpython stdlib — 671 files, 15k chunks, one cpu, default model:

```
first index      329s
re-index         0.5s    nothing changed
search           40ms    over 15k chunks
cold start       2.6s    python starting and the model loading
on disk          72mb
```

`fyc eval` scores retrieval against a file of query/expected-file pairs, and
`--sweep` compares settings on it:

```console
$ fyc eval examples/eval_stdlib.json --sweep

  setting           recall@1  recall@3  recall@10     MRR
  -------------------------------------------------------
  blend a=0             0.67      0.83       0.92   0.760
  blend a=0.25          0.69      0.86       0.92   0.789
  blend a=0.5           0.75      0.89       0.92   0.826
  blend a=0.75          0.83      0.92       0.92   0.870   <- default
  blend a=1             0.69      0.83       0.89   0.756
  semantic              0.69      0.83       0.89   0.756
  lexical               0.67      0.83       0.92   0.760
  rrf                   0.81      0.86       0.94   0.851
```

36 queries against the whole stdlib: 26 by meaning, 10 by exact identifier. split
them and the reason for two retrievers shows. on the meaning half, semantic alone
matches the blend — mrr 0.808 against 0.821. on the identifier half it collapses to
recall@1 0.50 while the blend gets all ten. a benchmark of one shape only would
have argued convincingly for deleting the branch that saves the other half.

36 hand-written queries against one corpus is a smoke test with a method attached,
not a benchmark — read it as evidence that the decisions were measured, not as a
quality claim.

recall@1 is 0.83, so roughly one query in six puts the right file below the top. the
ceiling is the default model's 128-token window — it reads the name, the docstring
and the file summary, not the body. swap in `intfloat/multilingual-e5-large` or
`voyage-code-3` and run the same eval on your own repo rather than trusting either
number.

don't take the table on faith — it is one command, and it works on your code too:

```bash
python scripts/benchmark.py                            # this python's stdlib
python scripts/benchmark.py --corpus ~/work/monorepo --cases my_cases.json
```

## rerank, measured and rejected

a cross-encoder reads the query and the chunk together, so it should beat a vector
computed without ever seeing the query. on this corpus it did not:

```
                                  recall@1  recall@3  recall@10     MRR
  no rerank                           0.83      0.92       0.92   0.870
  ms-marco-MiniLM-L-6-v2              0.78      0.86       0.92   0.828
  jina-reranker-v1-turbo-en           0.69      0.83       0.92   0.775
  jina-reranker-v2-base-multilingual  0.78      0.86       0.97   0.824
```

all three are trained on prose retrieval and code is out of distribution for them.
the multilingual one earns something in exactly one place — it lifts correct files
out of the tail of the shortlist into the top ten, 0.92 to 0.97 — and pays for it
in precision at one and 2.5 seconds a query. so it ships off by default, with the
numbers next to the flag:

```bash
fyc find "..." --rerank
fyc eval my_cases.json --rerank      # settle it on your corpus, not mine
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

the default model is multilingual: ask in any of ~50 languages and it still finds
english code, which is the point when the codebase is english and you are not.

## config

`.findyourcode.toml` in the project root, overridden by `FYC_<FIELD>` or a flag.

```toml
[findyourcode]
provider = "local"
max_chunk_lines = 110      # larger chunks, coarser addressing
alpha = 0.75               # weight of the semantic branch
per_file = 2               # so one file cannot own the page
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
pytest                     # 106 tests on the hash provider — offline, deterministic
FYC_TEST_REAL_MODEL=1 pytest tests/test_real_model.py   # the real model, ~220mb
cd examples/demo_repo && fyc index && fyc find "checking the password on sign in"
```

`scripts/benchmark.py` reproduces the numbers above and `scripts/record_demo.py`
re-records the gif at the top from live output.

`walker` picks files, `chunker` cuts them, `enrich` writes what gets embedded,
`store` is sqlite, `search` is the two retrievers and the blend, `evaluate` is
recall and mrr, `diagnose` is `doctor`, `mcp_server` is the agent surface, `cli`
is yours. see `contributing.md`.

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

what is actually different here: the enrichment step is explicit and documented
rather than incidental, the blend weight and the per-file cap were chosen by
measurement with the losing configurations published, bm25-only candidates get an
exact cosine before ranking, and the whole thing is one sqlite file with no server,
no daemon, no vector database and no api key.

## next

- cross-encoder rerank over the top of the blend
- query expansion for one-word queries
- publish to pypi

## stack

`python` `tree-sitter` `sqlite-vec` `fts5` `onnx` `numpy`

mit
