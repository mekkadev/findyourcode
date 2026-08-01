# benchmarks

every number here comes from one command on one corpus, and every command is in
this repository. nothing is averaged over runs it did not have, and the settings
that lost are printed next to the settings that won.

**corpus** — the cpython 3.11 standard library as it ships in this container:
672 files, 15 000 chunks, 14 439 symbols, 37 634 call sites.
**model** — the default `local` provider, `paraphrase-multilingual-MiniLM-L12-v2`
through onnx, one cpu.

```bash
python scripts/benchmark.py            # index this python's stdlib and score it
```

a hand-written set against one corpus is a smoke test with a method attached, not
a benchmark. read it as evidence that the decisions were measured, not as a claim
about your repository.

## cost

```
first index      305s     672 files, 15k chunks
re-index         2.4s     nothing changed
search            41ms    over 15k chunks
  without the graph 29ms
cold start       2.6s     python starting and the model loading
on disk         70.6mb    of which the call graph is 2.7mb
```

## fusion

`examples/eval_stdlib.json` — 26 questions phrased as meaning, 10 as an exact
identifier.

```console
$ fyc eval examples/eval_stdlib.json --sweep

  setting            recall@1   recall@3  recall@10     MRR
  ---------------------------------------------------------
  blend a=0              0.67       0.83       0.92   0.756
  blend a=0.25           0.69       0.83       0.92   0.776
  blend a=0.5            0.75       0.92       0.92   0.829
  blend a=0.75           0.81       0.89       0.92   0.850
  blend a=1              0.64       0.83       0.89   0.727
  no short blend         0.81       0.89       0.92   0.850
  default                0.83       0.92       0.92   0.870
  semantic               0.64       0.78       0.89   0.720
  lexical                0.67       0.81       0.92   0.748
  rrf                    0.81       0.86       0.94   0.850
  no graph               0.83       0.92       0.92   0.870
```

the alpha rows are measured with one blend for every query, or the sweep would be
reporting the short-query blend below instead of alpha. `default` is everything
as shipped; the rows around it are what each piece is worth.

split the set by shape and the reason for keeping two retrievers is plain:

```
                        recall@1  recall@3  recall@10     MRR
  ------------------------------------------------------------
  26 by meaning
    blend                   0.77      0.88       0.88   0.821
    semantic only           0.69      0.85       0.88   0.756
  10 by identifier
    blend                   1.00      1.00       1.00   1.000
    semantic only           0.50      0.60       0.90   0.625
    lexical only            0.90      0.90       1.00   0.925
```

neither branch survives alone: the vectors lose half the identifier queries, and
bm25 is the one that falls behind on meaning. the blend now beats both on both
halves — the identifier half only after the short-query blend below.

blend and rrf finish level on this corpus, 0.850 against 0.850 before that. blend
stays the default for a reason the aggregate cannot show: rrf ranks by position
only, so a single incidental keyword match at bm25 rank 1 can outrank the right
answer, and that is what it did to `card.Token` on the demo repository. the blend
sees the cosine as well and does not.

## how much query there is to read

`fyc find "checksum"` is the most common thing a person types and the worst
served. `examples/eval_oneword.json` — 16 one- and two-word queries, written
down before anything was measured:

```
                          recall@1  recall@3  recall@10     MRR
  ---------------------------------------------------------------
  one blend for everything      0.75      0.81       1.00   0.802
  short queries at alpha 0.55   0.88      1.00       1.00   0.938
```

a sentence is mostly ordinary english and the model reads it better than bm25
ever will. `epoll` is not a sentence: it is one rare token naming a thing that
lives in exactly one file, the model has never seen it used as a word, and it
answers `colorsys.py` — while bm25, which needs nothing but the posting list, had
`selectors.py` all along. so the two retrievers are weighted by how much query
there is to read: under three words, alpha 0.55 instead of 0.75.

it costs nothing — alpha is one multiplier applied after retrieval — and long
queries are untouched, checked literally rather than statistically: all 46
sentence-shaped queries in this repository return byte-identical pages, file,
line and score, with the blend on and off. 0.40 to 0.65 all score the same on 16
cases, so 0.55 is the middle of a plateau rather than a tuned value.

## the call graph

two properties matter and they pull against each other: the graph must not
disturb ordinary queries, and it must find things text cannot.

**it does not disturb ordinary queries.** the `no graph` row of the sweep is
identical to `default`, question for question. that is by construction: a chunk
reached through the graph enters the page below the best direct answer and never
above it, at most five join a page, and one that the query ranks nowhere at all
is dropped however loudly the structure argues for it. the text results keep
their order and their scores to every decimal the output prints; the one thing a
graph row does take from them is a slot under the per-file cap, which counts it
like any other result.

**multi-hop questions.** `examples/eval_multihop.json` — 17 questions whose
answer is a module one call away from the module the question describes.
`argparse` decides how wide to wrap help text by calling
`shutil.get_terminal_size`; the question is about help text, the answer is in
`shutil.py`.

```
                     multi-hop                ordinary
                  r@3   r@10     MRR       r@3     MRR
  ------------------------------------------------------
  no graph       0.24   0.41   0.238      0.92   0.870
  graph          0.29   0.59   0.280      0.92   0.870
```

ten results find what text alone needs twenty to find.

seventeen cases is a small set and each case is worth 0.06, so treat the gap as
a direction, not a measurement. how it was built, so you can judge it: a
candidate was written down as a (question, calling module, implementing module)
triple before anything was run, and kept if the index contained an edge between
the two modules. two of the twenty-two candidates were dropped because the call
turned out not to exist — `pickle` does not call `copyreg` by name, `inspect`
does not call `dis` by name. none was dropped for scoring badly.

## what makes the graph safe to listen to

for a while it was not. the graph had to be kept quiet — `graph_weight` 0.65 —
because turning it up traded one set against the other: at 0.8 the multi-hop set
went to recall@10 0.53 and the ordinary set gave up recall@3.

the number of candidates was never the problem. `graph_limit` from 3 to 20 does
not move either set by a single case: what binds is the score a graph-only chunk
is allowed, and turning that up let the wrong neighbours in with the right ones.
so the question was whether anything cheap tells them apart. most of what you
would reach for does not:

```
  1877 graph candidates over 53 queries, 141 of them a gold file (7.5%)

  predicate                              kept    gold   precision   vs base
  --------------------------------------------------------------------------
  reached by two or more results          330      25       0.076      1.0x
  it is called by a result                1281     86       0.067      0.9x
  it calls a result                        586     49       0.084      1.1x
  the strongest result was rank 1          352     21       0.060      0.8x
  the name resolved exactly, not guessed   752    116       0.154      2.1x
  the query ranks it in the top 400        299     82       0.274      3.7x
```

corroboration is worthless. direction is worthless. the rank of the result it
came from is worthless. what separates them is where the *query* puts the
candidate — not how strongly something points at it.

not its cosine, though: absolute thresholds, cosine relative to the top hit and
cosine relative to the retrieval floor all cost ordinary recall, because the
scale shifts from query to query. the rank does not. so the dense index is asked
for 400 instead of 80 — the first 80 are the ranking exactly as before, and the
rest is consulted only to ask whether a call neighbour is somewhere the query
points at all. the neighbours that used to do the damage sit at rank 700, 1500,
2000 or outside the top 3000; the ones worth having sit between 120 and 640.

with that gate the weight stops being a trade. `graph_weight` from 0.65 to 0.95
all leave the ordinary set at mrr 0.870 with not one question changing rank, so
the default is 0.85 and the multi-hop set keeps the whole gain.

it costs one deeper read, 29ms → 41ms per query, and it is not free of misses
either: 12 of the 17 multi-hop questions have their gold module among the graph
candidates and 4 of those sit outside the window, at rank 635, 640, 970 and
1417. widening it lets the damaging ones back in.

a window has to be able to leave something out to be worth anything, and twice
there is nothing it can leave out: `--mode lexical` embeds nothing, and a
repository smaller than the window has every chunk inside it. `-n 350` is the
same situation from the other end — the page has grown until the window is
barely wider than it. in all three the graph goes back to arguing at the 0.65 it
was measured safe at without a gate, rather than pretending to a gate it does
not have.

## a second hop, measured and rejected

propagation follows one call edge. following two looks obviously right — and the
premise checks out: for 5 of the 17 multi-hop questions the gold module is two
calls from the text match, not one, and a second hop does reach them.

it changes nothing, at any decay (measured against the `graph_weight = 0.65` of
the time, before the reach window existed):

```
  decay      multi-hop r@10   stdlib MRR   russian MRR
  -----------------------------------------------------
  0 (off)              0.47        0.850         0.611
  0.25                 0.47        0.850         0.609
  0.4                  0.47        0.850         0.607
  0.5                  0.47        0.850         0.606
  0.7                  0.47        0.850         0.607
```

not one case moves, because supply was never the problem. one hop offers about
40 candidates for five slots; two offer about 250, and the newly reached gold
arrives ranked 285th, 40th, 32nd, 15th and 7th of its own pool. handing out more
slots makes it worse rather than better — at `graph_limit = 20` the second hop
drops multi-hop recall@10 from 0.47 to 0.41 — and the one setting where the
graph genuinely bought recall gave all of it back:

```
                                     multi-hop r@10     MRR
  ------------------------------------------------------------
  graph_weight 0.8, one hop                     0.53   0.257
  graph_weight 0.8, second hop at 0.5           0.41   0.239
```

being two calls from a text match is barely evidence at all; on this corpus it
is mostly `os.path`. it cost about a millisecond, and it is not in the code.

## query expansion, measured and rejected

the obvious fix for a one-word query is to make it longer: run it, mine the top
few chunks for their most distinctive identifiers, add the best of them, run
again. terms scored rm3-style — occurrences across the feedback chunks times
idf, with the document frequencies read out of the fts5 vocabulary so the rarity
is the one bm25 already believes in.

```
  16 one- and two-word queries         recall@1   MRR   ms/query
  ---------------------------------------------------------------
  no expansion                             0.75  0.802        26
  +6 terms from 5 chunks                   0.44  0.568        56
  +3 terms from 3 chunks, vectors only     0.81  0.823        56
  +2 terms from 3 chunks, bm25 only        0.81  0.828        47
  +2 terms, required in 2 chunks           0.88  0.919       108
  the short-query blend instead            0.88  0.938        24
```

the failure is structural rather than a tuning miss. feedback comes from the
ranking, so it can only amplify a ranking that was already right — and twelve of
the sixteen were already at rank 1, where expansion has nothing to win and a page
of near-misses to lose. the four with headroom are exactly the ones where the
first pass is lost, so the terms it mines are lost too:

```
  mmap  -> _pydecimal.py, ElementTree.py   adds: logb, msd, magnitude
  epoll -> colorsys.py, turtle.py          adds: color, spectrum, hue
```

the best expansion setting also cost the ordinary set, 0.850 → 0.814. what the
sweep pointed at instead was the alpha, and that is what shipped.

## imports as a qualifier, measured and rejected

`linecache.getline(...)` says which of the four `getline` definitions is meant.
`from linecache import getline` followed by a bare `getline(...)` says exactly
the same thing one statement earlier, so reading the import statements into a
name → module map should settle a pile of otherwise ambiguous calls. it settles
almost none:

```
                          ambiguous cross-file calls   settled by a qualifier
  ---------------------------------------------------------------------------
  cpython stdlib, without the import map          1265                    118
  cpython stdlib, with it                         1265                    126
  npm's own source (110 js files), without         324                     95
  npm's own source, with it                        324                     95
```

eight calls out of 1265, and not one eval number moved. javascript, where the
pattern should have been strongest, gains nothing at all: npm is written in
commonjs, so the imports are `require()` calls and never look like imports to a
grammar. forty lines and a regex-based statement parser, deleted.

## asking in another language

the default model is multilingual, which is the point when the code is in
english and you are not. the same 26 questions, translated:

```
                     recall@1  recall@3  recall@10     MRR
  ---------------------------------------------------------
  english                 0.77      0.88       0.88   0.821
  russian                 0.54      0.69       0.81   0.632
  russian, bm25 only      0.12      0.15       0.27   0.154
```

the third row is the control: without the vectors a russian question finds
almost nothing, so the second row is the embedding working and not an accident
of shared tokens.

```bash
fyc eval examples/eval_stdlib_ru.json
```

## rerank, measured and rejected

a cross-encoder reads the query and the chunk together, so it should beat a
vector computed without ever seeing the query. on this corpus it does not:

```
                                    recall@1  recall@3  recall@10     MRR   per query
  -----------------------------------------------------------------------------------
  no rerank                             0.83      0.92       0.92   0.870       0.04s
  ms-marco-MiniLM-L-6-v2                0.72      0.89       0.94   0.814       1.7s
  jina-reranker-v1-turbo-en             0.69      0.81       0.92   0.769       1.9s
  jina-reranker-v2-base-multilingual    0.75      0.83       0.94   0.804      10.3s
```

all three are trained on prose retrieval, and code is out of distribution for
them. two of them earn something in exactly one place — they lift correct files
out of the tail of the shortlist into the top ten, 0.92 to 0.94 — and pay for it
at rank one, which is where a human looks. so it ships off by default, with the
numbers beside the flag:

```bash
fyc find "..." --rerank
fyc eval my_cases.json --rerank        # settle it on your corpus, not mine
```

## where the ceiling is

recall@1 is 0.83, so about one query in six puts the right file below the top.
the limit is the default model's 128-token window: it reads the name, the
docstring and the file summary, not the body. swap in
`intfloat/multilingual-e5-large` or `voyage-code-3` and run the same eval on
your own repository rather than trusting any of these numbers.
