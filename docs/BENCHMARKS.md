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

36 hand-written queries against one corpus is a smoke test with a method
attached, not a benchmark. read it as evidence that the decisions were measured,
not as a claim about your repository.

## cost

```
first index      305s     672 files, 15k chunks
re-index         2.4s     nothing changed
search            45ms    over 15k chunks
  without the graph 38ms
cold start       2.6s     python starting and the model loading
on disk         70.6mb    of which the call graph is 2.7mb
```

## fusion

`examples/eval_stdlib.json` — 26 questions phrased as meaning, 10 as an exact
identifier. one shape alone would have argued for deleting the branch that saves
the other.

```console
$ fyc eval examples/eval_stdlib.json --sweep

  setting            recall@1   recall@3  recall@10     MRR
  ---------------------------------------------------------
  blend a=0              0.67       0.81       0.92   0.748
  blend a=0.25           0.69       0.83       0.92   0.779
  blend a=0.5            0.75       0.92       0.92   0.829
  blend a=0.75           0.81       0.89       0.92   0.850   <- default
  blend a=1              0.64       0.81       0.89   0.720
  semantic               0.64       0.78       0.89   0.714
  lexical                0.67       0.81       0.92   0.748
  rrf                    0.81       0.89       0.94   0.853
  no graph               0.81       0.89       0.92   0.850
```

split the set by shape and the reason for keeping two retrievers is plain:

```
                        recall@1  recall@3  recall@10     MRR
  ------------------------------------------------------------
  26 by meaning
    blend                   0.77      0.88       0.88   0.821
    semantic only           0.69      0.85       0.88   0.755
  10 by identifier
    blend                   0.90      0.90       1.00   0.925
    semantic only           0.50      0.60       0.90   0.608
    lexical only            0.90      1.00       1.00   0.950
```

neither branch survives alone: the vectors lose half the identifier queries, and
bm25 is the one that falls behind on meaning. the blend is close to the better of
the two on both halves, which is the whole point of paying for both.

blend and rrf finish within noise of each other on this corpus — 0.850 against
0.853, one case either way. blend stays the default for a reason the aggregate
cannot show: rrf ranks by position only, so a single incidental keyword match at
bm25 rank 1 can outrank the right answer, and that is what it did to
`card.Token` on the demo repository. the blend sees the cosine as well and does
not.

## the call graph

two properties matter and they pull against each other: the graph must not
disturb ordinary queries, and it must find things text cannot.

**it does not disturb ordinary queries.** the `no graph` row above is identical
to the default row. that is by construction: a chunk reached through the graph
enters the page below the best direct answer and never above it, and at most
five join a page. an earlier version let call edges push text matches *up* as
well; it cost 0.016 mrr on this set for nothing and was removed.

**multi-hop questions.** `examples/eval_multihop.json` — 17 questions whose
answer is a module one call away from the module the question describes.
`argparse` decides how wide to wrap help text by calling
`shutil.get_terminal_size`; the question is about help text, the answer is in
`shutil.py`.

```
                     multi-hop                ordinary
                  r@3   r@10     MRR       r@3     MRR
  ------------------------------------------------------
  no graph       0.24   0.41   0.238      0.89   0.850
  graph          0.29   0.59   0.280      0.89   0.850
```

ten results find what text alone needs twenty to find, and the ordinary set is
not merely close but identical, question for question.

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
went to recall@10 0.53 and the ordinary set gave up recall@3, 0.89 → 0.86.

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
all leave the ordinary set at mrr 0.850 with not one question changing rank, so
the default is 0.85 and the multi-hop set keeps the whole gain.

it costs one deeper read: 38ms → 45ms per query on 15k chunks. it is not free of
misses either — 12 of the 17 multi-hop questions have their gold module among
the graph candidates and 4 of those sit outside the window, at rank 635, 640,
970 and 1417. widening it lets the damaging ones back in.

## a second hop, measured and rejected

propagation follows one call edge. following two looks obviously right — and the
premise checks out: for 5 of the 17 multi-hop questions the gold module is two
calls from the text match, not one, and a second hop does reach them.

it changes nothing, at any decay (measured against the `graph_weight = 0.65` of
the time, before the reach window above existed):

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
graph genuinely buys recall gives all of it back:

```
                                     multi-hop r@10     MRR
  ------------------------------------------------------------
  graph_weight 0.8, one hop                     0.53   0.257
  graph_weight 0.8, second hop at 0.5           0.41   0.239
```

being two calls from a text match is barely evidence at all; on this corpus it
is mostly `os.path`. it cost about a millisecond, and it is not in the code.

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
  no rerank                             0.81      0.89       0.92   0.850       0.04s
  ms-marco-MiniLM-L-6-v2                0.72      0.86       0.94   0.804       1.7s
  jina-reranker-v1-turbo-en             0.67      0.81       0.92   0.760       1.9s
  jina-reranker-v2-base-multilingual    0.75      0.86       0.94   0.808      10.3s
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

recall@1 is 0.81, so roughly one query in five puts the right file below the
top. the limit is the default model's 128-token window: it reads the name, the
docstring and the file summary, not the body. swap in
`intfloat/multilingual-e5-large` or `voyage-code-3` and run the same eval on
your own repository rather than trusting any of these numbers.
