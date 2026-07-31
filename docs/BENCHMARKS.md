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
search            40ms    over 15k chunks
  with graph      42ms
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
                       recall@10  recall@15     MRR
  ---------------------------------------------------
  no graph, n=10            0.41          -   0.238
  no graph, n=15            0.41       0.53   0.248
  graph, n=10               0.47          -   0.247
  graph, n=10, weight 0.8   0.53          -   0.257
```

read the first and third rows together: the graph finds in ten results what text
alone needs thirteen to find, and at `graph_weight = 0.8` what text needs fifteen
to find. the default is 0.65 because 0.8 costs recall@3 on the ordinary set
(0.89 → 0.86); if your questions look more like the multi-hop set than the
ordinary one, it is one line of `.findyourcode.toml`.

seventeen cases is a small set and each case is worth 0.06, so treat the gap as
a direction, not a measurement. how it was built, so you can judge it: a
candidate was written down as a (question, calling module, implementing module)
triple before anything was run, and kept if the index contained an edge between
the two modules. two of the twenty-two candidates were dropped because the call
turned out not to exist — `pickle` does not call `copyreg` by name, `inspect`
does not call `dis` by name. none was dropped for scoring badly.

of the 17, the gold module is reachable through the graph from the top-10 text
results in 13. the remaining gap is selection, not extraction: a strong result
calls twenty things and only five slots are given away.

## asking in another language

the default model is multilingual, which is the point when the code is in
english and you are not. the same 26 questions, translated:

```
                     recall@1  recall@3  recall@10     MRR
  ---------------------------------------------------------
  english                 0.77      0.88       0.88   0.821
  russian                 0.54      0.65       0.81   0.611
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
