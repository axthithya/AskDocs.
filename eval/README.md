# AskDocs evaluation

This measures **retrieval only**: does the retriever bring back the right chunk? It does not judge the LLM's answers.

## Status

`questions.jsonl` is **empty**: no PDFs were in `data/pdfs/` when this phase was built, and questions must come from real documents. Nothing has been measured yet.

## Dataset format

`questions.jsonl`, one JSON object per line:

```json
{"question": "What does the reliability pillar focus on?", "source": "doc.pdf", "keyword": "recover from failures"}
```

- `source`: the PDF that contains the answer.
- `keyword`: a specific word/phrase that appears in the chunk that answers the question (case-insensitive).

### Creating questions (do this by hand)

1. Put PDFs in `data/pdfs/` and open them.
2. Write 20-30 questions whose answers you can point to in the text: definitions, concepts, comparisons, technical explanations, facts, answers on different pages, specific sections.
3. For each, copy a distinctive keyword from the answer passage. Check the keyword actually occurs in the PDF's *extracted* text (pypdf can differ from what you see visually).
4. Never add a question whose answer is not in the documents.

## How a question is scored

Retrieve the top-k chunks. It is a **hit** if at least one chunk comes from `source` and its text contains `keyword`. **hit@k** = hits / total questions. Failures print the expected source/keyword and what was retrieved with scores.

Note: keyword matching is a proxy. A chunk may contain the keyword without really answering the question, so treat the number as an approximate signal, not ground truth.

## Commands

```bash
python -m app.ingest                # build the main index first
python -m eval.run_eval 4           # hit@4 on the main index
python -m eval.run_experiments      # chunk size (500/1000/1500) + top-k (2/4/6) -> eval/results.csv
python -m eval.run_generation       # calls Nova: writes answer_results.jsonl + no-answer check
```

`run_experiments` builds one index per chunk configuration in `data/experiments/` (cached; `--rebuild` to redo). Each build makes one Titan call per chunk, so it costs a few hundred to thousands of calls for larger PDFs. The main index in `data/index/` is untouched. The top-k experiment runs on the configuration with the highest hit@4 (ties go to chunk 1000).

## No-answer check

`no_answer_questions.jsonl` has questions unrelated to any technical document. `run_generation` checks whether the answer contains "I couldn't find that in the provided documents." This is a **basic behavioral check**, not a rigorous hallucination metric, and does not prove hallucination prevention. If your PDFs happen to cover one of these topics, replace the question.
