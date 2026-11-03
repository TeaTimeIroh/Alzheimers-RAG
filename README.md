# Alzheimer's Papers RAG

[![CI](https://github.com/TeaTimeIroh/Alzheimers-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/TeaTimeIroh/Alzheimers-RAG/actions/workflows/ci.yml)

A retrieval-augmented question-answering system over 19 Alzheimer's disease research papers.
Ask a question in plain English and get an answer written **only from the papers**, with
numbered citations pointing to the exact passage, section and page it came from. When the
papers don't contain the answer, it says so instead of guessing.

Built end to end in Python: PDF parsing → chunking → embeddings in **PostgreSQL + pgvector** →
**hybrid retrieval** (vector + full-text search, fused) → LLM answer with citations →
**FastAPI** REST API → an evaluation suite with known answers, used to compare retrieval
methods and a local LLM with a cloud LLM. On top of that: a **tool-using agent** that decides
its own searches (which fixed most questions that need two papers), and an **LLM judge** that
re-grades the saved answers as a check on the keyword scoring.

![Example answer with citations](docs/images/answer_gemini.png)

## How it works

```mermaid
flowchart LR
    A[19 PDFs +<br/>PubMed CSV] --> B[parse_pdfs.py<br/>clean page text]
    B --> C[chunk_docs.py<br/>~220-word chunks]
    C --> D[embed_store.py<br/>BGE-small embeddings]
    D --> E[(PostgreSQL<br/>+ pgvector)]
    Q[Question] --> S[search.py<br/>vector + full-text<br/>fused with RRF]
    E --> S
    S --> G[answer.py<br/>prompt with numbered sources]
    G --> L[LLM<br/>LM Studio or Gemini]
    L --> R[Answer + citations]
    API[api.py - FastAPI] -.-> S
    API -.-> G
```

| Step | File | What it does |
|---|---|---|
| Parse | [`parse_pdfs.py`](parse_pdfs.py) | PyMuPDF text extraction with reading-order fixes for two-column pages and sidebars, header/footer and reference-list removal, heading detection, de-hyphenation; attaches title/authors/year/DOI from the PubMed CSV |
| Chunk | [`chunk_docs.py`](chunk_docs.py) | Section-aware chunks (~220 words, hard cap 320, 40-word overlap); drops front matter, affiliations and boilerplate sections (funding, conflicts of interest, ...) |
| Embed & store | [`embed_store.py`](embed_store.py) | Embeds with `BAAI/bge-small-en-v1.5` (local, 384 dims) and upserts into Postgres/pgvector with an HNSW index; adds a full-text column (`tsvector`, GIN index); keeps the table identical to `chunks.jsonl` |
| Retrieve | [`search.py`](search.py) | Three modes: `vector` (cosine similarity), `keyword` (Postgres full-text) and `hybrid` (default: both, merged with Reciprocal Rank Fusion); also a CLI to inspect retrieval on its own |
| Answer | [`answer.py`](answer.py) | Numbers the retrieved passages, instructs the LLM to answer only from them and cite `[n]`, or reply that the papers don't contain the answer |
| Serve | [`api.py`](api.py), [`auth.py`](auth.py) | FastAPI: `POST /ask`, `GET /search`, `GET /health`, Swagger UI at `/docs`; API keys with scopes (`search`, `ask`), stored only as hashes |
| Agent | [`agent.py`](agent.py) | The LLM gets tools instead of fixed passages - `search` (all papers), `search_paper` (one paper), `cannot_answer` - and decides what to search, up to 6 steps; passages keep their `[n]` number across searches |
| Evaluate | [`evaluate.py`](evaluate.py) | Runs the test questions through the pipeline or the agent; scores retrieval (hit@k, MRR), answers, citations and refusals per question type |
| Judge | [`judge.py`](judge.py) | After a run: a separate LLM grades every saved answer against the reference and the passages it cites, and lists where it disagrees with the keyword scoring |

Any OpenAI-compatible chat API works as the LLM: the backend is chosen in `.env`, so the same
code runs against a local model in LM Studio or Google's Gemini API.

## Numbers

Measured with [`evaluate.py`](evaluate.py) on 38 questions with known answers
([`eval_questions.jsonl`](eval_questions.jsonl)):

- 11 about a paper's **findings**, 4 of them deliberately paraphrased so they don't share the
  paper's wording;
- 11 **lookups** of a specific number or name, most of them from the body text rather than
  the abstract;
- 10 that need **two papers** (a tracer, a mouse model or a cohort both studies used; which
  study was larger; whether two studies agree);
- 6 that **cannot** be answered from the papers - 3 of them near misses about a study that
  *is* in the collection but doesn't report the detail asked for.

The first 30 questions (with only 2 two-paper ones) were used for the retrieval comparison;
the 8 extra two-paper questions were added - and the single-search baseline run on them -
before the agent was written.

### Agent vs single search (38 questions)

The single-search pipeline retrieves 5 passages once and answers from them. For a question
about two studies, all 5 slots usually go to the paper that matches best. The agent instead
calls search tools itself - typically once per study - and answers from everything it found.

| | Pipeline<br/>Gemini | Pipeline<br/>Qwen | **Agent<br/>Gemini** | **Agent<br/>Qwen** |
|---|---|---|---|---|
| Single-paper questions correct (22) | 22 | 22 | **22** | 20 |
| Two-paper: evidence from both papers found (10) | 2 | 2 | **9** | 6 |
| Two-paper: correct (10) | 1 | 2 | **9** | 6 |
| Two-paper: right words, evidence never retrieved ("unsupported") | 2 | 6 | 1 | 2 |
| Unanswerable questions refused (6) | 6 | 6 | 5 \* | 6 |
| Answer cites a passage containing the answer (32) | 24 | 21 | 28 | 25 |
| LLM calls per answerable question | 1 | 1 | 2.2 | 2.2 |
| LLM time per answerable question | 3.2 s | 22.9 s | 4.9 s | 63.8 s |

*Gemini 3.1 Flash-Lite through the free API; Qwen3-Coder 30B-A3B (Q4_K_M) locally in
LM Studio. Hybrid search, top 5 per search. Scored by the keyword rules; the judge's
second opinion is below. \* U6: a correct refusal in its own words (see the judge section).*

- **Letting the model search fixed most two-paper questions**: Gemini 1 → 9 of 10, Qwen
  2 → 6. The cost is about 2.2 LLM calls instead of 1, with the whole conversation (including
  every passage found so far) re-sent on each call - for the local model that is ~64 s per
  question instead of ~23 s.
- **A cheaper fix was tried first and is kept as an option** (`--max-per-paper`): capping
  passages per paper raised two-paper retrieval from 2 to 3 (max 3 per paper) or 4 (max 2)
  of 10, but single-paper retrieval dropped from 22 to 21 or 18 of 22 - it takes slots from
  questions that need several passages of one paper.
- **Single-paper questions don't need an agent**: Gemini's agent searched exactly once for
  all 22. Qwen's two non-correct ones are about wording, not search: F8 found the right
  passage and listed the right artefacts, but opened with the refusal sentence; F11 is
  correct when read (the keyword rule was too strict - the judge caught it).
- **The local model is the weaker agent.** Qwen found both papers' evidence for only 6 of
  10 two-paper questions, and for M9 it stated that "only APOE" is found in both strata -
  a claim no passage makes (the answer is all three genes it was asked about).
- **The rules had to move into code.** Early versions of the agent answered some
  questions - including unanswerable ones - without searching, and Gemini paraphrased its
  refusals so they couldn't be counted. Now refusing is a tool (`cannot_answer`) with a
  reason, and a final answer or refusal before the first search is sent back with an
  instruction to search; in the final runs no question went unsearched.
- **Citations lag behind answers for two-paper questions**: Gemini's 9 correct two-paper
  answers cite a passage for both papers in only 6.

### LLM judge: a second opinion on the scoring

The keyword rules can't see a reworded refusal, a reversed comparison or a claim the cited
passage doesn't make. [`judge.py`](judge.py) runs after an evaluation, on the saved answers:
a separate model gets the question, the reference answer, the system's answer and the text
of the passages it cited, and returns a verdict (correct / partial / wrong / refused),
whether the cited passages support every claim, and a one-line reason. Nothing is
re-generated, so one judge grades every model and mode.

The judge here is **Gemini 3.5 Flash-Lite** - a newer, different model from the 3.1
Flash-Lite that answered, but from the same family, so some self-preference towards the
Gemini run can't be ruled out. The intended judge, Gemini 3.8 Flash, was not usable on the
free tier: 20 requests a day, ~4-5 minutes per call and repeated "high demand" errors. It
graded 17 of the (easy) single-paper answers before running out, agreeing with Flash-Lite on
all of them. Answers are graded 4 per call to fit free-tier quotas.

| | Agent Gemini | Agent Qwen |
|---|---|---|
| Judge: answerable questions correct (32) | 32 | 29 |
| Judge: unanswerable questions refused (6) | 6 | 5 |
| Cited passages back every claim (answers with claims) | 34 / 34 | 30 / 34 |
| Same verdict as the keyword rules | 37 / 38 | 33 / 38 |

Where the two disagree - and one case where both are wrong - reading the answers gives:

| Answer | Keyword rules | Judge | Reading the answer |
|---|---|---|---|
| Gemini U6 | wrong (answered) | correct (refused) | **Judge right**: "did not report a hazard ratio", without the fixed refusal sentence |
| Qwen F11 | partial | correct | **Judge right**: "not mainly frontal, predominantly temporoparietal" |
| Qwen M6 | partial | refused | **Judge right**: "I cannot definitively identify ..." |
| Qwen M9 | unsupported | wrong | **Judge right**: "only APOE" is false, not merely unsupported |
| Qwen U6 | correct | wrong | **Rules right**: same content as Gemini's U6 plus the refusal sentence - the judge contradicts itself |
| Qwen F8 | refused | correct | **Rules right**: the answer opens with "I could not find this"; the judge's reason ("common knowledge") is wrong |
| Gemini M8 | correct | correct | **Both wrong**: the answer first names the *smaller* study as larger, then corrects itself - partial at best |

Two more disagreements are still open: for Gemini M5 and Qwen M10 the rules say the
supporting passage was never retrieved, the judge says the cited passages do support the
answer - possibly passages missing from the questions' evidence lists, as happened before.

What this says: the judge fixed 4 rule errors and made 2 of its own, and neither method
caught the self-contradicting M8. A lite judge is not more reliable than the keyword rules
- it is lenient (all 38 Gemini answers "correct") - but the two fail on *different*
answers, so their disagreements are a short list of the answers worth reading. That list,
not the judge's score, is how the judge is used here.

### Single search: answers on the first 30 questions (hybrid, top 5 passages)


| | Gemini 3.1 Flash-Lite (API) | Qwen3-Coder 30B-A3B (local, LM Studio, Q4_K_M) |
|---|---|---|
| Answerable questions answered correctly | **22 / 24** | **22 / 24** |
| The two-paper questions | 0 / 2 (refused) | 0 / 2 (answered without support) |
| Answer cites a passage containing the answer | 22 / 24 | 19 / 24 \* |
| Unanswerable questions correctly refused | 6 / 6 | 6 / 6 |
| Time per answer | ~1 s (plus free-tier rate-limit waits) | ~21 s |

\* A lower bound: checking by hand, two of Qwen's "wrong" citations point to Results
passages that do contain the answer but aren't in the question's hand-made evidence list.

### Retrieval: vector vs keyword vs hybrid (first 30 questions)

| | Vector (cosine) | Keyword (full-text) | **Hybrid (RRF)** |
|---|---|---|---|
| Answer passage in the top 5 | 21 / 24 | 22 / 24 | **22 / 24** |
| Answer passage ranked first | 10 / 24 | 13 / 24 | **14 / 24** |
| MRR (top 5) | 0.604 | 0.691 | **0.717** |
| Answer passage in the top 3 | 21 / 24 | 21 / 24 | 21 / 24 |
| Correct answers, Gemini / Qwen (top 5) | 21 / 20 | - | **22 / 22** |
| Correct answers, Gemini / Qwen (top 3) | 21 / 21 | - | 21 / 21 |

*MRR: mean reciprocal rank of the answer passage (1.0 = always ranked first). Raw results with
every retrieved passage and answer are in [`results/`](results); files without a mode in
their name are vector search.*

What the numbers say:

- **Hybrid search mainly improves ranking.** Compared with vector search it moved the answer
  passage up for 9 questions, down for 1 (F8: 1st → 2nd) and left 14 unchanged; "ranked
  first" went from 10 to 14 of 24. With 24 questions, differences of one or two are noise -
  the ranking gain is the clear result.
- **Full-text search alone already beat vector search here.** Part of that is how the set was
  made (questions written while reading the papers tend to reuse their terms), but the
  paraphrased questions show no disadvantage for keyword search. The two methods often
  disagree on which passage is best, which is exactly what fusion exploits.
- **Fusion can also hurt**: for L11 ("60.9 months") full-text search ranks the passage 1st,
  vector search not even in its top 20, so RRF - which rewards passages both methods
  agree on - puts it 5th, and at top 3 it is lost again.
- **Two-paper questions fail in every mode** (the agent above fixes most). All five slots go to the paper that matches the
  question best and the second paper never makes the cut - a retrieval-design problem
  (diversity / splitting the question), not a ranking one. Gemini then refuses. Qwen doesn't
  use the refusal sentence: for M1 it names the tracer for one study and says the other is
  unknown; for M2 it attributes a finding to the second study using a passage from the first,
  citing the paper's internal reference numbers (`[62]`) instead of the sources.
- **Refusals only happen when the answer wasn't retrieved.** Neither model refused a question
  whose answer passage was in its context, and both refused all 6 unanswerable questions.
- **Keyword scoring has blind spots.** Qwen's L6 answer now passes ("CA1 neuron loss of over
  50%"), but "CA1" belongs to a *different* mouse model in the same passage (−14%): the right
  number with a wrong detail attached. Catching that needs a reader or an LLM judge.

<details>
<summary>Per-question rank of the answer passage (top 5; – = not retrieved)</summary>

| Question | Tags | Vector | Keyword | Hybrid |
|---|---|---|---|---|
| F1 | exact-term | 1 | 1 | 1 |
| F2 | abstract | 3 | 2 | 2 |
| F3 | abstract, exact-term | 3 | 4 | 2 |
| F4 | abstract | 2 | 1 | 1 |
| F5 | abstract | 2 | 3 | 1 |
| F6 | abstract, exact-term | 3 | 1 | 2 |
| F7 | paraphrase, abstract | 2 | 1 | 1 |
| F8 | body | 1 | 3 | 2 |
| F9 | paraphrase | 2 | 1 | 1 |
| F10 | paraphrase, body | 1 | 1 | 1 |
| F11 | paraphrase, body | 1 | 2 | 1 |
| L1 | body | 3 | 2 | 3 |
| L2 | abstract | 1 | 1 | 1 |
| L3 | abstract | 1 | 1 | 1 |
| L4 | abstract | 1 | 2 | 1 |
| L5 | body | 2 | 1 | 1 |
| L6 | body | 3 | 3 | 3 |
| L7 | exact-term, body | 1 | 1 | 1 |
| L8 | exact-term, body | 1 | 1 | 1 |
| L9 | body | 3 | 3 | 3 |
| L10 | exact-term, body | 1 | 1 | 1 |
| L11 | body | – | 1 | 5 |
| M1 | two papers | – | – | – |
| M2 | two papers | – | – | – |

</details>

**How the comparison was kept fair.** The 30 questions and their evidence passages were
fixed and the vector-search baseline was run *before* hybrid search was written. Two scoring
fixes came from reviewing the baseline (missing alternative wordings for F11 and M2, and the
new "unsupported" verdict below); all baseline runs were repeated with them. The full-text
settings (OR vs AND, `ts_rank` vs `ts_rank_cd`, title weighting) were compared on this same
set, which flatters the keyword numbers slightly; the choices that mattered were obvious
ones - AND-matching found the answer for only 2 of 24 questions.

An earlier 15-question version of the set (Gemini 12/12, Qwen 11/12, both refusing 3/3)
is kept in `results/` with the 2026-11-01 date.

![Evaluation run with Gemini (15-question version)](docs/images/eval_gemini-flash-lite.png)

## Design decisions

- **Local embeddings, swappable LLM.** Embeddings run locally (`bge-small-en-v1.5`, ~130 MB,
  CPU is enough), so indexing costs nothing. The LLM sits behind the OpenAI-compatible chat
  API, so a local model (LM Studio) and a hosted one (Gemini) are a `.env` change apart - which
  is what made the model comparison above cheap to run.
- **Chunk size follows the embedding model.** Biomedical text runs ~1.5 tokens per word;
  ~220-word chunks plus a `title | section` header stay well inside BGE's 512-token window
  (a 256-token model such as `all-MiniLM-L6-v2` would silently truncate them).
- **Chunks never cross section boundaries**, and every chunk keeps its paper, section and
  page range, so citations point somewhere a reader can check.
- **Refusal is part of the prompt contract.** The model must cite `[n]` for every claim or
  reply with a fixed "could not find" sentence, which makes refusals measurable.
- **Temperature 0** for answers, so evaluation runs are repeatable.
- **Postgres + pgvector in Docker** rather than a dedicated vector database: one familiar
  database for text, metadata and vectors, started with one command - and it already has
  full-text search, so hybrid retrieval needed no new service. The `tsvector` is a generated
  column, so Postgres keeps it in sync with the text by itself.
- **Reciprocal Rank Fusion** to combine the two searches: each passage scores
  1/(60 + rank) in every list it appears in (top 20 of each). It uses only ranks, so cosine
  similarities and `ts_rank` values - on unrelated scales - never have to be normalised.
- **Full-text query with OR, not AND.** Postgres' `plainto_tsquery` requires every word of a
  question to appear in a passage, which almost never happens for a whole question; joining
  the words with OR and letting `ts_rank` reward passages with more (and rarer) matches works.
- **Agent tools contain no LLM.** `search` and `search_paper` are plain database queries;
  only the agent loop calls the model. Every passage keeps the `[n]` it got when first found,
  so citations stay valid across several searches, and a passage found again is reported as
  "already shown" instead of being re-sent.
- **Refusal as a tool, rules in code.** `cannot_answer(reason)` makes a refusal structured
  rather than a sentence to pattern-match; searching before answering is checked by the loop,
  not just asked for in the prompt; after 6 steps the tools are withdrawn so it must answer.
- **Keyword rules first, LLM judge second.** The rules are fast, free and repeatable, and
  were reviewed by hand; the judge runs separately on saved answers, with its own model
  settings (`JUDGE_*`), and is checked against the rules and by reading the disagreements
  instead of being trusted.
- **API keys with scopes, stored as hashes.** `/search` and `/ask` need a key in the
  `X-API-Key` header; each key has the scopes it may use (`search` is retrieval only, `ask`
  also calls the LLM, which costs quota or money), so a demo key can search without spending
  anything. Missing or unknown keys get 401, a key without the scope 403. The key file holds
  only SHA-256 hashes - enough for random 256-bit keys, unlike human passwords, which need a
  slow salted hash - compared in constant time; requests are logged with the key's name,
  never the key. This is simple key-based access control, not a full identity system (no
  users, expiry or rate limits).

## Lessons learned

Things that only showed up by measuring, and what was done about them:

- **PDF parsing was the biggest source of error.** A first version silently lost most of the
  Nature Genetics paper (only 15% of its text reached the index): it's a "Letter" with no
  section headings, and Nature prints the reference list *before* Methods, so "cut everything
  after References" deleted the Methods too. Other fixes: columns read out of order when a
  whole column is a single PDF block (which also leaked reference lists into the text),
  affiliation sidebars interleaved line-by-line into abstracts, hundreds of words broken by
  hyphenation across lines, duplicated text layers, and table headers mistaken for section
  headings. After the fixes: Nature paper 2,102 → 7,470 words indexed; reference-list chunks
  10 → 0; boilerplate chunks 67 → 0; affiliation-list chunks 13 → 0.
- **Symbol fonts produce control characters.** Formulas and p-values in some papers use
  glyphs with no Unicode mapping, which come out as `\x00`–`\x05`; Postgres rejects `\x00`
  ([the error](docs/images/pipeline_embed-store.png)). Their meaning differs per font (`\x02`
  is "×" in one paper and "±" in another), so they are replaced with `�` rather than guessed.
- **Duplicate studies distort answers.** The collection first contained both the preprint and
  the published version of the APOE study; they report different numbers (33 vs 29 regions of
  interest) and took two of the five retrieval slots. The preprint was removed.
- **Question wording matters more than expected.** "How many regions of interest were
  tested?" retrieved brain-imaging papers (where "regions of interest" is a standard term);
  naming the study ranked the correct passage in the top 3.
- **A correct citation doesn't mean a correctly read source.** In a manual test Qwen claimed
  two loci "depend on APOE ε4" while citing the passage that says they don't
  ([screenshot](docs/images/answer_qwen-local.png)). The evaluation therefore checks the
  answer's content, not only that it cites something.
- **Evaluating is its own engineering problem.** The first evaluation flagged correct answers
  as retrieval misses because each fact appears several times in a paper in different words.
  The evidence for each question is now a hand-checked list of passages; a looser automatic
  rule was tried and rejected because it inflated the scores.
- **The right words aren't a correct answer.** Asked what *two* studies both report, Qwen
  answered "mean diffusivity" - correct - while only one of the papers had been retrieved,
  and labelled a passage of the first paper as the second study. Keyword scoring counted it
  as correct. Answers whose supporting passages were never retrieved are now scored
  **unsupported**, and refusals are no longer scored for facts (a refusal that explains what
  it *did* find used to show "1/1 facts").
- **Prompt rules are suggestions.** The local model sometimes answered - even unanswerable
  questions - from its own knowledge without calling a tool, despite the prompt saying it
  must search first. Enforcing it in the loop (send the answer back once, with an
  instruction to search) fixed it completely; the prompt alone never did.
- **An agent that reads more finds gaps in the test set.** Because the agent sees more
  passages, it surfaced correct answers backed by passages that weren't in the evidence
  lists (L4, M3, M7). Each was checked to belong to the right paper before being added, and
  saved runs were re-scored without new LLM calls (`--rescore`).
- **Free tiers shape the method.** A judge with 20 requests a day, minutes per call and
  frequent overload errors led to batching answers, saving after every batch, stopping
  (not retrying) on timeouts and overloads - and, in the end, to a lighter judge model.

## Build and run

Requirements: Python 3.11+, [Docker Desktop](https://www.docker.com/products/docker-desktop/),
and either [LM Studio](https://lmstudio.ai/) with a chat model or a
[Gemini API key](https://aistudio.google.com/apikey).

**Quick start** - after `pip install -r requirements.txt` and with Docker Desktop running and
the PDFs in `pdfs/` (see [Data](#data)):

```bat
python run.py
```

[`run.py`](run.py) checks the requirements and says what is missing, starts the database,
builds the index (parse → chunk → embed, skipping steps whose output is already up to date),
creates an API key the first time, and starts the API at `http://127.0.0.1:8000/docs`.
`python run.py check` only reports what is ready; `build` / `serve` do one half;
`--rebuild` redoes the index. It runs the same scripts as the steps below.

Step by step - commands for Windows (Anaconda Prompt); on macOS/Linux use `cp` instead of `copy`.

```bat
:: 1. install
pip install -r requirements.txt

:: 2. start Postgres + pgvector (host port 5433)
docker compose up -d

:: 3. put the PDFs in pdfs\ (see "Data" below), then build the index
python parse_pdfs.py --pdf-dir pdfs --csv pdfs\csv-alzheimer.csv --out parsed
python chunk_docs.py --parsed parsed --out chunks.jsonl
python embed_store.py --chunks chunks.jsonl

:: 4. configure the LLM (edit .env afterwards: LM Studio or Gemini)
copy .env.example .env
python answer.py --list-models

:: 5. ask
python search.py "sex differences in genetic risk for Alzheimer's pathology"
python search.py "interval between diagnosis and autopsy" --retrieval keyword
python answer.py "Which variants' effects depend on APOE e4?"

:: 6. evaluate (hybrid search unless --retrieval vector|keyword or RETRIEVAL in .env)
python evaluate.py --retrieval-only
python evaluate.py --retrieval-only --retrieval vector
python evaluate.py

:: 7. the agent: one question, then the evaluation (LM Studio: context length >= 16384)
python agent.py --check-tools
python agent.py "Which amyloid PET tracer did both the Beijing study and the ADNI analysis use?" --show-trace
python evaluate.py --mode agent

:: 8. a second opinion from the judge (JUDGE_MODEL / JUDGE_API_KEY in .env)
python judge.py results\eval_<model>_hybrid_agent_<timestamp>.json --batch 4

:: 9. create an API key (printed once), run the API, open http://127.0.0.1:8000/docs
::    and paste the key into "Authorize"
::    (or set API_AUTH=off in .env while experimenting locally)
python auth.py create me --scopes search,ask
uvicorn api:app --reload

:: when done: stop the database (keeps the data; "down -v" would delete it)
docker compose stop
```

### Tests

```bat
pip install -r requirements-dev.txt
ruff check .
pytest
```

197 tests cover the parser (including a PDF generated on the fly with a running header,
page numbers and a reference list), the chunker (size cap with overlap, front-matter and
boilerplate filtering, paragraphs split across pages), Reciprocal Rank Fusion and the search
modes, the per-paper cap, citation parsing, the LLM client, the agent loop (with a scripted
fake LLM: tool errors, refusals, the search-first rule, the step limit, rate-limit retries),
the evaluator's scoring and re-scoring, the judge (batching, reply parsing, resuming after a
quota stop), the API endpoints and error codes, API keys and scopes (401/403, revocation,
only hashes on disk), the setup script's rebuild decisions, and the format of `eval_questions.jsonl`. Most need no database, LLM, embedding model or
PDFs - those are faked. The full-text and hybrid SQL is tested against a real Postgres +
pgvector, in a throwaway schema; those tests run when `TEST_DATABASE_URL` is set
(`set TEST_DATABASE_URL=postgresql://rag:rag@localhost:5433/rag` with the Docker database up)
and are skipped otherwise. [GitHub Actions](.github/workflows/ci.yml) starts a pgvector
database and runs everything with ruff on Python 3.11, 3.13 and 3.14 for every push and pull
request. Several tests pin down bugs found during development (e.g.
two-column pages read out of order, heading-less papers losing their main text).

PyMuPDF is pinned to the version that produced the published results; with it, the pipeline
reproduces the same chunks byte for byte on Python 3.13 and 3.14.

### API

Create a key first (`python auth.py create me`; it is printed once and only its hash is
kept in `api_keys.json`). The easiest way to try it is the Swagger page at
`http://127.0.0.1:8000/docs` - click **Authorize** and paste the key:

![Swagger UI - POST /ask](docs/images/api_swagger-ask.png)

From a terminal (macOS/Linux/Git Bash):

```bash
curl -X POST 'http://127.0.0.1:8000/ask' \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: rag_...your key...' \
  -d '{"question": "Is Alzheimer'\''s disease more common in men or women?", "top_k": 5}'
```

Response (shortened):

```json
{
  "question": "Is Alzheimer's disease more common in men or women?",
  "answer": "Two-thirds of Alzheimer’s disease cases are female [1].",
  "not_found": false,
  "sources": [
    {
      "n": 1,
      "chunk_id": "awz206:0001",
      "title": "Sex differences in the genetic predictors of Alzheimer's pathology",
      "year": "2019",
      "section": "Introduction",
      "page_start": 2,
      "page_end": 2,
      "score": 0.8017,
      "cited": true
    }
  ],
  "llm_model": "qwen3-coder-30b-a3b-instruct",
  "seconds": {"retrieval": 0.03, "llm": 12.31}
}
```

`GET /search?q=...&k=5&mode=hybrid` returns the retrieved passages without calling the LLM;
`GET /health` reports the database status and configured models (no key needed). Keys:
`python auth.py list` / `revoke <name>`; a key with only the `search` scope gets 403 from
`/ask`:

| No key: 401 | Key with only the `search` scope calling `/ask`: 403 |
|---|---|
| ![401 Unauthorized in Swagger](docs/images/api_swagger-401.png) | ![403 Forbidden in Swagger](docs/images/api_swagger-403-scope.png) |

The server log names the key behind each request (never the key itself):

![Server log: a rejected request, then one by key 'me'](docs/images/api_auth-log.png)

Both `/search` (`mode`) and
`/ask` (`"retrieval"`) accept `vector`, `keyword` or `hybrid`, defaulting to the server's
`RETRIEVAL` setting (hybrid). Responses now also report the `retrieval` mode used; the score
is a cosine similarity, a `ts_rank` or an RRF score depending on the mode - only the order
matters. (The response above is from a vector-search run.)

<details>
<summary>More screenshots</summary>

| | |
|---|---|
| Parsing the PDFs | ![](docs/images/pipeline_parse-pdfs.png) |
| Chunking | ![](docs/images/pipeline_chunk-docs.png) |
| Embedding (with the NUL-byte error and the fixed re-run) | ![](docs/images/pipeline_embed-store.png) |
| Inspecting the vectors in psql | ![](docs/images/db_embeddings-psql.png) |
| Retrieval only (`search.py`) | ![](docs/images/search_sex-differences.png) |
| Refusing an unanswerable question | ![](docs/images/answer_unanswerable.png) |
| Local model answers (Qwen in LM Studio) | ![](docs/images/answer_qwen-local.png) |
| LM Studio server | ![](docs/images/lmstudio_server.png) |
| Swagger: entering an API key (Authorize) | ![](docs/images/api_swagger-authorize.png) |
| Gemini models available on the free tier | ![](docs/images/gemini_list-models.png) |
| Evaluation - retrieval only | ![](docs/images/eval_retrieval-only.png) |
| Evaluation - Qwen | ![](docs/images/eval_qwen3-coder.png) |

</details>

## Data

The 19 papers (2007–2026) are listed below. Their PDFs and extracted text are **not**
included in this repository - copyright stays with the publishers. Download them from the
DOI links (most are open access via PubMed Central) into `pdfs/` and run the pipeline; file
names don't matter, because each PDF is matched to its metadata by DOI.
[`pdfs/csv-alzheimer.csv`](pdfs/csv-alzheimer.csv) is the PubMed export used for titles,
authors and DOIs (it also lists the removed preprint, which is simply unused).

The pipeline isn't tied to these papers: any folder of text-based PDFs works, with an
optional PubMed CSV export for clean citations.

| Year | First author | Title | Journal |
|---|---|---|---|
| 2007 | Han X | [Potential mechanisms contributing to sulfatide depletion at the earliest clinically recognizable stage of Alzheimer's disease: a tale of shotgun lipidomics](https://doi.org/10.1111/j.1471-4159.2007.04708.x) | J Neurochem |
| 2009 | Tomic JL | [Soluble fibrillar oligomer levels are elevated in Alzheimer's disease brain and correlate with cognitive dysfunction](https://doi.org/10.1016/j.nbd.2009.05.024) | Neurobiol Dis |
| 2010 | Wirths O | [Neuron loss in transgenic mouse models of Alzheimer's disease](https://doi.org/10.4061/2010/723782) | Int J Alzheimers Dis |
| 2012 | Palesi F | [DTI and MR Volumetry of Hippocampus-PC/PCC Circuit: In Search of Early Micro- and Macrostructural Signs of Alzheimers's Disease](https://doi.org/10.1155/2012/517876) | Neurol Res Int |
| 2012 | Sweet RA | [Effect of Alzheimer's disease risk genes on trajectories of cognitive function in the Cardiovascular Health Study](https://doi.org/10.1176/appi.ajp.2012.11121815) | Am J Psychiatry |
| 2013 | Varghese T | [A review of neuroimaging biomarkers of Alzheimer's disease](https://pmc.ncbi.nlm.nih.gov/articles/PMC4243931/) | Neurol Asia |
| 2014 | Zhang B | [The role of diffusion tensor imaging in detecting microstructural changes in prodromal Alzheimer's disease](https://doi.org/10.1111/cns.12166) | CNS Neurosci Ther |
| 2015 | Ossenkoppele R | [The behavioural/dysexecutive variant of Alzheimer's disease: clinical, neuroimaging and pathological features](https://doi.org/10.1093/brain/awv191) | Brain |
| 2016 | Lauritzen I | [Intraneuronal aggregation of the β-CTF fragment of APP (C99) induces Aβ-independent lysosomal-autophagic pathology](https://doi.org/10.1007/s00401-016-1577-6) | Acta Neuropathol |
| 2016 | Tincer G | [Neural stem/progenitor cells in Alzheimer's disease](https://pmc.ncbi.nlm.nih.gov/articles/PMC4797833/) | Yale J Biol Med |
| 2017 | Alexiou A | [A Bayesian Model for the Prediction and Early Diagnosis of Alzheimer's Disease](https://doi.org/10.3389/fnagi.2017.00077) | Front Aging Neurosci |
| 2019 | Dumitrescu L | [Sex differences in the genetic predictors of Alzheimer's pathology](https://doi.org/10.1093/brain/awz206) | Brain |
| 2020 | Dumitrescu L | [Genetic variants and functional pathways associated with resilience to Alzheimer's disease](https://doi.org/10.1093/brain/awaa209) | Brain |
| 2022 | Verberk IMW | [Characterization of pre-analytical sample handling effects on a panel of Alzheimer's disease-related blood-based biomarkers: Results from the Standardization of Alzheimer's Blood Biomarkers (SABB) working group](https://doi.org/10.1002/alz.12510) | Alzheimers Dement |
| 2023 | Galeana-Ascencio RA | [β-Secretase-1: In Silico Drug Reposition for Alzheimer's Disease](https://doi.org/10.3390/ijms24098164) | Int J Mol Sci |
| 2023 | Wang D | [Multi-domain cognition dysfunction accompanies frontoparietal and temporal amyloid accumulation in the elderly](https://doi.org/10.1093/cercor/bhad369) | Cereb Cortex |
| 2025 | Haeger A | [Multimodal Perspective on the Influence of Physical Activity on Alzheimer’s Disease](https://doi.org/10.1002/alz.091460) | Alzheimers Dement |
| 2025 | Zamani J | [Mapping amyloid beta predictors of entorhinal tau in preclinical Alzheimer's disease](https://doi.org/10.1002/alz.14499) | Alzheimers Dement |
| 2026 | Thomassen JQ | [APOE-stratified genome-wide association analyses provide insights into the genetic etiology of Alzheimers's disease](https://doi.org/10.1038/s41588-026-02713-9) | Nat Genet |

This is a portfolio project, not a medical tool: answers summarise research papers and are
not medical advice.

## Known limitations and roadmap

Limitations:

- **Small test set.** 38 questions, written from the papers' text and checked against it
  rather than by a domain expert; differences of one or two questions are within noise, and
  the full-text settings were chosen on the same set. Answer scoring is keyword-based, with
  an LLM judge as a second opinion and the disagreements read by hand.
- **Two-paper questions need the agent**: the single-search pipeline (and the API) still
  fills all slots from one paper; the agent fixes most of them at ~2x the LLM calls, and is
  only available from the command line and the evaluator so far.
- **The judge is a lite model, not validated beyond these runs**: lenient, and from the same
  family as one of the answering models.
- **Ranking**: even with hybrid search the answer passage is ranked first for 14 of 24
  questions.
- **Tables and formulas** are flattened to text; some symbols are lost (`�`).
- **Single-turn**: no conversation memory; each question is independent.
- The Gemini free tier has unpublished, changeable rate limits, and Google may use free-tier
  prompts to improve its products.

Roadmap:

1. **Agent mode in the API** (`POST /ask` with `"mode": "agent"`), returning the tool trace.
   Per-key rate limits would come with it: an agent question costs several LLM calls.
2. **Reranker**: a cross-encoder over the fused top 20 to improve ranking further.
3. **A stronger judge** (outside the free tier), validated on a larger set of hand-labelled
   answers.
4. **Questions from someone who knows the field**, including more near-miss unanswerable ones.

## Repository layout

```
├── parse_pdfs.py          PDF -> clean page text (+ metadata)
├── chunk_docs.py          page text -> chunks.jsonl
├── embed_store.py         chunks -> embeddings -> Postgres/pgvector
├── search.py              retrieval (+ CLI)
├── answer.py              retrieval + LLM answer with citations (+ CLI)
├── api.py                 FastAPI app
├── auth.py                API keys with scopes (+ CLI to create / list / revoke)
├── run.py                 one-command setup: checks, database, index, API
├── agent.py               tool-using agent: search / search_paper / cannot_answer (+ CLI)
├── evaluate.py            evaluation runner (pipeline or agent; --rescore)
├── judge.py               LLM judge over saved results
├── eval_questions.jsonl   38 test questions with expected answers and evidence passages
├── tests/                 pytest suite (no database / LLM / model needed)
├── .github/workflows/     CI: ruff + pytest on Python 3.11, 3.13, 3.14
├── results/               evaluation outputs (all answers, per-question scores, judge verdicts)
├── docs/images/           screenshots used in this README
├── docker-compose.yml     Postgres 16 + pgvector
├── .env.example           LLM / database settings template
├── requirements*.txt      runtime / test dependencies
├── pyproject.toml         ruff and pytest settings
└── pdfs/csv-alzheimer.csv PubMed metadata of the papers (PDFs not included)
```

## Credits and AI assistance

Built on [PyMuPDF](https://pymupdf.readthedocs.io/) (PDF text extraction; AGPL-3.0),
[sentence-transformers](https://www.sbert.net/) with
[BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) (embeddings),
[PostgreSQL](https://www.postgresql.org/) and [pgvector](https://github.com/pgvector/pgvector)
(vector search), [psycopg](https://www.psycopg.org/), [FastAPI](https://fastapi.tiangolo.com/)
and [Uvicorn](https://www.uvicorn.org/) (API), [LM Studio](https://lmstudio.ai/) with
[Qwen3-Coder](https://huggingface.co/Qwen) (local LLM), the
[Gemini API](https://ai.google.dev/) (cloud LLM), and [PubMed](https://pubmed.ncbi.nlm.nih.gov/)
for the paper metadata. The research itself belongs to the authors of the 19 papers listed
above.

This project was built with Claude (Anthropic) as a coding assistant. Claude wrote most of
the Python code and the first draft of this README, and drafted the 38 evaluation questions
from the papers (30 at first, then 8 more two-paper questions for the agent stage), checking
each expected answer against the extracted text; I then reviewed the questions, answers and
evidence passages against the papers. Claude also wrote the agent loop, the judge, the API
keys, the setup script and their tests, and drafted the review of the judge's disagreements,
which I checked against the answers.
I defined the project and its scope, chose the papers, set up and ran the whole stack on my own machine
(Docker/pgvector, LM Studio, the Gemini API), ran every pipeline step and evaluation, fed the
outputs and errors back into the iterations, and made the project decisions - removing the
duplicate preprint, comparing a local and a cloud model, moving from a single search to an
agent, keeping the judge as a second opinion, choosing the judge model within the free tier,
and how the repository is organised.
The problems described under [Lessons learned](#lessons-learned) were found by running each
step and inspecting its output.

## License

Code: [MIT](LICENSE). The papers belong to their respective publishers and authors.
