# HydraMemory

An AI agent memory layer built on [HydraDB](https://github.com/hydra-db/hydradb), a graph
database. Built for the Hack Hydra hackathon, Track 03: Memory and Context Retrieval.

HydraMemory demonstrates two things a vector store structurally can't do well, using a graph
instead:

1. **Track chronology and overwritten facts.** If a user says "I live in Delhi" in one session
   and "I moved to Pune" weeks later, the graph holds an explicit `SUPERSEDES` edge between the
   two facts, so a query always returns the *current* one — not whichever is closest by
   embedding similarity.
2. **Abstain correctly.** If a question's answer was never stated in any session, the system
   says "I don't have that in memory" instead of letting an LLM guess from the nearest match.

Two things ship in this repo:

1. **A small, hand-crafted demo** (9 synthetic chat sessions, `ingest/ingest.py` + `api/app.py`)
   — a legible, convincing proof of the two properties above, on a closed 6-predicate vocabulary.
2. **A real LongMemEval-S benchmark run** (`eval/`) — open-vocabulary fact extraction and
   question-answering over actual multi-session chat histories from the
   [LongMemEval](https://github.com/xiaowu0162/LongMemEval) dataset, entirely on a local model.
   See [LongMemEval evaluation](#longmemeval-evaluation) below.

## How it works

```
data/sessions.json  --(LLM extraction)-->  ingest/ingest.py  --(writes)-->  HydraDB
                                                                                  |
ui/index.html  <--(JSON)--  api/app.py  --(Cypher query + LLM phrasing)---------+
```

**Schema:**

```
(:Session {id, key, index, date})
(:Fact {id, key, subject, predicate, object, text, timestamp, current})

(:Fact)-[:STATED_IN]->(:Session)
(:Fact)-[:SUPERSEDES]->(:Fact)     -- newer fact points at the one it overwrites
```

Every `Fact` carries a `current: true/false` flag. When ingestion sees a new fact whose
`(subject, predicate)` already has a current fact with a *different* value, it writes the new
fact, links it `-[:SUPERSEDES]->` the old one, and flips the old fact's `current` to `false` —
all via real Cypher queries against the live graph, not in-memory bookkeeping in the ingestion
script. Retrieval is then a single, trivially correct query:

```cypher
MATCH (f:Fact {subject: $s, predicate: $p, current: true})-[:STATED_IN]->(sess:Session)
RETURN f.text, f.object, sess.key, sess.date
```

Zero rows back means the graph has no current fact for that predicate — that's the abstention
signal the `/ask` endpoint acts on, before any LLM gets a chance to synthesize an answer.

## Real HydraDB behavior vs. the docs

While building this, direct testing against a running HydraDB node (not just its README)
turned up several load-bearing constraints worth documenting for anyone else building on it:

- **The Python `neo4j` Bolt driver refuses to connect at all.** It hard-rejects any server whose
  agent string doesn't start with `"Neo4j/"` — HydraDB reports `"SlateDBGraph/0.1.0"` — with no
  config flag to bypass it. HydraMemory talks to HydraDB entirely over its **HTTP JSON API**
  instead (`hydra_client.py`), which works cleanly.
- **Every write is exactly one relationship connecting two node patterns.** `CREATE` or `MERGE`
  of a bare, unconnected node is rejected ("only one-hop edge patterns are executable"), and so
  is any query mixing `MATCH` with `CREATE` ("write query is not executable by the mutation
  engine"). The fix: use `MERGE` for every write — it correctly matches-or-creates each side of
  the one-hop pattern by `id`, which is how ingestion "attaches a new Fact to an existing
  Session" without ever needing `MATCH` + `CREATE` together.
- **`id` must be an integer, and it's a global identity across the whole graph — not scoped per
  label.** Two nodes of different labels can't share an `id` value. HydraMemory uses a simple
  incrementing integer counter for `id` and a separate string `key` property (e.g. `"sess-01"`,
  `"fact-07"`) for human-readable references.
- **Pattern negation (`WHERE NOT (a)-[:R]->(b)`)** — the mechanism `PROJECT_CONTEXT.md`'s
  original schema sketch proposed for finding "the fact nothing supersedes" — **isn't
  supported.** That's exactly why the schema uses an explicit `current` boolean instead of
  computing it via negation at query time.
- **The request body key is `parameters`, not `params`**, for the HTTP API's parameterized
  queries.

## Setup

Requires Docker Desktop and Python 3.9+.

**1. Configure the LLM provider**

```bash
cp .env.example .env
```

Default is `LLM_PROVIDER=ollama` — no API key needed, but requires a local Ollama install with
`qwen2.5:3b-instruct` and `qwen2.5:7b-instruct` pulled (see [LongMemEval evaluation](#longmemeval-evaluation)
below). To use a hosted provider instead, set `LLM_PROVIDER=groq` (or `gemini` / `cerebras`) in
`.env` and add the matching API key (e.g. `GROQ_API_KEY`, from https://console.groq.com).

**2. Start HydraDB**

```bash
mkdir -p hydradb-data/store hydradb-data/cache
printf '%s\n' 'local-development-token-32-bytes' > hydradb-data/auth-token
docker compose up -d
```

Confirm it's up:

```bash
curl -sS http://127.0.0.1:8443/v1/graphs/default/query \
  -H "Authorization: Bearer local-development-token-32-bytes" \
  -H 'X-Graph-Namespace: default' -H 'Content-Type: application/json' \
  --data '{"cell_id":"cell-0","query":"MATCH (n {id: 1}) RETURN n.id"}'
```

**3. Install dependencies**

```bash
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash; use venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

**4. Ingest the synthetic dataset**

```bash
python ingest/ingest.py
```

This reads `data/sessions.json`, extracts facts via whichever LLM provider is configured in
`.env`, and writes them to HydraDB with supersession detection. It prints each fact as it's
written, including which ones superseded an
earlier value.

**5. Run the API**

```bash
python api/app.py
```

Serves `POST /ask` on `http://127.0.0.1:5000`:

```bash
curl -sS http://127.0.0.1:5000/ask -X POST -H 'Content-Type: application/json' \
  --data '{"question": "Where does the user currently live?"}'
# {"answer": "The user currently lives in Pune.", "found": true, "source_sessions": ["sess-07"]}
```

**6. Open the UI**

Open `ui/index.html` directly in a browser (or serve it: `python -m http.server 8080 -d ui`).
It's a static page that calls the API at `http://127.0.0.1:5000` — no build step.

**7. Verify end-to-end**

With HydraDB and the API both running:

```bash
python scripts/verify_e2e.py
```

Checks HydraDB and the API are reachable, then runs 7 fixed questions against whatever is
currently in the graph and asserts on the result: five should resolve to the *current* value of
a fact that was updated at least once (proving supersession), and two ask about things never
stated in any session (proving abstention). Exits non-zero if anything fails.

## LongMemEval evaluation

`eval/` runs the same graph-backed memory architecture against real [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval)
instances — each one 30-50 real chat sessions (~115k tokens) that a question's answer may
require synthesizing across, with facts that get restated, updated, or never mentioned at all.
It deliberately differs from the Part 1 demo above:

- **Open predicate vocabulary** instead of the fixed 6-value enum — real LongMemEval questions
  span arbitrary topics, so the model invents a short snake_case predicate per fact rather than
  being forced into a closed set.
- **Cumulative vs. replace tracking** — a fact is tagged `cumulative: true` if it's one entry in
  an open-ended list the user is building up (a restaurant tried, a book read), so repeated
  mentions of the same topic accumulate instead of overwriting each other the way a
  single-value fact (home city, job title) correctly does via `SUPERSEDES`.
- **Keyword-filtered candidate retrieval** — when a question's instance has 50-80+ current
  facts, only the ones sharing keywords with the question are handed to the answering model,
  instead of dumping the entire candidate set into context.
- **Runs entirely on a local Ollama model** (`qwen2.5:3b-instruct`, with automatic fallback to
  `qwen2.5:7b-instruct` only when the fast model's output fails schema validation) — no API
  key, no rate limits, no per-token cost.

**Result:** 7/8 (88%) on the LongMemEval-S subset in `eval/data/subset.json`, covering both
`knowledge-update` and `abstention` question categories. See `eval/results.json` for the
per-question breakdown.

**Running it:**

```bash
# 1. Pull the local models (one-time)
ollama pull qwen2.5:3b-instruct
ollama pull qwen2.5:7b-instruct

# 2. Point llm_client.py at Ollama
echo "LLM_PROVIDER=ollama" >> .env

# 3. Start HydraDB (see Setup above), then run the eval
python eval/run_eval.py
```

Prints per-question progress and a final accuracy table; writes `eval/results.json`. A
completed instance is cached — re-running after an interruption skips re-ingesting any question
whose facts are already in HydraDB.

The same ingestion/answering/grading logic (`eval/engine.py`) is also exposed live through
`api/app.py`'s `/eval/*` endpoints and `ui/index.html`'s eval tab, for an interactive walkthrough
of a single instance instead of a full batch run.

## Project structure

```
data/sessions.json     synthetic dataset: 9 sessions, 5 fact supersessions, 1 no-fact session
ingest/ingest.py        sessions.json -> LLM extraction -> HydraDB writes (closed vocabulary)
api/app.py               POST /ask -> Cypher query -> LLM-phrased answer, or abstain;
                          also serves the /eval/* endpoints for the LongMemEval flow
ui/index.html            static single-page demo UI (Part 1 demo + LongMemEval eval tab)
hydra_client.py          shared HydraDB HTTP API client
llm_client.py             shared LLM client -- groq/gemini/cerebras/ollama providers,
                          the fixed predicate vocabulary, schema-constrained JSON extraction
docker-compose.yml       local HydraDB node

eval/engine.py           LongMemEval ingestion/answering/grading (open vocabulary,
                          cumulative-fact tracking, keyword-filtered retrieval)
eval/run_eval.py         CLI batch runner over eval/data/subset.json -> eval/results.json
eval/data/subset.json    8-instance LongMemEval-S subset used for the 88% result above
```

## The dataset

`data/sessions.json` is 9 hand-written chat sessions spanning June-August 2026. Five facts get
updated across sessions (location, job + title, favorite food, travel plan), each a clean
old-value-to-new-value contradiction. One session (a book recommendation) deliberately contains
no fact matching the tracked vocabulary, to confirm ingestion doesn't force-extract noise.
Abstention is demonstrated by asking about anything outside the tracked vocabulary (e.g.
favorite color, birthplace) — the graph has no such predicate, so `/ask` correctly returns
`found: false`.

## Attribution

- [HydraDB](https://github.com/hydra-db/hydradb) (AGPL-3.0) — the graph database this project
  is built on.
- [LongMemEval](https://github.com/xiaowu0162/LongMemEval) — the benchmark dataset used in
  `eval/data/subset.json` for the evaluation results above.
- [Groq](https://groq.com) — hosts the `openai/gpt-oss-120b` model, one of several supported
  LLM providers (alongside Gemini, Cerebras, and local Ollama), used for fact extraction,
  question parsing, and answer phrasing via its OpenAI-compatible API.
- [Ollama](https://ollama.com) — runs the local `qwen2.5:3b-instruct` / `qwen2.5:7b-instruct`
  models used for the LongMemEval evaluation, with no API key or rate limit.
- Python packages: `flask`, `requests`, `openai`, `python-dotenv` (see `requirements.txt`).
- The synthetic dataset in `data/sessions.json` is original, written for this project.

## License

MIT — see [LICENSE](LICENSE).
