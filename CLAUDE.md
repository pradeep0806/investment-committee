# CLAUDE.md — The Investment Committee (Jnaara Take-Home, Problem A)

This file is the operating manual for whoever (human or Claude Code) builds this project.
Read it in full before writing code. It encodes decisions already made — if you think one is
wrong, say so explicitly and propose an alternative rather than silently changing it.

---

## 0. What we're actually being graded on

| Dimension | Weight | Looking for |
|---|---|---|
| Problem Decomposition | 25% | How you broke it apart, what abstractions you chose, what you decided *not* to build and why |
| Architecture & Extensibility | 25% | Can another engineer add an agent / pipeline step / strategy without rewriting core logic? |
| Systems Thinking | 20% | Failure modes, edge cases, concurrency, state, observability |
| Code Quality & Tests | 15% | Clean core paths, meaningful tests, type hints, error handling |
| Stretch & Real-Time | 10% | Does streaming actually help a user understand what's happening? |
| README & Reasoning | 5% | Honest tradeoffs, AI prompts documented |

The founders explicitly said they don't care about visual polish or exhaustive test coverage.
They care about the **explore-exploit budget logic**, **disagreement handling**, and whether
the system is genuinely pluggable. Every hour spent on infra ceremony that doesn't serve one
of the five rows above is an hour not spent on the thing being graded. Keep that tension
visible in every decision below.

---

## 1. Scope decisions — read before building anything

These are deliberate deviations from a "maximal" build, made because they serve the rubric
better than the alternative. Document the reasoning in the README verbatim where noted —
"what we decided not to build and why" is a *graded* line item, not a footnote.

1. **CLI is mandatory**, independent of the FastAPI layer. It's a Core requirement
   ("CLI entrypoint: accepts a thesis, budget, and configuration"). Build the orchestrator as
   a plain Python object with no web framework dependency; CLI and API both call into it.

2. **One business endpoint.** `POST /debate` returns the full trace + synthesis.
   `POST /debate?stream=true` returns the same computation as Server-Sent Events on the *same*
   route. `/health` and `/metrics` are infra, not business logic — they don't count against
   "one endpoint." Note for whoever builds a client: browsers' native `EventSource` only
   supports GET, so a streaming client must use `fetch()` with a `ReadableStream` reader, not
   `EventSource`.

3. **No separate cache-then-sync-worker pipeline.** A single debate run produces on the order
   of 10-20 writes over well under a minute — this is not a write-throughput problem, and a
   write-behind cache with an async sync service exists to solve write-throughput and
   durability problems this system doesn't have. Introducing one here adds a real failure mode
   (what happens when the sync worker dies mid-batch, leaving Redis and Mongo inconsistent?)
   without buying anything that's graded. Instead:
   - The orchestrator writes directly to MongoDB after each round (simple, correct, no
     eventual-consistency window).
   - The full trace is *also* always written to local JSON — this is the literal Core
     requirement ("full debate trace saved as structured JSON") and is the source of truth,
     independent of whether Mongo is reachable.
   - Redis is used only for what it's actually suited to here: a pub/sub channel per `run_id`
     so the streaming endpoint can fan out live events without polling Mongo, and a hot cache
     for the agent/prompt registry.
   - All storage sits behind a `TraceStore` protocol (see §5), so a write-behind cache + async
     sync worker could be dropped in later without touching orchestration logic — put this
     exact sentence in the README's "what we didn't build" section.

4. **"Accessible to other engineers" means swappable AI provider, not shared cloud infra.**
   Corrected from an earlier misreading — the actual requirement is that another engineer can
   point this at Claude, OpenAI, or anything LiteLLM supports without touching orchestrator or
   agent code, purely via config (`LLM_PROVIDER` / `LLM_MODEL` / `LLM_API_KEY` in `.env` or a
   ConfigMap/Secret). This is entirely the job of the `llm/client.py` provider-agnostic wrapper
   (§2, §5) — every agent calls that one interface, never a provider SDK directly.

5. **Observability is fused to the debate mechanics — it's how we deliver the Stretch
   real-time requirement, not a separate polish pass.** The Stretch section explicitly asks
   for "a real-time streaming interface showing: which agent is reasoning, running budget,
   explore/exploit mode, disagreement alerts, convergence signals." Rather than building a
   one-off UI just for that line item, deliver it through the same observability stack that's
   genuinely useful otherwise: the orchestrator emits Prometheus metrics for convergence
   score, budget spend per agent/round, mode transitions, and disagreement events as they
   happen; a Grafana panel visualizes the explore→exploit transition live while a debate is
   running. The `POST /debate?stream=true` SSE endpoint and the Grafana dashboard are two
   consumers of the same underlying event stream (Redis pub/sub for SSE, Prometheus scrape
   for Grafana) — the observability doesn't get built twice.

   This is also the direct answer to "Systems Thinking (20%): do you build for observability?"
   — the metrics are instrumented at exactly the points that make Problem A hard (convergence,
   budget allocation, disagreement), not generic HTTP request counters. That's what makes this
   read as understanding the problem rather than decoration.

   **MongoDB, Redis, Prometheus, Grafana, and MLflow are first-class parts of this build.**
   They get wired in as each orchestrator component is written (see §10 build order), not
   bolted on at the end. **Kubernetes is the one piece with no rubric line behind it at all**
   — no dimension scores deployment infra — so it stays a pure differentiation/portfolio
   signal, built last, and is the one thing to drop first if time runs short. Say this
   explicitly in the README rather than letting it look like scope creep: a short "Beyond the
   brief" paragraph naming which graded dimension each infra piece serves (and being upfront
   that K8s serves none of them, but demonstrates deployment maturity) is what turns this from
   "ignored the brief" into "showed judgment."

6. **The explore-exploit transition must be a computed signal, not a hardcoded round number.**
   This is the single most important piece of the assignment — it's the explicit stretch ask
   and the literal "question we're really asking." Do not shortcut it with `if round == 2:
   exploit_mode = True`.

---

## 2. Tech stack

- Python 3.11+, `pydantic` v2 for all structured data, `pydantic-settings` for config
- `typer` for the CLI
- `fastapi` + `uvicorn` for the API
- LLM access via a thin provider-agnostic wrapper (Anthropic / OpenAI / LiteLLM — selected by
  `.env`), with structured output enforced via tool-calling + Pydantic validation and a
  bounded retry-on-invalid loop (never trust raw JSON parsing of free text)
- `motor` (async Mongo driver) for MongoDB, `redis.asyncio` for Redis
- `structlog` for JSON logging, `prometheus-fastapi-instrumentator` + custom metrics,
  `mlflow` for run tracking
- `pytest` + `pytest-asyncio` + `respx`/mock LLM responses for tests
- Docker + docker-compose for the required local stack; `k8s/` manifests as an optional,
  lower-priority differentiation layer (see §1.5)

---

## 3. Folder structure

```
investment-committee/
├── CLAUDE.md
├── ARCHITECTURE.md
├── README.md
├── .env.example
├── pyproject.toml
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── src/committee/
│   ├── config.py                     # pydantic-settings, reads .env
│   ├── models/
│   │   ├── requests.py                # ThesisRequest, DebateConfig
│   │   ├── agent_output.py            # Stance enum, AgentOutput
│   │   ├── trace.py                   # RoundRecord, BudgetLedgerEntry, DebateTrace
│   │   └── synthesis.py               # DisagreementRecord, ConvergenceSignal, SynthesisMemo
│   ├── agents/
│   │   ├── base.py                    # AnalystAgent protocol
│   │   ├── registry.py                # plug-and-play registration, no core changes to add one
│   │   ├── fundamentals.py
│   │   ├── market_sentiment.py
│   │   ├── risk_contrarian.py
│   │   ├── macro_context.py
│   │   └── prompts/                   # one guardrailed template per agent
│   ├── orchestration/
│   │   ├── orchestrator.py
│   │   ├── budget_manager.py
│   │   ├── explore_exploit.py         # convergence scoring + mode policy
│   │   ├── disagreement.py            # detection, never averages
│   │   └── conflict_resolution/
│   │       ├── base.py                # ConflictResolutionStrategy protocol
│   │       ├── flag_unresolved.py     # default
│   │       ├── confidence_weighted.py
│   │       └── tie_breaker.py         # stretch: spawns an extra agent
│   ├── synthesis/synthesizer.py
│   ├── llm/
│   │   ├── client.py                  # provider-agnostic wrapper
│   │   └── structured_output.py       # validated call + retry-on-invalid
│   ├── storage/
│   │   ├── trace_store.py             # protocol: save_round, save_final, get_run
│   │   ├── json_store.py              # local JSON writer, source of truth
│   │   ├── mongo_store.py
│   │   └── redis_bus.py               # pub/sub streaming + config cache
│   ├── observability/
│   │   ├── logging.py
│   │   ├── metrics.py
│   │   └── mlflow_tracking.py
│   ├── api/
│   │   ├── app.py                     # FastAPI: /health, /metrics, POST /debate
│   │   └── schemas.py
│   └── cli/main.py                    # typer: run, replay, list-runs
├── observability/
│   ├── prometheus/prometheus.yml
│   └── grafana/{provisioning,dashboards/debate-overview.json}
├── k8s/                                 # differentiation layer, no rubric weight — build last
│   ├── README.md                       # how to build, push, and apply — not applied here
│   ├── kustomization.yaml
│   ├── namespace.yaml
│   ├── configmap.yaml                  # LLM_PROVIDER / LLM_MODEL live here — swap and roll
│   ├── secret.example.yaml             # copy to secret.yaml with real values, never commit
│   ├── api.yaml                        # Deployment + Service + HPA for committee-api
│   ├── dependencies.yaml               # Mongo StatefulSet, Redis Deployment (demo-scale only)
│   ├── observability.yaml              # Prometheus + Grafana + MLflow
│   └── ingress.yaml
├── tests/
│   ├── test_models.py
│   ├── test_agents.py
│   ├── test_budget_manager.py
│   ├── test_explore_exploit.py
│   ├── test_disagreement.py
│   ├── test_orchestrator.py
│   ├── test_storage.py
│   └── test_api.py
├── examples/sample_run_trace.json      # required "example output" deliverable
└── scripts/run_five_sample_debates.sh
```

---

## 4. Core domain model (all Pydantic)

- `ThesisRequest` — thesis text, entity/ticker context (optional), any priors
- `DebateConfig` — total_token_budget, num_rounds (2-3), agent roles (or `None` = use
  defaults), conflict_resolution_strategy, convergence thresholds
- `Stance` enum — `Buy | Hold | Sell | Pass`
- `AgentOutput` — agent_id, round, stance, confidence (0-100), key_factors (list[str], tagged
  for overlap comparison), top_risk, rebuttals (references to prior round's other agents,
  optional), tokens_used
- `BudgetLedgerEntry` — round, agent_id, tokens_allocated, tokens_used, mode (explore/exploit)
- `ConvergenceSignal` — round, stance_agreement, factor_overlap, confidence_spread,
  composite_score, mode_selected
- `DisagreementRecord` — round detected, agents involved, opposing stances, contested factors,
  resolution_strategy_applied, resolved (bool)
- `SynthesisMemo` — final recommendation, confidence, supporting_agents, dissenting_agents,
  dissent_appendix, full reasoning trace references
- `DebateTrace` — top-level: request, config, all rounds, all AgentOutputs, budget ledger,
  convergence signals, disagreements, synthesis, timing, total tokens used

---

## 5. Orchestration flow

1. **Round 1 — explore.** Equal token allocation per agent. Prompts explicitly instruct
   independent reasoning ("give your own view; do not try to agree with anyone").
2. **Compute convergence score** after every round:
   `score = 0.5 * stance_agreement + 0.3 * factor_overlap + 0.2 * (1 - confidence_spread)`
   - `stance_agreement`: fraction of agents matching the majority stance
   - `factor_overlap`: normalized overlap of cited `key_factors` across agents (tag-based
     Jaccard is enough; don't reach for embeddings unless time allows)
   - `confidence_spread`: normalized stddev of confidence scores
3. **Mode policy** (`ExploreExploitController`, swappable):
   - `score < 0.4` → stay explorative; inject a rotating "argue against the majority" prompt
     into one agent to actively force divergence
   - `0.4 ≤ score ≤ 0.75` → balanced round, moderate rebalancing
   - `score > 0.75` → exploit; shift 2-3x the baseline token allocation to the contested or
     minority-view agent(s), optionally spawn a tie-breaker (stretch)
4. **Disagreement detection** runs every round: if final-round stances split with no
   majority ≥ 3/4 (or any Buy-vs-Sell direct conflict at high confidence on both sides), do
   **not** average into a blended recommendation. Create a `DisagreementRecord` and route to
   the active `ConflictResolutionStrategy`.
5. **Conflict resolution strategies** (pick one via config, all implement the same protocol
   so adding a 4th needs no orchestrator change):
   - `flag_unresolved` (default) — synthesis explicitly states the split and both positions
   - `confidence_weighted` — majority wins, weighted by stated confidence, but the losing
     side's reasoning still appears as a dissent appendix
   - `tie_breaker` (stretch) — spawns one additional agent with the two opposing arguments as
     context, extra budget drawn from a reserve pool
6. **Synthesis** produces the committee memo: recommendation, confidence, and — critically —
   an explicit dissent section any time `DisagreementRecord`s exist, rather than a single
   averaged number.
7. **Every round writes to**: local JSON trace file (always), MongoDB (best-effort, logged if
   it fails — never blocks the debate on a DB write), Redis pub/sub channel (for live
   streaming, best-effort). At the same point, emit the Prometheus metrics and MLflow log
   entries for that round (§8) — these are not a separate pass over the trace afterward, they
   fire from the same code path that just computed the convergence score, budget spend, and
   any disagreement, because that's the only place those numbers exist without recomputation.

---

## 6. Default agent lenses

Config-driven; adding a 5th means one new class + one registry line, no orchestrator change.

| Agent | Lens | Blind spot (deliberately) |
|---|---|---|
| Fundamentals/Valuation | Financials, unit economics, valuation multiples | Ignores narrative/momentum entirely |
| Market Sentiment | Price action, positioning, narrative momentum | Weak on balance-sheet detail |
| Risk Contrarian | Downside, tail risk, what could go wrong | Deliberately skeptical even of strong theses |
| Macro/Industry Context | Sector trends, competitive dynamics, macro headwinds/tailwinds | Weak on company-specific execution detail |

If `DebateConfig.agent_roles` is unset, use these four. If the user supplies custom roles in
config, use those. A "derive N lenses from the thesis via a meta-prompt" mode is a good
stretch extension but is *not* Core — don't build it before the four above work end to end.

---

## 7. `.env` schema

```
# LLM
LLM_PROVIDER=anthropic            # anthropic | openai | litellm
LLM_MODEL=claude-sonnet-4-6
LLM_API_KEY=
LLM_MAX_RETRIES=3
LLM_TIMEOUT_SECONDS=60

# Debate defaults
DEFAULT_TOKEN_BUDGET=50000
DEFAULT_NUM_ROUNDS=3
CONVERGENCE_HIGH_THRESHOLD=0.75
CONVERGENCE_LOW_THRESHOLD=0.4
CONFLICT_RESOLUTION_STRATEGY=flag_unresolved

# Storage
MONGO_URI=mongodb://localhost:27017
MONGO_DB=investment_committee
REDIS_URL=redis://localhost:6379/0
TRACE_JSON_DIR=./traces

# Observability
MLFLOW_TRACKING_URI=./mlruns
PROMETHEUS_PORT=9100
LOG_LEVEL=INFO
LOG_FORMAT=json

# API
API_HOST=0.0.0.0
API_PORT=8000
```

---

## 8. Observability spec

Every metric below is emitted from inside the orchestration loop itself (§5, step 7), not
computed afterward from the trace — that's what makes this "fused" rather than bolted on.

- **Logging**: structured JSON via `structlog`, every log line carries `run_id` as a
  correlation id. Loki is optional (docker-compose driver) for aggregation into Grafana.
- **Prometheus metrics**:
  - `debate_tokens_used_total{agent,round}` — budget spend, the raw material for a
    stacked-bar "budget by agent" panel
  - `debate_convergence_score{run_id}` (gauge) — updated after every round; this is the
    single number the explore-exploit controller is deciding on, and it's the metric a
    "convergence over time, with threshold lines at 0.4/0.75" panel is built from
  - `debate_active_agent{run_id,agent}` (gauge, 1 while reasoning else 0) — this is what
    lets a panel answer "which agent is reasoning right now," one of the literal Stretch
    asks
  - `disagreements_detected_total{run_id}` and `mode_transitions_total{from,to}` — feed a
    "disagreement alerts" and "explore→exploit transition" panel respectively
  - `llm_call_errors_total{provider}`, `llm_call_latency_seconds` — ordinary API health,
    not domain-specific, but cheap to add alongside the rest
- **Grafana**: one pre-provisioned dashboard, checked in as JSON. Panels: convergence score
  over time (with threshold lines), token budget by agent, mode timeline, disagreement
  alerts, currently-active agent. Together these panels *are* the real-time interface the
  Stretch section asks for — viewed live on a short refresh interval while a debate runs,
  not assembled after the fact. `POST /debate?stream=true` covers the same ground for anyone
  who wants it inline in a terminal or a custom UI without standing up Grafana.
- **MLflow**: one run per debate. Params: thesis, budget, model, num_agents, strategy.
  Metrics: tokens_used, rounds_run, disagreements_count, convergence_at_final_round, plus the
  full convergence-score-per-round series logged as a metric history so a completed run's
  explore→exploit trajectory is inspectable after the fact, not just live in Grafana.
  Artifacts: full trace JSON, synthesis memo.

---

## 9. Testing requirements (Core)

- Structured output validation (invalid LLM output triggers retry, not a crash)
- Budget enforcement (orchestrator never exceeds total_token_budget even under
  worst-case allocation)
- Explore-exploit controller (convergence score math, mode transitions at boundary values)
- Disagreement detection (opposing stances flagged, never silently averaged)
- Conflict resolution strategy swapping (same disagreement, different strategy → different
  synthesis)
- Orchestrator end-to-end with a mocked LLM client (deterministic fixture responses)

---

## 10. Build order

Correctness first, but observability gets wired in as each piece lands, not tacked on at the
end — a metric added the day its underlying logic is written is trivial; retrofitted after
the fact it's guesswork about what the code was doing.

1. Domain models + config loading
2. One agent working end-to-end against a real LLM call, with its own test
3. Orchestrator with fixed equal budget across a hardcoded 2-round debate (get the loop right
   before making it smart). Wire `structlog` + `debate_duration_seconds` here — the logging
   skeleton should exist before there's meaningful state to log.
4. Budget manager + disagreement detection. Add `debate_tokens_used_total` and
   `disagreements_detected_total` the same day each is built, from inside the code that
   already computed those numbers.
5. Explore-exploit convergence controller (the centerpiece — don't rush this). Add
   `debate_convergence_score` and `mode_transitions_total` here — this is the metric the
   eventual Grafana dashboard is really about, so get the number right before worrying about
   how it's displayed.
6. Synthesis step with dissent handling
7. CLI
8. Storage layer behind `TraceStore` (JSON always, Mongo/Redis best-effort). Start an MLflow
   run per debate here, logging params/metrics/artifacts as the trace is written.
9. FastAPI single endpoint, including the `?stream=true` SSE path. Add `debate_active_agent`
   here — it needs the same "an agent just started/finished reasoning" event the SSE stream
   is already emitting.
10. docker-compose + the Grafana dashboard JSON — by this point every panel is just wiring
    up metrics that already exist, not inventing new instrumentation.
11. Tests throughout (not deferred to the end), then README (including the "Beyond the
    brief" paragraph from §1.5) + `examples/sample_run_trace.json`

**Lower priority, build after everything above is solid:** the `k8s/` manifest set (§1.5).
Do not start this while any Core requirement is unfinished — an unfinished disagreement
detector graded at 20-25% loses far more than a missing k8s folder, which carries no rubric
weight at all.

---

## 11. Non-goals (state these in the README, don't apologize for them)

No auth, no multi-tenancy, no horizontal-scale load testing, no real brokerage/market-data
integration beyond whatever the thesis text provides, no fine-tuning. The `k8s/` manifests are
a differentiation layer with no rubric weight (see §1.5) — no live cluster deployment or
TLS/cert-manager/NetworkPolicy hardening is expected regardless of whether that folder gets
built. MongoDB, Redis, Prometheus, Grafana, and MLflow are *not* non-goals — they're fused
into the orchestration loop per §1.5, §8, and §10.
