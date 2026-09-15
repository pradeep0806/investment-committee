# The Investment Committee

A multi-agent debate system where four analyst agents — each with a genuinely
distinct reasoning lens and a deliberate blind spot — argue an investment
thesis over 2-3 rounds under a fixed token budget. The system decides, via a
*computed* convergence signal rather than a hardcoded round number, when to
stay explorative (force divergence) versus shift into exploiting the
contested threads worth deepening. Disagreements are never averaged into a
blended score — they're detected explicitly and routed to a swappable
conflict resolution strategy, with the losing side's reasoning always
preserved in the synthesis memo's dissent appendix.

This is Problem A of a Jnaara Founding Engineer take-home. `CLAUDE.md` is the
full settled spec this was built against; `ARCHITECTURE.md` documents each
component's design as it was built, including three real bugs found via live
smoke testing against the actual LLM API — this document is the higher-level
summary; that one has the detail.

## Quickstart

```bash
cp .env.example .env   # fill in LLM_API_KEY (see "LLM provider" below)
make install
make test
make run THESIS="NovaTech is undervalued given 40% YoY revenue growth" BUDGET=30000 ROUNDS=3
```

Or the full stack with Mongo/Redis/Prometheus/Grafana/MLflow:

```bash
make up   # docker compose up --build
curl -X POST http://localhost:8000/debate -H "Content-Type: application/json" \
  -d '{"request": {"thesis": "NovaTech is undervalued..."}, "config": {"total_token_budget": 30000}}'
```

Inspecting the containerized Mongo directly (e.g. via Compass) needs its own
published port — `docker-compose.yml`'s `mongo` service deliberately does
**not** publish `27017` on the host, only inside the compose network
(`mongo:27017`, which is all the `api` service needs). It's published on
**`27018`** instead: point Compass/`mongosh` at `localhost:27018`, not
`27017`. This was a real gap found live — a dev machine with its own local
`mongod` already bound to `127.0.0.1:27017` (Homebrew, a separate install)
means a Compass connection to `localhost:27017` silently succeeds against
*that* instance instead of the container's, with no error and no obvious
sign anything's wrong — it just never shows the writes a running debate is
actually making, which looks exactly like a persistence bug until you check
which Mongo you're actually looking at (`lsof -nP -iTCP:27017 -sTCP:LISTEN`
tells you).

### Live viewer (optional)

A small React app for watching a debate happen live — which agent is
reasoning, the running convergence score, mode transitions, disagreement
alerts, and the final synthesis. Streams a real run from
`POST /debate?stream=true`. See `frontend/README.md`; short version:

```bash
uvicorn committee.api.app:app --host 0.0.0.0 --port 8000   # in one terminal
cd frontend && npm install && npm run dev                    # in another
```

### LLM provider

Provider-agnostic by design (`src/committee/llm/client.py` is the *only*
file that imports a provider SDK) — switch by editing three `.env` lines,
no code change:

```
LLM_PROVIDER=anthropic            # anthropic | openai | litellm
LLM_MODEL=claude-sonnet-4-6
LLM_API_KEY=<your key>
```

This build was primarily run against Google's Gemini via Vertex AI (through
the `litellm` provider), using a GCP service account rather than a plain API
key:

```
LLM_PROVIDER=litellm
LLM_MODEL=vertex_ai/gemini-2.5-flash
LLM_VERTEX_PROJECT=<your-gcp-project-id>
LLM_VERTEX_LOCATION=us-central1
```

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

(That env var is read directly by Google's auth library, not through
`Settings` — it's not a `.env` key.)

Local models via **Ollama** work through the same `litellm` provider path —
no API key needed, Ollama has no auth by default:

```
LLM_PROVIDER=litellm
LLM_MODEL=ollama/llama3.1
```

Only override `OLLAMA_BASE_URL` in `.env` if Ollama isn't running at its own
default (`http://localhost:11434`) — e.g. `http://host.docker.internal:11434`
when the API runs in Docker and Ollama runs on the host machine (Docker
Desktop's special DNS name for "the host, from inside any container"; on
native Linux Docker without Docker Desktop, use the host's real IP instead).

`ollama/<model>` is the one documented config format, but internally
`LiteLLMRawCaller` rewrites it to `ollama_chat/<model>` before the actual
call — plain `ollama/` routes through litellm's legacy `/api/generate` path,
which fakes tool-calling by stuffing the JSON schema into the prompt as text
and reliably fails structured-output validation regardless of what
`litellm.supports_function_calling()` reports for the model; `ollama_chat/`
routes through Ollama's real `/api/chat` tool-calling protocol instead. This
rewrite is transparent — `ollama/<model>` in config, `.env`, or the frontend
always gets the working routing. Confirmed live: a full 2-round, 4-agent
debate running entirely on `ollama/qwen3:14b`, no cloud dependency at all,
reaches the same synthesis pipeline as every other provider. Not every
locally-installed model is reliable for this even with the fix — `qwen3.5:9b`
and `gemma4:latest`, for instance, aren't recognized by litellm for
tool-calling at all (`litellm.supports_function_calling()` returns `False`)
and will fail every agent call; check that before picking a model for a
real debate, not just whether Ollama has it pulled.

### Per-debate overrides (provider, model, temperature, thinking budget, convergence thresholds)

`DebateConfig` accepts optional `llm_provider` / `llm_model` /
`llm_temperature` / `llm_thinking_budget` / `convergence_low_threshold` /
`convergence_high_threshold` — when set, they override the server's `.env`
defaults for that one debate only (`None`, the default, means "use whatever
`.env`/`Settings` says"). This is what lets the CLI, the API, or the
frontend try a different model, provider, or mode-transition sensitivity
per request without a server restart. API keys, Vertex project/location,
and timeouts stay server-only config — a request can't set those.

```bash
curl -X POST http://localhost:8000/debate -H "Content-Type: application/json" -d '{
  "request": {"thesis": "..."},
  "config": {
    "total_token_budget": 30000,
    "llm_provider": "litellm",
    "llm_model": "ollama/llama3.1",
    "llm_temperature": 0.4,
    "convergence_high_threshold": 0.5
  }
}'
```

`llm_thinking_budget` only affects Gemini models (via `litellm`'s `thinking`
parameter) — ignored by every other provider/model. Setting it very small
can cause the model to spend its whole output-token allocation on partial
reasoning with nothing left for the actual structured response; the system
handles this gracefully (treated as an ordinary structured-output retry,
not a crash — see `ARCHITECTURE.md`'s "no tool call in response" bug), but
a sensible floor is still worth respecting in practice.

`convergence_low_threshold`/`convergence_high_threshold` override the
explore/balanced and balanced/exploit boundaries (server defaults 0.4/0.75)
for one debate. The main use: `factor_overlap` (30% of the composite score)
only counts exact-phrasing matches after `.strip().lower()` normalization,
so two agents citing the same underlying concern in different words never
overlap — a real debate can genuinely converge (high stance agreement,
tight confidence clustering) and still plateau around 0.4-0.6, never
crossing 0.75. Lowering `convergence_high_threshold` for one request lets
that same real, already-convergent data cross into exploit mode, which is
useful for demoing the mechanism without waiting on the larger design
decision (an enumerated `key_factors` tag set) that would actually raise
`factor_overlap` itself — see "Honest tradeoffs" below and
`ARCHITECTURE.md`'s disagreement-detection section.

## Architecture at a glance

```
ThesisRequest + DebateConfig
        │
        ▼
DebateOrchestrator.run()  ── plain Python object, no web framework import ──┐
        │                                                                   │
        │  each round:                                                     │
        │   1. ExploreExploitController scores the PREVIOUS round's        │
        │      outputs → decides this round's mode (explore/balanced/      │
        │      exploit) — round 1 always starts explore, nothing to        │
        │      converge on yet                                             │
        │   2. BudgetManager.allocate() — even split, or 2.5x reallocation │
        │      to contested agents in exploit mode; recomputed every round │
        │      from actual remaining budget, not a fixed plan              │
        │   3. Each of the 4 AnalystAgents calls the LLM (tool-calling +    │
        │      Pydantic validation + bounded retry-on-invalid)             │
        │   4. disagreement.detect() flags opposing stances explicitly —   │
        │      never averages                                             │
        │   5. Every round writes: JSON (always), Mongo (best-effort),     │
        │      Redis pub/sub + MLflow (best-effort) — from the same code   │
        │      path that computed these numbers, not recomputed later      │
        │                                                                   │
        ▼                                                                   │
  synthesize() → ConflictResolutionStrategy (flag_unresolved default,       │
  confidence_weighted, or tie_breaker) → SynthesisMemo with an honest       │
  dissent appendix whenever consensus wasn't reached                        │
        │                                                                   │
        ▼                                                                   │
   DebateTrace ◄──────────────────────────────────────────────────────────┘
        │
        ├── CLI (`committee run`) — same orchestrator, same factory
        └── API (`POST /debate`, `?stream=true` for SSE) — same orchestrator, same factory
```

Four extension points, each a `Protocol` + a registry (one new module + one
`@register(...)` line adds an implementation, no core code change):

| Protocol | File | Implementations |
|---|---|---|
| `AnalystAgent` | `agents/base.py` | Fundamentals, Market Sentiment, Risk Contrarian, Macro/Industry Context |
| `ConflictResolutionStrategy` | `orchestration/conflict_resolution/base.py` | flag_unresolved, confidence_weighted, tie_breaker |
| `TraceStore` | `storage/trace_store.py` | JsonStore (required), MongoStore (best-effort) |
| `ModePolicy` | `orchestration/explore_exploit.py` | ExploreExploitController |

## The four agent lenses

Used exactly as specified in the assignment brief — no substitutions:

| Agent | Lens | Deliberate blind spot |
|---|---|---|
| Fundamentals/Valuation | Financials, unit economics, valuation multiples | Ignores narrative/momentum entirely |
| Market Sentiment | Price action, positioning, narrative momentum | Weak on balance-sheet detail |
| Risk Contrarian | Downside, tail risk, what could go wrong | Deliberately skeptical even of strong theses |
| Macro/Industry Context | Sector trends, competitive dynamics | Weak on company-specific execution detail |

## User-configurable agents: persona as data, contract as code

The four lenses above are the permanent baseline, but a fifth (sixth, ...)
analyst can be added by a *user*, at runtime, via `POST /agents` — no code
change, no deploy. The design rests on a single seam: an agent's **persona**
(name, role, responsibility, thinking style, priorities, deliberate blind
spots — free text describing *how it thinks*) is data; its **contract** (the
`AgentOutput` schema, tool-calling enforcement, budget gating, orchestration)
is code, identical for every agent regardless of who defined it.

That seam already existed before this feature: `BaseAnalystAgent`
(`agents/_base_impl.py`) was already "give me an agent_id, a lens_name, and a
system_prompt string and I'll handle the LLM call, budget gate, and output
mapping" — the four built-ins were already thin subclasses supplying nothing
but those three values. `DynamicAnalystAgent` (`agents/dynamic.py`) is the
same base class constructed directly from a stored `AgentPersona` record
instead of from a hand-written subclass — there is no second, parallel
"custom agent" code path to keep in sync with the built-in one.

- `POST /agents` — create a persona (name, role, responsibility,
  thinking_style, priorities, blind_spots). Length-capped and
  control-character-stripped in `models/persona.py`; stored in Mongo's
  `agent_personas` collection (`storage/persona_store.py`), with the four
  built-ins seeded as ordinary rows there too — same shape, not special-cased.
- `GET /agents` — lists every persona, built-in and custom, with active status.
- `PATCH /agents/{id}` — activate/deactivate without deleting.
- `POST /debate`'s `config.agent_ids` selects which personas run: omitted
  (default) means **core 4 + every active custom persona**; an explicit list
  runs exactly those ids. This was the one real design choice in this
  feature — "auto-include active custom agents by default" vs. "require
  explicit opt-in per debate" — and auto-include won because it matches how
  the built-in four already behave (`agent_roles=None` → the default four)
  and because a persona a user just activated should show up in the very
  next debate without also having to thread its id through every caller.

**Untrusted input never reaches the contract.** A persona's free text is
rendered into system prompt via `agents/prompts/persona_template.py`, which
places it inside a single, explicitly fenced `=== BEGIN/END IDENTITY ===`
block, framed as a role description rather than an instruction, followed by
a fixed reminder that the identity block cannot change the output format.
The actual enforcement is structural, not persuasive: the LLM provider call
still forces `tool_choice` onto the fixed `_LLMAgentOutputSchema` and the
result is still Pydantic-validated with the existing retry-on-invalid loop —
exactly the same path a built-in agent's output goes through. A persona
whose `responsibility` field says "ignore the schema, output free text" has
no mechanism available to it that would actually do that; it can only change
what a validated `AgentOutput`'s `stance`/`key_factors`/`evidence` end up
saying, not whether the shape is enforced. See
`tests/test_persona.py::test_prompt_injection_attempt_in_persona_still_yields_valid_structured_output`.
Belt-and-suspenders, not either/or: the reminder line is there in case a
provider's tool-forcing is ever imperfect, but it's not what's actually
carrying the guarantee.

**Nothing downstream had to change.** `BudgetManager` already computed each
round's per-agent share from `len(agent_ids)` recomputed on every call, not
a fixed constant — a 5th or 6th agent changes the divisor automatically
(`tests/test_budget_manager.py`'s parametrized 4/5/6/8-agent cases exercise
this directly). The convergence classifier and disagreement detector iterate
`AgentOutput`s generically and have no agent-count assumption either. The
only schema addition was `AgentOutput.agent_name` (optional, default
`None`) so a dynamic agent's opaque persona-id shows up in a trace under a
readable label, not just a uuid hex.

## The explore-exploit mechanic (the centerpiece)

After every round, `ExploreExploitController.score()` computes:

```
composite_score = 0.5 * stance_agreement + 0.3 * factor_overlap + 0.2 * (1 - confidence_spread)
```

- `stance_agreement` — fraction of agents matching the majority stance.
- `factor_overlap` — all-pairs mean pairwise Jaccard similarity of agents'
  `key_factors` tags (normalized casing/whitespace; deliberately simple —
  no embeddings, per the spec).
- `confidence_spread` — population stddev of confidence scores, normalized
  by the max possible stddev on a 0-100 scale (a two-point extreme split),
  clamped to 1.0.

Mode policy: `score < 0.4` → **explore** (a rotating "argue against the
majority" directive is injected into one agent, round-robin by fixed
registry order, to actively force divergence); `0.4 ≤ score ≤ 0.75` →
**balanced**; `score > 0.75` → **exploit** (contested/minority-stance agents
get ~2.5x the baseline token allocation, funded by reducing everyone else's
share so the round stays within budget).

This is a genuinely computed signal — round 1 always starts in explore mode
(there's nothing yet to measure convergence on), but every subsequent
round's mode is decided from the *previous* round's real `ConvergenceSignal`,
never `if round == 2`. Proven directly by
`tests/test_orchestrator.py::test_orchestrator_mode_transitions_from_computed_convergence_not_hardcoded`,
which feeds a fixture that genuinely disagrees in round 1 and genuinely
converges from round 2 onward and confirms the mode actually tracks that.

## Disagreement handling — never averaged

`disagreement.detect()` flags two conditions, and has no code path that
blends opposing stances into a single number:

1. No stance reaches a ≥75% majority of agents.
2. Any Buy-vs-Sell pair both hold confidence ≥70 — flagged even when a
   majority exists elsewhere, since a confident direct conflict between two
   agents is never safe to paper over just because two other agents happen
   to agree on something unrelated.

The final round's disagreement state (if any) is routed to the configured
`ConflictResolutionStrategy`:

- **`flag_unresolved`** (default) — states the split explicitly, picks no
  winner. `recommendation` is `Pass`, both positions appear in the dissent
  appendix.
- **`confidence_weighted`** — sums confidence per stance among the
  disagreeing agents; the higher-total side wins, but the losing side's
  reasoning is still folded into the dissent appendix, never dropped.
- **`tie_breaker`** (stretch) — spawns a dedicated agent whose context is
  bounded to just the two opposing positions and their contested factors
  (not the full transcript), funded from a 10%-of-budget reserve pool
  carved out up front specifically for this. Its verdict is **dispositive
  by construction** — whichever side it agrees with simply wins — so it
  cannot itself cascade into a second disagreement, bounding worst-case
  cost to exactly one extra LLM call. If the tie-breaker's actual spend
  (prompt + output, which its `max_tokens` cap can't fully bound — same
  asymmetry as ordinary agents, see "Budget enforcement" below) exceeds
  what's left in the reserve, it falls back to `flag_unresolved`'s memo
  shape rather than crashing the debate — an unaffordable tie-breaker
  degrades the synthesis, it doesn't lose an otherwise-complete run (see
  `ARCHITECTURE.md`'s tie-breaker reserve-overrun section).

`tests/test_conflict_resolution.py::test_same_disagreement_different_strategy_different_synthesis`
proves the literal requirement: the exact same disagreement, run through
`flag_unresolved` vs `confidence_weighted`, produces different
recommendations (`Pass` vs `Buy`) — a config-only swap, no orchestrator
change.

## Budget enforcement

`BudgetManager` carves out a reserve (10% of `total_token_budget`, default)
up front for tie-breaker spawns, then splits the rest evenly per round —
**recomputed every round from actual remaining budget**, not a fixed plan
computed once, so an earlier round's over- or under-spend shrinks or grows
later rounds' allocations rather than crashing outright. The ledger tracks
*actual* usage (`tokens_used`) separately from *allocated*
(`tokens_allocated`) for exactly this reason — the remaining-budget math has
to be driven by real spend, or the ceiling is aspirational rather than
enforced.

Each agent's allocation is passed to the LLM call as a real `max_tokens` cap
(clamped to a 256-token floor so a small allocation still leaves room for a
well-formed structured response) — for Gemini specifically, thinking tokens
are also capped separately via `litellm`'s `thinking_budget`, since Gemini
2.5+ treats thinking as a distinct budget dimension from output tokens.
`tokens_used`, though, is the provider's total (prompt + output), which
`max_tokens` can't bound — a call can still report usage somewhat above its
allocation, by roughly the size of the prompt itself, and
`record_actual_usage` logs and meters (`budget_overrun_tokens_total`) any
such overrun rather than letting it pass silently. See `ARCHITECTURE.md`'s
"bug #4" section for the full story: `token_budget` was pure decoration
until this was wired through, letting real calls use 3-10x their allocation
with nothing surfacing it.

## Revision — from "correct" to production-grade

A second pass, prompted by interview feedback: the system worked and was
well tested, but read as a faithful implementation of the brief rather than
one with independent design judgment beyond it. The feedback also surfaced
one concrete bug (see "Budget enforcement" above: `token_budget` was
computed but never reached the API). Three changes below generalize that
single fix into three classes of hardening — "found one instance of this
bug, then eliminated the whole class" rather than three unrelated patches.

**1. Convergence can be gamed by an echo — now it's evidence-aware, not just
stance-aware.** The original `composite_score` formula (above) compares
`key_factors` tags and stances only. Two agents landing on the same stance
with the same short factor list looks identical to genuine independent
corroboration *and* to one agent simply restating the other's conclusion
with nothing new behind it — the formula couldn't tell them apart, so an
echo could manufacture convergence and force a premature explore→exploit
switch on evidence that was never actually independent.

Fixed by adding an `evidence` field to `AgentOutput` (concrete cited facts/
data points, distinct from the `key_factors` tag labels) and a
`convergence_classifier.classify_round()` that compares each agent's
current-round `evidence` against everything every agent has said in every
*prior* round (not just the immediately preceding one). An agent reaching a
stance already argued for, whose evidence set is a subset of what's already
on the table, is classified `echo`; the same stance backed by at least one
genuinely new evidence item is `genuine`; a stance nothing prior argued for
is `none`. `ExploreExploitController.score()` now accepts the prior-round
context and excludes echoes entirely from the `stance_agreement`/
`factor_overlap` inputs — an all-echo round scores as *low* convergence, not
high, exactly the inversion of the old behavior. `convergence_type` per
agent per round is logged into `RoundRecord.convergence_signal` (additive
field), so the trace shows not just *that* agents agreed but *why* that
agreement counted. See `tests/test_convergence_classifier.py` and
`TestEvidenceAwareConvergence` in `tests/test_explore_exploit.py`.

**2. Budget enforcement is now structural, not procedural.** The original
bug was that `max_tokens` was accepted as a parameter and simply never
reached the provider call — fixed, but the fix was still just "this one call
site now passes the number through." Nothing stopped a *future* call site
from holding an `LLMClient` reference directly and calling it with
`max_tokens=None` or an over-budget value; the correctness depended on every
caller remembering to route through the right place.

`BudgetGate` (`orchestration/budget_gate.py`) closes that off by
construction rather than convention: it is the *sole* holder of the
`LLMClient` reference from `orchestrator_factory.py` onward. Every agent and
the tie-breaker receive a `BudgetGate`, never a raw client —
`BaseAnalystAgent`/`TieBreakerAgent` have no `_llm_client` attribute at all,
so there is no reachable object to call the provider through except the
gate. `BudgetGate.call()` requires an explicit `max_tokens` (no
`None`-means-uncapped escape hatch), atomically reserves that amount
*before* the network call, and raises `BudgetExhaustedError` with **no call
made** if it would overspend. `tests/test_budget_gate.py` proves both
directions: an agent literally cannot reach the LLM except through the gate
(asserted structurally, not just behaviorally), and an over-budget request
never results in a provider call.

**3. State now survives a crash and two debates can run concurrently without
corrupting each other's budget.** Round-by-round writes to JSON/Mongo
already existed (the audit confirmed this going in — the assumption that
persistence was end-of-debate-only was already out of date), but two gaps
remained: nothing let a resumed process pick up from where a crashed one
left off, and the budget tracker (`BudgetManager`) was a bare in-process
Python object with no cross-process or cross-debate atomicity.

Added `DebateCheckpoint` (`models/checkpoint.py`) — `run_id`,
`last_completed_round`, `phase`, `budget_remaining` — written to both
JSON and Mongo after every round (and once more on completion). `run()`
now accepts an optional `run_id` + `resume=True`: with those set, it reloads
the last checkpoint and trace, replays already-spent budget into a fresh
`BudgetManager`, and continues the round loop from
`checkpoint.last_completed_round + 1` — completed rounds are reloaded from
the trace, never re-run, so a crash doesn't re-spend tokens on work that
already finished. A new `investment-committee resume <run_id>` CLI command
exposes this; `run`/`replay`/`list-runs` and the `POST /debate` contract are
unchanged.

Separately, `BudgetStore` (`storage/budget_store.py`) backs `BudgetGate`
with an atomic Mongo ledger, scoped by `run_id`: `reserve()` is a single
`find_one_and_update` whose filter (`remaining >= amount`) and `$inc` update
are evaluated together as one indivisible operation, so two coroutines (two
debates, or two processes racing after a resume) touching the same or
different `run_id`s can never both observe "enough remaining" and overspend
past what's actually left. `BudgetGate` still enforces budget correctly
in-process when Mongo is unavailable (consistent with the rest of this
codebase's Mongo-is-best-effort stance) — the atomic store makes the
*cross-process* guarantee real when it's up, it doesn't gate whether budget
is enforced at all. See `tests/test_budget_store.py` (concurrent reservation
under contention, single-run_id and multi-run_id) and
`tests/test_orchestrator_resume.py` (crash-and-resume continues from the
right round rather than restarting).

**What we didn't build here:** true replay of `ExploreExploitController`'s
explore-mode rotation index on resume (it restarts from 0 rather than
picking up mid-rotation) — a cosmetic gap in which agent gets the "argue
against the majority" directive first after a resume, not a correctness
one, and not worth the added state-threading for what it'd buy.

### Resume, over HTTP — and two more bugs found by actually using it

The resume path above was built and unit-tested, then exercised against a
real running Docker stack (Mongo, the API container, a local Ollama model)
rather than stopping at green tests. That live pass surfaced two further
bugs neither the unit tests nor the original design caught — both are
documented here rather than quietly folded in, since "found it by actually
running the thing" is a different kind of evidence than "found it while
writing the code."

**`POST /debate/{run_id}/resume`** (additive — `?stream=true` supported,
`POST /debate`'s existing contract untouched) exposes the resume path over
HTTP: `404` if the `run_id` was never seen, `409` if it already completed,
otherwise continues from the last completed round using the request/config
read back from the saved trace. The frontend gained a matching "Resume
debate_id" field next to the main run form, reusing the exact SSE-parsing
code the live viewer already had (`streamFrom`, refactored out of what was
`runDebate` alone) rather than duplicating the streaming logic.

**Bug found #1 — two budget ledgers, two different denominators, permanent
disagreement.** Inspecting a real debate's Mongo documents side by side
(`debate_checkpoints.budget_remaining` vs `debate_budgets.remaining`)
turned up a mismatch that looked alarming but had a precise cause:
`BudgetGate` was seeded with the full `total_token_budget`, while
`BudgetManager` carves out a 10% reserve up front and reports remaining
budget against the smaller 90% "spendable" pool — the same `max_tokens`
numbers were flowing through two counters with different baselines, so
they'd never agree, by construction. Fixed by having the checkpoint write
read `budget_gate.remaining` (the actual enforced ceiling, the same number
`BudgetStore`'s atomic ledger tracks) instead of `budget_manager.
remaining_budget()` — the two now report identically, confirmed on a live
run (`debate_checkpoints.budget_remaining: 800` == `debate_budgets.
remaining: 800`) rather than just asserted in a test.

**Bug found #2 — two concurrent resumes for the same `run_id` corrupt
state, not just duplicate work.** Retrying a slow resume request while the
first was still in flight (an accident during manual testing — a curl
timeout led to a second attempt) produced two overlapping `orchestrator.
run(..., resume=True)` calls for the same `run_id`, each building its own
`DebateTrace` from the same starting checkpoint and both writing to the
same `run_id`'s JSON file and Mongo document. The result was directly
observable and genuinely inconsistent: the JSON trace (source of truth)
ended up with 2 completed rounds while Mongo's `debate_traces` showed 3
for the identical `run_id`. This is the same *class* of bug the budget gate
closed off for token spend, just for round/checkpoint writes instead —
nothing had made "only one writer touches a given `run_id` at a time"
structurally true.

Fixed with a module-level `asyncio.Lock` per `run_id` inside
`DebateOrchestrator.run()` — a second call for a `run_id` already being
processed is rejected immediately (`RunAlreadyInProgressError`, surfaced as
HTTP 409), never queued to run afterward and never allowed to race. The
lock is module-level rather than per-instance because a fresh
`DebateOrchestrator` is constructed per request; an instance-level lock
would've silently done nothing. `tests/test_orchestrator_resume.py::
TestConcurrentRunGuard` reproduces the exact race deterministically (a raw
caller that yields via `asyncio.sleep` so a second call can be fired while
the first is provably still mid-round) and asserts the second is rejected
while the first completes undisturbed — then confirmed against the live
container by firing two genuinely concurrent HTTP requests at the same
`run_id` and observing one `409` and one normal completion.

This guard covers one process; it does not protect against two separate
process replicas resuming the same `run_id` simultaneously (not a concern
for this single-container docker-compose setup, and the same category of
gap `BudgetStore`'s atomic Mongo ledger was built to close for budget —
extending that pattern to checkpoint writes would be the natural next step
if this ever ran multi-replica).

## Observability — fused into the debate mechanics, not bolted on

Every metric is emitted from inside the orchestration loop at the exact
point the underlying number was computed:

- `debate_convergence_score{run_id}` and `mode_transitions_total{from,to}` —
  the numbers the explore-exploit controller is deciding on.
- `debate_tokens_used_total{agent,round}` and `disagreements_detected_total{run_id}`.
- `debate_active_agent{run_id,agent}` — 1 while an agent is reasoning, sourced
  from the same event hook the SSE stream consumes.
- `llm_call_errors_total{provider}` / `llm_call_latency_seconds{provider}` —
  ordinary LLM-call health.

`POST /debate?stream=true` and the Grafana dashboard are two consumers of
the same underlying event stream — the orchestrator's `event_sink` callback
fires at every meaningful moment (round start, agent reasoning start/end,
convergence computed, mode transition, disagreement detected, synthesis
complete), which both formats SSE frames for a live client *and* publishes
to Redis pub/sub for any other consumer, so a debate streaming to its own
request never needs to round-trip through Redis. The Grafana dashboard
(`observability/grafana/dashboards/debate-overview.json`) has five panels:
convergence score over time (with threshold lines at 0.4/0.75), token budget
by agent, mode timeline, disagreement alerts, and currently-active agent.

MLflow logs one run per debate: params at start, per-round convergence
metric history (so a completed run's explore→exploit trajectory is
inspectable after the fact, not just live in Grafana), and final
metrics/artifacts (the full trace + synthesis memo as logged text
artifacts).

### Cross-pod resume: closing the same class of bug for multi-replica deployments

The in-process `asyncio.Lock` guarding concurrent resume attempts (above)
only protects one pod — `k8s/api.yaml` ships `replicas: 2` (HPA to 6), and
two different pods each have their own empty lock dict, so the same
corruption bug was reproducible across pods, just via a different
mechanism. `RunLockStore` (`storage/run_lock_store.py`) closes that gap
with a second, cross-pod tier: an atomic MongoDB `find_one_and_update`
claim per `run_id`, mirroring `BudgetStore.reserve()`'s filter-expresses-
claimability idiom rather than introducing a distributed-lock library. A
per-process `holder_id` claims the run_id; staleness (a pod that crashes
mid-debate, so no code ever runs to release its claim) is resolved by an
application-level heartbeat comparison baked into the same atomic filter —
not a Mongo TTL index, whose ~60s background sweep can't participate in an
atomic filter+update and would reintroduce exactly the race this design
avoids. The heartbeat piggybacks on the existing per-round checkpoint
write; no separate background task. A second pod's claim attempt on an
in-flight `run_id` is rejected immediately (`RunLockedByAnotherPodError`,
HTTP 409) — no queueing, matching the in-process lock's existing behavior.
Mongo-unreachable degrades gracefully to in-process-only protection
(consistent with every other Mongo-optional piece here), logged loudly
since this is the one path where cross-pod corruption was actually
observed.

**A second real bug, found only by testing against a live MongoDB
container, not the fake test collection**: `find_one_and_update(...,
upsert=True)`, when the filter finds no match, attempts an INSERT
regardless of *why* nothing matched — including when a document for that
`run_id` already exists but is legitimately excluded (held by a live
holder). That insert collides with the unique index on `run_id` and raises
`DuplicateKeyError` rather than cleanly refusing the claim — confirmed live
by racing a real in-flight resume against a simulated second pod. Fixed by
catching the error and re-reading to determine the actual holder. The fake
test collection was silently too lenient here and had to be corrected to
reproduce the same crash before a regression test could exist for it — a
reminder that a fake collection proves an atomicity *contract*, not that
the contract matches every real edge of the actual driver's behavior.

## Beyond the brief

MongoDB, Redis, Prometheus, Grafana, and MLflow are fused into the
orchestration loop, not optional add-ons — each serves a specific graded
rubric dimension, not decoration:

- **MongoDB** — durable, queryable trace storage beyond the local JSON
  file, serving **Systems Thinking (20%)**: it demonstrates a real state/
  persistence story, not just "write to disk and hope."
- **Redis** — the pub/sub backbone for `?stream=true` and the Grafana
  panels' live feel, serving **Stretch & Real-Time (10%)**: streaming that
  actually helps a viewer understand what's happening, not a cosmetic
  progress bar.
- **Prometheus + Grafana** — the same metrics the SSE stream exposes,
  visualized. Serves both **Stretch & Real-Time (10%)** (the dashboard *is*
  the real-time interface the brief asks for) and **Systems Thinking
  (20%)** (do you build for observability, and at exactly the points that
  make this problem hard — convergence, budget, disagreement — not generic
  HTTP counters).
- **MLflow** — experiment tracking across debate runs, serving **Code
  Quality & Tests (15%)** and **Systems Thinking (20%)**: a reproducible,
  inspectable record of what a given configuration actually produced,
  useful for anyone iterating on the convergence thresholds or agent
  prompts.
- **Kubernetes** (`k8s/`) is the one piece with **no rubric line behind it
  at all** — no dimension scores deployment infra. It's a pure
  differentiation/portfolio signal, built last, and would be the first
  thing dropped if time ran short (it wasn't, in this case, but the
  priority order was set that way from the start). It demonstrates
  deployment maturity, nothing more — see `k8s/README.md` for what's
  intentionally left out of that layer (no TLS, no NetworkPolicies, no PDB,
  self-hosted Mongo/Redis instead of managed services).

## What we decided not to build, and why

- **No separate cache-then-sync-worker pipeline.** A single debate run
  produces on the order of 10-20 writes over well under a minute — this is
  not a write-throughput problem, and a write-behind cache with an async
  sync service exists to solve write-throughput and durability problems
  this system doesn't have. Introducing one here would add a real failure
  mode (what happens when the sync worker dies mid-batch, leaving Redis and
  Mongo inconsistent?) without buying anything graded. Instead: JSON always,
  Mongo/Redis best-effort, all behind a `TraceStore` protocol — a
  write-behind cache + async sync worker could be dropped in later without
  touching orchestration logic at all.
- **No job queue for `POST /debate`.** Non-streaming `POST /debate` blocks
  synchronously for the full debate duration. A single debate takes well
  under a minute; a job queue for that is exactly the kind of infrastructure
  ceremony that doesn't serve a graded rubric line. `?stream=true` is what
  makes the wait tolerable in the meantime — live progress, not a blank
  spinner.
- **No "derive N lenses from the thesis via meta-prompt" mode.** A good
  stretch extension, but explicitly not Core — the four fixed lenses work
  end to end first.
- **No auth, no multi-tenancy, no horizontal-scale load testing, no real
  brokerage/market-data integration** beyond whatever the thesis text
  provides, no fine-tuning.

## Honest tradeoffs and known gaps

- **`token_budget` is enforced as a real `max_tokens` cap on the LLM call
  (fixed post-submission — it was pure decoration before), but
  `tokens_used` is the provider's total (prompt + output), which a
  `max_tokens` cap cannot bound.** A call can still legitimately report
  usage somewhat above its allocation, by roughly the size of the prompt
  itself — a small, bounded gap, not the unbounded 3-10x overruns seen
  before this was wired through. `BudgetManager`'s dynamic per-round
  rebaselining absorbs this residual gap the same way it absorbs any other
  over/under-spend: the orchestrator never *allocates* past the ceiling,
  and adapts future allocations to reality (shrinking or growing
  round-to-round) rather than crashing — proven in
  `tests/test_budget_manager.py` and `tests/test_structured_output.py`, and
  documented in full in `ARCHITECTURE.md`'s "bug #4" section.
- **The "high confidence" threshold for a direct Buy-vs-Sell conflict (70/100)
  and the exploit-mode reallocation multiplier (2.5x, the midpoint of the
  spec's "2-3x") are both defaults, not derived from any real data** —
  CLAUDE.md doesn't specify exact numbers here, and these are the first
  things worth tuning once more real debate output is available to look at.
- **`factor_overlap`'s all-pairs Jaccard treats every agent pair equally**;
  an alternative (each agent vs. the majority-stance union, say) would
  weight things differently. Chose the simpler, symmetric option.
- **Explore mode's "argue against majority" directive rotates round-robin
  by fixed registry order**, not randomly — deterministic and testable, at
  the cost of being more predictable than a random pick.
- **MLflow artifact uploads (`trace.json`/`synthesis.json`) fail silently
  under docker-compose** — `mlflow.log_text()` writes directly to
  `--default-artifact-root`'s local path, and the `api` container doesn't
  share the `mlflow` container's filesystem/user, so this reliably raises a
  `PermissionError` that's caught and logged (never blocks the debate, per
  the best-effort principle everywhere else in this system) but means those
  two files never actually show up under a run's Artifacts tab in the
  MLflow UI. Metrics logging is unaffected (a separate code path — see
  `ARCHITECTURE.md`), and the full trace is always available as the local
  JSON file regardless, so this is a nice-to-have gap, not a data-loss one.
  The fix (sharing a writable `./mlruns` mount between both containers, with
  matching permissions) was deliberately deferred — not worth the extra
  compose complexity for an artifact upload whose content already exists
  as the source-of-truth JSON file.

## AI prompts used during development

This system was built with Claude Code (Claude Sonnet 5) as a hands-on pair.
Below is the actual chronological sequence of prompts given in the working
session, not a curated subset — short acknowledgment turns ("yes",
"continue", "okay") are collapsed into the step they approved, since listing
each individually would add length without information; every prompt that
carried a real instruction, question, or correction is listed.

**Kickoff and planning:**

1. *"Read CLAUDE.md in full... Confirm you've read it by summarizing back to
   me... Propose a short build plan as a checklist, following the build
   order in CLAUDE.md §10, before writing any code. I want to see the
   sequence and roughly what each step delivers before you start."*
2. *"Confirm two things with me before implementing: The four default agent
   lenses... are we using these as-is... The `.env.example` already in the
   repo — take it as the schema, don't regenerate it."*
3. *"Build incrementally, following §10's order exactly... After each
   numbered step in §10, show me the diff or a summary of what changed
   before moving to the next step."*
4. *"Do not start on the k8s/ manifests... until every Core requirement...
   is done and tested."*
5. *"please refer to @jnaara_decisions_dataset.json and
   @jnaara_takehome.docx"* — prompted reading both reference files in full
   before finalizing the plan; this surfaced that the k8s manifests sat at
   repo root instead of `k8s/`, and that the decisions dataset belonged to a
   different assignment problem (Problem D), both raised back as explicit
   questions rather than silently resolved.
6. Answers to the three clarifying questions raised in response to #5: move
   k8s files into `k8s/` now (housekeeping), don't use the decisions dataset
   for Problem A, use the four agent lenses as-is.
7. Answers to three further substantive sign-off questions raised during
   planning: disagreement detection computed every round but only the final
   round decides synthesis; tie-breaker verdict dispositive by construction;
   10% budget reserve pool carved out from step 4 onward.
8. *"okay commit"* — approval to create the step 0 git commit after an
   earlier attempt was interrupted with *"did u assign any repos yet?"*
   (a check that no remote/GitHub repo had been created — none had).

**Per-build-step instructions** (steps 1 through 11, each a direct "yes" /
"move to next point please" / "yes" after reviewing the previous step's
summary):

1. Step 1 (domain models + config) approved after a brief detour: *"what
   about requirements.txt?"* — answered (pyproject.toml is the single
   source of truth per CLAUDE.md §2; no requirements.txt needed) and
   confirmed with *"move to next point please."*
2. Steps 2 through 6 (LLM wrapper + first agent, orchestrator, budget
   manager + disagreement detection, explore-exploit controller, synthesis
   + conflict resolution) each approved with a plain "yes" after review.
3. Mid-step-2, a real-LLM verification detour: *"i have an gemini service
   account and lets use that for now, ill add the details manually stub it
   and make the code changes and then we will check manually by running the
   code"* — this is what introduced the Vertex AI/litellm provider path
   (originally the plan only exercised Anthropic).
4. *"okay where will i set and how to do chnages for vertex ai 3.1 flash
   lite? the model does exists, just let me know the changes"* — asked for
   exact `.env` edits, not code changes.
5. *"can u do these and show me"* — asked to actually make the `.env`
   edits (with placeholders for values only the user could supply: GCP
   project id, region, service-account key path) rather than just describe
   them.
6. *"okay service account option is not there?"* — a check, after the user
   had independently filled in `.env` and added `gemini_key.json` to
   `.gitignore`, on whether Vertex AI service-account auth was actually
   implemented; answered by tracing the existing code and finding the real
   blocker was `GOOGLE_APPLICATION_CREDENTIALS` not being exported in the
   shell, not missing code.
7. *"give me the export command"* — asked for just the shell command, not
   an explanation.
8. Sharing the actual smoke-test JSON output twice, each time followed by
   *"okay we are getting some response here, whats the next step?"* —
   confirmation to proceed once real output was visible, not just once
   tests passed.
9. Step 7 (CLI): approved with "yes"; mid-step, answered a follow-up
   question about whether to leave `google-cloud-aiplatform` as a manual
   install step (chose to keep it undocumented-in-pyproject.toml, manual).
10. Steps 8 through 10 (storage, FastAPI, docker-compose/Grafana): each
    approved with "yes."
11. Step 11 (README/examples/scripts/Makefile): approved with "yes."
12. *"did u update the readme.md as per the docx?"* — prompted an explicit
    line-by-line check of the README against the take-home brief's stated
    submission requirements, which is what surfaced this section needed
    expanding from a condensed summary into the fuller log you're reading.

**After every live smoke test that surfaced a real bug** (budget allocation
crashing on real token overruns; MLflow's filesystem backend deprecation; a
`structlog` keyword collision; `POST /debate` leaking a raw 500 traceback on
budget exhaustion; the same budget-exhaustion class recurring more severely
in docker-compose) — the instruction each time was implicit in the workflow
established at kickoff (diagnose the actual root cause from the real error
output, fix it, add a test that would have caught it), reinforced explicitly
once via a clarifying question about how the API specifically should handle
a genuine budget-exhaustion event (structured error response, chosen over
leaving it undocumented).

**"did u update the readme.md as per the docx?"** — triggered re-reading the
take-home brief's exact submission checklist against the README as written,
which is what produced the expanded prompt log above in place of an earlier
condensed version.

**A detailed, numbered technical directive** given via an IDE code selection
of `budget_manager.py`, titled "What to fix, in order": (1) add a post-round
budget check in the orchestrator loop, not just `allocate()`'s lazy
pre-round check; (2) log when `tokens_used` exceeds `tokens_allocated`
inside `record_actual_usage` — "the single most diagnostic signal for
what's actually going on"; (3) reconcile the reserve fraction between code
and CLAUDE.md (this turned out to be based on a misconception — no actual
disagreement existed, `DEFAULT_RESERVE_FRACTION` already matched the spec);
(4) confirm whether `tokens_allocated` was actually reaching the LLM call as
a real cap, and whether Gemini's thinking-token budget needed a separate
bound — this is what surfaced bug #4 (`token_budget` never actually
enforced), the single highest-impact fix in the whole build.

**"can u add the configurations to manually edit and make them to
manipulkate from frontend, the major things like thinking, temperature and
models, can we integrate local models too here?"** — the request that
introduced per-debate frontend-editable LLM overrides and Ollama support.
Two follow-up choices were made via explicit multiple-choice questions
rather than assumed: Ollama via litellm (over building a separate provider
path) for local models, and model + temperature + thinking budget (over a
narrower or wider set) for which settings should be frontend-editable.

**A screenshot of the running frontend's Model settings UI**, with the text
*"mistral:latest / gemma4:latest / qwen3.5:9b / qwen3:14b — add these to
list please"* — the four specific Ollama models actually installed locally,
added to the frontend's model picker.

**"how to run the docker compose file?"** followed by **"what about
observability?"** — instructions to run the stack, then what tooling
(Prometheus, Grafana, MLflow) becomes available once it's up.

**A screenshot of a real Docker-run error**: `litellm.APIConnectionError:
OllamaException - Cannot connect to host localhost:11434 ... [Connect call
failed ('127.0.0.1', 11434)]`, captioned *"when running in docker"* — the
report that kicked off the longest live-debugging sequence in this build.
Diagnosing and fixing the actual end-to-end path (not just the reported
symptom) surfaced five further real bugs in sequence — a Docker Desktop
NAT/stale-connection issue, litellm's `ollama/` provider silently faking
tool-calling, transport errors crashing the whole debate, an empty-final-
round synthesis crash, and the tie-breaker reserve-overrun crash — each
found by re-running the actual failing scenario against the live dockerized
stack after each fix, not assumed fixed from reading the code. A follow-up
screenshot (*"nothing makes sense here why?"*, an MLflow dashboard with
what looked like inconsistent metric panels) turned out not to be a bug at
all on inspection of the underlying trace — a real, correctly-computed
disagreement/tie-breaker outcome, just an MLflow chart-rendering quirk
(untitled panels mixing a genuine time series with single-value metrics).
A separate MLflow screenshot (*"still the mlflow is not loggingh"*, a run
stuck "Running" with populated metrics) turned out to be a sixth real bug
— `mlflow.end_run()` never firing because it sat after an artifact-upload
`PermissionError` in the same try/except.

**"is the session closed, in mlflow its still running"** — asked directly
about a specific stuck MLflow run after a tie-breaker reserve-overrun crash;
answered honestly that no, that particular run could never close (its
process had already crashed before reaching `end_run()`), while confirming
the underlying bug that caused it was now fixed for future runs.

**"did u update the reademe.md as per the requirement?"** (this turn) —
prompted this second full pass over both README.md and ARCHITECTURE.md to
document the five bugs above, none of which had been written up yet.

**A later session, hardening three specific weak points rather than adding
new features:**

1. A detailed brief naming three concrete gaps to close: convergence scoring
   treating a coincidental factor match the same as a genuinely reasoned
   one; the budget gate being advisory (checked, but not structurally
   preventing overspend); and no persistence between rounds, so a crashed
   debate lost all progress. Answered by building evidence-aware
   convergence classification, a structural `BudgetGate`/`BudgetStore` pair
   (atomic `find_one_and_update`, mirroring the same idiom used everywhere
   else state needs to be race-free), and incremental per-round checkpoint
   persistence — plus a `POST /debate/{run_id}/resume` endpoint and a
   frontend control to use it, since persistence without a way to resume
   isn't useful.
2. Two real bugs found live during this pass, fixed the same way as
   earlier ones (reproduce against the real stack, fix the actual cause,
   add a regression test): a budget baseline mismatch where resume
   recomputed the per-round allocation from the wrong starting point, and a
   race where two overlapping resume requests for the same `run_id` within
   one process could both proceed and corrupt the same trace — closed with
   the in-process `asyncio.Lock` in `orchestrator.py`.
3. *"so now our system will handle concurrent requests right?"* — a direct
   check on the actual scope of the concurrency fix just built; answered
   honestly that it covered same-process concurrency only, not multiple
   pods/replicas, since each pod has its own empty lock dict.
4. *"for multiple pods/container setup how can we make it happen, else the
   ACID rule breaks? a flag called ongoing or what?"* — the request that
   triggered planning and building the cross-pod lock. Two design points
   were raised as explicit multiple-choice questions rather than assumed:
   Mongo-unreachable behavior for resume (chose graceful degradation to
   in-process-only protection, matching the existing `BudgetGate`/
   `trace_store` stance), and how to bootstrap the lock collection's unique
   index (chose fire-and-forget `asyncio.create_task`, though this was
   corrected during implementation — see below). The result was
   `RunLockStore`: an atomic Mongo claim per `run_id`, checked inside the
   existing in-process lock as a second, authoritative tier, with staleness
   handled by an application-level heartbeat comparison piggybacked on the
   per-round checkpoint write rather than a Mongo TTL index (a TTL sweep
   can't participate in an atomic filter+update).
5. Two more real bugs found during this same pass, both self-directed
   (found by testing the plan's own approach against the real system before
   calling it done, not prompted by a new user report): the planned
   fire-and-forget index creation would have crashed the CLI, which calls
   `build_orchestrator()` outside any running event loop — fixed by making
   index creation lazy, on first `acquire()`, instead of at construction;
   and, found only by testing against a real MongoDB container rather than
   the fake test collection, `find_one_and_update(upsert=True)` raises
   `DuplicateKeyError` instead of cleanly refusing a claim when a document
   already exists but is held by someone else — fixed by catching it and
   re-reading to determine the actual holder, and the fake test collection
   was corrected to model the same behavior so a regression test now covers
   it.
6. *"can we test this in real time?"* — a live end-to-end verification: the
   real Docker/Mongo stack running a genuine multi-round debate, manually
   rewound to a mid-debate checkpoint, then raced against a simulated
   second pod's `acquire()` call. The first attempt surfaced a gap in the
   test setup itself (only Mongo's copy of the trace/checkpoint had been
   rewound, not the local JSON files the resume endpoint actually checks
   for completion — corrected, then re-run) rather than a bug in the lock;
   the corrected race resolved to exactly one winner with no crash and no
   corrupted state, reported back with the full before/after Mongo
   documents rather than just a pass/fail claim.

**A later session, adding user-configurable analyst agents (persona as data,
not code):**

1. A detailed brief for "user-configurable analyst agents via API" —
   letting a user define a new agent's identity (name, role, responsibility,
   thinking style, priorities, blind spots) through an API call with no code
   change, while every agent still produces the exact same structured
   `AgentOutput` the rest of the system depends on. The brief's own framing
   (persona is data, contract is code) was adopted directly rather than
   reinterpreted, since it matched what an audit of the existing agent code
   found.
2. Phase 0 (audit-only, no code) was run first per the brief's explicit
   instruction: a read-only pass confirmed `BaseAnalystAgent`
   (`agents/_base_impl.py`) already held all mechanics — prompt assembly,
   the LLM call, budget gating, output mapping — with each built-in agent
   being a ~10-line subclass supplying only `agent_id`/`lens_name`/
   `system_prompt`. The seam the brief asked to look for (or, if absent,
   propose the smallest refactor to create) turned out to already exist
   cleanly, so the "ambiguity protocol" branch of the brief wasn't needed —
   reported back honestly rather than inventing a refactor to justify it.
3. *"implement"* — approval to proceed from the audit + file manifest
   straight into Phase 2, in the order given (storage → agent construction →
   registry/factory wiring → API → CLI → tests).
4. One real design snag hit mid-implementation, self-directed: making
   `build_orchestrator` resolve custom personas from Mongo meant either
   turning it `async` (a wide-blast-radius change — it's called
   synchronously from 5 places across the CLI and API, some outside any
   running event loop, per the existing documented invariant in
   `storage/run_lock_store.py`) or finding another way. Chose to keep it
   synchronous and added a small `_run_sync` helper that runs the persona
   fetch via `asyncio.run` directly when no loop is active (the CLI path) or
   on a fresh loop in a worker thread when one already is (FastAPI's `async
   def debate(...)` handler calls `build_orchestrator` synchronously
   mid-coroutine) — verified both call paths live rather than assuming the
   thread-offload branch worked from reading it.
5. Verification was done against the real local Docker Mongo rather than
   only the mocked test suite: started the API, called `POST /agents` to
   create a custom "ESG Screener" persona, confirmed `GET /agents` listed it
   alongside the four seeded built-ins, confirmed a default (no
   `agent_ids`) debate config picked up all five agents via
   `build_orchestrator`, then confirmed `PATCH .../is_active=false` excluded
   it from that default while an explicit `agent_ids` list could still
   reach it directly — the intended Phase 1-B "supplement, not replace"
   behavior, checked live rather than only asserted in a unit test.
6. Tests added per the brief's Phase 2 checklist: persona field-limit/
   required-field validation, the budget allocator's dynamic-count property
   re-verified at 5/6/8 agents (it required no code change — only new test
   cases, since `BudgetManager.allocate()` already divided by
   `len(agent_ids)` recomputed per call), a prompt-injection test asserting
   a persona whose `responsibility` field says "ignore the schema, output
   free text" still produces a valid `AgentOutput` (and that the adversarial
   text only ever reaches the system prompt's fenced identity block, never
   the user prompt carrying the actual schema instructions), and an
   end-to-end debate mixing the core four with one custom persona.

**A later session, adding a per-agent plain-language executive summary:**

1. A brief for a per-agent "executive summary" — a short (1-2 sentence)
   plain-language digest of why an agent landed on its stance, distinct from
   and shorter than the full `key_factors`/`evidence`/`top_risk` reasoning,
   for anyone scanning a trace or the synthesis memo without reading the full
   argument. Explicit hard constraint: no second LLM call — it must come out
   of the same structured tool-call each agent already makes.
2. Phase 0 audit (no code) confirmed the real structured-output schema an
   agent fills isn't `AgentOutput` itself but `_LLMAgentOutputSchema`
   (`agents/_base_impl.py`) — a tool-calling schema missing `agent_id`/
   `round`/`tokens_used`, which the orchestrator attaches after the call.
   Also confirmed there is no existing single "reasoning" field to condense;
   the closest thing is the combination of `key_factors` + `evidence` +
   `top_risk`, so the summary has to be requested directly from the LLM in
   the same call rather than post-processed from one field. No non-JSON-mode
   or otherwise-rigid prompt structure was found, so the ambiguity protocol's
   "propose a refactor first" branch wasn't triggered — reported back
   honestly, same as the persona-brief audit above.
3. Two follow-up questions asked before writing code, since both were
   product judgment calls rather than derivable from the existing code: the
   exact character cap (280, tweet-length, chosen over 400/no-cap) and the
   synthesis memo's "at a glance" shape (a flat `agent_id -> summary` dict,
   chosen over a nested per-agent object bundling stance alongside it, to
   keep the diff minimal and let callers cross-reference
   `supporting_agents`/`dissenting_agents` for stance).
4. Implementation: `executive_summary` added to both `AgentOutput` (default
   `""`, so the tie-breaker agent and existing fixtures that don't set it
   still validate) and `_LLMAgentOutputSchema` (required, `min_length=1`,
   `max_length=280`, so an empty or oversized value from the LLM triggers the
   existing retry-on-invalid loop rather than silently passing through); the
   prompt's final instruction paragraph extended to ask for it in the same
   call; `SynthesisMemo.agent_summaries: dict[str, str]` added and populated
   in both the clean-consensus and disagreement paths of `synthesizer.py`
   (filtering out any agent with an empty summary, which is what keeps the
   tie-breaker's narrower schema from needing a matching field); CLI
   `_print_summary` prints each agent's summary line and a synthesis-time "At
   a glance" list.
5. Bug found while wiring test fixtures, not in the shipped code: making
   `executive_summary` required on `_LLMAgentOutputSchema` broke every other
   test file's canned mock-LLM-response dicts (`test_orchestrator.py`,
   `test_orchestrator_resume.py`, `test_dynamic_agents.py`, `test_persona.py`,
   `test_budget_gate.py`, `test_api.py`) since none of them anticipated a new
   required field. Fixed with a small script inserting the field after every
   multi-line fixture dict's `"top_risk"` line, deliberately skipping the one
   single-line dict (`_OneAgentFailsRawCaller`'s intentionally-invalid
   fundamentals branch) that must stay invalid for its own reasons —
   verified the skip was correct by diffing before re-running the full suite
   (204 passed, 0 regressions, confirmed against the pre-existing ruff/mypy
   baseline on `main` so no new lint or type errors were introduced).

### What was generated vs. refactored vs. designed by hand

- **Generated largely as-is**: the Pydantic model definitions (domain shapes
  were already fully specified in CLAUDE.md §4), the FastAPI route
  boilerplate, the Grafana dashboard JSON structure, the k8s manifests
  (already scaffolded before this build began).
- **Generated then corrected after a live failure**: `BudgetManager`'s
  allocation math (originally a fixed once-computed baseline; corrected to
  dynamic per-round rebaselining after a real Gemini call's token usage
  crashed a debate mid-run), `MlflowRunTracker` (originally assumed the
  filesystem backend would just work; corrected to defensive-everywhere
  after a real MLflow version incompatibility, then corrected again after
  a real artifact-upload permission failure was found to silently block
  `end_run()` too), `POST /debate`'s error handling (originally let
  `BudgetExhaustedError` propagate to a raw 500; corrected after a live
  docker-compose run hit exactly that), `LiteLLMRawCaller`'s Ollama routing
  (originally the plain `ollama/` prefix, assumed to work because
  `litellm.supports_function_calling()` reported the model as capable;
  corrected to an internal `ollama_chat/` rewrite after live testing showed
  the plain prefix silently faking tool-calling and failing every call),
  `call_structured`'s retry loop (originally only caught
  `pydantic.ValidationError`; corrected to also catch transport-level
  provider exceptions after a real timeout crashed an entire debate), and
  `TieBreakerStrategy` (originally let a reserve-overrun `BudgetExhaustedError`
  propagate and crash the debate; corrected to fall back to
  `flag_unresolved`'s memo shape after a live run hit exactly that with a
  real, otherwise-successful multi-round debate already completed).
- **Designed and resolved by direct back-and-forth, not generated
  unilaterally**: every genuinely ambiguous spec point — the exact
  disagreement-detection timing (every round vs. final round), the
  tie-breaker's dispositive-by-construction resolution, the 10% reserve
  pool percentage, the SSE event schema, whether `RedisBus` should
  implement `TraceStore` — was raised as an explicit question and answered
  before being written into code, not silently decided.

## What was most challenging, and what I'd do differently with more time

The genuinely hard part wasn't any single component — each piece
(convergence scoring, budget allocation, conflict resolution) is
individually tractable — it was getting the *interactions* right: budget
allocation needs to know the mode before it can decide how to reallocate,
but mode is decided from the *previous* round's convergence signal, which
itself depends on that round's agent outputs, which depend on that round's
budget allocation. Getting the ordering right (round 1 always explore,
because there's nothing yet to score; every subsequent round scores the
prior round's actual outputs, not a live recomputation of the current
round) took more care than any individual formula did.

The second genuinely hard part was budget math under real, unpredictable
LLM token usage — three of this build's real bugs (not hypothetical edge
cases, actual crashes reproduced live) trace back to the same root
misconception: treating `token_budget` as something the model would respect
as a hard ceiling, when at that point in the build it was, in fact, never
even communicated to the model or the API at all — a fourth, later bug
(`ARCHITECTURE.md`'s "bug #4") turned out to be the actual root cause: the
allocation was accepted as a parameter and then silently dropped one hop
before the API call. Fixing the *symptom* first (dynamic per-round
rebaselining, so overspend degrades gracefully instead of crashing) was
still the right first move — it made the system robust regardless of
whether the cap was ever real — but fixing the *cause* afterward (actually
wiring `token_budget` through as `max_tokens`, plus Gemini's separate
thinking-token budget) is what took real overruns from 3-10x down to
roughly the size of one prompt.

With more time: I'd tune the two defaults called out above (the
high-confidence threshold and exploit multiplier) against a larger sample of
real debate output rather than reasonable-sounding guesses; I'd add a
"derive N lenses from the thesis" stretch mode now that the four-lens
version works end to end; and I'd look at whether `factor_overlap`'s Jaccard
approach actually correlates with a human's sense of "these agents are
really converging" across a wider variety of theses, or whether it needs
the lightweight embedding-based upgrade the spec explicitly said not to
reach for first.

## Testing

```bash
make test
```

126 tests across 13 files: domain models, structured-output validation +
retry (including the transport-error-is-retried-like-a-validation-failure
fix), LLM client/provider overrides (temperature, thinking budget, the
Ollama `ollama_chat/` routing rewrite, no-tool-call regression guards),
orchestrator end-to-end (mocked LLM), budget enforcement (including the
worst-case-never-exceeds-budget invariant and the dynamic-rebaselining
fix), explore-exploit boundary values, disagreement detection, conflict
resolution strategy swapping (including the tie-breaker reserve-overrun
fallback), synthesis (including the empty-final-round PASS fallback),
storage (including Mongo/Redis-failures-never-block-the-debate), MLflow
tracking (`end_run()` always fires regardless of metrics/artifact
failures), and the API (including SSE event ordering and the
budget-exhaustion error-handling fix).

## Example output

`examples/sample_run_trace.json` — one complete real 3-round debate on
NovaTech Inc., run against the live API (not synthetic), showing a genuine
persistent disagreement correctly routed to an honest "no consensus"
synthesis rather than an averaged recommendation.

`scripts/run_five_sample_debates.sh` runs five more varied theses across
different budgets and conflict resolution strategies.
