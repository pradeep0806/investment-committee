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

## Post-interview changes

Everything below was added after the interview submission (commit `4462b22`,
"Harden debate system: evidence-aware convergence, structural budget gate,
resumable persistence") — kept as a separate, explicit index so a reviewer
can tell submitted work apart from follow-on work at a glance, without
diffing commit hashes. Each entry is also logged verbatim, prompt-by-prompt,
in [`PROMPTS.md`](PROMPTS.md); this section is the scannable summary, that
file is the full record. Update both every session a change lands here
(CLAUDE.md §13).

- **`b316a05`** — Resume API endpoint (`POST /debate/{run_id}/resume`,
  matching the CLI's existing `resume` command), plus a fix for a budget
  baseline mismatch and a race between two concurrent resume attempts on the
  same run.
- **`a11460b`** — Cross-pod mutual exclusion for debate resume: a
  Mongo-backed `RunLockStore` so two API pods can't both resume the same
  `run_id` at once in a multi-replica deployment.
- **`fd03fa4`** — User-configurable analyst agents ("persona as data, contract
  as code"): `POST/GET/PATCH /agents` lets a caller define a new agent's
  identity (name, role, responsibility, thinking style, priorities, blind
  spots) with no code change, while every agent — built-in or custom — still
  returns the exact same structured `AgentOutput` schema. `DebateConfig.agent_ids`
  lets a debate pick a specific subset (built-in and/or custom) instead of
  always running the default four.
- **`db1db7c`** — Per-agent `executive_summary`: a short (≤280 char)
  plain-language digest of *why* an agent landed on its stance, produced in
  the same structured tool-call as everything else (no second LLM round
  trip), surfaced in the trace JSON and CLI output.
- **`69bf577`** — Structured `dissenting_view` on the synthesis memo: every
  final-round agent whose stance differs from the committee's recommendation
  is named individually (agent, stance, reason — reusing `executive_summary`
  or falling back to `top_risk`), with an explicit `dissenting_view_note`
  stating "no dissent" when the committee fully agreed, rather than that
  absence being implicit. Also fixed a real bug found while building this:
  the clean-consensus path's `dissenting_agents` was hardcoded to `[]`, which
  silently dropped a minority agent's disagreement on a plurality (not
  unanimous) majority decision.
- **Frontend viewer wiring** (same session as `69bf577`, folded into that
  commit) — the existing `frontend/` Vite/React debate viewer extended to
  render `executive_summary` per agent (live, via a new field on the
  `agent_reasoning_end` SSE event) and the synthesis memo's `dissenting_view`/
  `agent_summaries`.
- **`6f57cef`** — Primary/fallback LLM provider on transient failure:
  `LLMClient` now retries the primary provider with exponential backoff on a
  transient (429/503) error before falling through to a configured fallback
  provider, entirely inside the LLM client layer; `BudgetGate`'s
  reservation/deduction is unchanged structurally (no second, looser path to
  the LLM), token accounting across providers is a documented approximation
  (not normalized), and every `AgentOutput`/`BudgetLedgerEntry` now records
  `provider_used` so a reviewer can see exactly when a fallback fired.
- **Depth pass on convergence detection and the budget gate** (uncommitted
  as of this note) — an explicit interview follow-up asking for depth on a
  couple of existing safeguards rather than new breadth. Echo-vs-genuine
  convergence classification proven against two named adversarial fixtures
  (a "lazy echo" that trivially rewords already-stated evidence, and a
  "genuinely persuaded" agent citing independent, non-overlapping evidence
  for the same conclusion) and given a real downstream consequence: an
  echoed argument is now excluded from the synthesis majority vote,
  `supporting_agents`, and the average confidence, rather than only being
  logged — proven by a test where excluding the echo *changes the
  recommendation itself*. `SynthesisMemo.echoed_agents` keeps the echoed
  agent visible rather than silently dropped. Separately, the budget
  invariant — cumulative spend never exceeds `total_token_budget` — is now
  proven with property-based tests (Hypothesis, a new dev dependency) across
  randomized agents/rounds/per-call requests and simulated failures, which
  surfaced two real, previously-untested behaviors worth documenting rather
  than "bugs": (1) a single call's actual usage can legitimately exceed its
  own reservation by an unbounded amount when that reservation is below
  `MIN_MAX_TOKENS` (256), since `call_structured` clamps any smaller request
  up to that floor before it ever reaches the provider; (2) `BudgetGate`'s
  real, provable guarantee is that it never *admits* a call whose request
  exceeds what's remaining at that instant — not a hard cap on cumulative
  actual usage regardless of per-call overrun, which is a stronger claim the
  gate was never built to make. Finally, a static AST scan
  (`scripts/check_budget_gate_bypass.py`) now fails the build if any code
  path calls `LLMClient.call()` outside `BudgetGate`, wired into a new
  GitHub Actions CI workflow (none existed before this) alongside the full
  `pytest` suite.
- **Closed the provider-dependent token cap enforcement gap** — a real live
  debate against local Ollama (`qwen3.5:9b`) surfaced `tokens_used=7044`
  against `tokens_allocated=1853` (a 3.8x overrun). Audited before writing
  any code: per-call `max_tokens`/`num_predict` enforcement was verified
  correct with real, deliberately-tight-budget calls against *both*
  Ollama and Vertex AI Gemini directly (never a per-call cap problem); the
  actual root cause was `call_structured`'s retry loop re-issuing the same
  `max_tokens` on every attempt with no ceiling on cumulative spend across
  attempts, which a weaker model needing several retries exposed far more
  than Gemini ever did. Fixed with a `RETRY_BUDGET_MULTIPLIER` cap on total
  spend across all attempts combined (each retry's own cap shrinks by what
  was already spent; the loop stops early rather than force one more
  doomed attempt), re-verified live against the exact original scenario
  (2289 tokens_used post-fix, down from 7044, under the 3706 ceiling).
  Along the way, fixed two related accounting bugs in `BudgetGate`: a
  failed call previously credited back its *full* reservation even though
  real tokens were spent across failed retries (`LLMValidationError` now
  carries `total_tokens_used`), and any overrun beyond a call's own
  reservation was never debited from `remaining` at all, silently
  overstating what was actually left.
- **Followed by a real-time UI report of empty round-3 agent cards**, traced
  to a second, related gap in `BudgetManager.allocate()`: its only floor
  was 1 token/agent — far below what any real model needs, and completely
  disconnected from the retry-budget fix above. By round 3 of the same
  kind of debate, exploit-mode's 2.5x reallocation toward the contested
  agent drove non-contested agents down to 723 tokens each — above 1, so
  `allocate()` never refused, but not enough for `qwen3.5:9b` to reliably
  produce a valid tool call, so those agents were excluded every remaining
  round (empty cards in the live viewer, not a UI bug). Fixed with
  `min_viable_allocation` on `BudgetManager` — every agent (contested or
  not) is now guaranteed a real floor per round, funded by shrinking the
  exploit-mode multiplier's boost if needed, or by `allocate()` raising
  `BudgetExhaustedError` cleanly (stopping the debate with fewer, valid
  rounds) if remaining budget genuinely can't cover it for everyone. This
  needed a second calibration pass, not a first-try number: an initial
  `MIN_MAX_TOKENS * RETRY_BUDGET_MULTIPLIER` (512) formula was still below
  the 723 that actually failed live; a hardcoded empirical floor (2048,
  matching what `qwen3.5:9b` needed in practice) then broke 33 existing
  tests built around what a *capable* hosted model needs (well under 1000
  tokens for this schema) — one constant can't be right for every
  provider. Landed on a small, safe default (256, matching
  `structured_output.py`'s own `MIN_MAX_TOKENS`) with an explicit
  per-debate override (`DebateConfig.min_viable_allocation_per_agent` /
  `Settings.min_viable_allocation_per_agent`, same None-means-server-default
  pattern as the LLM provider/model overrides) — raise it explicitly when
  running against a model observed to need more headroom, rather than
  baking one model's requirement into every debate.
- **Truncate a lone over-length free-text field instead of excluding the
  whole agent over it** — a live debate against Gemini showed a fully
  valid `risk_contrarian` response (correct stance, confidence,
  key_factors, top_risk) losing its entire output because
  `executive_summary` alone ran a few characters past its 280-char cap on
  every retry attempt. `call_structured` now recognizes the narrow case
  where every validation error is `string_too_long` on a top-level string
  field, truncates just that field to its `max_length`, and re-validates
  — skipped entirely if any other error is present, so a response that's
  wrong in some other way still goes through the normal retry path
  unchanged. See "Honest tradeoffs and known gaps" for the full writeup.

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

### Proving the budget invariant, not just asserting it

An interview follow-up asked for depth on the existing safeguards rather
than new breadth: is "total spend never exceeds `total_token_budget`" true
by inspection of a handful of example tests, or true across the space of
what could actually happen? `tests/test_budget_invariant.py` answers this
with property-based tests (`hypothesis`, a new dev dependency) generating
randomized numbers of agents/rounds, per-call `max_tokens` requests, and
simulated call failures, then asserting the invariant holds regardless.

Two runs of Hypothesis against an initial, more naively-stated version of
this invariant found real, previously-untested behavior — not bugs to fix,
but gaps in what the test suite (and, honestly, this README) had claimed
`BudgetGate` guarantees:

1. **The 256-token floor is a second, wider overrun channel than the
   already-documented prompt-size one.** Any `max_tokens` `BudgetGate`
   reserves below `MIN_MAX_TOKENS` (256) is clamped *upward* to 256 by
   `call_structured` before it's ever told to the provider — so a call
   reserving as little as 1 token still asks the provider for up to 256,
   and real usage can land anywhere in that range. This is distinct from
   the "prompt-size" overrun described above (which is bounded and small);
   this one is bounded only by the 256-token floor itself, and it's
   unavoidable for any allocation below that floor — not a residual bug,
   but worth stating plainly rather than letting the existing "small,
   bounded overrun" framing quietly cover a case it doesn't actually
   describe.
2. **The actual, provable guarantee is admission control, not a hard cap on
   cumulative usage.** `BudgetGate` never *admits* a call whose requested
   `max_tokens` exceeds what's remaining at that instant — that's the real
   invariant, and it's the one the property tests prove directly, call by
   call. "Cumulative actual tokens_used never exceeds the budget" is true
   *whenever every call's usage stays within its own reservation*, which is
   the common case — but it's a corollary of admission control plus the
   documented overrun mechanisms, not a separate, stronger promise the gate
   makes on its own. Stating the weaker, actually-true invariant is more
   honest than a stronger-sounding one that a single adversarial example
   (a 1-token budget, one overrun) can falsify.

The refund-on-failure path (a reservation is released in full when the
underlying call raises before reporting any usage) is proven separately and
explicitly, since the task called it out as its own case worth isolating.

### Down-weighting echoed convergence, not just detecting it

The same follow-up asked for the convergence classifier to be proven
against adversarial cases and given a real consequence, not just a log
entry. `tests/test_convergence_classifier.py` now includes two named,
paired fixtures run through the same round together — a "lazy echo" agent
that restates a prior argument's evidence with only trivial rewording
(casing/whitespace, the exact normalization `classify_round` already
applies) and a "genuinely persuaded" agent that reaches the same conclusion
via independent, non-overlapping evidence — asserting the classifier
correctly flags the first as `ECHO` and the second as `GENUINE`. One honest
limitation the fixture-building surfaced: the classifier compares
*normalized literal text*, not meaning, so a paraphrase using different
wording for the same fact currently isn't caught — semantic echo detection
via embedding similarity is exactly the Stretch item that would close this,
left as such rather than silently expanding Core's scope.

That classification now has a real downstream consequence in synthesis
(`synthesis/synthesizer.py`): an agent whose final-round argument is
classified `ECHO` is excluded from the majority-stance vote,
`supporting_agents`, and the average confidence on the clean-consensus
path — down-weighted, not silently counted as an equal vote alongside a
genuinely independent argument. `tests/test_synthesizer.py` proves this
isn't cosmetic with a case where excluding the echo *flips the
recommendation itself* (a 2-1 Buy majority becomes a 1-1 tie broken toward
Sell once the echo is correctly excluded). The echoed agent is still named
in a new `SynthesisMemo.echoed_agents` field rather than erased from the
record entirely, and the whole feature is additive: `synthesize()`'s new
`convergence_types` parameter defaults to `None`/unfiltered, so every
existing call site keeps its prior behavior unless it explicitly opts in
by passing the final round's classification through (which
`orchestrator.py` now does, since it already computed it).

### Bypassing the budget gate is now a build-time failure

The same follow-up's fourth ask: make "no code path can reach the LLM
without going through `BudgetGate`" a property of the codebase, not just an
observation that happens to be true of the code paths that exist today.
`scripts/check_budget_gate_bypass.py` is a plain AST scan (no type
inference) over every file in `src/committee` that fails if it finds either
a `.call(...)` invoked on anything whose receiver expression looks like an
`LLMClient` outside `budget_gate.py`/`llm/client.py` themselves, or a direct
`LLMClient(...)` construction outside the one legitimate construction site
(`orchestrator_factory.py`, which immediately hands the client to
`BudgetGate` and never calls `.call()` on it directly). It's wired into a
new GitHub Actions workflow (`.github/workflows/ci.yml` — no CI existed
before this) as its own named step, and also runs as an ordinary pytest
test (`tests/test_no_budget_gate_bypass.py`) so it's visible locally too,
including a test that plants a real violation in a throwaway tree to prove
the checker isn't vacuously passing.

### Primary/fallback provider on transient failure

Real production experience on a separate project surfaced Gemini returning
503 (overloaded) and 429 (rate limited) under normal load — this is a genuine
failure mode `LLMClient` (`llm/client.py`) now handles directly, entirely
inside the LLM client layer (`BudgetGate` itself is untouched structurally):

- **Retry before fallback.** On a transient error, `call_structured`'s own
  inner loop already retries immediately with the same provider (see the
  budget-enforcement section above); on top of that, `LLMClient.call()` adds
  an *outer* retry layer specifically for 429/503 (`is_transient_error` in
  `llm/structured_output.py` — checks `exc.status_code`, which every
  reachable SDK's APIStatusError-rooted exceptions expose, anthropic/openai/
  litellm alike) with exponential backoff (`LLM_RETRY_BACKOFF_SECONDS`,
  `LLM_FALLBACK_MAX_RETRIES`). Only after that outer retry budget is
  exhausted — and only if the last error was still transient — does a
  configured fallback provider (`LLM_FALLBACK_PROVIDER`/`LLM_FALLBACK_MODEL`/
  `LLM_FALLBACK_API_KEY`) get one attempt. A non-transient error (auth, bad
  request) is never retried this way and never triggers a fallback — no
  amount of retrying or switching providers fixes a bad API key.
- **Fallback still goes through the same gate.** The retry-then-fallback
  decision happens entirely *inside* the single `self._llm_client.call(...)`
  that `BudgetGate.call()` already makes — there is no second reservation,
  no separate call path, and no way to reach either provider except through
  the gate's existing reserve-before-call/deduct-after-call accounting
  (`orchestration/budget_gate.py`). Whichever provider ends up serving the
  call, the gate's `remaining` budget is debited exactly once, by that
  call's actual `tokens_used`.
- **Token accounting is a documented approximation, not normalized.**
  Different providers/models don't cost the same per token, and token counts
  aren't strictly comparable across tokenizers — this codebase does not
  convert to a common unit or dollar cost. A fallback-provider token is
  deducted 1:1 against the same `total_token_budget` as a primary-provider
  token, stated here as a known, deliberate simplification rather than
  silently assumed. What *is* built to make that approximation auditable:
  every `AgentOutput`/`BudgetLedgerEntry` now carries `provider_used`, so a
  reviewer can see exactly which calls were served by the fallback (and
  therefore where the approximation was actually in effect) rather than
  having to infer it.
- **Observable.** `llm_fallback_invocations_total{from_provider,to_provider}`
  (Prometheus) increments the moment a fallback is actually used; `.env`'s
  fallback block is entirely optional (unset = today's primary-only
  behavior, unchanged) and server-only — no per-debate override, same
  reasoning as the API key/Vertex project (operational resilience config,
  not something a request should redirect).

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
  `max_tokens` cap cannot bound.** A single call can still legitimately
  report usage somewhat above its own request, by roughly the size of the
  prompt itself. **Cumulative spend across retry attempts is separately
  capped** (`structured_output.py`'s `RETRY_BUDGET_MULTIPLIER`, added after
  a real 3.8x overrun — 7044 tokens_used against a 1853 allocation — was
  observed in a live debate against a local Ollama model, qwen3.5:9b):
  each individual attempt's own `max_tokens`/`num_predict` cap was verified
  correct on both Vertex AI Gemini and Ollama via direct, deliberately
  tight-budget calls against both real APIs — this was never a per-call
  enforcement problem — but `call_structured`'s retry loop previously
  re-issued the *same* `max_tokens` on every attempt with no ceiling on the
  running total, so a weaker model needing several attempts to produce a
  valid tool call could spend roughly `max_retries`x its allocation. Now
  each subsequent attempt's own cap shrinks by what prior attempts already
  spent, and the loop stops retrying early (excluding the agent, same as
  exhausting `max_retries`) once the remaining allowance can't support
  another viable attempt — bounding total spend at 2x the original
  allocation, verified against the exact live scenario that surfaced the
  bug (re-run afterward: 2289 tokens_used, well under the 3706 ceiling).
  `BudgetGate` also now debits any overrun beyond a call's own reservation
  from `remaining` (a related accounting bug: it previously only ever
  decremented `remaining` by what was *reserved*, so an overrun silently
  overstated what was actually left), and reconciles real spend even when
  every retry attempt fails validation (`LLMValidationError` now carries
  `total_tokens_used`, so an excluded agent's genuine cost is recorded in
  the ledger and debited from the budget instead of being silently
  credited back as if nothing had been spent). `BudgetManager`'s dynamic
  per-round rebaselining still absorbs whatever residual gap remains the
  same way it absorbs any other over/under-spend: the orchestrator never
  *allocates* past the ceiling, and adapts future allocations to reality
  rather than crashing — proven in `tests/test_budget_manager.py`,
  `tests/test_structured_output.py`, `tests/test_budget_gate.py`, and
  `tests/test_budget_invariant.py`, and documented in full in
  `ARCHITECTURE.md`'s "bug #4" section.
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
- **A weak local model can fail tool-calling entirely, returning an empty
  `{}` with none of the schema's required fields — this is a genuine
  model-compliance failure, not a token-starvation bug, and the system's
  correct response is to exclude that agent for the round, not to retry
  forever or crash.** Observed live against `qwen3.5:9b` via Ollama
  (docker-compose run, round 2, exploit mode): `risk_contrarian` was
  allocated a healthy 3130 tokens, well above `min_viable_allocation`, and
  still returned `{}` on every attempt — `stance`, `confidence`,
  `key_factors`, `evidence`, `top_risk`, and `executive_summary` all
  "Field required." `call_structured`'s retry loop worked exactly as
  designed here: it retried with the validation error fed back each time,
  spent up to its `RETRY_BUDGET_MULTIPLIER`-capped allowance (8349 tokens
  across attempts against a 6260 ceiling — stopped rather than force a
  doomed extra attempt), then raised `LLMValidationError` and the
  orchestrator excluded the agent for that round, logged
  `agent_excluded_invalid_output`, and the debate continued with the
  remaining agents rather than stalling or crashing. No code change made
  in response — this is the intended graceful-degradation path, and
  forcing a "fix" here (e.g. looping until the schema is satisfied) would
  trade a bounded, logged exclusion for an unbounded retry against a model
  that may simply not comply. If this turns out to recur often against
  `qwen3.5:9b` specifically, the real levers are outside this module:
  raising `LLM_MAX_RETRIES`, simplifying the exploit-mode prompt
  (rebuttal context may be pushing a small model past its tool-calling
  reliability), or accepting it as a documented characteristic of running
  this system against sub-10B local models.
- **A response that's fully valid except one free-text field running a few
  characters over its `max_length` used to cost the whole agent its
  output, not just that field — fixed by truncating instead of failing.**
  Found live against Gemini: `risk_contrarian`'s round-3 response had a
  correct stance, confidence, key_factors, and top_risk, but
  `executive_summary` (capped at 280 chars) came back slightly over, on
  every one of 3 retry attempts — the model didn't reliably self-correct a
  character-count constraint even with the exact Pydantic error fed back
  verbatim, plausibly because nothing in its own generation loop counts
  characters as it writes. The old behavior discarded the *entire* agent
  output (stance and all) over this one cosmetic overage.
  `call_structured` now recognizes the narrow case where every validation
  error is `string_too_long` on a top-level string field, truncates
  exactly that field to its schema's `max_length` (with a trailing `...`
  so a truncated summary is visibly incomplete rather than looking like a
  naturally short one), and re-validates — accepted on the same attempt
  instead of burning the whole retry budget. Deliberately narrow: if any
  other error is present alongside the length one (missing field, wrong
  type, nested error), the repair is skipped and the normal
  retry-with-feedback path runs unchanged, so this never masks a response
  that's genuinely wrong in some other way.

## AI prompts used during development

The full, verbatim chronological log of prompts used to build this system
has moved to [`PROMPTS.md`](PROMPTS.md), to keep this document focused on
architecture and tradeoffs rather than session-by-session history. See that
file for the complete record, including every bug found live and how it was
diagnosed and fixed.

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
