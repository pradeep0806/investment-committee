# Architecture

This document is written incrementally as the system is built, one section per
major component, rather than reconstructed after the fact — the same
"instrument it as you build it" principle CLAUDE.md applies to observability
applies here too: describing a design decision the day it's made is more
accurate than describing it in hindsight.

## Extension points (the "add X without touching core" contracts)

The system has three swappable extension points, each a `Protocol` with a
registry that maps a string identifier to an implementation class. Adding a
new implementation is always "one new module + one registration line" — the
orchestrator, CLI, and API never branch on which concrete implementation is
in use.

### `AnalystAgent` (`src/committee/agents/base.py`)

```python
class AnalystAgent(Protocol):
    agent_id: str
    lens_name: str

    async def analyze(
        self, request, round, token_budget, prior_round_outputs, directive=None
    ) -> AgentOutput: ...
```

Every concrete agent (`fundamentals.py`, `market_sentiment.py`,
`risk_contrarian.py`, `macro_context.py`) subclasses a shared internal base
(`agents/_base_impl.py::BaseAnalystAgent`) that handles prompt assembly and the
LLM call, so a concrete agent module only needs to supply `agent_id`,
`lens_name`, and a system prompt — see `agents/prompts/*.py` for the four
lens-specific prompts, each written to embed a genuine, deliberate blind spot
(CLAUDE.md §6) rather than just a different persona label.

**To add a 5th lens**: write `agents/my_lens.py` subclassing `BaseAnalystAgent`
with `@register("my_lens")` (from `agents/registry.py`), add its prompt module,
and add one line to `AGENT_MODULES` in `registry.py`. Nothing in
`orchestrator.py` changes — `build_agents()` is what turns config into
instances, and it iterates over whatever's registered.

The registry's `AGENT_MODULES` import order is also what fixes the
deterministic agent ordering later used by the explore-exploit controller's
round-robin "argue against majority" directive (step 5) — order matters for
reproducibility/testability, so it's pinned here rather than left to dict
iteration order or import side effects elsewhere.

## LLM provider wrapper (`src/committee/llm/`)

`LLMClient` (`llm/client.py`) is the only place in the codebase that imports a
provider SDK. Every agent calls `LLMClient.call(system_prompt, user_prompt,
response_model, max_retries)` and gets back `(validated_pydantic_instance,
tokens_used)` — never raw text, never a provider-specific response object.

Switching providers is a `.env` edit only (`LLM_PROVIDER` /
`LLM_MODEL` / `LLM_API_KEY`), per CLAUDE.md §1.4:

- `LLM_PROVIDER=anthropic` → `llm/providers/anthropic_provider.py` (native tool-calling)
- `LLM_PROVIDER=openai` → `llm/providers/openai_provider.py` (function-calling)
- `LLM_PROVIDER=litellm` → `llm/providers/litellm_provider.py` (unified interface,
  covers everything else — including Gemini via plain API key, `gemini/<model>`,
  or via GCP service-account/Vertex AI auth, `vertex_ai/<model>`)

Each provider module implements a narrow `RawCaller` shape (one async
`__call__` returning `(raw_tool_call_arguments, tokens_used)`) — `LLMClient`
and `structured_output.py` never see provider-specific request/response
objects, only this shape.

### Structured output validation + retry (`llm/structured_output.py`)

Raw JSON parsing of free text is never trusted. Every call goes through
tool-calling so the provider is constrained to a JSON schema derived from the
target Pydantic model, and the result is *still* re-validated through Pydantic
before the caller sees it. If validation fails, `call_structured` feeds the
Pydantic `ValidationError` text back to the model as a "your previous response
failed schema validation: ... correct it" follow-up message and retries, up to
`LLM_MAX_RETRIES` total attempts. Exhausting retries raises
`LLMValidationError` rather than crashing silently or returning invalid data —
the orchestrator (steps 4+) is what decides how to handle an agent that
couldn't produce valid output after all retries (excluded from that round,
per the plan's resolved decision, not an abort or synthetic abstain).

## Orchestrator (`src/committee/orchestration/orchestrator.py`)

`DebateOrchestrator` is a plain Python object with zero imports from
`fastapi`/`typer` — CLI (step 7) and API (step 9) both construct one and call
`.run(request)`, so the debate logic is identical regardless of entry point
(CLAUDE.md §1.1).

As of step 3, `.run()` is intentionally the "dumb" version CLAUDE.md §10 calls
for: a hardcoded 2-round loop, agents called in registry order each round, an
equal token slice (`total_token_budget // (num_agents * 2)`) allocated to every
agent every round regardless of what happens during the debate, and a
placeholder `ConvergenceSignal` (all zeros) attached to each round purely so
`RoundRecord`'s shape is already correct — no real convergence math exists yet.
This scaffolding gets *replaced* in step 5 (real convergence-driven mode and
budget reallocation), not layered underneath a separate "smart" path.

Every log line emitted during a run is bound with that run's `run_id` via
`observability/logging.py::get_run_logger`, and `debate_duration_seconds`
(the first Prometheus metric, `observability/metrics.py`) wraps the whole
`.run()` call — both wired in at this step per CLAUDE.md's "logging skeleton
should exist before there's meaningful state to log."

## What's proven so far

- `pytest tests/test_agents.py` proves the retry-on-invalid loop actually
  retries with the validation error fed back, and that it raises (not
  crashes) after exhausting attempts.
- `pytest tests/test_orchestrator.py` proves the orchestrator calls all four
  registered agents in both hardcoded rounds, splits budget evenly, and passes
  round 1's outputs into round 2's calls (enabling rebuttals).
- `scripts/smoke_test_fundamentals_agent.py` has been run manually against a
  real Vertex AI (Gemini) endpoint via the `litellm` provider path, confirming
  the wrapper works end-to-end against a live API, not just against mocks.

## Budget manager (`src/committee/orchestration/budget_manager.py`)

`BudgetManager` is the single source of truth for token allocation and the
hard invariant that spend never exceeds `total_token_budget`. Two design
choices worth calling out:

**The ledger tracks actual usage, not just allocation.** `record_actual_usage`
debits `tokens_used` (what the LLM call actually cost) against
`remaining_budget()`, not `tokens_allocated` (what was planned). If the
ledger only tracked allocation, a `total_token_budget` ceiling would be
aspirational rather than enforced — an LLM call can legitimately use more or
fewer tokens than requested, and only real usage protects the ceiling.

**A reserve pool is carved out at construction time**, before any round-level
allocation happens (10% of `total_token_budget` by default,
`DEFAULT_RESERVE_FRACTION`). This exists specifically to fund tie-breaker
agent spawns (step 6) without needing to retrofit the budget math once
conflict resolution is built — `draw_from_reserve()` debits against the
reserve independently of `remaining_budget()`, and raises
`BudgetExhaustedError` rather than allowing tie-breaker spend to spill into
the spendable pool.

`allocate()` currently supports even splits (explore/balanced mode) and a
contested-agent reallocation (exploit mode, 2.5x — the midpoint of CLAUDE.md
§5's "2-3x" — funded by giving non-contested agents a reduced share so the
round's total allocation still respects the per-round baseline rather than
overspending). The real explore-exploit controller that decides which mode
and which agents are "contested" lands in step 5; as of step 4 the
orchestrator always passes `mode="explore"`.

## Disagreement detection (`src/committee/orchestration/disagreement.py`)

`detect()` has no code path that blends or averages opposing stances into a
single number — it only ever returns explicit `DisagreementRecord`s (or an
empty list). Two conditions trigger a record:

1. No stance reaches a `>= 0.75` majority fraction of agents.
2. Any Buy-vs-Sell pair both hold confidence `>= 70` — flagged even if a
   majority otherwise exists elsewhere, since a confident direct conflict
   between two agents is never safe to paper over just because two other
   agents happen to agree on something else.

Computed every round (feeding the live `disagreements_detected_total`
metric), but only the final round's result is what synthesis/conflict
resolution (step 6) actually acts on — an explicit, documented reading of
CLAUDE.md's slightly ambiguous "runs every round" / "if final-round stances
split" wording, confirmed with the project owner during planning.

Agents excluded earlier in the round (failed structured-output validation
after all retries — `LLMValidationError`, handled in `orchestrator.py`) never
reach `detect()`'s `agent_outputs` list at all, so they don't count toward
the majority-fraction denominator; the orchestrator filters them out before
disagreement detection runs, not inside `detect()` itself.

## What's proven so far (step 4 additions)

- `pytest tests/test_budget_manager.py` proves the reserve-pool carve-out,
  exploit-mode reallocation favoring contested agents, and — the hard
  invariant CLAUDE.md §9 calls for — that cumulative actual usage across a
  worst-case 3-round exploit-mode simulation never exceeds
  `total_token_budget`.
- `pytest tests/test_disagreement.py` proves majority detection, the
  high-confidence Buy-vs-Sell override, and that no averaging code path
  exists (checked directly: `DisagreementRecord` has no blended-stance or
  average-confidence field for a bug to accidentally populate).
- `pytest tests/test_orchestrator.py` now also proves an agent that
  exhausts retries and raises `LLMValidationError` is excluded from that
  round rather than crashing the whole debate — the debate continues with
  the remaining three agents, and the excluded agent never appears in that
  round's `agent_outputs` or budget ledger.

## Explore-exploit controller (`src/committee/orchestration/explore_exploit.py`) — the centerpiece

This is the direct answer to the assignment's actual question ("how do you
build a system that's not just orchestrating agents, but making meta-decisions
about how to allocate limited reasoning resources?"). CLAUDE.md §1.6 is
explicit that this must be a *computed* signal, never `if round == N`, and the
orchestrator now proves it: round 1 always starts in explore mode (there's
nothing yet to measure convergence on), but every subsequent round's mode is
decided from the *previous* round's actual `ConvergenceSignal` — a genuinely
divergent round 1 followed by a genuinely convergent round 2 produces
different modes in rounds 2 and 3 respectively, with no hardcoded round
number anywhere in the decision path
(`test_orchestrator_mode_transitions_from_computed_convergence_not_hardcoded`
proves exactly this with a fixture that splits opinion in round 1 and agrees
from round 2 onward).

**Composite score** — computed by `ExploreExploitController.score()` exactly
as CLAUDE.md §5 step 2 specifies:

```text
composite_score = 0.5 * stance_agreement + 0.3 * factor_overlap + 0.2 * (1 - confidence_spread)
```

- `stance_agreement`: fraction of agents matching the majority stance.
- `factor_overlap`: all-pairs mean pairwise Jaccard similarity of agents'
  `key_factors`, normalized (lowercased, whitespace-stripped) before
  comparison so trivial formatting differences don't undercount genuine
  overlap. Deliberately simple, per CLAUDE.md's explicit "tag-based Jaccard
  is enough; don't reach for embeddings."
- `confidence_spread`: population stddev of confidence scores (0-100 scale),
  divided by 50 (the stddev of the most extreme possible two-point split,
  `{0, 100}`) and clamped to 1.0 — a normalization choice that keeps the
  metric in `[0, 1]` without inventing a more elaborate scheme than the
  problem needs.

**Mode policy** (`decide_mode_from_score`) — the exact three-tier boundary
CLAUDE.md specifies, with the boundary values themselves resolved by reading
CLAUDE.md's own inequality notation literally: `score < 0.4` is explore,
`0.4 <= score <= 0.75` is balanced (so *both* 0.4 and 0.75 exactly belong to
balanced, not to their neighboring tier), `score > 0.75` is exploit. Verified
directly at both boundary values in `test_explore_exploit.py`.

**Explore mode's forced divergence**: `build_directives()` injects a rotating
"argue explicitly against the current majority" instruction into one agent
per explore round, cycling through agents in the registry's fixed import
order (round-robin) rather than randomly — this keeps the behavior
deterministic and testable, at the cost of being slightly more predictable
than a random pick would be, a tradeoff worth noting given CLAUDE.md doesn't
specify a rotation rule.

**Exploit mode's reallocation target**: `contested_agents()` returns whichever
agents hold a minority stance; the orchestrator passes this list into
`BudgetManager.allocate(mode="exploit", contested_agents=...)`, which is where
the actual 2-3x reallocation happens (§4's budget manager section above) —
the controller identifies *who* is contested, the budget manager decides *how
much* extra they get, keeping the two concerns separate.

**Formalized as a protocol** (`ModePolicy`) even though only one
implementation exists, for consistency with `AnalystAgent` and
`ConflictResolutionStrategy` as genuine (not merely asserted) extension
points — a future alternative mode policy is a drop-in replacement for
`ExploreExploitController` in the orchestrator's constructor.

## What's proven so far (step 5 additions)

- `pytest tests/test_explore_exploit.py` (19 tests) proves the composite
  score formula exactly, both mode boundaries at the precise threshold
  values, the round-robin explore directive rotation (and its wraparound),
  and that contested-agent identification is exactly "everyone not holding
  the majority stance."
- `pytest tests/test_orchestrator.py::test_orchestrator_mode_transitions_from_computed_convergence_not_hardcoded`
  proves the end-to-end integration: a fixture that genuinely disagrees in
  round 1 and genuinely converges from round 2 onward produces a
  `ConvergenceSignal` with `stance_agreement < 1.0` in round 1 and
  `stance_agreement == 1.0` with `composite_score` above the exploit
  threshold in round 2 — mode really does track the debate's actual state.

## Synthesis + conflict resolution (`src/committee/synthesis/`, `src/committee/orchestration/conflict_resolution/`)

`synthesize()` (`synthesis/synthesizer.py`) is the single entry point the
orchestrator calls once, after the final round. It never branches on which
conflict resolution strategy is configured — that's entirely
`ConflictResolutionStrategy`'s job (protocol in
`conflict_resolution/base.py`, registry mirroring `agents/registry.py`'s
pattern in `conflict_resolution/registry.py`). Two paths:

- **No final-round disagreements** → `_clean_consensus_memo()` builds the
  memo directly from the majority stance, `dissent_appendix` stays `None`.
- **One or more final-round disagreements** → delegates entirely to
  `strategy.resolve(...)`.

Three strategies ship, selected purely via `DebateConfig.conflict_resolution_strategy`
(no orchestrator code change to add a 4th):

- **`flag_unresolved`** (default) — picks no winner. `recommendation` is
  `Stance.PASS`, both positions are stated explicitly in
  `dissent_appendix`, `DisagreementRecord.resolved` stays `False`.
- **`confidence_weighted`** — sums confidence per stance among the
  disagreeing agents; the higher-total-confidence side wins, its own
  confidence expressed as a fraction of the total (not just copied from one
  agent). The losing side's reasoning is never dropped — it's folded into
  `dissent_appendix` — but `resolved` is `True` since a recommendation was
  actually reached.
- **`tie_breaker`** (stretch) — spawns a dedicated `TieBreakerAgent`
  (`agents/tie_breaker.py`, deliberately *not* registered among the four
  standing committee lenses — it's summoned on demand, not a fifth
  permanent member) whose context is bounded to just the opposing
  `AgentOutput`s and their contested factors, not the full multi-round
  transcript, keeping its prompt and cost predictable. **Its verdict is
  dispositive by construction**: whichever stance it agrees with simply
  wins, full stop — it cannot itself produce a new disagreement, which
  bounds worst-case runtime and budget to exactly one extra LLM call rather
  than a potentially unbounded cascade of tie-breakers-resolving-tie-breakers.
  Its actual token cost is drawn from `BudgetManager`'s reserve pool
  (`draw_from_reserve`, §4 above) via `orchestrator.py`'s `spawn_agent_fn`
  closure — never from the per-round spendable budget, so a tie-breaker
  spawn can never itself blow the `total_token_budget` ceiling.

`test_conflict_resolution.py::test_same_disagreement_different_strategy_different_synthesis`
proves the literal CLAUDE.md §9 requirement: the exact same
`DisagreementRecord` and `AgentOutput`s, run through `flag_unresolved` vs
`confidence_weighted`, produce different `recommendation` values (`Pass` vs
`Buy`) — swapping strategy is a config change, and it visibly changes the
outcome.

## What's proven so far (step 6 additions)

- `pytest tests/test_conflict_resolution.py` (7 tests) proves each strategy's
  behavior in isolation: `flag_unresolved` never picks a winner,
  `confidence_weighted` genuinely weighs confidence (verified by flipping
  which side has higher confidence and confirming the winner flips too, not
  just asserting a single fixed case), `tie_breaker` is dispositive by
  construction and requires its spawn function, and the registry builds all
  three by id.
- `pytest tests/test_orchestrator.py` gained three end-to-end integration
  tests: clean-consensus debates populate `DebateTrace.synthesis` with no
  dissent; a persistently-disagreeing debate run twice with only
  `DebateConfig.conflict_resolution_strategy` changed produces two
  different synthesis recommendations with no orchestrator code touched;
  and the `tie_breaker` strategy, exercised through the real orchestrator
  (not just the isolated strategy unit test), successfully spawns
  `TieBreakerAgent`, resolves the disagreement, and marks it `resolved`.

## CLI (`src/committee/cli/main.py`)

`typer`-based, with three subcommands per CLAUDE.md §3: `run`, `replay`,
`list-runs`. `run` is fully functional; `replay` and `list-runs` are
deliberate stubs (clear message, exit code 1) until step 8 builds
`TraceStore` — there's nothing to replay or list without a storage layer to
read from, so this reflects CLAUDE.md §10's literal build order rather than
working around it.

`run` never re-implements orchestration — it calls
`orchestrator_factory.build_orchestrator(settings, config)`
(`src/committee/orchestrator_factory.py`), a small factory that constructs
the `LLMClient`, the four default agents, and the `ExploreExploitController`
from `Settings` + `DebateConfig`, then hands back a `DebateOrchestrator`. The
same factory is what the FastAPI layer (step 9) will call too — this is the
concrete mechanism behind CLAUDE.md §1.1's "CLI is mandatory, independent of
the FastAPI layer... CLI and API both call into it": neither entrypoint
constructs agents or the LLM client itself, both just call the factory and
then `.run(request)`.

## Bug found and fixed via live smoke test: budget allocation crashing on real token overruns

Running `committee run` against the real Vertex AI (Gemini) endpoint — not
just mocked tests — surfaced a genuine bug the mocks couldn't: `BudgetManager`
originally computed its per-agent baseline *once*, at construction, from
`spendable_budget // (num_agents * num_rounds)`. A live Gemini call used 1243
tokens against a ~675-token baseline (over 100 tokens is entirely normal —
`token_budget` is advisory to the model, not a hard cap on its response size),
and by round 2, cumulative actual usage had already pushed
`remaining_budget()` negative, raising `BudgetExhaustedError` and crashing the
whole debate.

**Fix**: `BudgetManager.allocate()` now recomputes the baseline on every call,
from whatever budget is actually left divided by however many rounds are
actually left (`remaining // (len(agent_ids) * remaining_rounds)`, floored at
1). An overspend in an earlier round genuinely shrinks later rounds'
allocations instead of crashing; an underspend genuinely grows them. The hard
invariant survives unchanged — cumulative actual usage still never exceeds
`total_token_budget` (`test_cumulative_usage_still_never_exceeds_budget_even_with_dynamic_rebaselining`)
— `BudgetExhaustedError` is now reserved for a genuine out-of-budget event
(remaining budget can't cover even a 1-token floor per agent), not a rigid
per-round math artifact that fires the moment reality diverges from a
fixed plan.

Worth naming honestly in the README: `total_token_budget` is a planning
ceiling the orchestrator allocates against and gracefully degrades around —
it is not a hard runtime cutoff that truncates or rejects an individual LLM
call's response. If every agent in every round happens to badly overrun its
allocation, cumulative *actual* usage can still exceed the nominal
`total_token_budget` the user configured, since nothing in this system
forcibly cuts off an in-flight LLM response mid-generation. What's guaranteed
is that the orchestrator never *allocates* past the ceiling, and always
adapts its future allocations to reality rather than crashing.

## What's proven so far (step 7 additions)

- `python -m committee.cli.main --help` shows all three subcommands;
  `replay`/`list-runs` correctly exit 1 with a clear message pointing at
  step 8.
- `python -m committee.cli.main run --thesis ... --budget 6000 --rounds 2`
  was run twice against the real Vertex AI (Gemini) endpoint: the first run
  crashed with `BudgetExhaustedError` (the bug above), the second — after
  the `BudgetManager` fix — completed cleanly across both rounds (14,209
  total tokens actually used, a genuine 4-way stance split in round 1, a
  persisting 3-way split in round 2, correctly routed through
  `flag_unresolved` to an honest "no consensus" synthesis rather than an
  averaged fake recommendation).
- `pytest tests/test_budget_manager.py` gained three tests locking in the
  dynamic-rebaselining fix: allocations shrink after an overspend, grow
  after an underspend, and cumulative usage still never exceeds
  `total_token_budget` under a mixed over/under-spend simulation across 3
  rounds.

## Storage (`src/committee/storage/`)

Four pieces, each with a genuinely different shape of responsibility:

- **`TraceStore` protocol** (`trace_store.py`): `save_round`, `save_final`,
  `get_run`, `list_runs`. This is what §1.3 means by "a write-behind cache +
  async sync worker could be dropped in later without touching orchestration
  logic" — any future backend just implements this protocol.
- **`JsonStore`**: the only implementation the orchestrator requires to
  succeed. One file per `run_id`, *rewritten* (not appended) on every
  `save_round` so the file on disk always reflects the trace's current state
  even if the process crashes mid-debate — this is the literal Core
  requirement ("full debate trace saved as structured JSON") and the source
  of truth, independent of whether Mongo is reachable.
- **`MongoStore`**: a real `motor`-based `TraceStore` implementation — never
  called directly by the orchestrator, always wrapped in
  **`BestEffortTraceStore`**, which catches and logs write failures (never
  raises) but lets read failures (`get_run`/`list_runs`) propagate, since a
  caller explicitly asking to read a run deserves to know the read failed —
  unlike a write that happens as an unavoidable side effect of a debate
  already in progress. `orchestrator_factory._DualTraceStore` fans
  `save_round`/`save_final` out to both `JsonStore` (blocking, source of
  truth) and the wrapped Mongo store (best-effort) in one call, while reads
  go to JSON only.
- **`RedisBus`**: deliberately *not* shaped as a `TraceStore` — §1.3 describes
  it as "a pub/sub channel per run_id" plus "a hot cache for the
  agent/prompt registry," a genuinely different concern (transient event
  fan-out, not durable persistence), so it gets its own narrow interface
  (`publish_round_event`, `get_cached_registry`/`set_cached_registry`)
  instead of being force-fit into `save_round`/`save_final`/`get_run`
  semantics it doesn't actually need.

**Defense in depth on every best-effort call**: `RedisBus` and
`MlflowRunTracker` both swallow their own errors internally, *and* every
call site in `orchestrator.py` wraps the call in its own try/except too —
deliberately redundant. The contract ("never blocks the debate") is the
orchestrator's to guarantee, not something to trust a specific
implementation to uphold; a caller could hand the orchestrator any
`redis_bus`-shaped object, including a broken one that doesn't swallow its
own errors (proven directly by
`test_debate_completes_even_when_redis_bus_itself_raises`), and the debate
must still complete.

**MLflow is not named in CLAUDE.md's explicit "best-effort" list** (that's
Mongo/Redis specifically) — but the same reasoning applies, and a real bug
proved it matters in practice, not just in theory: newer MLflow versions
deprecated the plain filesystem tracking backend
(`MLFLOW_TRACKING_URI=./mlruns` from `.env.example`) and now raise on
`FileStore.__init__` unless `MLFLOW_ALLOW_FILE_STORE=true` is set or a
database URI is used. This surfaced from a live CLI run against the real
Vertex AI endpoint, not from any mocked test — `MlflowRunTracker.start_run()`
was crashing the whole debate on an observability sidecar's version
incompatibility. Fixed by making every `MlflowRunTracker` method defensive
internally (log and disable further tracking on failure, never raise) plus
the same orchestrator-level try/except used for Redis.

## What's proven so far (step 8 additions)

- `pytest tests/test_storage.py` (11 tests) proves `JsonStore` round-trips a
  full trace correctly, `save_round` requires prior registration (fails
  loudly rather than silently no-op-ing), `BestEffortTraceStore` swallows
  write failures but propagates read failures, and — the hard CLAUDE.md
  §1.3 requirement — a debate with Mongo *and* Redis both permanently
  failing still completes and still produces a correct on-disk JSON trace.
  A second integration test proves the orchestrator's own defensive
  try/except catches a `redis_bus` that doesn't swallow its own errors,
  not just the well-behaved `RedisBus` implementation.
- **Two real bugs found via live smoke testing** (`committee run` against
  the actual Vertex AI/Gemini endpoint, with real storage wiring, no mocks):
  (1) the MLflow filestore-deprecation crash described above, and (2) a
  `structlog` keyword collision (`logger.warning(..., event="...")` —
  `event` is structlog's own reserved first positional argument, so passing
  it as a keyword raised `TypeError: got multiple values for argument
  'event'`) inside the redis-failure log call, caught by
  `test_debate_completes_even_when_redis_bus_itself_raises` before it ever
  reached a real run. Both are fixed; both fixes are covered by tests, not
  just patched and left unverified.
- A full live run (`committee run --thesis ... --budget 8000 --rounds 2`
  against real Gemini, real — and, for this run, genuinely unreachable —
  Mongo/Redis) completed cleanly end-to-end: `redis_publish_failed` warnings
  logged and ignored exactly as designed, a real JSON trace written to
  `./traces/<run_id>.json`, and both `committee list-runs` and
  `committee replay <run_id>` (previously stubbed since step 7) now
  correctly read that real file back and reproduce the same summary.

## FastAPI layer (`src/committee/api/app.py`)

Three routes: `/health`, `/metrics` (both infra, don't count against
CLAUDE.md §1.2's "one business endpoint"), and `POST /debate` — the one
business endpoint, with `?stream=true` returning the same computation as
Server-Sent Events on the *same* route rather than a separate endpoint.
Since this is POST, browsers' native `EventSource` (GET-only) can't consume
it — a streaming client needs `fetch()` with a `ReadableStream` reader, noted
directly in the module docstring for whoever builds a client next.

**The event hook** (`orchestrator.py`'s `event_sink` callback, added this
step) is what makes SSE possible without duplicating the orchestration loop:
`_stream_debate()` in `app.py` gives the orchestrator an `event_sink` that
pushes onto an `asyncio.Queue`, and a generator drains that queue and formats
each event as an SSE frame (`event: <type>\ndata: <json>\n\n`) as it arrives
— genuinely live, not buffered until the debate finishes. The same hook is
what `RedisBus.publish_round_event` is called from (wired in
`orchestrator_factory.build_orchestrator`), so a debate running in this
process streams to its own request directly (no pointless Redis round-trip)
while *also* publishing to Redis for any other consumer (e.g. a second
process, or eventually Grafana-adjacent tooling) — resolving the
one open design question from planning about whether SSE should
exclusively tap Redis even for a locally-running debate.

**Event schema**: `event: <type>` where `<type>` is one of `round_start`,
`agent_reasoning_start`, `agent_reasoning_end`, `convergence_computed`,
`mode_transition`, `disagreement_detected`, `synthesis_complete`, `done`
(stream end, carries the full trace), `error`. Every `data:` payload is a
JSON object with at least `event` and `run_id`, plus type-specific fields —
proven directly by a live run against the real Vertex AI/Gemini endpoint
(see below), not just the mocked API test.

**`debate_active_agent{run_id,agent}`** (the literal Stretch ask: "which
agent is reasoning right now") is set to 1 right before an agent's
`analyze()` call and back to 0 immediately after (success or exclusion) —
sourced from the exact same two points in the loop that emit
`agent_reasoning_start`/`agent_reasoning_end`, per CLAUDE.md §8's explicit
"it needs the same... event the SSE stream is already emitting."

**`llm_call_errors_total{provider}` / `llm_call_latency_seconds{provider}`**
wrap `LLMClient.call()` (not the API layer — these are about the LLM call
itself, which is where "provider" as a label actually means something).
`llm_call_errors_total` only increments for genuinely unexpected failures
(network errors, auth failures); `LLMValidationError` is explicitly excluded
since that's an expected, already-handled outcome (the orchestrator excludes
the agent for that round), not a call failure.

**Non-streaming `POST /debate` blocks synchronously** for the full debate
duration — no job queue, no polling endpoint. This is a deliberate,
named tradeoff (not an oversight): CLAUDE.md §1.3 explicitly argues against
adding infrastructure that doesn't serve a graded rubric line, and a job
queue is exactly that kind of addition for a system whose single debate run
takes well under a minute. The real UX cost — a client has to hold a
connection open for the whole debate — is exactly what `?stream=true`
exists to make tolerable in the meantime, since the client at least sees
live progress rather than staring at a blank spinner.

## What's proven so far (step 9 additions)

- `pytest tests/test_api.py` (4 tests) proves `/health` returns 200,
  `/metrics` exposes the custom registry (verified by checking
  `debate_duration_seconds` appears in the scraped text), non-streaming
  `POST /debate` returns a full trace as JSON, and `?stream=true` returns
  well-formed SSE frames in the correct order (`round_start` before any
  `agent_reasoning_start`, `convergence_computed` after all of a round's
  `agent_reasoning_end`s, ending in `done`) with every `data:` payload valid
  JSON.
- **Live end-to-end verification against the real API, not just mocks**:
  started `uvicorn committee.api.app:app` and hit both routes against a real
  Vertex AI/Gemini thesis. `POST /debate` (non-streaming) returned a
  complete, correctly-shaped trace. `POST /debate?stream=true` streamed
  every event type in the expected order for a real 2-round debate,
  including genuine round-2 rebuttals (agents explicitly referencing and
  countering each other's round-1 arguments in their `rebuttals` field) and
  the budget graceful-degradation from step 7/8 correctly shrinking round
  2's baseline allocation after round 1's real token usage overran it — all
  while Mongo, Redis, and MLflow were genuinely unavailable in this
  environment, confirmed via the server log to have failed exactly as
  designed (logged warnings, zero impact on either request's success).

## docker-compose + Grafana dashboard (`observability/{prometheus,grafana}/`)

`observability/prometheus/prometheus.yml` scrapes `api:8000/metrics` on a
5s interval — the same interval the Grafana dashboard refreshes on, so the
explore→exploit transition is genuinely visible live, not just after a
debate finishes. `observability/grafana/provisioning/` auto-loads a
`Prometheus` datasource and the dashboard JSON on container start (no manual
click-through setup needed). The dashboard
(`observability/grafana/dashboards/debate-overview.json`) has the five
panels CLAUDE.md §8 names: convergence score over time (with threshold
lines at 0.4/0.75), token budget by agent, mode timeline, disagreement
alerts, and currently-active agent — this is deliberately "just wiring up
metrics that already exist" (per CLAUDE.md §10 step 10's framing), since
every metric these panels query was instrumented in steps 3-9 as its
underlying orchestrator logic was written, not invented here.

### Real bug #3, found via a full `docker compose up --build`, not local testing

The exact `BudgetExhaustedError` scenario described in step 7/8 above — an
LLM using more tokens than its allocation — recurred in a more severe form
during the first live docker-compose run: round 1's *total* actual usage
across all four agents exceeded the *entire* spendable budget outright (not
just one round's baseline), so there was genuinely nothing left for round 2.
That part is correct, intentional `BudgetManager` behavior (a real
out-of-budget event, not a rebaselining bug) — but `POST /debate` had no
handling for it, so the exception propagated all the way to a raw HTTP 500
with a full Python traceback leaked to the client.

**Fix**: `api/app.py`'s non-streaming path now catches `BudgetExhaustedError`
specifically and returns a clean `422` with a short `detail` message — never
the raw exception. The streaming path already handled this correctly by
construction (its broad `except Exception` around `orchestrator.run()`
already converts any exception, budget-related or not, into a structured
`error` SSE event) — this bug only existed in the non-streaming branch.
Covered by two new tests
(`test_post_debate_non_streaming_returns_422_on_budget_exhaustion`,
`test_post_debate_stream_emits_error_event_on_budget_exhaustion`) using a
fixture that reports unrealistically large token usage on every call,
reproducing the exact failure mode without needing a live LLM.

### A genuine, unrelated host-environment finding (not a bug in this project)

While verifying the Grafana panels rendered real data, `http://localhost:3000`
initially returned datasources (`grafana-pyroscope-datasource`, `tempo`,
`tempo-UAT`) that don't belong to this project at all — a pre-existing,
unrelated Grafana instance already listening on port 3000 on the development
machine used to build this, verified directly (`docker port` showed our
container's own port mapping was empty; retrying `docker compose up -d
grafana` produced Docker's own explicit `port is already allocated` error).
This is a host-environment conflict specific to whichever machine runs
`docker compose up`, not a defect in `docker-compose.yml`'s `3000:3000`
mapping — the compose file keeps the conventional Grafana port, since that's
what another engineer running this fresh will expect, and it works correctly
on any host where port 3000 is free. Verified our own dashboard and
datasource provisioning correctly by temporarily running Grafana on an
alternate host port (3300) against the same compose network — the
`Prometheus` datasource (pointed at `http://prometheus:9090`, matching the
compose network) and the `Investment Committee — Debate Overview` dashboard
both loaded exactly as provisioned, with real query results
(`debate_convergence_score`, `sum by (agent) (debate_tokens_used_total)`)
matching the actual debate that had just run.

## What's proven so far (step 10 additions)

- Live `docker compose up --build`: all six services (api, mongo, redis,
  prometheus, grafana, mlflow) built and started; `api` reported healthy;
  Prometheus's own target-health API confirmed it was successfully scraping
  `api:8000/metrics` on schedule with zero scrape errors.
- A real debate run through the compose stack's API (not local uvicorn)
  reproduced, and then (after the fix) resolved, the `BudgetExhaustedError`
  →500 bug above — confirmed via direct comparison of the same request
  before and after rebuilding the image.
- Prometheus's own query API confirmed `debate_convergence_score` and
  `debate_tokens_used_total` held correct, live values matching that real
  debate's actual outcome, and (via the temporary alternate-port Grafana
  container) confirmed the dashboard's provisioned datasource and panels
  query those same values correctly.

## Step 11 — README, examples, scripts, Makefile

`README.md` is the reader-facing summary of everything documented
incrementally above — architecture overview, the explore-exploit mechanic,
disagreement handling, budget enforcement, the "Beyond the brief" paragraph
(CLAUDE.md §1.5), what wasn't built and why (§11), honest tradeoffs
(including the `total_token_budget`-is-advisory point that drove real bugs
in steps 7-10), AI prompts used, and what was most challenging. This
document stays the deeper per-component record; that one is the entry
point.

`examples/sample_run_trace.json` is a real 3-round debate on NovaTech Inc.,
run against the live Vertex AI/Gemini endpoint (not synthetic), validated
directly against `DebateTrace.model_validate_json()`. It shows a genuine
persistent disagreement (never converging past 0.41 composite score across
3 rounds) correctly routed to an honest "no consensus" synthesis via
`flag_unresolved`, rather than an averaged fake recommendation.

`scripts/run_five_sample_debates.sh` runs five more varied theses (growth,
commodity-cyclical, digital-transformation, biotech-binary-event with
`tie_breaker`, customer-concentration-risk with the default strategy) across
different budgets, demonstrating the system isn't tuned to work on just one
kind of thesis.

The full test suite — 91 tests across the 9 files CLAUDE.md §3 names
(`test_models`, `test_agents` — folded in as agents were built rather than
kept as a separate file, `test_budget_manager`, `test_explore_exploit`,
`test_disagreement`, `test_orchestrator`, `test_storage`, `test_api`, plus
`test_conflict_resolution` for the swappable strategies) — passes cleanly.

Total real bugs found and fixed via live smoke testing across the build (not
hypothetical, not caught by any mock, each reproduced and then verified
fixed against the real API or real docker-compose stack): three. Each is
documented in its own step's section above with the actual failure, the
root cause, the fix, and the test written to lock it in.

## Post-submission fix: `token_budget` was never actually enforced (bug #4)

A later review of `budget_manager.py` prompted the question of whether
`tokens_allocated` was reaching the LLM call as a real cap at all. It
wasn't. `AnalystAgent.analyze()` accepted `token_budget` as a parameter, but
`_base_impl.py` never passed it to `LLMClient.call()` — the whole plumbing
stopped one hop short of the actual API request. Concretely:
`AnthropicRawCaller` hardcoded `max_tokens=4096` regardless of allocation;
`OpenAIRawCaller` and `LiteLLMRawCaller` set no `max_tokens` at all. For
Gemini 2.5 Flash specifically (the model this project was primarily run
against, via `litellm`/Vertex AI), this is worse than it sounds: Gemini's
"thinking" tokens are a distinct budget dimension from output tokens, so
even capping `max_tokens` alone would have left thinking spend completely
unbounded. This fully explains every large overrun observed throughout live
testing (a 254-token allocation producing a 3902-token response, etc.) —
`token_budget` had been pure decoration the entire time, not an
under-enforced soft target as earlier documentation here assumed.

**Fix**, threaded through the full call chain:
- `AnalystAgent.analyze()`'s `token_budget` now reaches `LLMClient.call()`
  as a new `max_tokens` parameter, which reaches `call_structured()`, which
  clamps it to a floor (`MIN_MAX_TOKENS = 256`, so a small allocation still
  gets enough headroom for a well-formed tool call) before passing it to
  every `RawCaller` implementation.
- `AnthropicRawCaller` and `OpenAIRawCaller` now pass the real value through
  as `max_tokens`/`max_completion_tokens` instead of a hardcoded constant or
  nothing at all.
- `LiteLLMRawCaller` passes `max_tokens` directly, and additionally — when
  the target model is Gemini — sets `thinking={"type": "enabled",
  "budget_tokens": max_tokens // 2}`, which `litellm` translates to Gemini's
  native `thinkingBudget` field (confirmed by reading
  `VertexGeminiConfig._map_thinking_param` in the installed `litellm`
  package before wiring this in, rather than guessing the parameter shape).
  Splitting the cap roughly in half between thinking and output is a
  deliberate simplification — neither this project nor litellm exposes a
  single "total including thinking" knob for Gemini — but it bounds both
  dimensions instead of leaving one open.
- `TieBreakerAgent.resolve()` gained the same `max_tokens` parameter,
  fed `budget_manager.remaining_reserve()` by the orchestrator's
  `spawn_agent_fn` — the natural cap, since a tie-breaker's spend is drawn
  from exactly that reserve.

**Residual, expected gap, not a bug**: `tokens_used` (what
`record_actual_usage`/the ledger/every budget calculation is driven by) is
the provider's `total_tokens` — prompt/input tokens plus output tokens.
`max_tokens` only bounds the output (and, for Gemini, thinking) side; it
cannot and does not bound input tokens, since those are simply "however long
the prompt is." A call can still legitimately report `tokens_used` somewhat
above its `max_tokens` allocation, by roughly the size of the prompt itself
— confirmed live (a 400-token cap produced 660 total: ~260 prompt + ~400
output). This converts what was an *unbounded* 3-10x overrun into a *small,
bounded* one that scales with prompt length (which is fixed and modest per
agent), not a residual bug worth chasing further — documented directly in
`structured_output.py`'s `RawCaller.__call__` docstring and
`budget_manager.py`'s module docstring rather than left implicit.

**Two more small fixes landed alongside this**, both surfaced by direct
code review rather than a live failure:
- `record_actual_usage` now logs a `budget_allocation_overrun` warning and
  increments a new `budget_overrun_tokens_total{agent}` counter whenever
  `tokens_used > tokens_allocated` — previously this was invisible except by
  diffing the ledger by hand. Confirmed live: an 8000-budget run that used
  to show 4-8x overruns on every single agent call now shows exactly one
  64-token overrun across both rounds, logged precisely.
- The orchestrator's per-round loop now checks `remaining_budget() < 0`
  immediately after each round's usage is tallied, not just relying on the
  next round's `allocate()` call to lazily raise `BudgetExhaustedError`.
  This lets a debate whose round genuinely exhausts the budget stop cleanly
  with a partial trace (a `budget_exhausted` event, then normal
  synthesis/`done`) instead of crashing on the following iteration.
  `BudgetExhaustedError` is now reserved for the narrower case where even
  the *first* agent of a round can't get a floor allocation — verified with
  two separate API tests distinguishing the two outcomes (`test_api.py`).

Verified live end-to-end after the fix: the same 8000-token-budget thesis
that previously totaled 11,000-27,000+ actual tokens across a 2-round debate
(3-8x its nominal budget) now totals 7098 tokens — landing *within* budget,
with round-by-round usage tracking its allocation closely (706-1119 tokens
against 900-1055 allocations) instead of blowing past it by thousands.

All 100 tests pass (up from 91 — nine new tests: two in
`test_budget_manager.py` for the overrun metric, two in `test_orchestrator.py`
for the early-stop path and for proving `max_tokens` genuinely reaches the
raw caller, four in a new `test_structured_output.py` for the `MIN_MAX_TOKENS`
floor clamp, and the API tests' budget-exhaustion pair was rewritten rather
than just patched, since the early-stop fix changed which scenario actually
reaches `BudgetExhaustedError`).

## Post-submission feature: per-debate LLM overrides (provider, model, temperature, thinking budget) + local models via Ollama

Prompted by a direct question: could a user pick a different provider,
model, temperature, or Gemini thinking budget per debate — including local
models — without editing `.env` and restarting the server? Not previously
possible: `LLMClient` (and every `RawCaller`) was constructed once at
startup from `Settings`, with no `temperature` parameter wired at all (every
call used the provider's own default), and no per-request override path
existed on `DebateConfig`.

**What changed**:

- `DebateConfig` gained four optional fields — `llm_provider`, `llm_model`,
  `llm_temperature`, `llm_thinking_budget` — all `None` by default (meaning
  "use the server's `.env`"). Deliberately scoped to what's reasonable for a
  request to set: API keys, Vertex project/location, and timeouts stay
  server-only config, never client-suppliable.
- `build_llm_client_from_settings` (in `llm/client.py`) gained matching
  optional parameters, called from `orchestrator_factory.build_orchestrator`
  with `config`'s override fields — this is the one place a per-debate
  `LLMClient` is actually constructed, so it's the natural seam for this.
- `temperature` is now threaded through every provider (`AnthropicRawCaller`,
  `OpenAIRawCaller`, `LiteLLMRawCaller`) to the real API call — a genuine gap
  closed alongside this, not just new plumbing for a new field.
- `LiteLLMRawCaller` gained an explicit `thinking_budget` override that wins
  over the `max_tokens // 2` default derivation from the earlier budget-
  overrun fix, and a `base_url` parameter (mapped to litellm's `api_base`)
  for pointing at a local model server.
- **Ollama support**: `model="ollama/<name>"` routes through the exact same
  `litellm` provider path as Gemini/vertex_ai — no new dependency, no new
  code path. Confirmed via `litellm.supports_function_calling("ollama/...")`
  returning `True` before relying on it (Ollama models need to support tool-
  calling for this system's structured-output approach to work at all — not
  every local model does, but the ones with a `tools` capability, verified
  directly via `ollama`'s own `/api/tags` listing on the development
  machine, do). Ollama has no auth by default, so the `api_key` kwarg is
  omitted entirely on this path rather than sent empty.
- `Settings` gained `ollama_base_url` (default `http://localhost:11434`,
  Ollama's own default) — server-side config, not a per-request override,
  since letting a request point the server at an arbitrary URL isn't
  something to accept from outside.
- The frontend (`frontend/src/App.jsx`) gained a collapsible "Model
  settings" section (Provider, Model, Temperature, Thinking budget) whose
  values, when non-empty, ride along in the same `POST /debate?stream=true`
  request body as the existing budget/rounds/strategy fields — no new
  endpoint, no new request shape beyond the four new optional keys.

### Real bug found via live testing: no tool call in the response crashes with an unhandled TypeError/StopIteration

Testing the override path live (`llm_thinking_budget=100` against a real
Vertex AI/Gemini debate) hit `TypeError: 'NoneType' object is not
subscriptable` inside `LiteLLMRawCaller`. The model can legitimately return
a response with no tool call at all — it hit `max_tokens` mid-thought before
emitting one, or (with an unusually tight thinking budget) spent its entire
allocation on reasoning with nothing left for the actual structured call.
`response.choices[0].message.tool_calls` is then `None`, and indexing `[0]`
into that raised an unhandled `TypeError`. The same fragility existed in
`OpenAIRawCaller` (identical `tool_calls[0]` indexing) and, in a different
shape, in `AnthropicRawCaller` (`next(block for block in response.content if
block.type == "tool_use")` with no default — an unhandled `StopIteration` if
no such block exists, e.g. a response truncated by `max_tokens` during
extended thinking).

**Fix**, identical in shape across all three providers: when no tool call
comes back, return `{}` as the raw arguments rather than crashing. Pydantic's
own `model_validate({})` then rejects it as a missing-required-fields error —
an ordinary `ValidationError` that `call_structured`'s existing retry loop
already knows how to handle (feed the error back to the model, try again),
rather than a provider-specific crash needing its own handling path. This
converts "the model happened not to call the tool this time" from a fatal
error into exactly the same kind of recoverable event a malformed tool call
already was.

Verified live: reran the exact scenario that crashed (`llm_thinking_budget:
100` against Vertex AI/Gemini) three times after the fix — all three
succeeded (HTTP 200), and log inspection confirmed no agent was excluded or
retried in those particular runs (the crash was intermittent, consistent
with a model occasionally not emitting a tool call under tight generation
constraints — but the fix means when it does happen, the system recovers
instead of crashing, regardless of how often it recurs). Separately verified
a complete 2-round, 4-agent debate running entirely on a local Ollama model
(`ollama/mistral:latest`, no API key, no cloud dependency) reached a clean
consensus synthesis through the exact same code path.

All 115 tests pass (up from 100 — fifteen new tests in
`test_llm_client_overrides.py`: override plumbing through
`build_llm_client_from_settings`, Ollama `api_base`/no-`api_key` resolution,
explicit vs. derived `thinking_budget`, `temperature` reaching all three
providers' actual API calls, and the no-tool-call regression guard for all
three providers).

## Post-submission fixes: a full `docker compose up` + Ollama run surfaced five more real bugs

Running a real debate against a local Ollama model *through the dockerized
API* (not just locally) — the actual end-to-end path a user hits when they
follow the docker-compose quickstart with `llm_provider: litellm`,
`llm_model: ollama/<model>` — surfaced five distinct, previously-latent bugs
in quick succession. None of these showed up in local (non-Docker) testing
or in the test suite as it stood; each needed the real network topology,
the real slower local-model latency, or a real reserve-exhaustion scenario
to trigger. Documenting all five together since they were found and fixed
in one continuous live-debugging session.

### Bug: stale pooled Ollama connections hang forever under Docker Desktop

Symptom: a debate running against `ollama/<model>` through the dockerized
API (with `OLLAMA_BASE_URL=http://host.docker.internal:11434` pointing back
at the host) would work for the first several agent calls, then go
completely silent — no error, no timeout, just an open request that never
returned, for minutes. Direct diagnosis: a **fresh** connection to the exact
same `host.docker.internal:11434` succeeded instantly (tens of
milliseconds) while the stuck request sat dead. Root cause: Docker
Desktop's NAT can silently drop an idle container→host connection mapping
without either side observing a close — `litellm`'s pooled `httpx.AsyncClient`
then reuses what looks like a live keep-alive connection but is actually
dead, and the read blocks forever (the `timeout=` parameter passed to
`litellm.acompletion` bounds request *execution*, not a stuck connection-pool
acquisition/reuse on a half-dead socket).

**Fix** (`llm/providers/litellm_provider.py`): Ollama-bound calls now send
`extra_headers={"Connection": "close"}`, forcing a fresh connection on every
single call rather than reusing the pool. This only matters for Ollama
specifically — cloud providers behind a real load balancer don't have this
failure mode, and forcing connection-per-request there would just add
needless TLS-handshake overhead.

### Bug: litellm's `ollama/` provider fakes tool-calling and reliably fails structured output

Even after the connection fix, every single agent call against
`ollama/qwen3:14b` (a model litellm itself reports as
`supports_function_calling() == True`) failed structured-output validation
after all retries — always returning an empty `{}`, in 1-8 seconds (too
fast to be a real generation attempt). Direct investigation with
`litellm._turn_on_debug()` showed the actual outbound request: `litellm`'s
plain `ollama/` provider talks to Ollama's **legacy `/api/generate`
endpoint** and fakes tool-calling by embedding the JSON schema as literal
prompt text, asking the model to reply with raw JSON — not Ollama's real
tool-calling protocol at all. A direct test against Ollama's native
`/api/chat` endpoint with the same tools/schema produced a proper
`tool_calls` response on the first try, with real "thinking" content —
proving the model and the schema were both fine; only litellm's `ollama/`
routing was broken.

**Fix**: `litellm` has a separate provider prefix, `ollama_chat/`, that
routes through `/api/chat` and uses real tool-calling. Rather than requiring
`ollama_chat/<model>` in config/the frontend (a litellm implementation
detail no user of this system should need to know), `LiteLLMRawCaller`
detects `ollama/` (or `ollama_chat/`, accepted as-is) and rewrites the model
string to `ollama_chat/<model>` internally before the actual `acompletion`
call. `ollama/<model>` stays the one documented, stable config format.

### Bug: transport-level failures (timeouts, dropped connections) crashed the whole debate

With tool-calling now actually working, a slower local model (real
generation taking 9-40+ seconds per call) occasionally exceeded
`LLM_TIMEOUT_SECONDS` under Docker's added latency. `litellm` correctly
raised a real, bounded `APIConnectionError`/`Timeout` this time (not a
silent hang) — but `call_structured`'s retry loop only ever caught
`pydantic.ValidationError`. A transport exception from `raw_caller`
propagated straight through, uncaught, all the way to a raw HTTP 500 —
crashing the entire multi-agent debate over one slow agent call, when the
system already has a perfectly good mechanism (exclude the agent after
`max_retries`) for "this one agent didn't produce usable output."

**Fix** (`llm/structured_output.py`): `call_structured` now also catches
`openai.APIError` and `anthropic.APIError` — verified that `litellm`'s
entire exception hierarchy (`APIConnectionError`, `Timeout`,
`RateLimitError`, etc.) subclasses `openai.APIError`, so this one type
covers `LiteLLMRawCaller`, `OpenAIRawCaller`, and any future litellm-routed
provider; `anthropic.APIError` covers `AnthropicRawCaller` directly. A
transport failure is retried on the same terms as a validation failure, and
if every attempt fails this way, the agent is excluded exactly like a
validation failure — `LLMValidationError.last_error` now accepts either a
`ValidationError` or the underlying transport exception, since from the
orchestrator's perspective both mean the same thing: no usable output after
`max_retries`.

### Bug: an empty final round crashed synthesis with an `IndexError`

A weak local model failing structured output for *every* agent in the
final round (not just one) — a real scenario once tool-calling started
actually working and surfaced genuine model-quality limits rather than a
broken integration — left `final_round.agent_outputs` empty.
`_clean_consensus_memo`'s `Counter(...).most_common(1)[0]` then raised
`IndexError` on the empty sequence, crashing the debate with a raw 500
immediately after a fully successful two-round run (all the LLM calls
had actually happened; the crash was purely in synthesis).

**Fix** (`synthesis/synthesizer.py`): `_clean_consensus_memo` special-cases
the empty-outputs case and returns a degraded-but-valid `SynthesisMemo`
(`Stance.PASS`, confidence 0, empty supporting/dissenting lists, and a
`dissent_appendix` explaining that every agent was excluded) instead of
crashing. "The committee reached no conclusion because every agent failed"
is a legitimate outcome this system should be able to represent, not a bug
to guard against with a crash.

### Bug: the tie-breaker's real cost can exceed its reserve allocation, and that used to crash the whole debate

The tie-breaker strategy's `spawn_agent_fn` passes
`max_tokens=budget_manager.remaining_reserve()` to bound the spawned agent's
*output* — but, exactly like the ordinary per-agent case documented above in
bug #4, the agent's actual reported `tokens_used` is prompt + output, which
`max_tokens` cannot bound. `draw_from_reserve` correctly detects when actual
spend exceeds what's left in the reserve and raises `BudgetExhaustedError` —
but that exception used to propagate straight out of `TieBreakerStrategy.resolve`,
through `synthesize()`, through the orchestrator, to a raw, request-crashing
error — for what was, in the live case that surfaced this
(`total_token_budget=30000`, reserve exhausted by ~6%, i.e. 3187 tokens
spent against a 3000 reserve), a single-digit-percent overrun. Worse: the
tie-breaker's real tokens and real latency (a genuine LLM call, tens of
seconds) were already spent by the time the check fires — `draw_from_reserve`
is necessarily post-hoc, the same as every other budget check in this
system, so there's no way to know the actual cost before spending it.
Crashing an otherwise-complete, multi-round debate over this was wildly
disproportionate to the actual problem.

**Fix** (`orchestration/conflict_resolution/tie_breaker.py`):
`TieBreakerStrategy.resolve` now catches `BudgetExhaustedError` from the
`spawn_agent_fn` call and falls back to `FlagUnresolvedStrategy`'s memo
shape (explicit unresolved split, no averaging, `resolved=False`) instead
of letting the debate crash. This keeps the same downstream contract every
other conflict-resolution failure path already honors — a spec-correct
synthesis that plainly states the committee couldn't resolve the split —
rather than losing an entire otherwise-successful debate over one
unaffordable tie-breaker call.

### Bug: MLflow runs stuck "Running" forever with metrics recorded but never closed

Separately from the crash-causing bugs above, MLflow runs were observed
stuck in "Running" status indefinitely in the UI — but, confusingly, *with*
their final metrics (`tokens_used`, `rounds_run`, `disagreements_count`,
`convergence_at_final_round`) already populated. Root cause:
`MlflowRunTracker.log_final()` wrapped `log_metrics`, `log_text` (the
`trace.json`/`synthesis.json` artifacts), and `mlflow.end_run()` in one
single `try/except`. With `--default-artifact-root` set to a local
filesystem path (`/mlruns/artifacts`) and the `api` container **not**
sharing the `mlflow` container's filesystem or user (`api` runs as uid 999,
`mlflow`'s container runs as root, and `./mlruns` is only mounted into the
`mlflow` service) — `mlflow.log_text()` reliably raised
`PermissionError: [Errno 13] Permission denied: '/mlruns'`. The broad except
caught this and logged a warning, but that also meant `mlflow.end_run()`
right after it never executed — leaving the run's server-side status stuck
`RUNNING` forever, with no live process left anywhere to close it.

**Fix** (`observability/mlflow_tracking.py`): split `log_final` into three
independent `try/except` blocks — metrics, artifacts, then `end_run()` —
so a failure in any one can no longer block the next. `end_run()` now
always fires once `log_final` is reached, regardless of whether metrics or
artifact logging succeeded. The underlying artifact-write permission issue
itself is left as a known gap (see "Honest tradeoffs" in the README) rather
than fixed by mounting `./mlruns` into the `api` container too, since the
full trace is already the JSON file under `TRACE_JSON_DIR` regardless — a
failed `trace.json`/`synthesis.json` *artifact* upload to MLflow is a
nice-to-have, not the source of truth. Runs that got stuck under the old
code stay stuck (there's no live process left to retroactively close them);
every run through the fixed code closes correctly.

### Verification

All five fixes were verified against the real dockerized stack, not just
the test suite: rebuilt the `api` image after each fix, restarted the
container, and re-ran the exact failing scenario via `curl` against
`POST /debate` until it produced a real, complete synthesis. The final
end-to-end run (`ollama/qwen3:14b`, 8000 token budget, 2 rounds) completed
in ~14 minutes (this model's real generation latency under Docker, not a
hang), used 7039 of its 8000 token budget, and produced a genuine `Buy`
recommendation — confirming every fix in the chain (`Connection: close` →
`ollama_chat/` routing → transport-error retry → PASS-fallback synthesis)
together, not in isolation.

11 new tests added across `test_llm_client_overrides.py` (Ollama prefix
rewrite), `test_structured_output.py` (transport-error retry/exclusion),
`test_synthesizer.py` (new file — empty-final-round PASS fallback),
`test_conflict_resolution.py` (tie-breaker reserve-overrun fallback), and
`test_mlflow_tracking.py` (new file — `end_run()` always fires). All 126
tests pass.
