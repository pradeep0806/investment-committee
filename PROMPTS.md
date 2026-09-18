# AI prompts used during development

This is the full, verbatim chronological log of prompts used to build The
Investment Committee, moved out of `README.md` to keep that document
readable — see `README.md` for the project overview and `ARCHITECTURE.md`
for per-component design detail.

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

**A later session, adding a `dissenting_view` field to the synthesis memo:**

1. A brief for a dedicated, named dissent field on the synthesis memo —
   listing every agent whose final-round stance differs from the committee's
   final decision, with stance and a short reason, reusing existing
   `executive_summary`/reasoning rather than a new LLM call, and stating an
   explicit "no dissent" sentence rather than an empty/omitted field when
   everyone agrees.
2. Phase 0 audit (no code) found `synthesizer.py`'s `synthesize()` already
   receives full per-agent `final_round_outputs` regardless of path (clean
   consensus or any conflict-resolution strategy), so the ambiguity
   protocol's "retain per-agent data through to synthesis" branch wasn't
   needed — every strategy already computes its own winners/losers from the
   same data. Also found the `agent_summaries` field added the prior session
   already established a working pattern for exactly this: a post-hoc
   `memo.model_copy(update={...})` patch applied after whichever strategy
   ran, independent of which one it was.
3. One design question asked before coding: whether the explicit zero-dissent
   sentence should live as a display-time fallback or as a persisted schema
   field. Chose a dedicated `dissenting_view_note: str` field (always set,
   alongside the `dissenting_view: list[DissentEntry]` that's empty on full
   agreement) so the explicit "no dissent" statement is part of the trace
   JSON itself, not something reconstructed only at render time.
4. Implementation: `DissentEntry` (`agent_id`, `agent_name`, `stance`,
   `reason`) added to `models/synthesis.py`; a single `_build_dissenting_view`
   helper in `synthesizer.py` diffs each final-round agent's stance against
   the computed `recommendation` and is called from both the clean-consensus
   path and patched onto the strategy-resolved memo in the disagreement path,
   so no individual conflict-resolution strategy file needed changes;
   `reason` reuses `executive_summary` when non-empty, else falls back to
   `top_risk`, per the hard constraint of zero new LLM calls; CLI prints the
   note and each dissenter.
5. One real gap found while writing this (not by a test failing after the
   fact, but by tracing the existing majority-stance logic before reusing
   it): `_clean_consensus_memo`'s majority pick via
   `Counter(...).most_common(1)` only guarantees a plurality, not unanimity,
   yet `dissenting_agents` was hardcoded to `[]` on that path — a minority
   agent disagreeing on the "clean consensus" branch (no `DisagreementRecord`
   raised, e.g. 3-1 with the 1 below the disagreement-detection threshold)
   was silently invisible. Fixed by computing `dissenting_agents` from the
   same diff `dissenting_view` uses, instead of assuming that path only ever
   runs on full agreement — a correctness fix surfaced by this feature, not
   a pre-existing failing test.
6. Verified with new tests covering: full agreement (empty list + explicit
   note), a minority dissenter on the clean-consensus path, the
   executive_summary-to-top_risk fallback, the disagreement/strategy path
   (confidence_weighted) still naming the losing side even though the
   strategy itself picked a winner, and — per the brief's explicit ask —
   that `synthesize()` never reaches for an LLM: since `spawn_agent_fn` is
   the only parameter through which it could ever reach one (used solely by
   the tie_breaker strategy), passing a fail-if-called stub as
   `spawn_agent_fn` on a run with no disagreement proves that path is
   provably unreached. Full suite (208 tests) and the pre-existing ruff/mypy
   baseline both stayed clean.
7. *"can u add the appropriate front end change to view this?"* — a follow-up
   to surface `dissenting_view`/`agent_summaries` (and the earlier session's
   `executive_summary`) in the existing `frontend/` Vite/React debate viewer,
   discovered via a directory scan rather than assumed to not exist. Read
   `App.jsx`/`App.css` first: `SynthesisCard` already rendered
   `dissenting_agents`/`dissent_appendix`, so this was additive, not a
   rebuild. One real gap found in that read: the live `agent_reasoning_end`
   SSE event (`orchestrator.py`) didn't carry `executive_summary` at all —
   only the final `done` event's embedded `trace.synthesis` would have had
   it, since `SynthesisMemo` serializes as-is with no separate SSE schema.
   Asked whether to add it to the live per-agent event (so `RoundCard` shows
   *why* as each agent finishes, not only at the very end) or leave it
   synthesis-only; chose live, one field added to the existing event payload.
   `SynthesisCard` extended with a `dissenting_view` list (agent, stance,
   reason) under the existing dissent line, the explicit `dissenting_view_note`
   sentence always shown, and an "At a glance" list from `agent_summaries` —
   no new SSE event types, since `agent_summaries`/`dissenting_view` already
   arrive inside the `done` event's trace with no backend change needed.
   Verified via `npm run build` (clean) and `npm run lint` (only the one
   pre-existing `set-state-in-effect` warning already on `main`, confirmed by
   diffing lint output against a stash of the unmodified tree) plus a full
   backend re-run (208 tests) after the one-line orchestrator SSE payload
   change.

**A later session, adding a primary/fallback LLM provider on transient
failure:**

1. A brief grounded in real production experience from a separate OCR
   project: Gemini returning 503 (overloaded) and 429 (rate limited) under
   normal load, a genuine failure mode rather than a hypothetical one. The
   ask was a bounded retry-with-backoff on the primary, falling through to a
   configured fallback provider only on exhaustion, routed through the exact
   same `BudgetGate` as every other call — explicitly not a second path to
   the LLM that bypasses budget enforcement.
2. Phase 0 audit (no code) found `call_structured`'s existing `TRANSPORT_ERRORS`
   catch-all already retried *any* transport exception immediately, no
   backoff, no distinction between transient (429/503) and non-transient
   (401, 400) — a real gap this task's Phase 1-A explicitly asks to fix, not
   something to build from scratch. Also found the ambiguity protocol's
   "propose a smaller abstraction" branch wasn't needed: the existing
   `RawCaller` Protocol (already used for exactly this — swapping providers
   with zero orchestrator changes) is already the correct seam for a second,
   fallback `RawCaller`. Confirmed via a live Python check that `litellm`'s
   exception classes subclass `openai`'s (e.g. `litellm.RateLimitError`
   inherits from `openai.RateLimitError`/`openai.APIStatusError`), and that
   `anthropic`'s own exceptions separately root at `anthropic.APIStatusError`
   — both expose `.status_code`, which is what let transient detection stay
   a single provider-agnostic `getattr(exc, "status_code", None) in {429,
   503}` check instead of enumerating each SDK's own RateLimitError/
   OverloadedError/ServiceUnavailableError class by name.
3. Also found during audit: token accounting today deducts every provider's
   raw `tokens_used` uniformly, with no per-provider cost normalization —
   exactly the approximation Phase 1-C says must be normalized-or-documented,
   and today it was neither (silently assumed equivalent). Asked whether to
   build real cross-provider cost normalization (a maintained $/token
   multiplier per model) or document the approximation as a stated, known
   simplification; chose documenting it, given the hard constraint to keep
   this change minimal and scoped to resilience, not cost accounting — a
   maintained pricing table is a different, larger feature.
4. Two more design questions asked before coding: where transient detection
   should live (chose the generic `status_code` check over enumerating named
   SDK exception classes, per point 2) and where retry/fallback orchestration
   should live (chose inside `LLMClient.call()` over inside `BudgetGate`,
   since the gate's entire purpose is budget enforcement, not provider
   routing — keeping them separate is what let `BudgetGate` end up
   completely unchanged structurally, just returning one more value).
5. One real design snag surfaced while wiring `provider_used` through to the
   trace, not anticipated at audit time: `record_actual_usage` (where
   `BudgetLedgerEntry` — the structure Phase 2 names as "whatever already
   logs per-call metadata" — gets built) only ever sees the orchestrator's
   already-built `AgentOutput`, never the gate's raw return value directly,
   since `agent.analyze()` is what calls the gate and returns only an
   `AgentOutput`. Raised as an explicit question (add `provider_used` to
   `AgentOutput` too, vs. an out-parameter/callback on `BudgetGate.call()`)
   rather than silently picking one; chose extending `AgentOutput`, since
   `tokens_used` already makes exactly this same trip (gate → agent →
   `AgentOutput` → ledger) and a callback-based side channel would be an
   unusual calling convention for a codebase that otherwise threads state
   through plain return values.
6. Implementation: `LLMClient.call()` now runs two independent, deliberately
   unmerged retry layers — `call_structured`'s existing inner loop (no
   backoff, validation + any transport error) is untouched; a new outer loop
   in `LLMClient.call()` only engages when `call_structured`'s
   `LLMValidationError.last_error` is specifically transient, retries the
   *primary* with exponential backoff up to `LLM_FALLBACK_MAX_RETRIES`
   times, then makes exactly one attempt against the fallback `RawCaller` —
   still through `call_structured`, so validation/retry-on-invalid-output
   behaves identically no matter which provider ends up serving the call.
   `BudgetGate.call()` needed only a tuple-arity change (2 → 3 elements) to
   propagate `provider_used`; its reservation/deduction logic is completely
   untouched, which is exactly the guarantee the hard constraints asked for.
   `llm_fallback_invocations_total{from_provider,to_provider}` added
   alongside the existing `llm_call_errors_total`/`llm_call_latency_seconds`
   metrics.
7. A widely-scattered test-fixture bug found while wiring this in, similar
   in shape to the executive_summary session's fixture breakage: every test
   file that builds an `LLMClient` via `LLMClient.__new__(LLMClient)`
   (bypassing `__init__` entirely, a pattern used in 8 test files before
   this session) needed the new `retry_backoff_seconds`/`fallback_provider`/
   `fallback_model`/`fallback_max_retries`/`_fallback_raw_caller` attributes
   set manually, or every existing test calling `.call()` would hit an
   `AttributeError` the moment the new outer retry loop read
   `self.fallback_max_retries`. Fixed with the same script-based approach as
   before — insert the new attribute assignments right after each fixture's
   existing `client.max_retries = ...` line — verified by diffing before
   re-running the full suite (214 passed: 208 existing + 5 new
   `test_llm_fallback.py` tests + 1 new `test_budget_gate.py` test, 0
   regressions).
8. Verified with new tests covering every Phase 2-6 case: transient error
   with a successful retry never engages the fallback path at all; primary
   retries exhausted correctly falls through to the fallback with the gate
   still deducting real usage from the shared budget (proving no bypass);
   a non-transient (401) error is never retried or escalated to the
   fallback; and — since no fallback is configured for the vast majority of
   existing debates — a persistent transient error with no fallback
   configured still raises exactly as it did before this feature, proving
   the new code path is fully opt-in and changes nothing when unconfigured.
   Real `anthropic.APIStatusError`/`AuthenticationError` instances (built
   via `httpx.Response`) were used as the fake raw callers' raised errors
   rather than a bespoke fake exception class, so `is_transient_error`'s
   `.status_code` check is exercised against the exact shape a real call
   would raise. `mypy`/`ruff` both confirmed clean against the pre-existing
   baseline on `main` (68 ruff errors after vs. 69 before — no regression;
   2 real new mypy errors in `client.py` from `fallback_provider`'s
   `str | None` typing were fixed with an explicit narrowing guard rather
   than suppressed, which also closed a latent gap where
   `_fallback_raw_caller` and `fallback_provider` could theoretically
   diverge).

**A later session, adding a standing "post-interview changes" convention:**

1. *"from this commit, whatever changes, test and stuffs added, i want it to
   be documented as post interview changes... you should update here after
   in both readme.md and this file as well, to do this add the prompt in
   claude.md"* — pointing at commit `4462b22` ("Harden debate system:
   evidence-aware convergence, structural budget gate, resumable
   persistence") as the interview-submission boundary. Everything after it —
   7 commits (resume API, cross-pod resume locking, custom agent personas,
   executive summaries, dissenting view + frontend wiring, and the LLM
   fallback work above) plus whatever was still uncommitted — needed a
   standing label, not a one-time note that would drift out of date.
2. Asked two clarifying questions before writing anything: whether the
   README should get a new, separate scannable index section versus
   splitting the existing prompts log in place (chose a new section, to keep
   the existing verbatim log intact and add a skim-first summary above it),
   and whether the CLAUDE.md instruction should be a new numbered section
   versus folded into the existing §12 (chose a new §13, since §12 already
   has a specific, narrow job — the prompts log — and conflating "log every
   prompt" with "track the interview boundary" would have muddied both
   instructions).
3. Implementation: README gained a "Post-interview changes" section right
   after Quickstart, listing each of the 5 hash-bearing commits plus the two
   most recent still-uncommitted-at-the-time pieces of work (frontend wiring,
   LLM fallback) as short summaries; CLAUDE.md gained §13, which names the
   boundary commit explicitly and requires both README sections (the new
   index and the existing §12 prompts log) to be updated together every
   session that lands a post-boundary change, with an explicit note that a
   future interview round would mean updating the boundary hash itself
   rather than letting it go stale.

**A later session, a depth-over-breadth interview follow-up on convergence
detection and the budget gate:**

1. *"Pick a few of these angles and go deep. Depth on one beats surface
   coverage of four."* — explicit interview feedback, applied to two systems
   already implemented (echo detection, the token budget gate) with the
   instruction that this task proves the existing safeguards actually hold
   and gives them real consequences, rather than adding new ones. Split into
   Core (do these) and Stretch (only after Core is solid), with a hard
   constraint to touch nothing outside the convergence classifier, synthesis
   weighting, and BudgetGate.
2. Phase 0 audit (no code) found the actual gap behind Core item B before
   any code was written: `classify_round`'s `ConvergenceType` output already
   fed `ExploreExploitController.score()` (excluding echoes from the
   convergence signal), but it never reached `synthesize()` at all —
   `orchestrator.py` passed only the raw `AgentOutput` list, so an echoed
   argument voted in the final majority on equal footing with a genuine one.
   This triggered the task's own ambiguity protocol ("if synthesis doesn't
   have a clean seam... stop and propose the smallest structural change");
   proposed threading `convergence_types` through as a new optional
   parameter (default `None`, preserving every existing call site's
   behavior) rather than a larger refactor, confirmed before implementing.
3. Two clarifying questions asked before writing code: how an echo should
   be down-weighted (chose full exclusion from the vote over a fractional
   weight, since exclude-or-include is the pattern already used everywhere
   else disagreement is handled in this codebase, with no existing
   precedent for a tunable partial weight) and confirmation to add
   `hypothesis` as a new dev dependency for the property-based test.
4. Building the two adversarial classifier fixtures surfaced a real,
   honest limitation immediately: an initial "lazy echo" fixture used
   heavier paraphrasing (different wording, same fact) and the classifier
   correctly did *not* flag it as an echo, since `classify_round` compares
   normalized literal text, not meaning — semantic echo detection is the
   task's own Stretch item, not something Core was ever asked to cover.
   Fixed the fixture to exercise trivial rewording (casing/whitespace) —
   the literal-overlap case the classifier is actually built to catch — and
   documented the paraphrase gap explicitly in both the test and the README
   rather than either silently working around it or quietly overclaiming
   Core's coverage.
5. The property-based budget test went through two full redesigns as
   Hypothesis found genuine issues, each investigated and resolved before
   moving on rather than patched around:
   - First failure: a 1-token budget, one call requesting `max_tokens=1`
     but reporting `tokens_used=2` — a real, already-documented overrun
     (prompt size can push usage above a call's own reservation), just not
     one the test's first draft of the invariant accounted for. Asked
     whether to treat this as a bug to fix in `BudgetGate` now or restate
     the test's claim to match what the gate actually guarantees; chose
     restating, per the task's own "minimal diff, proof not redesign"
     constraint.
   - Second failure, after narrowing the claim to "reservations never
     exceed the budget in total": Hypothesis found a call that used less
     than it reserved (correctly credited back) enabling a *later* call to
     be legitimately re-admitted using that freed capacity — meaning
     reservations aren't cumulative-and-permanent, so summing raw requested
     amounts across every call was itself the wrong invariant to state, not
     a bug anywhere.
   - Third, most valuable finding, traced by hand rather than assumed: a
     single-call case reported `tokens_used=255` against a reservation of
     `1`. Root cause was `call_structured`'s own `MIN_MAX_TOKENS=256` floor
     clamping the *provider-facing* request upward regardless of what
     `BudgetGate` actually reserved — a distinct, wider overrun channel
     from the already-documented prompt-size one, previously untested and
     unremarked on anywhere in the README. Resolved by scoping the "usage
     stays within its own reservation" tests to `max_tokens >=
     MIN_MAX_TOKENS` (the regime the codebase's own logic actually
     supports) and stating the sub-256 interaction explicitly as a finding
     in both the test file and the README, rather than either asserting
     something false or quietly avoiding small budgets in every test.
   - Separately found and fixed a bug in the test double itself (not
     production code): an early version popped from a pre-built per-call
     script list indexed by position in the call plan, which silently
     desynced the moment any call was refused before ever reaching the raw
     caller (a refusal makes no call at all) — the next admitted call would
     then consume the *refused* call's scripted outcome. Fixed by making
     the raw caller's outcome a pure function of the `max_tokens` it's
     actually invoked with, eliminating the shared mutable state that could
     desync in the first place.
6. Core item D (the AST bypass check) was verified, not just written: after
   building `scripts/check_budget_gate_bypass.py`, deliberately planted a
   real bypass (a direct `LLMClient(...)` construction plus a raw
   `.call()`) in a temporary file, confirmed the script caught both and
   exited non-zero, then removed the temp file and reconfirmed a clean
   pass — the same "prove it, don't just claim it" standard applied to the
   checker meant to prove the codebase's own property, not just to the
   convergence/budget work.
7. Ran the full test suite (229 passed, up from 217) and the project's
   existing `ruff`/`mypy` commands after every phase, confirmed against the
   pre-existing baseline each time (no new lint or type errors introduced)
   rather than only at the very end.

**A later session, verifying the system against real infrastructure — a
local Ollama debate, and closing the gap it surfaced:**

1. *"did u check by running the service and checked in realtime, please
   use local ollama model to check, qwen 3.5 model"* — a direct challenge
   to actually run the system rather than rely on the mocked test suite,
   using local Ollama's `qwen3.5:9b`. Confirmed Ollama was running and the
   model available, temporarily pointed `.env` at it (never committed —
   `.env` is git-ignored), and ran a real 4-agent, 2-round debate via the
   CLI's actual production `orchestrator_factory.build_orchestrator` path,
   not a smoke-test script.
2. The debate completed successfully in ~10m 41s with real, divergent
   agent reasoning (stances shifted round-to-round based on specific cited
   numbers), a genuinely computed convergence score, correct best-effort
   degradation when Redis/MLflow weren't reachable, and a coherent
   synthesis with an explicit dissenting view — but it also surfaced a real
   problem live: `budget_allocation_overrun` warnings fired on nearly every
   call, with `tokens_used` reaching up to 3.8x `tokens_allocated`
   (7044 vs. 1853). Restored the original `.env` afterward.
3. A follow-up task treated that overrun as a genuine bug to root-cause,
   not patch over: "Close the Provider-Dependent Token Cap Enforcement
   Gap," hypothesizing Ollama's `num_predict` vs. `max_tokens` parameter
   mismatch as the likely cause and asking for an explicit audit before any
   code changes — including verifying Gemini's own enforcement with a real
   deliberately-tight-budget call rather than assuming it worked because no
   overrun had been observed there.
4. The audit (real calls, not just reading code) found the task's own
   hypothesis didn't hold: `litellm`'s `ollama_chat/` transformation
   correctly maps `max_tokens`→`num_predict` (confirmed in `litellm`'s
   source), and three independent real-call reproductions — against
   `litellm.acompletion` with tools, against Ollama's raw `/api/chat` with
   tools, and against Vertex AI Gemini with a deliberately tight 80-token
   budget — all showed the per-call cap correctly enforced server-side on
   *both* providers (Gemini: `completion_tokens=77` against `max_tokens=80`,
   `finish_reason: length`). The real cause, found by tracing
   `call_structured`'s retry loop line by line against the observed
   numbers: `effective_max_tokens` was passed unchanged to every retry
   attempt with no ceiling on the cumulative total, so a model needing
   multiple attempts (confirmed live: `qwen3.5:9b` failed to produce a
   valid tool call within 300 tokens) could spend roughly `max_retries`x
   its allocation. A second, related bug was found while tracing the
   accounting: `LLMValidationError` discarded `total_tokens_used` entirely,
   so `BudgetGate` credited back the *full* reservation on a failed call
   even though real tokens were spent — the inverse problem, both rooted in
   the same missing per-attempt-vs-total accounting.
5. Flagged this divergence from the task's premise explicitly before
   writing code, since Phase 1/2 as written assumed a hard/soft *per-call*
   provider-enforcement split (with streaming-abort machinery) that the
   audit showed wasn't the actual mechanism; asked whether to build the
   originally-scoped streaming/capability-flag machinery anyway or fix the
   real, verified root cause instead — chose the latter, adapting the
   spirit of the capability-flag/severity-escalation asks (Phase 1-D,
   2-5) to the real mechanism (a `retry_budget_exhausted` event and
   escalated logging severity) rather than building unneeded streaming
   support for a failure mode that hadn't actually occurred.
6. Implementation surfaced a further, non-obvious interaction needing its
   own fix mid-session: correctly debiting overruns from `BudgetGate`'s
   `remaining` (previously silently absorbed) meant `remaining` could now
   go negative *mid-round*, which caused a *second* agent's call to be
   refused by `_reserve` and raise `BudgetExhaustedError` — an exception
   type the orchestrator's per-agent loop had never needed to catch before
   (previously unreachable, since the old accounting bug always let a full
   round complete before the post-round graceful-stop check could fire).
   Root-caused via a failing existing test rather than dismissed as
   unrelated flakiness, then fixed by adding an explicit
   `except BudgetExhaustedError` branch alongside the existing
   `LLMValidationError` one, ending the round cleanly instead of letting
   the exception crash the whole debate.
7. Two existing tests failed as a direct, expected consequence of the
   accounting fixes rather than being treated as regressions to work
   around: one asserted an excluded agent's failed attempts "still
   consumed no budget-ledger entry" (the literal old bug, now fixed —
   updated to assert the ledger entry now exists with `excluded=True` and
   real `tokens_used`); the other (a property-based test from the previous
   session) asserted `BudgetGate.remaining >= 0` after any sequence of
   calls, which is no longer universally true now that overruns are
   correctly debited rather than silently absorbed — updated to document
   why a negative `remaining` is the correct, honest outcome in that case,
   not a new bug.
8. Verified end-to-end against the real, live scenario that started this
   task: re-ran the exact call shape that had produced `tokens_used=7044`
   directly against local Ollama with the fix applied — it now completes
   at `2289` tokens (excluded after retries, but capped correctly), safely
   under the `RETRY_BUDGET_MULTIPLIER`-derived 3706-token ceiling. Full
   test suite (238 passed, up from 229), `ruff`, and `mypy` all confirmed
   against the pre-existing baseline (no new issues; one real new mypy
   error from `min()`'s type-narrowing on an `int | None` was fixed
   properly rather than suppressed) before considering the task done.

**The same session, continued: a real-time UI screenshot surfaces a second,
related budget gap.**

1. *"why there is no summary kind of thing? in the final round?"* — a
   screenshot showing round 3 of a live debate with 3 of 4 agent cards
   completely empty. Investigated the actual trace file (`run_id` visible
   in the screenshot's raw event log) rather than guessing at a UI cause:
   the agents genuinely produced no output that round (`excluded: true` in
   the ledger) — not a missing summary field, not a rendering bug.
2. Traced why: by round 3, `total_tokens_used` (41532) had nearly exhausted
   `total_token_budget` (40000), and exploit-mode reallocation had shrunk
   non-contested agents down to 723 tokens each — enough to clear
   `MIN_MAX_TOKENS`'s 256-token clamp (so the earlier retry-budget fix's
   own floor never engaged) but not enough for `qwen3.5:9b` to reliably
   produce a valid tool call. Recognized this as the "per-agent budget
   floor" Stretch item from an earlier task's own brief, never built.
   Confirmed before writing any code (`AskUserQuestion`, not assumed) that
   this was worth fixing at the `BudgetManager.allocate()` level rather than
   only documenting as a total_token_budget sizing issue.
3. First attempt at a floor (`MIN_MAX_TOKENS * RETRY_BUDGET_MULTIPLIER` =
   512) was re-verified against the exact live trace numbers and found
   still insufficient — `512 < 723`, so it would never have engaged for the
   very case that motivated it. Escalated to an explicit second question
   rather than silently picking a bigger number: whether to hardcode an
   empirically-observed floor (2048, matching what `qwen3.5:9b` needed
   live) or make it configurable; chose hardcoding first to verify the fix
   actually worked end-to-end against the real scenario.
4. That hardcoded 2048 then broke 33 of 240 existing tests — proof, not
   assumption, that one constant can't serve every provider: those tests
   used budgets sized around a *capable* hosted model (which this same
   schema succeeds against in well under 1000 tokens), and a `qwen3.5:9b`-
   calibrated floor made ordinary debates against a strong model
   artificially budget-starved from round 1. Raised this explicitly as its
   own question rather than papering over the test failures; chose a
   small, safe default (256, reusing `structured_output.py`'s own
   `MIN_MAX_TOKENS`) with an explicit per-debate override
   (`DebateConfig.min_viable_allocation_per_agent` /
   `Settings.min_viable_allocation_per_agent`), matching the existing
   None-means-server-default convention already used for the LLM
   provider/model/temperature overrides in the same config model.
5. Fixing the floor surfaced two further, non-obvious exception-handling
   gaps found only by running the full suite after each change, not
   anticipated up front: (a) `allocate()` can now raise
   `BudgetExhaustedError` on round 1 itself (previously essentially
   impossible with the old 1-token floor), which propagated unhandled and
   crashed the debate — fixed by extending the round loop's existing
   post-round graceful-stop handling to the pre-round case; (b) with zero
   completed rounds possible for the first time, `trace.rounds[-1]` could
   raise `IndexError` — fixed by routing that case through
   `synthesize()`'s existing "no agent outputs" degraded-PASS handling
   instead of assuming at least one round always exists.
6. Two existing tests failed as an expected consequence, not a regression:
   one (`test_claim_is_released_even_when_the_round_loop_raises`) had used
   `BudgetExhaustedError` specifically *because* it used to be an
   unhandled crash — now that it's gracefully handled, the test needed a
   different genuine exception (a raw-caller bug) to still prove its real
   point (the run-lock releases on any exception). A 422-expecting API
   test's whole premise (a case that "survives the early-stop fix") was
   directly superseded by this session's own fix — rewritten to assert the
   new, better behavior (200 with a zero-round degraded trace) rather than
   the older, worse one. Separately, two unrelated dormant test fixtures
   (missing `executive_summary`, stale since that field became required)
   were only exposed because the floor fix made round 2 correctly refuse
   to run rather than silently limping forward — fixed the fixtures
   directly rather than working around the newly-surfaced failure.
7. Re-verified against the exact original live-trace numbers twice — once
   confirming the 512 floor didn't help, once confirming the final
   256-default-plus-2048-override design does (round 3 now cleanly stops
   with `BudgetExhaustedError` at the true configured floor instead of
   silently allocating 723 to agents that would fail) — rather than
   trusting the unit tests alone for a bug that was originally found via a
   live run. Full suite (240 passed), `ruff`, and `mypy` confirmed clean
   against the pre-existing baseline before considering this done.

**Same session, rebuilt the docker image with the floor fix and re-ran
against Ollama live: a second, distinct empty-card report.**

1. *"i rebuild the docker after these changes now and i executed, the next
   agent satrted but this summary is excluded? why?"* — a screenshot of a
   live round 2 showing Risk Contrarian's card empty and Macro/Industry
   Context mid-spinner. Investigated the real container logs and the live
   trace/checkpoint files for the exact `run_id` rather than assuming the
   floor fix hadn't worked: confirmed round 1 had completed cleanly
   (`budget_remaining: 28812` in the checkpoint) and round 2 was genuinely
   still in progress — Macro/Industry Context's spinner was real, not
   stuck (it finished moments later with a valid "Buy" stance in the logs).
2. Risk Contrarian, however, was a genuine exclusion — but a *different*
   failure mode from the one just fixed. It was allocated 3130 tokens,
   comfortably above `min_viable_allocation`, and still returned an empty
   `{}` (all six required fields "Field required") on every attempt. This
   is a model tool-calling compliance failure, not a budget-starvation
   bug: `call_structured`'s retry loop, `RETRY_BUDGET_MULTIPLIER` cap, and
   the orchestrator's exclusion-and-continue handling all did exactly what
   they were built to do (retried, capped cumulative spend at 8349 against
   a 6260 ceiling, excluded the agent, logged
   `agent_excluded_invalid_output`, kept the debate running). Concluded no
   code change was warranted — asked the user directly, via
   `AskUserQuestion`, whether "log this bug" meant a README
   known-limitations note, a tracked GitHub issue, or both; the user chose
   the README note. Documented it under "Honest tradeoffs and known gaps"
   with the real numbers from this run, rather than filing it as an
   actionable defect that doesn't exist.

**Same session, continued: a second screenshot from a Gemini run, same
symptom, different root cause.**

1. *"why in gemini model this happened?"* — a screenshot of round 3
   against Gemini, again showing Risk Contrarian's card empty. Rather than
   assuming it was the same qwen3.5-style empty-`{}` failure, pulled the
   real container logs for that run's `risk_contrarian`/round 3 events
   directly, since the two providers turned out to fail for unrelated
   reasons.
2. The actual error was `executive_summary: String should have at most
   280 characters` — every other field (stance, confidence, key_factors,
   top_risk) was fully valid. Traced this to a real, fixable gap rather
   than another instance of "model just doesn't comply": the retry loop
   fed the exact Pydantic error back verbatim for all 3 attempts, but
   Gemini never reliably shortened the field, and `call_structured` had no
   path other than full retry-and-exclude for a response that was
   otherwise completely correct.
3. Asked via `AskUserQuestion` whether to (a) truncate the over-long field
   and accept the response, (b) only strengthen the retry prompt's
   wording, or (c) just document it like the qwen3.5 case; the user chose
   truncation. Implemented `_truncate_over_long_strings()` in
   `structured_output.py`, deliberately narrow: it only fires when *every*
   validation error is `string_too_long` on a top-level field (checked via
   `ValidationError.errors()`'s `type`/`loc`/`ctx.max_length`), so a
   response that's wrong in any other way still goes through the normal
   retry path untouched — this was a conscious design choice to avoid
   silently masking a genuinely different problem behind a length-repair
   heuristic.
4. Found and fixed the one existing test whose premise the new behavior
   directly superseded
   (`test_fundamentals_agent_rejects_executive_summary_over_max_length`,
   renamed to `..._truncates_...`) rather than leaving it failing or
   deleting it — rewrote it to assert the new correct behavior (agent
   succeeds, summary truncated to 280 chars with a trailing `...`). Added
   two new tests: one proving the repair path (over-long field alone is
   truncated and accepted on the first attempt, no retry spent), one
   proving the guard rail (an over-long field *plus* an unrelated error,
   e.g. a missing field, still genuinely retries rather than being
   silently patched). Full suite (242 passed), `ruff`, and `mypy` verified
   against the pre-existing baseline (68 ruff errors, 20 mypy errors,
   both confirmed identical via `git stash` comparison) before considering
   this done.

**A later session, relocating this log out of README.md:**

1. A detailed brief to move this full prompt log to its own `PROMPTS.md`
   file, leaving a short pointer in README.md, and updating CLAUDE.md's
   §12/§13 logging instructions to target the new file — motivated by the
   log having grown large enough to read as breadth to a reviewer rather
   than depth, the opposite of what interview feedback asked for. A Phase 0
   audit (quoting CLAUDE.md's exact current §12/§13 wording, confirming the
   "Post-interview changes" index section is structurally separate and
   stays in place) was required before any edit, per the brief's own
   ambiguity protocol.
2. One follow-up question asked before editing: README.md's
   "Post-interview changes" section (which the brief said must stay
   untouched) contains one sentence pointing readers "below" to this
   section — since the pointer's target was moving, asked whether to update
   just that one sentence despite the section otherwise being marked
   hands-off; chose to fix it, since leaving a known-stale cross-reference
   inside an otherwise-accurate section serves no one.

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
