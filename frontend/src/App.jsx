import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_BASE = "http://localhost:8000";

const AGENT_COLORS = {
  fundamentals: "#3b82f6",
  market_sentiment: "#f59e0b",
  risk_contrarian: "#ef4444",
  macro_context: "#10b981",
  tie_breaker: "#8b5cf6",
};

const STANCE_COLORS = {
  Buy: "#16a34a",
  Hold: "#ca8a04",
  Sell: "#dc2626",
  Pass: "#6b7280",
};

// Quick-pick list for common setups — the Model field stays free text, so
// any other model/provider string still works by typing it directly. These
// are just shortcuts for models confirmed to support tool-calling (checked
// via litellm.supports_function_calling before relying on any Ollama model
// in this system — not every local model declares that capability).
const MODEL_PRESETS = [
  { label: "(custom — type below)", provider: "", model: "" },
  { label: "Vertex AI — Gemini 2.5 Flash", provider: "litellm", model: "vertex_ai/gemini-2.5-flash" },
  { label: "Ollama — mistral:latest", provider: "litellm", model: "ollama/mistral:latest" },
  { label: "Ollama — mistral-small:24b", provider: "litellm", model: "ollama/mistral-small:24b" },
  { label: "Ollama — llama3.1:70b", provider: "litellm", model: "ollama/llama3.1:70b" },
  { label: "Ollama — qwen3:14b", provider: "litellm", model: "ollama/qwen3:14b" },
  { label: "Ollama — qwen3.5:9b", provider: "litellm", model: "ollama/qwen3.5:9b" },
  { label: "Ollama — gemma4:latest", provider: "litellm", model: "ollama/gemma4:latest" },
];

function StancePill({ stance }) {
  return (
    <span
      className="pill"
      style={{ background: STANCE_COLORS[stance] || "#6b7280" }}
    >
      {stance}
    </span>
  );
}

function ModePill({ mode }) {
  const colors = { explore: "#0ea5e9", balanced: "#a3a3a3", exploit: "#f97316" };
  return (
    <span className="pill" style={{ background: colors[mode] || "#a3a3a3" }}>
      {mode}
    </span>
  );
}

function ConvergenceBar({ score }) {
  const pct = Math.max(0, Math.min(1, score)) * 100;
  let color = "#0ea5e9";
  if (score > 0.75) color = "#f97316";
  else if (score >= 0.4) color = "#a3a3a3";
  return (
    <div className="convergence-track">
      <div className="convergence-fill" style={{ width: `${pct}%`, background: color }} />
      <div className="convergence-threshold" style={{ left: "40%" }} />
      <div className="convergence-threshold" style={{ left: "75%" }} />
      <span className="convergence-label">{score.toFixed(2)}</span>
    </div>
  );
}

function RoundCard({ round, agentLabels }) {
  return (
    <div className="round-card">
      <div className="round-header">
        <h3>Round {round.round}</h3>
        <ModePill mode={round.mode} />
        {round.composite_score != null && <ConvergenceBar score={round.composite_score} />}
      </div>
      <div className="agent-grid">
        {round.agents.map((a) => (
          <div key={a.agent_id} className="agent-card" style={{ borderLeftColor: AGENT_COLORS[a.agent_id] || "#666" }}>
            <div className="agent-card-header">
              <strong>{agentLabels[a.agent_id] || a.agent_id}</strong>
              {a.reasoning ? <span className="spinner" /> : a.stance ? <StancePill stance={a.stance} /> : null}
            </div>
            {a.confidence != null && <div className="agent-confidence">confidence: {a.confidence}</div>}
            {a.tokens_used != null && <div className="agent-tokens">{a.tokens_used} tokens</div>}
            {a.executive_summary && <div className="agent-summary">{a.executive_summary}</div>}
          </div>
        ))}
      </div>
      {round.disagreementCount > 0 && (
        <div className="disagreement-banner">⚠ {round.disagreementCount} disagreement(s) detected</div>
      )}
    </div>
  );
}

function SynthesisCard({ synthesis, agentLabels }) {
  if (!synthesis) return null;
  const label = (id) => agentLabels[id] || id;
  const dissentingView = synthesis.dissenting_view || [];
  const agentSummaries = synthesis.agent_summaries || {};
  return (
    <div className="synthesis-card">
      <h2>Synthesis</h2>
      <div className="synthesis-headline">
        <StancePill stance={synthesis.recommendation} />
        <span className="synthesis-confidence">confidence {synthesis.confidence}</span>
      </div>
      <div className="synthesis-row">
        <strong>Supporting:</strong> {synthesis.supporting_agents.map(label).join(", ") || "none"}
      </div>
      <div className="synthesis-row">
        <strong>Dissenting:</strong> {synthesis.dissenting_agents.map(label).join(", ") || "none"}
      </div>
      {synthesis.dissent_appendix && (
        <div className="dissent-appendix">{synthesis.dissent_appendix}</div>
      )}

      {synthesis.dissenting_view_note && (
        <div className="dissenting-view">
          <div className="dissenting-view-note">{synthesis.dissenting_view_note}</div>
          {dissentingView.length > 0 && (
            <ul className="dissenting-view-list">
              {dissentingView.map((entry) => (
                <li key={entry.agent_id}>
                  <span className="dissenting-view-agent">
                    {entry.agent_name || label(entry.agent_id)}
                  </span>
                  <StancePill stance={entry.stance} />
                  <span className="dissenting-view-reason">{entry.reason}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {Object.keys(agentSummaries).length > 0 && (
        <div className="agent-summaries">
          <h3>At a glance</h3>
          <ul className="agent-summaries-list">
            {Object.entries(agentSummaries).map(([agentId, summary]) => (
              <li key={agentId}>
                <span className="dissenting-view-agent">{label(agentId)}</span>
                <span className="dissenting-view-reason">{summary}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

// Agent roster: lists every persona (built-in + custom) from GET /agents,
// lets a user create a new one via POST /agents (persona fields only — the
// output contract/schema is never exposed here, since it isn't something a
// persona can affect), and toggle active/inactive via PATCH /agents/{id}.
// `selectedIds`/`onToggleSelected` drive which agents the *next* debate run
// includes (config.agent_ids) — independent from a persona's own is_active
// flag, so a user can preview an inactive persona in one debate without
// changing its default-inclusion status.
function AgentRosterPanel({ agents, loading, error, onRefresh, onCreate, onSetActive, selectedIds, onToggleSelected }) {
  const [showForm, setShowForm] = useState(false);
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState("");
  const [form, setForm] = useState({
    name: "",
    role: "",
    responsibility: "",
    thinking_style: "",
    priorities: "",
    blind_spots: "",
  });

  function updateField(field, value) {
    setForm((prev) => ({ ...prev, [field]: value }));
  }

  async function handleCreate(e) {
    e.preventDefault();
    setCreateError("");
    const priorities = form.priorities.split(",").map((s) => s.trim()).filter(Boolean);
    const blindSpots = form.blind_spots.split(",").map((s) => s.trim()).filter(Boolean);
    if (!priorities.length || !blindSpots.length) {
      setCreateError("At least one priority and one blind spot are required (comma-separated).");
      return;
    }
    setCreating(true);
    try {
      await onCreate({
        name: form.name,
        role: form.role,
        responsibility: form.responsibility,
        thinking_style: form.thinking_style,
        priorities,
        blind_spots: blindSpots,
      });
      setForm({ name: "", role: "", responsibility: "", thinking_style: "", priorities: "", blind_spots: "" });
      setShowForm(false);
    } catch (err) {
      setCreateError(err.message);
    } finally {
      setCreating(false);
    }
  }

  return (
    <details className="agent-roster" open>
      <summary>
        Agent roster ({agents.length}) — choose who debates, or add a new persona
      </summary>

      {error && <div className="error-banner small">Failed to load agents: {error}</div>}

      <div className="agent-roster-list">
        {agents.map((persona) => (
          <div key={persona.id} className={`roster-row${persona.is_active ? "" : " inactive"}`}>
            <label className="roster-checkbox">
              <input
                type="checkbox"
                checked={selectedIds.includes(persona.id)}
                onChange={() => onToggleSelected(persona.id)}
              />
            </label>
            <div className="roster-info">
              <div className="roster-name">
                {persona.name}
                {persona.is_builtin && <span className="badge">built-in</span>}
                {!persona.is_active && <span className="badge muted">inactive</span>}
              </div>
              <div className="roster-role">{persona.role}</div>
            </div>
            {!persona.is_builtin && (
              <button
                type="button"
                className="secondary small"
                onClick={() => onSetActive(persona.id, !persona.is_active)}
              >
                {persona.is_active ? "Deactivate" : "Activate"}
              </button>
            )}
          </div>
        ))}
        {loading && <div className="roster-hint">Loading agents…</div>}
        {!loading && agents.length === 0 && <div className="roster-hint">No agents found.</div>}
      </div>

      <div className="roster-actions">
        <button type="button" className="secondary small" onClick={onRefresh}>
          Refresh
        </button>
        <button type="button" className="secondary small" onClick={() => setShowForm((s) => !s)}>
          {showForm ? "Cancel" : "+ New persona"}
        </button>
      </div>

      {showForm && (
        <form className="roster-create-form" onSubmit={handleCreate}>
          <p className="roster-create-hint">
            Only identity fields — how this agent thinks. It still returns the same structured
            stance/confidence/evidence output every agent does; nothing here can change that.
          </p>
          <div className="form-row">
            <label>
              Name
              <input value={form.name} onChange={(e) => updateField("name", e.target.value)} required maxLength={80} />
            </label>
            <label>
              Role
              <input value={form.role} onChange={(e) => updateField("role", e.target.value)} required maxLength={120} />
            </label>
          </div>
          <label>
            Responsibility
            <textarea
              value={form.responsibility}
              onChange={(e) => updateField("responsibility", e.target.value)}
              rows={2}
              required
              maxLength={600}
            />
          </label>
          <label>
            Thinking style
            <textarea
              value={form.thinking_style}
              onChange={(e) => updateField("thinking_style", e.target.value)}
              rows={2}
              required
              maxLength={600}
            />
          </label>
          <div className="form-row">
            <label>
              Priorities (comma-separated)
              <input
                value={form.priorities}
                onChange={(e) => updateField("priorities", e.target.value)}
                placeholder="e.g. Regulatory exposure, Litigation risk"
                required
              />
            </label>
            <label>
              Deliberate blind spots (comma-separated)
              <input
                value={form.blind_spots}
                onChange={(e) => updateField("blind_spots", e.target.value)}
                placeholder="e.g. Ignores valuation entirely"
                required
              />
            </label>
          </div>
          {createError && <div className="error-banner small">{createError}</div>}
          <div className="form-actions">
            <button type="submit" disabled={creating}>
              {creating ? "Creating…" : "Create persona"}
            </button>
          </div>
        </form>
      )}
    </details>
  );
}

export default function App() {
  const [thesis, setThesis] = useState(
    "NovaTech Inc. is showing exceptional revenue momentum at 40% YoY growth with expanding enterprise customer count."
  );
  const [entity, setEntity] = useState("");
  const [budget, setBudget] = useState(30000);
  const [rounds, setRounds] = useState(3);
  const [strategy, setStrategy] = useState("flag_unresolved");

  // LLM overrides — all optional; an empty value means "use the server's
  // .env default," matching DebateConfig's llm_* fields (None by default).
  const [llmProvider, setLlmProvider] = useState("");
  const [llmModel, setLlmModel] = useState("");
  const [temperature, setTemperature] = useState("");
  const [thinkingBudget, setThinkingBudget] = useState("");

  // Convergence thresholds — optional overrides of DebateConfig's
  // convergence_low_threshold/convergence_high_threshold (server defaults
  // 0.4/0.75). Mainly useful for demoing exploit mode on demand: real
  // debates often plateau well under 0.75 because factor_overlap (30% of
  // the composite score) only counts exact-string matches after
  // normalization, so two agents phrasing the same concern differently
  // never overlap — lowering the high threshold lets a genuinely
  // convergent-but-not-quite-0.75 round still cross into exploit mode,
  // without touching what the score itself measures.
  const [convergenceLowThreshold, setConvergenceLowThreshold] = useState("");
  const [convergenceHighThreshold, setConvergenceHighThreshold] = useState("");

  // debate_id to resume an incomplete run — POST /debate/{run_id}/resume
  // (additive endpoint, doesn't touch POST /debate's request/response
  // contract). Reads the request/config the debate was originally started
  // with back from its saved trace server-side, so nothing above needs to
  // be resent for a resume.
  const [debateId, setDebateId] = useState("");

  const [status, setStatus] = useState("idle"); // idle | running | done | error
  const [errorMessage, setErrorMessage] = useState("");
  const [roundsById, setRoundsById] = useState({});
  const [roundOrder, setRoundOrder] = useState([]);
  const [synthesis, setSynthesis] = useState(null);
  const [rawEvents, setRawEvents] = useState([]);

  // Agent roster (GET/POST/PATCH /agents) — persona-as-data agents, see
  // README.md's "User-configurable agents" section. selectedAgentIds is
  // null until the roster loads or the user makes an explicit choice; null
  // means "let the server apply its own default" (core 4 + all active
  // custom personas) rather than sending an empty/wrong list before the
  // roster is even known.
  const [agents, setAgents] = useState([]);
  const [agentsLoading, setAgentsLoading] = useState(false);
  const [agentsError, setAgentsError] = useState("");
  const [selectedAgentIds, setSelectedAgentIds] = useState(null);

  const abortRef = useRef(null);

  async function fetchAgents() {
    setAgentsLoading(true);
    setAgentsError("");
    try {
      const response = await fetch(`${API_BASE}/agents`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setAgents(data.agents);
      // Default selection mirrors the server's own default (core 4 + all
      // active custom agents) so the checkboxes reflect what a debate would
      // actually run if nothing is touched.
      setSelectedAgentIds((prev) => prev ?? data.agents.filter((a) => a.is_active).map((a) => a.id));
    } catch (err) {
      setAgentsError(err.message);
    } finally {
      setAgentsLoading(false);
    }
  }

  useEffect(() => {
    fetchAgents();
  }, []);

  async function createAgent(persona) {
    const response = await fetch(`${API_BASE}/agents`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(persona),
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`HTTP ${response.status}: ${text}`);
    }
    const created = await response.json();
    setAgents((prev) => [...prev, created]);
    setSelectedAgentIds((prev) => (prev ? [...prev, created.id] : prev));
  }

  async function setAgentActive(agentId, isActive) {
    const response = await fetch(`${API_BASE}/agents/${encodeURIComponent(agentId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ is_active: isActive }),
    });
    if (!response.ok) return;
    const updated = await response.json();
    setAgents((prev) => prev.map((a) => (a.id === agentId ? updated : a)));
  }

  function toggleAgentSelected(agentId) {
    setSelectedAgentIds((prev) => {
      const current = prev ?? agents.map((a) => a.id);
      return current.includes(agentId) ? current.filter((id) => id !== agentId) : [...current, agentId];
    });
  }

  // A custom agent's agent_id is an opaque uuid hex (see AgentOutput.agent_id
  // vs. agent_name in the backend) — SSE events only carry agent_id, so this
  // map (built from the roster already fetched via GET /agents) is what lets
  // RoundCard show "ESG Screener" instead of a hex string for custom agents.
  // Built-ins already have readable agent_ids, so this is a no-op for them.
  const agentLabels = Object.fromEntries(agents.map((a) => [a.id, a.name]));

  function resetState() {
    setStatus("running");
    setErrorMessage("");
    setRoundsById({});
    setRoundOrder([]);
    setSynthesis(null);
    setRawEvents([]);
  }

  function upsertRound(roundNum, patch) {
    setRoundsById((prev) => {
      const existing = prev[roundNum] || { round: roundNum, mode: "explore", agents: [], disagreementCount: 0 };
      return { ...prev, [roundNum]: { ...existing, ...patch } };
    });
    setRoundOrder((prev) => (prev.includes(roundNum) ? prev : [...prev, roundNum]));
  }

  function upsertAgent(roundNum, agentId, patch) {
    setRoundsById((prev) => {
      const existing = prev[roundNum] || { round: roundNum, mode: "explore", agents: [], disagreementCount: 0 };
      const agents = existing.agents.some((a) => a.agent_id === agentId)
        ? existing.agents.map((a) => (a.agent_id === agentId ? { ...a, ...patch } : a))
        : [...existing.agents, { agent_id: agentId, ...patch }];
      return { ...prev, [roundNum]: { ...existing, agents } };
    });
  }

  // Shared by both a fresh run (POST /debate?stream=true) and a resume
  // (POST /debate/{run_id}/resume?stream=true) — identical SSE frame
  // parsing either way, only the URL/body differ per call site.
  async function streamFrom(url, body) {
    resetState();

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: body === undefined ? undefined : JSON.stringify(body),
      });

      if (!response.ok || !response.body) {
        const text = await response.text();
        throw new Error(`HTTP ${response.status}: ${text}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let frameEnd;
        while ((frameEnd = buffer.indexOf("\n\n")) !== -1) {
          const frame = buffer.slice(0, frameEnd);
          buffer = buffer.slice(frameEnd + 2);
          handleFrame(frame);
        }
      }
      setStatus((s) => (s === "error" ? s : "done"));
    } catch (err) {
      if (err.name === "AbortError") return;
      setStatus("error");
      setErrorMessage(err.message);
    }
  }

  function runDebate(e) {
    e.preventDefault();
    streamFrom(`${API_BASE}/debate?stream=true`, {
      request: { thesis, entity: entity || null },
      config: {
        total_token_budget: Number(budget),
        num_rounds: Number(rounds),
        conflict_resolution_strategy: strategy,
        llm_provider: llmProvider || null,
        llm_model: llmModel || null,
        llm_temperature: temperature === "" ? null : Number(temperature),
        llm_thinking_budget: thinkingBudget === "" ? null : Number(thinkingBudget),
        convergence_low_threshold: convergenceLowThreshold === "" ? null : Number(convergenceLowThreshold),
        convergence_high_threshold: convergenceHighThreshold === "" ? null : Number(convergenceHighThreshold),
        agent_ids: selectedAgentIds,
      },
    });
  }

  function resumeDebate(e) {
    e.preventDefault();
    if (!debateId.trim()) return;
    streamFrom(`${API_BASE}/debate/${encodeURIComponent(debateId.trim())}/resume?stream=true`);
  }

  function handleFrame(frame) {
    const lines = frame.split("\n");
    const eventLine = lines.find((l) => l.startsWith("event: "));
    const dataLine = lines.find((l) => l.startsWith("data: "));
    if (!eventLine || !dataLine) return;

    const eventType = eventLine.slice("event: ".length);
    let payload;
    try {
      payload = JSON.parse(dataLine.slice("data: ".length));
    } catch {
      return;
    }

    setRawEvents((prev) => [...prev, { eventType, payload }]);

    switch (eventType) {
      case "round_start":
        upsertRound(payload.round, { mode: payload.mode });
        break;
      case "agent_reasoning_start":
        upsertAgent(payload.round, payload.agent_id, { reasoning: true, stance: null });
        break;
      case "agent_reasoning_end":
        upsertAgent(payload.round, payload.agent_id, {
          reasoning: false,
          stance: payload.stance,
          confidence: payload.confidence,
          tokens_used: payload.tokens_used,
          excluded: payload.excluded || false,
          executive_summary: payload.executive_summary,
        });
        break;
      case "convergence_computed":
        upsertRound(payload.round, { composite_score: payload.composite_score });
        break;
      case "disagreement_detected":
        upsertRound(payload.round, { disagreementCount: payload.count });
        break;
      case "synthesis_complete":
        break;
      case "done":
        setSynthesis(payload.trace.synthesis);
        setStatus("done");
        break;
      case "error":
        setStatus("error");
        setErrorMessage(payload.message || "Unknown error");
        break;
      default:
        break;
    }
  }

  function stopDebate() {
    abortRef.current?.abort();
    setStatus("idle");
  }

  return (
    <div className="app">
      <header>
        <h1>The Investment Committee</h1>
        <p className="subtitle">Live multi-agent debate viewer — streams a real run from the API.</p>
      </header>

      {/* Outside the debate <form> deliberately — it has its own nested
          <form> (create-persona) and nested <form>s are invalid HTML that
          silently breaks the inner form's submit behavior (confirmed live:
          the browser never fires the inner onSubmit at all). */}
      <AgentRosterPanel
        agents={agents}
        loading={agentsLoading}
        error={agentsError}
        onRefresh={fetchAgents}
        onCreate={createAgent}
        onSetActive={setAgentActive}
        selectedIds={selectedAgentIds ?? []}
        onToggleSelected={toggleAgentSelected}
      />

      <form className="debate-form" onSubmit={runDebate}>
        <label>
          Thesis
          <textarea value={thesis} onChange={(e) => setThesis(e.target.value)} rows={3} required />
        </label>
        <div className="form-row">
          <label>
            Entity (optional)
            <input value={entity} onChange={(e) => setEntity(e.target.value)} placeholder="NovaTech Inc." />
          </label>
          <label>
            Token budget
            <input type="number" value={budget} onChange={(e) => setBudget(e.target.value)} min={1000} step={1000} />
          </label>
          <label>
            Rounds
            <select value={rounds} onChange={(e) => setRounds(e.target.value)}>
              <option value={2}>2</option>
              <option value={3}>3</option>
            </select>
          </label>
          <label>
            Conflict strategy
            <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
              <option value="flag_unresolved">flag_unresolved</option>
              <option value="confidence_weighted">confidence_weighted</option>
              <option value="tie_breaker">tie_breaker</option>
            </select>
          </label>
        </div>

        <details className="model-settings">
          <summary>Model settings (optional — leave blank to use the server's .env defaults)</summary>
          <div className="form-row">
            <label>
              Quick pick
              <select
                defaultValue=""
                onChange={(e) => {
                  const preset = MODEL_PRESETS[Number(e.target.value)];
                  if (preset) {
                    setLlmProvider(preset.provider);
                    setLlmModel(preset.model);
                  }
                }}
              >
                {MODEL_PRESETS.map((preset, i) => (
                  <option key={preset.label} value={i}>
                    {preset.label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Provider
              <select value={llmProvider} onChange={(e) => setLlmProvider(e.target.value)}>
                <option value="">(server default)</option>
                <option value="anthropic">anthropic</option>
                <option value="openai">openai</option>
                <option value="litellm">litellm (Gemini / Ollama / etc.)</option>
              </select>
            </label>
            <label>
              Model
              <input
                value={llmModel}
                onChange={(e) => setLlmModel(e.target.value)}
                placeholder="e.g. vertex_ai/gemini-2.5-flash or ollama/llama3.1"
              />
            </label>
            <label>
              Temperature (0–2)
              <input
                type="number"
                value={temperature}
                onChange={(e) => setTemperature(e.target.value)}
                min={0}
                max={2}
                step={0.1}
                placeholder="default"
              />
            </label>
            <label>
              Thinking budget (Gemini only)
              <input
                type="number"
                value={thinkingBudget}
                onChange={(e) => setThinkingBudget(e.target.value)}
                min={0}
                step={64}
                placeholder="auto"
              />
            </label>
            <label>
              Explore→balanced threshold (0–1)
              <input
                type="number"
                value={convergenceLowThreshold}
                onChange={(e) => setConvergenceLowThreshold(e.target.value)}
                min={0}
                max={1}
                step={0.05}
                placeholder="0.4"
              />
            </label>
            <label>
              Balanced→exploit threshold (0–1)
              <input
                type="number"
                value={convergenceHighThreshold}
                onChange={(e) => setConvergenceHighThreshold(e.target.value)}
                min={0}
                max={1}
                step={0.05}
                placeholder="0.75"
              />
            </label>
          </div>
          <p className="model-settings-hint">
            For a local model via Ollama, set Provider to <code>litellm</code> and Model to{" "}
            <code>ollama/&lt;model-name&gt;</code> (e.g. <code>ollama/llama3.1</code>) — Ollama must be running
            locally (or wherever <code>OLLAMA_BASE_URL</code> points server-side). No API key needed for Ollama.
            Thinking budget only affects Gemini models; a very small value can cause the model to run out of
            output tokens before finishing its answer. The two convergence thresholds control when the debate
            shifts mode — below the first, agents are pushed to diverge; above the second, budget shifts 2.5x
            toward the contested/minority agent. Real debates often plateau in the 0.4–0.6 range even when
            reasoning genuinely converges (factor_overlap only counts exact-phrasing matches, so agents wording
            the same concern differently never overlap) — lowering the exploit threshold is a legitimate way to
            see exploit mode trigger on that same real data, not a way to fake convergence.
          </p>
        </details>

        <div className="form-actions">
          <button type="submit" disabled={status === "running"}>
            {status === "running" ? "Debating…" : "Run debate"}
          </button>
          {status === "running" && (
            <button type="button" className="secondary" onClick={stopDebate}>
              Stop
            </button>
          )}
        </div>
      </form>

      <form className="debate-form resume-form" onSubmit={resumeDebate}>
        <div className="form-row">
          <label>
            Resume debate_id
            <input
              value={debateId}
              onChange={(e) => setDebateId(e.target.value)}
              placeholder="run_id of an incomplete debate, e.g. 59174cfd-20f5-4798-aa81-1733f5bda008"
            />
          </label>
          <button type="submit" disabled={status === "running" || !debateId.trim()}>
            {status === "running" ? "Debating…" : "Resume debate"}
          </button>
        </div>
      </form>

      {status === "error" && <div className="error-banner">Error: {errorMessage}</div>}

      <div className="rounds">
        {roundOrder.map((r) => (
          <RoundCard key={r} round={roundsById[r]} agentLabels={agentLabels} />
        ))}
      </div>

      <SynthesisCard synthesis={synthesis} agentLabels={agentLabels} />

      {rawEvents.length > 0 && (
        <details className="event-log">
          <summary>Raw event log ({rawEvents.length} events)</summary>
          <pre>{rawEvents.map((e, i) => `${i}. ${e.eventType}: ${JSON.stringify(e.payload)}`).join("\n")}</pre>
        </details>
      )}
    </div>
  );
}
