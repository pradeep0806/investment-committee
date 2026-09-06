import { useRef, useState } from "react";
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

function RoundCard({ round }) {
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
              <strong>{a.agent_id}</strong>
              {a.reasoning ? <span className="spinner" /> : a.stance ? <StancePill stance={a.stance} /> : null}
            </div>
            {a.confidence != null && <div className="agent-confidence">confidence: {a.confidence}</div>}
            {a.tokens_used != null && <div className="agent-tokens">{a.tokens_used} tokens</div>}
          </div>
        ))}
      </div>
      {round.disagreementCount > 0 && (
        <div className="disagreement-banner">⚠ {round.disagreementCount} disagreement(s) detected</div>
      )}
    </div>
  );
}

function SynthesisCard({ synthesis }) {
  if (!synthesis) return null;
  return (
    <div className="synthesis-card">
      <h2>Synthesis</h2>
      <div className="synthesis-headline">
        <StancePill stance={synthesis.recommendation} />
        <span className="synthesis-confidence">confidence {synthesis.confidence}</span>
      </div>
      <div className="synthesis-row">
        <strong>Supporting:</strong> {synthesis.supporting_agents.join(", ") || "none"}
      </div>
      <div className="synthesis-row">
        <strong>Dissenting:</strong> {synthesis.dissenting_agents.join(", ") || "none"}
      </div>
      {synthesis.dissent_appendix && (
        <div className="dissent-appendix">{synthesis.dissent_appendix}</div>
      )}
    </div>
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

  const [status, setStatus] = useState("idle"); // idle | running | done | error
  const [errorMessage, setErrorMessage] = useState("");
  const [roundsById, setRoundsById] = useState({});
  const [roundOrder, setRoundOrder] = useState([]);
  const [synthesis, setSynthesis] = useState(null);
  const [rawEvents, setRawEvents] = useState([]);

  const abortRef = useRef(null);

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

  async function runDebate(e) {
    e.preventDefault();
    resetState();

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch(`${API_BASE}/debate?stream=true`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          request: { thesis, entity: entity || null },
          config: {
            total_token_budget: Number(budget),
            num_rounds: Number(rounds),
            conflict_resolution_strategy: strategy,
            llm_provider: llmProvider || null,
            llm_model: llmModel || null,
            llm_temperature: temperature === "" ? null : Number(temperature),
            llm_thinking_budget: thinkingBudget === "" ? null : Number(thinkingBudget),
          },
        }),
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
          </div>
          <p className="model-settings-hint">
            For a local model via Ollama, set Provider to <code>litellm</code> and Model to{" "}
            <code>ollama/&lt;model-name&gt;</code> (e.g. <code>ollama/llama3.1</code>) — Ollama must be running
            locally (or wherever <code>OLLAMA_BASE_URL</code> points server-side). No API key needed for Ollama.
            Thinking budget only affects Gemini models; a very small value can cause the model to run out of
            output tokens before finishing its answer.
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

      {status === "error" && <div className="error-banner">Error: {errorMessage}</div>}

      <div className="rounds">
        {roundOrder.map((r) => (
          <RoundCard key={r} round={roundsById[r]} />
        ))}
      </div>

      <SynthesisCard synthesis={synthesis} />

      {rawEvents.length > 0 && (
        <details className="event-log">
          <summary>Raw event log ({rawEvents.length} events)</summary>
          <pre>{rawEvents.map((e, i) => `${i}. ${e.eventType}: ${JSON.stringify(e.payload)}`).join("\n")}</pre>
        </details>
      )}
    </div>
  );
}
