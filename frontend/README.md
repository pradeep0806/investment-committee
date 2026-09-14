# Investment Committee — Live Debate Viewer

A small React app (Vite, no framework beyond that) for watching a debate
happen — which agent is reasoning, the running convergence score with
threshold markers at 0.4/0.75, mode transitions, disagreement alerts, and
the final synthesis. It's a viewer for understanding the system's output,
not a product build — no routing, no state library, one component.

It streams a real debate from the API's `POST /debate?stream=true` SSE
endpoint via `fetch()` + a `ReadableStream` reader (browsers' native
`EventSource` only supports GET, so it can't be used against a POST route).

## Run it

1. Start the API (from the repo root):
   ```bash
   source .venv/bin/activate
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json   # if using Vertex AI
   uvicorn committee.api.app:app --host 0.0.0.0 --port 8000
   ```
2. Start this app:
   ```bash
   cd frontend
   npm install   # first time only
   npm run dev
   ```
3. Open the printed `http://localhost:5173` URL, fill in a thesis, and click
   "Run debate."

### Model settings

The "Model settings" section (collapsed by default, click to expand) lets
you override the provider/model/temperature/thinking-budget/convergence-
thresholds for that one debate, without touching the server's `.env` or
restarting anything:

- **Provider + Model**: leave blank to use the server default, or set
  Provider to `litellm` and Model to `ollama/<model-name>` (e.g.
  `ollama/llama3.1`) to run entirely on a local model via
  [Ollama](https://ollama.com) — no API key needed. Ollama must be running
  locally (`ollama serve`, or it's already running if you installed it
  normally) with that model pulled (`ollama pull llama3.1`).
- **Temperature**: 0–2, passed straight through to whichever provider is
  selected.
- **Thinking budget**: only affects Gemini models (via `litellm`'s
  `thinking` parameter) — ignored otherwise. Leave blank to let the backend
  derive a sensible default from the token budget.
- **Explore→balanced / Balanced→exploit thresholds**: 0–1, override the
  composite convergence score's mode boundaries (server defaults 0.4/0.75).
  Mainly useful for demoing exploit mode on demand — a real debate's score
  often plateaus below 0.75 even when the reasoning genuinely converges,
  because `factor_overlap` (30% of the composite score) only counts exact-
  phrasing matches after casing/whitespace normalization, so two agents
  wording the same concern differently (e.g. "unproven margin sustainability"
  vs. "unproven sustainability of margin expansion") never overlap. Lowering
  the exploit threshold lets that same real, already-convergent data cross
  into exploit mode instead of waiting on a design fix to `key_factors`
  matching (see `ARCHITECTURE.md`).

All six are optional per-request overrides on `DebateConfig` — see the main
`README.md`'s "Per-debate overrides" section for the full picture, including
what stays server-only (API keys, Vertex project/location, timeouts).

The API URL is hardcoded to `http://localhost:8000` in `src/App.jsx`
(`API_BASE`) — edit that constant if running the API somewhere else. CORS is
wide open (`allow_origins=["*"]`) on the API side since this is a local
dev/demo pairing, not a public deployment — see `src/committee/api/app.py`.
