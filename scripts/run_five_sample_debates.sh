#!/usr/bin/env bash
# Runs 5 varied sample debates via the CLI, demonstrating the system across
# different theses, budgets, and conflict resolution strategies. Each run
# writes its own trace to TRACE_JSON_DIR (./traces by default) — inspect
# those afterward, or use `committee list-runs` / `committee replay <run_id>`.
#
# Usage:
#   source .venv/bin/activate
#   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json   # if using Vertex AI
#   ./scripts/run_five_sample_debates.sh

set -euo pipefail

run() {
    echo ""
    echo "=== $1 ==="
    python -m committee.cli.main run "${@:2}"
}

run "1/5: NovaTech Inc. — growth thesis, default strategy" \
    --thesis "NovaTech Inc. is showing exceptional revenue momentum at 40% YoY growth with expanding enterprise customer count and a newly hired AI infrastructure CTO." \
    --entity "NovaTech Inc." \
    --budget 30000 --rounds 3

run "2/5: Vantage Energy — commodity cyclical, confidence-weighted" \
    --thesis "Vantage Energy is a compelling buy given oil prices above 80 dollars a barrel and industry-low breakeven costs, despite recent management credibility concerns." \
    --entity "Vantage Energy" \
    --budget 30000 --rounds 3 --strategy confidence_weighted

run "3/5: Meridian Healthcare — digital transformation, default strategy" \
    --thesis "Meridian Healthcare's digital health transformation under new CEO leadership justifies a premium valuation despite the stock trading near all-time highs." \
    --entity "Meridian Healthcare" \
    --budget 20000 --rounds 2

run "4/5: Forge Therapeutics — biotech binary event, tie-breaker" \
    --thesis "Forge Therapeutics' FT-400 Phase 3 data, even after the NEJM reanalysis showed a smaller effect size, still supports a position given the large addressable market." \
    --entity "Forge Therapeutics" \
    --budget 25000 --rounds 3 --strategy tie_breaker

run "5/5: Arcadia Robotics — customer concentration risk, default strategy" \
    --thesis "Arcadia Robotics remains attractive despite TerraMotors' bankruptcy, since the Boeing contract and broader aerospace diversification provide a floor." \
    --entity "Arcadia Robotics" \
    --budget 20000 --rounds 2

echo ""
echo "=== Done. Saved runs: ==="
python -m committee.cli.main list-runs
