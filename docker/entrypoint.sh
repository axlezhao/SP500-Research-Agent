#!/bin/sh
# Build the research data on first start (or when REFRESH_ON_START=1), then serve the dashboard.
#   DATA_SOURCE    live (default) | sample | kaggle
#   PIPELINE_ARGS  extra run_pipeline.py flags, e.g. "--limit 100" for a faster first start
#   PORT           port to listen on (many hosts set this)
set -e
cd "${SP500_HOME:-/app}"
if [ ! -f data/processed/research_features.parquet ] || [ "${REFRESH_ON_START:-0}" = "1" ]; then
  echo "Building research data (source: ${DATA_SOURCE}). The first live run takes a few minutes..."
  # shellcheck disable=SC2086
  python run_pipeline.py --source "${DATA_SOURCE}" ${PIPELINE_ARGS:-}
fi
exec python -m sp500_agent.api.server --host 0.0.0.0 --port "${PORT:-8000}"
