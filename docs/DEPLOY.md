# Deploying a public demo

The repository builds into one Docker image: the API, the research pipeline and the dashboard. Any host that runs a Dockerfile can serve it. CI builds the image and smoke-tests it on every push.

## Before you deploy: two decisions

**1. Which data the public sees.**
- **Wikipedia, SEC EDGAR and FRED:** public; showing them is fine.
- **Yahoo Finance prices:** Yahoo's terms don't allow redistributing its data, and a public dashboard of price charts arguably does that. Your options:
  - keep the deployment private, or behind a password at your host;
  - use `DATA_SOURCE=sample`, a synthetic dataset that is instant to build, though the numbers are meaningless;
  - accept the risk for a small portfolio demo, as many projects do. This choice is yours.

**2. Whether the AI agent runs on your key.**
- **With a key:** every visitor's question costs a fraction of a cent on your DeepSeek or Anthropic key. `DEMO_MODE=1` caps this at 20 questions per visitor per hour and 300 per day by default.
- **Without a key:** the chat uses the free rule-based assistant.

## Settings

| Variable | Purpose | Demo suggestion |
|---|---|---|
| `DEMO_MODE` | Turns on chat limits and locks admin endpoints | `1` |
| `DATA_SOURCE` | `live`, `sample` or `kaggle` | `live` (or `sample`, see above) |
| `PIPELINE_ARGS` | Extra pipeline flags | `--limit 150 --skip-experiments` for a ~1-minute first start |
| `SEC_USER_AGENT` | Contact SEC requires, e.g. `ProjectName you@example.com` | required for live fundamentals |
| `DEEPSEEK_API_KEY` / `ANTHROPIC_API_KEY` | LLM for the agent | optional |
| `CHAT_LIMIT_PER_HOUR`, `CHAT_LIMIT_PER_DAY` | Override the demo limits | defaults 20 / 300 |
| `ADMIN_TOKEN` | Allows `POST /api/reload` with header `X-Admin-Token` | a long random string |
| `PORT` | Port to listen on | set by most hosts |

The container builds its research data on first start, then serves on `$PORT`. Health check: `GET /api/health`. With `--limit`, the dashboard covers only the first N current members, with no survivorship handling; the full live run takes about 4 minutes.

## Render (example)

1. In Render, create a **Web Service** from the GitHub repository. Render detects the `Dockerfile`.
2. Add the environment variables above. Keys go in as secrets, never in the repository.
3. Set the health check path to `/api/health`.
4. Free instances sleep when idle and have no persistent disk, so the pipeline reruns on each cold start. To keep data between restarts, attach a disk mounted at `/app/data` (paid plans), or use `DATA_SOURCE=sample`.

## Hugging Face Spaces (example)

Create a Space with the **Docker** SDK, push this repository to it, and add `app_port: 7860` to the Space's README front matter. In the Space settings, set `PORT=7860` plus the variables above as secrets.

## Any server with Docker

```bash
cp .env.example .env    # fill in SEC_USER_AGENT, an LLM key, DEMO_MODE=1, ...
docker compose up -d --build
```

Put a reverse proxy with HTTPS (Caddy, nginx, a cloud load balancer) in front of port 8000. Data persists in the compose volumes; refresh it with `docker compose exec app python run_pipeline.py`, then `curl -X POST -H "X-Admin-Token: $ADMIN_TOKEN" localhost:8000/api/reload`.
