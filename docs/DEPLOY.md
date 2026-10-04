# Deployment

The project builds into a single Docker image containing the API, the research pipeline and the dashboard. Any host that can run a Dockerfile can serve it. CI builds the image and smoke-tests it on every push.

## Decide first

**1. What data the public sees.**
- **Wikipedia, SEC EDGAR and FRED:** public; showing them is fine.
- **Yahoo Finance prices:** Yahoo's terms don't allow redistributing its data, and a public dashboard of price charts arguably does that. Options:
  - **Keep it private:** deploy behind your host's password or access controls.
  - **Synthetic data:** `DATA_SOURCE=sample` starts instantly, but the numbers are meaningless.
  - **Live data publicly:** a judgement call some portfolio projects make. It's your decision.

**2. Who pays for the AI agent.**
- **With a key:** every question a visitor asks costs a fraction of a cent on your DeepSeek or Anthropic account. Demo mode caps this at 20 questions per visitor per hour and 300 in total per day by default.
- **Without a key:** the chat uses the free rule-based assistant.

## Settings

| Variable | Purpose | Suggested for a demo |
|---|---|---|
| `DEMO_MODE` | Chat limits on; admin endpoints locked | `1` |
| `DATA_SOURCE` | `live`, `sample` or `kaggle` | `live` or `sample` |
| `PIPELINE_ARGS` | Extra flags for the first-start pipeline | `--limit 150 --skip-experiments` (first start ~1 minute) |
| `SEC_USER_AGENT` | Contact SEC requires, e.g. `ProjectName you@example.com` | Required for live fundamentals |
| `DEEPSEEK_API_KEY` or `ANTHROPIC_API_KEY` | The AI agent | Optional; store as a secret |
| `CHAT_LIMIT_PER_HOUR`, `CHAT_LIMIT_PER_DAY` | Override the demo limits | Defaults 20 and 300 |
| `ADMIN_TOKEN` | Allows `POST /api/reload` with header `X-Admin-Token` | A long random string |
| `PORT` | Listening port | Usually set by the host |

**How the container starts.** On first start the container builds its research data, then serves on `$PORT`.
- **Health check:** `GET /api/health`.
- **Without `--limit`:** a full live build takes about 4–5 minutes.
- **With `--limit N`:** the demo covers only the first N current members, with no survivorship handling.

## Render

1. Create a **Web Service** from the GitHub repository; Render detects the Dockerfile.
2. Add the settings above as environment variables, with keys as secrets.
3. Set the health check path to `/api/health`.
4. Free instances sleep when idle and have no persistent disk, so the pipeline reruns on every cold start. Either attach a disk at `/app/data` (paid plans) or use `DATA_SOURCE=sample`.

## Hugging Face Spaces

1. Create a Space with the **Docker** SDK and push this repository to it.
2. Add `app_port: 7860` to the Space's README front matter.
3. In the Space settings, set `PORT=7860` and the variables above, with keys as secrets.

## Your own server

```bash
cp .env.example .env            # SEC_USER_AGENT, an LLM key, DEMO_MODE=1, ADMIN_TOKEN=...
docker compose up -d --build
```

- **HTTPS:** put a reverse proxy (Caddy, nginx or a cloud load balancer) in front of port 8000.
- **Persistence:** data lives in the compose volumes.
- **Refreshing data:** to update without downtime, run:

```bash
docker compose exec app python run_pipeline.py
curl -X POST -H "X-Admin-Token: $ADMIN_TOKEN" http://localhost:8000/api/reload
```

A cron job running those two commands daily keeps the demo current.
