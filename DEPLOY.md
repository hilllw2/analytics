# Deploying DataChat on Digital Ocean

This app runs as a **single Docker container**: the backend serves the API and the built frontend (static files).

## Quick start (local with Docker)

```bash
# Build and run
docker compose up --build

# Open http://localhost:8000
```

For local runs with a Gemini API key, create `backend/.env` (see `backend/.env.example`) and set `GEMINI_API_KEY`. If `backend/.env` doesn't exist, comment out the `env_file` section in `docker-compose.yml` or create an empty `.env`.

---

## Deploy on Digital Ocean App Platform

### 1. Push your code

Push this repo to GitHub (or GitLab). Digital Ocean will build from the repo.

From the project root, after you have committed your code:

```bash
# If origin doesn't exist yet:
git remote add origin https://github.com/hilllw2/analytics.git
git branch -M main
git push -u origin main
```

If `origin` already exists (e.g. you pushed to another repo first), either change it:

```bash
git remote set-url origin https://github.com/hilllw2/analytics.git
git push -u origin main
```

Or add the second repo under a different remote and push there:

```bash
git remote add hilllw2 https://github.com/hilllw2/analytics.git
git push -u hilllw2 main
```

### 2. Create an App

1. In [Digital Ocean](https://cloud.digitalocean.com/) go to **Apps** → **Create App**.
2. Connect your GitHub/GitLab and select this repository.
3. Choose the branch to deploy (e.g. `main`).

### 3. Configure the app

- **Type:** Docker (Digital Ocean will use the repo’s `Dockerfile`).
- **Resource type:** Basic or higher (e.g. Basic $5/mo).
- **HTTP Port:** `8000` (the app listens on 8000).

### 4. Environment variables

In the app’s **Settings** → **App-Level Environment Variables**, add:

| Variable | Required | Example |
|----------|----------|---------|
| `GEMINI_API_KEY` | Yes (for chat/LLM) | Your Google AI API key |
| `SESSION_TIMEOUT_MINUTES` | No | `180` (default: 3 hours) |
| `MAX_SESSIONS` | No | `100` |
| `PORT` | No | Set by DO; default 8000 |

Do **not** commit real API keys. Use DO’s env vars (or DO’s “Encrypted” option).

### 5. Deploy

Save and deploy. DO will:

1. Build the image from the `Dockerfile` (frontend build + backend).
2. Run the container with `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
3. Expose the app on the URL DO assigns (e.g. `https://your-app-xxxxx.ondigitalocean.app`).

### 6. Health check (optional)

In App Platform → your component → **Settings** → **Health Check**:

- **Path:** `/health`
- **Initial delay:** 30 s
- **Period:** 30 s

---

## Deploy with Docker only (any VPS)

On any server with Docker (e.g. DO Droplet):

```bash
# Clone and build
git clone <your-repo-url> datachat && cd datachat
docker build -t datachat .

# Run (set your API key)
docker run -d -p 8000:8000 \
  -e GEMINI_API_KEY=your_key_here \
  -e SESSION_TIMEOUT_MINUTES=180 \
  --name datachat \
  datachat
```

Open `http://<server-ip>:8000`. For HTTPS, put a reverse proxy (e.g. Caddy or Nginx) in front.

---

## Single Dockerfile summary

- **Stage 1:** Node builds the frontend (`npm run build` → `frontend/dist`).
- **Stage 2:** Python image gets backend code + `frontend/dist` copied as `backend/static`. FastAPI serves `/` from `static` and `/api` from the app.
- **Port:** 8000 (overridable with `PORT`).
- **Secrets:** Pass `GEMINI_API_KEY` (and any other config) via environment, not files in the image.
