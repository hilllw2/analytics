# -----------------------------------------------------------------------------
# Stage 1: Build frontend
# -----------------------------------------------------------------------------
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 2: Production image (backend + static frontend)
# -----------------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Install runtime deps for weasyprint (PDF export); use correct Debian package names
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf-2.0-0 libffi-dev shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./
COPY --from=frontend-build /app/frontend/dist ./static

# Create temp dirs
RUN mkdir -p /tmp/datachat /tmp/datachat/exports

ENV PORT=8000
ENV HOST=0.0.0.0

EXPOSE 8000

# Run with uvicorn (no reload in production)
CMD ["sh", "-c", "uvicorn app.main:app --host $HOST --port $PORT"]
