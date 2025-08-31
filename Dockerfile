# Multi-stage build: Frontend first, then backend
FROM node:20-alpine AS frontend

WORKDIR /app/frontend
COPY web-app/package*.json ./
RUN npm ci
COPY web-app/ ./
RUN npm run build

# Backend stage
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ ./backend/
COPY models/ ./models/
COPY config/ ./config/

# Copy built frontend from previous stage
COPY --from=frontend /app/frontend/dist ./web-app/dist/

# Environment variables
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

EXPOSE 5001

CMD ["python", "backend/app.py"]
