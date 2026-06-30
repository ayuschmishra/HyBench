#!/usr/bin/env bash
# Verify that the PostgreSQL + pgvector environment is correctly configured.
# Run this after `docker compose up -d` to confirm everything is ready.

set -euo pipefail

HOST="${POSTGRES_HOST:-localhost}"
PORT="${POSTGRES_PORT:-5432}"
USER="${POSTGRES_USER:-hybench}"
DB="${POSTGRES_DB:-hybench}"

echo "Waiting for PostgreSQL to be ready..."
until psql -h "$HOST" -p "$PORT" -U "$USER" -d "$DB" -c "SELECT 1;" > /dev/null 2>&1; do
  sleep 1
done

echo "PostgreSQL is ready."

echo "Checking pgvector extension..."
psql -h "$HOST" -p "$PORT" -U "$USER" -d "$DB" \
  -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"

echo "Checking products table..."
psql -h "$HOST" -p "$PORT" -U "$USER" -d "$DB" \
  -c "SELECT COUNT(*) AS row_count FROM products;" 2>/dev/null || \
  echo "Table 'products' not yet populated — run python run_experiments.py first."

echo "Done."
