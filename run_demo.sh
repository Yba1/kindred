#!/usr/bin/env bash
# One-shot demo launcher. Run from the repo root: bash run_demo.sh
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "== installing backend deps =="
(cd backend && python -m pip install -q -r requirements.txt)

echo "== installing frontend deps =="
(cd frontend && npm install --silent)

echo "== starting backend on :8000 =="
(cd backend && python -m uvicorn app.main:app --port 8000 > /tmp/kindred_backend.log 2>&1 &)

echo "== starting frontend on :5173 =="
(cd frontend && npm run dev -- --port 5173 > /tmp/kindred_frontend.log 2>&1 &)

echo "== starting village on :4190 =="
(python -m http.server 4190 --directory "$ROOT" > /tmp/kindred_village.log 2>&1 &)

sleep 2
echo ""
echo "Graph:    http://localhost:5173"
echo "Backend:  http://localhost:8000/health"
echo "Village:  http://localhost:4190/viz/village.html"
echo ""
echo "Logs: /tmp/kindred_backend.log /tmp/kindred_frontend.log /tmp/kindred_village.log"
