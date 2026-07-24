# Running Kindred on a new computer

**Nothing below is required to run the app.** Every sponsor integration has a
built-in fallback, so `git clone` + the three commands in each section below
gets you a fully working demo with zero keys. The keys only upgrade specific
pieces from a local fallback to the real sponsor service. Skip straight to
"Just running it" if you don't need those upgrades.

## Just running it (no keys, ~2 minutes)

```bash
# backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --port 8000        # -> http://localhost:8000

# frontend (separate terminal)
cd frontend
npm install
npm run dev -- --port 5173              # -> http://localhost:5173

# village (separate terminal, needs no backend at all)
python -m http.server 4190              # from the repo root
# -> http://localhost:4190/viz/village.html
```

`GET /health` on the backend tells you exactly which mode everything is
running in — check that instead of guessing:
```bash
curl -s localhost:8000/health
# {"status":"ok","profiler":"heuristic","embeddings":"hash-fallback","actian":"numpy","gemini":false,"people":30}
```

## Upgrading to the real sponsor services

| Sponsor | What it upgrades | Status if you skip it |
|---|---|---|
| **Gemini** | Profiler (proper reasoning instead of regex) + embeddings (real semantic vectors instead of hashing) | Runs on a real regex/keyword heuristic + deterministic hash embeddings. Fully functional, just less nuanced. |
| **Actian** | Real vector-store retrieval instead of an in-process index | Runs on an in-memory numpy cosine index. Functionally identical for a 30-person demo dataset — this only matters at real scale. |
| **Pioneer** | The match scorer's training/inference path | `pioneer/` runs entirely locally already (local logistic-regression-style scorer, real F1 numbers vs baseline) — the key only matters if you want it hitting the actual Pioneer/Fastino hosted API instead of the local head. |
| **BAND** | Nothing yet | No code exists for this anywhere in the repo. Setting the key does nothing until someone builds the Introducer's actual BAND client. |

### Gemini — three things, not just the key

Setting `GEMINI_API_KEY` alone is not enough; all three of these have to be true:

1. **Put the key in `backend/.env`** (gitignored — this never comes through
   `git pull`, every machine needs its own copy):
   ```
   GEMINI_API_KEY=your-key-here
   ```
2. **Install the package** — it's commented out in `backend/requirements.txt`
   by default:
   ```bash
   pip install google-generativeai
   ```
   Skip this and the key does nothing — the code catches the ImportError
   silently and falls back to the heuristic with no error message.
3. **Run uvicorn from inside `backend/`** (or wherever `.env` actually sits).
   `load_dotenv()` only searches the current directory and its *parents*, not
   subfolders — launch from the repo root instead and it won't find the file.

Verify with `/health` afterward — `"gemini": true` is the only real proof it
worked; don't assume from "I set the key."

### Actian

```
ACTIAN_HOST=...
ACTIAN_PORT=...
ACTIAN_DB=...
ACTIAN_USER=...
ACTIAN_PASSWORD=...
```
in `backend/.env`. **Nobody has wired the real client yet** — `backend/app/actian.py`
has a `_ActianBackend` class with `_connect`/`_upsert`/`_query` stubbed out for
whoever picks this up; until then, any connection error (or no host set at all)
transparently falls back to the numpy index.

### Pioneer

```
PIONEER_API_KEY=pio_sk_...
```
in `pioneer/.env`. Note: `pioneer/` has no `python-dotenv` dependency, so this
file is **not auto-loaded** — either `export PIONEER_API_KEY=...` in your shell
before running anything in `pioneer/`, or add `python-dotenv` + a `load_dotenv()`
call to whichever module reads it first.

## Running the evolution loop

No keys needed — pure stdlib, deterministic:
```bash
python -m loop.run                       # regenerates run.json
python -m loop.run --replay run.json     # replays the last run frame-for-frame
python -m unittest tests.test_loop       # 9 tests
```

## Sanity-checking everything at once

```bash
python scripts/integration_check.py
```
Validates the graph payload contract, `run.json`'s schema, every phase-card
file's dialogue rules, and the SSE agent-event shape — run this after any
merge, not just before the demo.
