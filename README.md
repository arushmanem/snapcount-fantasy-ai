# SnapCount

An NFL fantasy football start/sit assistant. Pick two players at the same position, and SnapCount predicts each one's output for a given week, then explains the call in plain language.

The interesting part is not the prediction itself. It is that the first version was a hand-tuned rule engine, and replacing it with a learned model meant building the feature pipeline, the training loop, and the evaluation harness to prove the swap was actually an improvement.

## What it does

- Pulls weekly player statistics and schedules from `nfl_data_py`
- Builds position-specific feature sets for QB, RB, WR, and TE
- Predicts fantasy output with a PyTorch MLP regressor
- Compares two players and returns a start/sit recommendation
- Generates a natural-language rationale via the Gemini API

## Architecture

```
nfl_data_py ─> load + cache ─> position feature builders ─> MLP regressor
                                  (qb / rb / wr / te)          (PyTorch)
                                                                   │
                          FastAPI ◄──────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    v                 v
              React frontend     Gemini API
              (Vite)             (rationale text)
```

### Backend (`backend/`)

| Path | Responsibility |
|---|---|
| `main.py` | FastAPI app: `/weeks`, `/players/{position}`, `/compare`, `/predict` |
| `ml/features/` | Per-position feature engineering (`qb.py`, `rb.py`, `wr.py`, `te.py`) over a shared `base.py` |
| `ml/preprocessing/` | Encoders and scalers, fit on training data only |
| `ml/models/` | `MLPRegressor` and a model registry for versioned checkpoints |
| `ml/build_dataset.py` | Assembles the training set from raw weekly data |
| `ml/train.py` | Training loop with dropout regularization |
| `ml/evaluate.py` | Held-out evaluation |
| `ml/utils/split.py` | Train/test splitting |

Positions are separated deliberately. A quarterback's production and a tight end's production are driven by different signals, and one shared feature vector with mostly-zero columns performed worse than four smaller specialized ones.

### Frontend (`frontend/`)

React and Vite. `PlayerSearch.jsx` handles player lookup and the comparison view.

## Running it

Backend:

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo "GOOGLE_API_KEY=your-key-here" > .env
uvicorn main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Training a model from scratch:

```bash
cd backend
python -m ml.build_dataset
python -m ml.train
python -m ml.evaluate
```

## Design notes

**Why a model instead of rules.** The original version scored matchups with hand-weighted heuristics. Those weights were guesses, and there was no way to tell whether changing one made the tool better or worse. A learned model with a held-out evaluation set made that question answerable.

**Caching.** Weekly data is cached locally rather than refetched per request; `nfl_data_py` pulls are slow and the underlying data only changes once a week.

**Gemini integration.** The LLM writes the explanation, not the prediction. It receives the model's numeric output and the feature context, and its job is to phrase the reasoning. Keeping the numbers out of the LLM's hands means a hallucinated sentence cannot change the recommendation. Retry logic handles API rate limits.

## Limitations and next steps

- No injury or depth-chart signal, which is the largest single omission for a start/sit tool.
- Weather and Vegas lines are both known to be predictive and are not used.
- The model is retrained manually rather than on a schedule.
- No backtest across multiple seasons, so the evaluation is narrower than it should be.
- Frontend has no loading or error states worth the name.

## Stack

Python, FastAPI, PyTorch, Pandas, nfl_data_py, Google Gemini API, React, Vite
