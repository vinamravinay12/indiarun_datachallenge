# IndiaRun — Intelligent Candidate Discovery & Ranking

Solution for the Redrob "Intelligent Candidate Discovery & Ranking Challenge": rank the
top 100 of 100,000 candidates against a Senior AI Engineer job description, seeing
**semantic fit beyond keywords**, integrating profile / career / behavioral signals, and
excluding the planted **honeypot** candidates.

## Approach

A **hybrid ranker** — structured evidence gates domain fit, semantics fine-rank:

```
final = (0.6 * struct_norm + 0.4 * sem_norm) * behavioral_multiplier   # honeypots excluded
```

- **Structured fit** — weighted evidence: experience band, applied-ML years, verified Redrob
  assessment scores, retrieval/eval prose evidence, ML title, product-company, graded
  location; penalties for recent-LLM-only and title-chasing; disqualifiers → 0.
- **Semantic fit** — `bge-small-en-v1.5` cosine of the JD query vs each candidate's **career
  prose** (summary + titles + descriptions, not the noise skill list). Encoded offline + cached.
- **Behavioral multiplier** (0.45–1.0) — response rate, recency, open-to-work (JD directive #3).
- **Honeypot exclusion** — impossibility checks (skill/experience anachronisms, overlapping
  jobs, experience > tool age, expert-with-0-months, etc.); rare contradictions = deliberate
  plants and are dropped before ranking.

## Repo layout

| Path | What |
|------|------|
| `explore.ipynb` | Phase 1 — feature extraction, honeypot detection, structured scoring |
| `ranker.ipynb`  | Phase 2 — semantic encode, hybrid blend, top-100 + audit |

## Running

The 487MB `candidates.json` is supplied separately and symlinked at `data/candidates.jsonl`
(gitignored). Generated artifacts (embeddings, features, exports) land in `artifacts/`
(gitignored). Run `explore.ipynb` first, then `ranker.ipynb`.

Requires: `pandas`, `numpy`, `torch`, `sentence-transformers`, `pyarrow`.
