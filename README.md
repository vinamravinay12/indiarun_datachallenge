# IndiaRun — Intelligent Candidate Discovery & Ranking

Solution for the Redrob **Intelligent Candidate Discovery & Ranking Challenge**: rank the top 100
of 100,000 candidates against the Senior AI Engineer JD — seeing **semantic fit beyond keywords**,
integrating profile / career / behavioral signals, and excluding the planted **honeypot** candidates.

## Approach — a hybrid ranker

Structured evidence **gates** domain fit; the embedding model **fine-ranks**; behavioral and
logistics signals **modulate** availability. Honeypots and disqualified profiles are removed first.

```
fit        = 0.75 · struct_norm + 0.25 · sem_norm
base_final = fit · behavioral · logistics · early_band_mult        # honeypots/disqualified excluded
final      = 0.95 · base_norm(top-150) + 0.05 · assessment_refine   # top-150 → top-100
```

- **Structured fit** (`struct_score`) — explicit JD-derived weights: experience band (peaks at the
  JD's ideal 6–8 yrs), applied-ML years, strong-Python evidence, verified Redrob IR assessment,
  production-retrieval / vector-infra / ranking-evaluation prose, ML title, product-company context,
  graded location. Hard **disqualifiers** (→ tier 0): under-min experience, consulting-only career,
  non-eng title without ML, CV/speech-without-NLP, pure-research-without-production, impossible YOE.
  **Penalties**: title-chasing, framework-demo-only, recent-LLM-only, closed-source-no-validation,
  CV-primary-with-weak-IR.
- **Semantic fit** — `bge-small-en-v1.5` cosine of the JD query vs each candidate's **career prose**
  (summary + titles + descriptions, *not* the noise skill list). Embeddings are precomputed + cached.
- **Behavioral multiplier** (≈0.19–0.93) — 6 redrob signals: recruiter-response rate, recency,
  interview-completion, response-speed, open-to-work, offer-acceptance (JD directive #3).
- **Logistics multiplier** — India eligibility, JD-city / relocation, notice period; no visa sponsorship.
- **Assessment refinement + gates** — a 40% verified-assessment pass floor; missing scores imputed
  from structured strength; a stricter hard gate for the 4.0–4.5 early-career band.
- **Honeypot exclusion** — an impossibility battery (skill/experience anachronisms, company-founded-
  after dates, overlapping jobs, expert-skill-with-0-months, tenure/YOE contradictions). Rare
  contradictions = deliberate plants and are dropped before ranking (0 of ~80 reach the top-100).

## Repo layout

| Path | What |
|------|------|
| `rank.py` | **Canonical reproducer** — the ≤5-min CPU/no-network step that produces `submission.csv` |
| `precompute.py` | Offline embedding generation (the slow step that may exceed 5 min) |
| `ranker_core.py` | Single source of truth — constants, features, honeypot battery, scoring, blend, reasoning |
| `requirements.txt` | Pinned dependencies |
| `notebooks/explore.ipynb` | **Design record** — feature extraction, honeypot analysis, structured scoring, diagnostics |
| `notebooks/ranker.ipynb` | **Design record** — semantic encode, hybrid blend, top-100 audit |
| `artifacts/` | Cached embeddings + generated outputs (git-ignored; embeddings shipped via Git LFS) |

`rank.py` is authoritative for reproduction; the notebooks are the exploration/methodology record and
are kept in sync (`rank.py` reproduces the notebook `submission.csv` byte-for-byte).

## Reproduce the submission

The 487 MB candidate pool is supplied separately; symlink or copy it to `data/candidates.jsonl`
(`.jsonl` or `.jsonl.gz` both work).

```bash
pip install -r requirements.txt

# (one-time, offline — encodes 100k candidates with bge-small; GPU/MPS used if available)
python precompute.py --candidates ./data/candidates.jsonl --artifacts ./artifacts

# the ranking step — produces the submission
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
```

If the cached embedding artifacts are already present (`artifacts/cand_emb_256.npy`,
`cand_ids.json`, `jd_emb_256.npy`), **skip `precompute.py`** — `rank.py` recomputes every structured
/ behavioral / blend score live and only loads the embeddings.

### Compute-constraint compliance (spec §3)

| Constraint | Limit | This solution |
|---|---|---|
| Ranking runtime | ≤ 5 min | **~34 s** (`rank.py`) |
| Memory | ≤ 16 GB | well under |
| Compute | CPU only | `rank.py` is CPU-only (embeddings precomputed) |
| Network | off | `rank.py` makes no network/API calls |

Only embedding *pre-computation* uses a model/GPU and may exceed 5 min — explicitly permitted by the
spec. The ranking step that emits the CSV is pure CPU feature-extraction + a dot-product blend.

## Dependencies

`rank.py` needs only `numpy` + `pandas`. `precompute.py` additionally needs `torch` +
`sentence-transformers` (one-time embedding build). Versions pinned in `requirements.txt`.
