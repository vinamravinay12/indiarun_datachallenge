# Notebooks

These notebooks are the canonical design and audit record for the ranker.

- `explore.ipynb` documents candidate schema inspection, JD-derived feature design, honeypot analysis, structured scoring, and behavioral/logistics diagnostics.
- `ranker.ipynb` documents semantic embedding experiments, hybrid ranking calibration, top-100 audits, and submission diagnostics.

For final reproduction, use the root pipeline:

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
```

`rank.py` and `ranker_core.py` are the source of truth for the generated CSV. The notebooks are kept as methodology evidence for review and interview discussion, not as the Stage-3 execution path.