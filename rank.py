#!/usr/bin/env python3
"""
rank.py - produce the top-100 submission CSV from candidates.jsonl.

This is the constrained ranking step (challenge spec section 3): <=5 min wall-clock,
<=16 GB RAM, CPU-only, no network. It:
  1. loads the PRECOMPUTED embedding artifacts (see precompute.py),
  2. rebuilds structured features + honeypot flags + scores LIVE from candidates.jsonl,
  3. runs the hybrid blend (structured x semantic x behavioral x logistics, gated),
  4. writes submission.csv (and optionally top100.json).

    python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

Only the embeddings are precomputed (encoding 100k candidates can't fit the 5-min CPU budget);
every score is recomputed here, so the CSV genuinely reproduces from the candidates file.
"""
import argparse
import csv
import json
import os
import time

try:
    import numpy as np
except ImportError:
    import sys
    print("Error: numpy is not installed. Install it with: pip install numpy", file=sys.stderr)
    sys.exit(1)

import ranker_core as rc


def main():
    ap = argparse.ArgumentParser(description="Rank top-100 candidates for the Redrob JD.")
    ap.add_argument('--candidates', default='data/candidates.jsonl',
                    help='Path to candidates .jsonl or .jsonl.gz')
    ap.add_argument('--out', default='submission.csv', help='Output submission CSV path')
    ap.add_argument('--artifacts', default='artifacts',
                    help='Dir holding cand_emb_<L>.npy, cand_ids.json, jd_emb_<L>.npy')
    ap.add_argument('--top100-json', default=None,
                    help='Optional path to also write the rich top100.json')
    args = ap.parse_args()

    t0 = time.time()
    L = rc.MAX_SEQ_LENGTH
    cand_emb = np.load(os.path.join(args.artifacts, f'cand_emb_{L}.npy'))
    cand_ids = json.load(open(os.path.join(args.artifacts, 'cand_ids.json')))
    jd_emb = np.load(os.path.join(args.artifacts, f'jd_emb_{L}.npy'))
    print(f'loaded embeddings {cand_emb.shape} | ids {len(cand_ids)} | jd {jd_emb.shape}')

    # Live: features + honeypot battery + structured scoring + behavioral multiplier.
    df, hp_ids = rc.build_feature_frame(rc.iter_candidates(args.candidates))
    print(f'features {df.shape} | honeypots {int(df.honeypot.sum())}')
    feat = df.set_index('candidate_id')

    top = rc.rank(feat, cand_emb, cand_ids, jd_emb)
    print(f'top-100 selected | final {top.final.min():.4f}..{top.final.max():.4f}')

    # Second streaming pass: pull the 100 raw records for reasoning text.
    top_ids = set(top.index)
    recs = {c['candidate_id']: c for c in rc.iter_candidates(args.candidates)
            if c['candidate_id'] in top_ids}

    out, submission_rows = rc.build_outputs(top, recs)

    with open(args.out, 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=['candidate_id', 'rank', 'score', 'reasoning'])
        w.writeheader()
        w.writerows(submission_rows)
    print(f'wrote {args.out} ({len(submission_rows)} rows)')

    if args.top100_json:
        json.dump(out, open(args.top100_json, 'w'), indent=2)
        print(f'wrote {args.top100_json}')

    print(f'done in {time.time() - t0:.1f}s (ranking step)')


if __name__ == '__main__':
    main()
