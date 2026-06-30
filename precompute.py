#!/usr/bin/env python3
"""
precompute.py — build the cached embedding artifacts the ranker depends on (offline step).

Encoding 100k candidates with bge-small cannot fit the 5-min CPU ranking budget, so it is done
here once, offline, where GPU/MPS and a one-time model download are allowed. rank.py then only
loads these arrays (a dot product) and never touches a model or the network.

    python precompute.py --candidates ./data/candidates.jsonl --artifacts ./artifacts

Outputs:
    artifacts/cand_emb_<L>.npy   (100000 x 384 float32, L2-normalized)
    artifacts/cand_ids.json      (candidate_id order matching the rows above)
    artifacts/jd_emb_<L>.npy     (384-dim JD query embedding)
"""
import argparse
import json
import os
import time

import numpy as np

import ranker_core as rc


def main():
    ap = argparse.ArgumentParser(description="Precompute bge embeddings for the ranker.")
    ap.add_argument('--candidates', default='data/candidates.jsonl',
                    help='Path to candidates .jsonl or .jsonl.gz')
    ap.add_argument('--artifacts', default='artifacts', help='Output directory for artifacts')
    ap.add_argument('--model', default=rc.EMB_MODEL)
    ap.add_argument('--max-seq-length', type=int, default=rc.MAX_SEQ_LENGTH)
    ap.add_argument('--device', default=None, help='mps | cuda | cpu (auto-detected if omitted)')
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    device = args.device or ('mps' if torch.backends.mps.is_available()
                             else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = args.max_seq_length
    L = args.max_seq_length
    print(f'device: {model.device} | max_seq_length: {L} | model: {args.model}')

    os.makedirs(args.artifacts, exist_ok=True)

    # JD query embedding (bge wants the retrieval-query prefix, already in JD_QUERY).
    jd_emb = model.encode(rc.JD_QUERY, normalize_embeddings=True)
    np.save(os.path.join(args.artifacts, f'jd_emb_{L}.npy'), jd_emb)
    print(f'JD embedded: {jd_emb.shape}')

    cand_ids, texts = [], []
    for c in rc.iter_candidates(args.candidates):
        cand_ids.append(c['candidate_id'])
        texts.append(rc.work_text(c))

    bs = 256 if model.device.type in ('mps', 'cuda') else 64
    print(f'encoding {len(texts)} candidates on {model.device} (one-time)...')
    t = time.time()
    cand_emb = model.encode(texts, normalize_embeddings=True, batch_size=bs, show_progress_bar=True)
    print(f'encoded in {round(time.time() - t)}s')

    np.save(os.path.join(args.artifacts, f'cand_emb_{L}.npy'), cand_emb)
    json.dump(cand_ids, open(os.path.join(args.artifacts, 'cand_ids.json'), 'w'))
    print(f'wrote cand_emb_{L}.npy {cand_emb.shape}, cand_ids.json, jd_emb_{L}.npy')


if __name__ == '__main__':
    main()
