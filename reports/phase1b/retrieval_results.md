# Phase 1B Retrieval Validity Pilot

> Pilot results only. These are not formal paper main results.

All baselines used the same per-dataset query set, evidence-memory pool, top-k, and 256-token selection budget. QASPER is an internal diagnostic and was never used for training.

## Primary aggregate (SciREX + SciFact)

| Baseline | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.1835 | 0.3196 | 0.3544 | 0.2505 | 0.2668 |
| EviMem_full_retrieval_score | 0.6076 | 0.9367 | 0.9684 | 0.7454 | 0.7991 |
| TF-IDF | 0.3608 | 0.6677 | 0.7310 | 0.4949 | 0.5476 |
| dense_certificate_aware_reranker | 0.6139 | 0.9494 | 0.9747 | 0.7541 | 0.8079 |
| fine_tuned_dense_seed_13 | 0.6139 | 0.9494 | 0.9747 | 0.7541 | 0.8079 |
| fine_tuned_dense_seed_42 | 0.6234 | 0.9462 | 0.9778 | 0.7602 | 0.8133 |
| fine_tuned_dense_seed_97 | 0.6139 | 0.9399 | 0.9715 | 0.7544 | 0.8070 |
| frozen_dense | 0.5443 | 0.9272 | 0.9620 | 0.7030 | 0.7654 |
| frozen_scientific_dense | 0.3987 | 0.7658 | 0.8797 | 0.5621 | 0.6335 |

## Primary aggregate at fixed 256-token budget

| Baseline | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.1835 | 0.3196 | 0.3513 | 0.2414 | 0.2659 |
| EviMem_full_retrieval_score | 0.6076 | 0.9367 | 0.9684 | 0.7442 | 0.7991 |
| TF-IDF | 0.3608 | 0.6677 | 0.7310 | 0.4903 | 0.5476 |
| dense_certificate_aware_reranker | 0.6139 | 0.9494 | 0.9715 | 0.7528 | 0.8069 |
| fine_tuned_dense_seed_13 | 0.6139 | 0.9494 | 0.9715 | 0.7528 | 0.8069 |
| fine_tuned_dense_seed_42 | 0.6234 | 0.9462 | 0.9715 | 0.7588 | 0.8114 |
| fine_tuned_dense_seed_97 | 0.6139 | 0.9399 | 0.9684 | 0.7532 | 0.8060 |
| frozen_dense | 0.5443 | 0.9272 | 0.9620 | 0.7019 | 0.7654 |
| frozen_scientific_dense | 0.3987 | 0.7658 | 0.8766 | 0.5600 | 0.6326 |

## Three-seed fine-tuning variance

```json
{
  "mrr": {
    "mean": 0.7562319558707997,
    "sample_std": 0.003407105476769689,
    "values": {
      "13": 0.7541308971805063,
      "42": 0.7601630289171929,
      "97": 0.7544019415146999
    }
  },
  "ndcg_at_10": {
    "mean": 0.8093946274636231,
    "sample_std": 0.003403046294100565,
    "values": {
      "13": 0.8079220485752179,
      "42": 0.8132859696241617,
      "97": 0.8069758641914897
    }
  },
  "recall_at_1": {
    "mean": 0.6170886075949368,
    "sample_std": 0.005481173441673627,
    "values": {
      "13": 0.6139240506329114,
      "42": 0.6234177215189873,
      "97": 0.6139240506329114
    }
  },
  "recall_at_10": {
    "mean": 0.9746835443037974,
    "sample_std": 0.0031645569620253333,
    "values": {
      "13": 0.9746835443037974,
      "42": 0.9778481012658228,
      "97": 0.9715189873417721
    }
  },
  "recall_at_5": {
    "mean": 0.9451476793248945,
    "sample_std": 0.00483394060649348,
    "values": {
      "13": 0.9493670886075949,
      "42": 0.9462025316455697,
      "97": 0.939873417721519
    }
  }
}
```

## Certificate-aware interpretation

Not estimable in this pilot: the authorized retrieval views have no certificate or memory-type gold. The fail-closed reranker therefore leaves dense scores unchanged, and no certificate labels were fabricated.

Rejected/conflict/certificate-mismatch metrics are null because the authorized retrieval views contain no certificate or memory-type gold. No such labels were fabricated.
