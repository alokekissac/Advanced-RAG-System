| Retriever | Keyword Qs Hit@1 | Hit@3 | MRR@10 | Paraphrased Qs Hit@1 | Hit@3 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 (sparse) | 100.0% | 100.0% | 1.000 | 53.3% | 73.3% | 0.630 |
| Dense (LSA embeddings) | 95.7% | 100.0% | 0.978 | 46.7% | 66.7% | 0.580 |
| **Hybrid (RRF + MMR)** | 97.8% | 100.0% | 0.989 | 46.7% | 66.7% | 0.578 |

Abstention threshold chosen on dev split: **0.65**

| Test split | Value |
|---|---:|
| Answer accuracy (answerable questions) | 95.7% (22/23) |
| Correct abstentions (unanswerable) | 100.0% (6/6) |
| False abstentions | 4.3% |
| Overall accuracy | 96.6% |
| Cited sentences supported by their source | 100.0% |

Paraphrased questions (extractive answers): 6.7% correct, 86.7% abstained.
