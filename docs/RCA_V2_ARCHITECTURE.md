# RCA v2 Framework

## Scope and invariants

The framework predicts exactly one boundary label: `L1` (400G endpoint), `L2` (200G endpoint), or `fiber` (inter-endpoint medium). It keeps the legacy experiments intact while isolating the new implementation in `rca_framework/`.

The following invariants are enforced:

1. Source data is never overwritten. A SHA-256 manifest freezes all 366 original files.
2. Only one-400G/one-200G cases enter v2. All side-scoped values move together when the endpoint order is reversed.
3. Training artifacts use training cases only. A target label is removed before feature extraction, retrieval, prompting, and rule matching.
4. The knowledge graph contains root-cause centers and anomaly nodes only. Normal observations never create edges.
5. Each symbolic antecedent is owned by at most one of the L1, L2, and fiber rule sets.

## End-to-end flow

```text
raw case
  -> source checksum + privacy transformation + L1/L2 canonicalization
  -> training-only robust anomaly thresholds
  -> abnormal behavior nodes
       |-> label-centered anomaly KG -> path scoring -> train-case RAG -> LLM path reasoning
       |-> exclusive symbolic rule sets -> rule matching score
  -> agreement completion / confidence-based conflict resolution
  -> L1 | L2 | fiber + evidence + missing-information list + review status
```

## Method 1: KG + RAG + LLM

The three root causes are graph centers. Typed anomaly nouns include signal drop, low/high signal, lane imbalance, device status fault, directional loss, coupled TX/RX fault, and bidirectional loss. An aggregate edge stores training count, root-cause frequency, precision, lift, and weight.

A new case does not become training knowledge. Its anomaly nodes are projected onto paths of the form:

```text
query case -> EXHIBITS -> anomaly noun -> INDICATES -> root cause
```

RAG uses IDF-weighted anomaly overlap against training cases only. The LLM prompt receives the target anomalies, ranked paths, and retrieved training cases, never the target label. With `--backend none`, the same path scores produce a deterministic fallback and the complete LLM prompt is retained for audit.

## Method 2: KG + RCA symbolic matching

The rule learner considers single anomalies and two-anomaly conjunctions. Each antecedent is assigned to only its most discriminative root cause using confidence, lift, support, and an exclusivity margin. This produces three physically separate rule sets with a machine-checked zero-overlap audit.

For a new case, a rule matches only when every `all_of` anomaly is present. Rule strength, matched-rule count, and anomaly coverage determine the symbolic confidence.

## Agreement and conflict policy

When both methods agree, the framework keeps the common label and merges graph paths with matched rules to complete the explanation.

When they disagree:

- a calibrated confidence gap of at least 0.20 lets the stronger method decide;
- otherwise, normalized class evidence is fused with 0.55 KG/RAG/LLM weight and 0.45 symbolic weight;
- if the fused top-two margin is below 0.10, a provisional L1/L2/fiber result is still returned but marked `manual_review_recommended`.

Every result preserves supporting evidence, conflicting evidence, data coverage, and a list of fields whose collection could resolve uncertainty.

## Commands

```bash
# Optional: set this for reproducible pseudonyms across runs.
export RCA_ANONYMIZATION_SECRET='store-this-outside-the-repository'

python -m rca_framework.cli prepare \
  --input-dir data \
  --output-dir datasets/rca_v2 \
  --archive-manifest archive/legacy_exploration/source_data_manifest.json

python -m rca_framework.cli train-evaluate \
  --data-dir datasets/rca_v2 \
  --train-size 200 \
  --output-dir artifacts/rca_v2_baseline \
  --backend none

python -m rca_framework.cli infer \
  --model artifacts/rca_v2_baseline/model \
  --case datasets/rca_v2/case_000268.json \
  --output artifacts/single_case_result.json
```

To enable the actual LLM path reasoner, set `--backend vllm` or `--backend transformers` and pass `--model-path`. Model loading is lazy; the deterministic path baseline has no third-party runtime dependency.
