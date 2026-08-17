# Expanded dataset contract

```json
{
  "old_train_path": "datasets/organized_rca_v2_stratified_60_40_seed42",
  "old_train_range": "first 126 positional cases; cleaned to 122",
  "old_test_path": "datasets/organized_rca_v2_stratified_60_40_seed42",
  "old_test_range": "remaining 85 positional cases",
  "new_dataset_path": "datasets/all_data_rca_v2_stratified_60_40_seed42",
  "added_count": 258,
  "expanded_test_path": "experiments/20260816_expanded-pattern-conflict/clean_expanded_test.jsonl",
  "clean_train_path": "experiments/20260816_expanded-pattern-conflict/clean_train.jsonl",
  "data_contract_path": "experiments/20260816_expanded-pattern-conflict/data_contract.json",
  "deduplication_basis": "physical-content-sha256-v1",
  "cleanup": "removed 6 old cases absent from new dataset",
  "warning": "new dataset is not a strict superset: 6 old cases were not found"
}
```
