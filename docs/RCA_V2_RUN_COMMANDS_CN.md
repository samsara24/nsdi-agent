# RCA v2 代码运行手册

所有命令均从项目根目录执行：

```bash
cd /home/shibinpeng/luoyu/huangzeshun/nsdi27
```

## `rca_framework/cli.py`

查看帮助：

```bash
python -m rca_framework.cli --help
```

### 生成脱敏数据集

```bash
python -m rca_framework.cli prepare \
  --input-dir data \
  --output-dir datasets/rca_v2_new \
  --archive-manifest archive/legacy_exploration/source_data_manifest_new.json
```

需要让不同运行产生一致的匿名 ID 时：

```bash
RCA_ANONYMIZATION_SECRET='your-fixed-secret' \
python -m rca_framework.cli prepare \
  --input-dir data \
  --output-dir datasets/rca_v2_new \
  --archive-manifest archive/legacy_exploration/source_data_manifest_new.json
```

`--output-dir` 必须使用尚不存在的新目录。

### 不使用大模型训练并评估

```bash
python -m rca_framework.cli train-evaluate \
  --data-dir datasets/rca_v2 \
  --train-size 200 \
  --output-dir artifacts/rca_v2_baseline_new \
  --backend none
```

`--output-dir` 必须为空或尚不存在。

### 使用 vLLM 训练并评估

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python -m rca_framework.cli train-evaluate \
  --data-dir datasets/rca_v2 \
  --train-size 200 \
  --output-dir artifacts/rca_v2_vllm_new \
  --backend vllm \
  --model-path /path/to/local/model \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-new-tokens 512
```

纯 PCIe 多卡环境如果 custom all-reduce 初始化失败，再增加：

```bash
  --enforce-eager \
  --disable-custom-all-reduce
```

### 使用 Transformers 训练并评估

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m rca_framework.cli train-evaluate \
  --data-dir datasets/rca_v2 \
  --train-size 200 \
  --output-dir artifacts/rca_v2_transformers_new \
  --backend transformers \
  --model-path /path/to/local/model \
  --max-new-tokens 512
```

### 推理一条 case

不使用大模型：

```bash
python -m rca_framework.cli infer \
  --model artifacts/rca_v2_baseline/model \
  --case datasets/rca_v2/case_000268.json \
  --output artifacts/single_case_result.json \
  --backend none
```

使用大模型：

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m rca_framework.cli infer \
  --model artifacts/rca_v2_baseline/model \
  --case datasets/rca_v2/case_000268.json \
  --output artifacts/single_case_llm_result.json \
  --backend vllm \
  --model-path /path/to/local/model
```

## `scripts/run_rca_v2.py`

它与 `python -m rca_framework.cli` 等价。

```bash
python scripts/run_rca_v2.py --help
```

生成数据：

```bash
python scripts/run_rca_v2.py prepare \
  --input-dir data \
  --output-dir datasets/rca_v2_new \
  --archive-manifest archive/legacy_exploration/source_data_manifest_new.json
```

训练评估：

```bash
python scripts/run_rca_v2.py train-evaluate \
  --data-dir datasets/rca_v2 \
  --train-size 200 \
  --output-dir artifacts/rca_v2_baseline_new \
  --backend none
```

单条推理：

```bash
python scripts/run_rca_v2.py infer \
  --model artifacts/rca_v2_baseline/model \
  --case datasets/rca_v2/case_000268.json
```

## `scripts/prepare_rca_v2_dataset.py`

```bash
python scripts/prepare_rca_v2_dataset.py \
  --input-dir data \
  --output-dir datasets/rca_v2_new \
  --archive-manifest archive/legacy_exploration/source_data_manifest_new.json
```

## `scripts/debug_rca_case.py`

只查看异常提取：

```bash
python scripts/debug_rca_case.py \
  --model artifacts/rca_v2_baseline/model \
  --case datasets/rca_v2/case_000268.json \
  --channel evidence
```

只查看 KG-RAG：

```bash
python scripts/debug_rca_case.py \
  --model artifacts/rca_v2_baseline/model \
  --case datasets/rca_v2/case_000268.json \
  --channel kg-rag \
  --backend none
```

只查看符号规则：

```bash
python scripts/debug_rca_case.py \
  --model artifacts/rca_v2_baseline/model \
  --case datasets/rca_v2/case_000268.json \
  --channel kg-rca
```

运行完整双路推理与融合：

```bash
python scripts/debug_rca_case.py \
  --model artifacts/rca_v2_baseline/model \
  --case datasets/rca_v2/case_000268.json \
  --channel full \
  --backend none \
  --output artifacts/debug_case_result.json
```

## `scripts/repair_l2_lane_width.py`

```bash
python scripts/repair_l2_lane_width.py \
  datasets/rca_v2 \
  datasets/rca_v2_l2_repaired
```

第二个位置参数必须是尚不存在的新目录。

## `tests/`

运行全部测试：

```bash
pytest -q
```

分别运行每个测试文件：

```bash
pytest -q tests/test_data_pipeline.py
pytest -q tests/test_graph_rules.py
pytest -q tests/test_pipeline_and_fusion.py
```

运行单个测试函数：

```bash
pytest -q tests/test_graph_rules.py::test_three_symbolic_rule_sets_are_disjoint
```

查看全部测试名称：

```bash
pytest --collect-only -q
```

## `rca_framework` 其余模块

`__init__.py`、`types.py`、`data.py`、`anomaly.py`、`graph.py`、`rules.py`、`llm.py`、`fusion.py` 和 `pipeline.py` 没有独立命令行入口，不使用 `python rca_framework/rules.py` 这种方式运行。

在 Python 中运行完整 pipeline：

```python
from pathlib import Path
from rca_framework.data import load_cases
from rca_framework.pipeline import RCAPipeline

cases = load_cases(Path("datasets/rca_v2"))
pipeline = RCAPipeline().fit(cases[:200])
result = pipeline.infer(cases[200], llm_backend="none")
print(result)
```

保存和加载模型：

```python
from pathlib import Path
from rca_framework.pipeline import RCAPipeline

pipeline.save(Path("artifacts/my_model"))
loaded = RCAPipeline.load(Path("artifacts/my_model"))
```

单独导入各模块：

```python
from rca_framework.types import Anomaly, CaseEvidence
from rca_framework.anomaly import fit_thresholds, extract_evidence
from rca_framework.graph import AnomalyKnowledgeGraph
from rca_framework.rules import SymbolicRuleEngine
from rca_framework.llm import PathLLMReasoner
from rca_framework.fusion import fuse_results
```

## 第一次完整运行

```bash
# 1. 测试
pytest -q

# 2. 生成数据
python -m rca_framework.cli prepare \
  --input-dir data \
  --output-dir datasets/rca_v2_new \
  --archive-manifest archive/legacy_exploration/source_data_manifest_new.json

# 3. 训练并评估
python -m rca_framework.cli train-evaluate \
  --data-dir datasets/rca_v2_new \
  --train-size 200 \
  --output-dir artifacts/rca_v2_first_run \
  --backend none

# 4. 单条推理
python -m rca_framework.cli infer \
  --model artifacts/rca_v2_first_run/model \
  --case datasets/rca_v2_new/case_000268.json \
  --output artifacts/rca_v2_first_run/single_case_result.json
```
