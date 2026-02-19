# Compaction

Code for [Fast KV Compaction via Attention Matching](https://arxiv.org/abs/2602.16284). Attention Matching (AM) compacts a KV cache in latent space by constructing a smaller set of keys and values that reproduce the original attention behavior.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ad98f32b-dfc6-43d1-84c4-698507b68a2b" alt="Attention Matching" width="600">
</p>

## Repository Layout

| Path | Purpose |
| --- | --- |
| `compaction/` | Core methods (`compaction_methods/`, `algorithms/`), chunking strategies, and query generation helpers. |
| `evaluation/` | `run_qa_evaluation.py`, `run_reasoning_evaluation.py`, configs, and scoring utilities. |
| `scripts/` | Shell entry points for main experiments plus plotting and aggregation helpers. |
| `head_budget_optimization/` | Tools for computing nonuniform head budgets for models. |
| `models/` | Model-specific generation and caching utilities (Qwen3, Llama, Gemma3). |
| `examples/` | Demo scripts. |
| `data/` | Dataset and cached reference generation artifacts. |

### Try Compaction

```bash
python -m examples.qa_demo --model Qwen/Qwen3-4B --target-size 0.1
```

This prefills a short article, compacts its KV cache to 10% with Attention Matching, and compares QA accuracy before and after. See [`examples/qa_demo.py`](examples/qa_demo.py) for details.

### Evaluate QA Tasks
Run the evaluator with one or more compaction methods and datasets:

```bash
python -m evaluation.run_qa_evaluation \
  --algorithm-config default \
  --methods original AM-HighestAttnKeys \
  --dataset-name quality \
  --n-articles 1 \
  --compute-stats 1
```

For
non-uniform budgets, point `--precomputed-budget-path` to one of the JSON files
under [`head_budget_optimization/head_budgets/`](head_budget_optimization/head_budgets/).


## Development Status

We might continue developing this into a more polished, installable package and are open to contributions. If you're interested in collaborating, feel free to open an issue or PR. See [`TODO`](TODO) for current plans.

## Citation

If you found this work useful, please cite:

```
@misc{zweiger2026fastkvcompactionattention,
      title={Fast {KV} Compaction via {Attention Matching}}, 
      author={Adam Zweiger and Xinghong Fu and Han Guo and Yoon Kim},
      year={2026},
      eprint={2602.16284},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.16284}, 
}
```
