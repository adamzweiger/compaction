#!/bin/bash
#SBATCH --job-name=head-budget-opt
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-9
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=6:00:00
#SBATCH --requeue

# -------- Environment ------------------------------------------------ #
export HOME=/home/$USER
source ~/.bashrc
cd ~/compaction
conda activate compaction

start_time=$(date +%s)

# Each array task processes one article
# Articles 90-99
ARTICLE_IDX=$((90 + SLURM_ARRAY_TASK_ID))

python -u -m head_budget_optimization.run \
  --baseline-schedule head_budget_optimization/head_budgets/gemma-3-12b-it/uniform.json \
  --target-ratio 0.05 \
  --model-name google/gemma-3-12b-it \
  --dataset-name "quality" \
  --n-articles 1 \
  --start-article "$ARTICLE_IDX" \
  --n-eval-points 51 \
  --max-ratio 0.5 \
  --solve-ratios 0.01,0.02,0.05,0.1 \
  --algorithm-config best \
  --method highest_attn_keys_rms_nnls2_-3_3_lsq \
  --query-config ss-plus-repeat \
  --skip-solve

end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
