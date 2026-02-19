#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-14
#SBATCH --partition=mit_preemptable
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

perplexity_only="1"
# MODEL=meta-llama/Llama-3.1-8B-Instruct
# DATASET=lqa32k
budget_path="head_budget_optimization/head_budgets/Qwen3-4B/optimized_agnostic.json"

# Array: name n_articles start_article compute_stats methods target_size query_config algorithm_config use_budget ignore_article_indices
configs=(
  "t0.01_ablation     10  0 1 all 0.01 ss-plus-repeat ablation 1 0"
  "t0.02_ablation     10  0 1 all 0.02 ss-plus-repeat ablation 1 0"
  "t0.05_ablation     10  0 1 all 0.05 ss-plus-repeat ablation 1 0"
  "t0.1_ablation      10  0 1 all 0.1  ss-plus-repeat ablation 1 0"
  #
  "t0.01_ablation     10  0 1 omp_nnls0_-inf_7_drop-7_lsq_on-policy 0.01 repeat ablation 1 0"
  "t0.02_ablation     10  0 1 omp_nnls0_-inf_7_drop-7_lsq_on-policy 0.02 repeat ablation 1 0"
  "t0.05_ablation     10  0 1 omp_nnls0_-inf_7_drop-7_lsq_on-policy 0.05 repeat ablation 1 0"
  "t0.1_ablation      10  0 1 omp_nnls0_-inf_7_drop-7_lsq_on-policy 0.1  repeat ablation 1 0"
  #
  "t0.01_ablation     10  0 1 omp_nnls0_-inf_7_drop-7_lsq_on-policy 0.01 ss-plus-repeat ablation 0 0"
  "t0.02_ablation     10  0 1 omp_nnls0_-inf_7_drop-7_lsq_on-policy 0.02 ss-plus-repeat ablation 0 0"
  "t0.05_ablation     10  0 1 omp_nnls0_-inf_7_drop-7_lsq_on-policy 0.05 ss-plus-repeat ablation 0 0"
  "t0.1_ablation      10  0 1 omp_nnls0_-inf_7_drop-7_lsq_on-policy 0.1  ss-plus-repeat ablation 0 0"
  #
  "t0.01_ignore-article-idx   10  0 1 omp_nnls0_-inf_7_drop-7_lsq_progressive_on-policy 0.01  ss-plus-repeat ablation 1 1"
  "t0.02_ignore-article-idx   10  0 1 omp_nnls0_-inf_7_drop-7_lsq_progressive_on-policy 0.02  ss-plus-repeat ablation 1 1"
  "t0.05_ignore-article-idx   10  0 1 omp_nnls0_-inf_7_drop-7_lsq_progressive_on-policy 0.05  ss-plus-repeat ablation 1 1"
  "t0.1_ignore-article-idx      10  0 1 omp_nnls0_-inf_7_drop-7_lsq_on-policy 0.1  ss-plus-repeat ablation 1 1"
)

# Select configuration based on SLURM array task ID
config="${configs[$SLURM_ARRAY_TASK_ID]}"
read -r name n_articles start_article compute_stats methods target_size query_config algorithm_config use_budget ignore_article_indices <<< "$config"

model_flag=""
if [ -n "$MODEL" ]; then
  model_flag="--model-name $MODEL"
fi

dataset_flag=""
if [ -n "$DATASET" ]; then
  dataset_flag="--dataset-name $DATASET"
fi

perplexity_only_flag=""
if [ -n "$perplexity_only" ]; then
  perplexity_only_flag="--perplexity-only $perplexity_only"
fi

budget_flag=""
if [ -n "$budget_path" ] && [ "$use_budget" = "1" ]; then
  budget_flag="--precomputed-budget-path $budget_path"
fi

ignore_article_indices_flag=""
if [ "$ignore_article_indices" = "1" ]; then
  ignore_article_indices_flag="--ignore-article-indices"
fi

methods_formatted=$(echo "$methods" | tr ',' ' ')

python -u -m evaluation.run_qa_evaluation --name "$name" $dataset_flag --n-articles "$n_articles" --start-article "$start_article" --compute-stats "$compute_stats" --methods $methods_formatted --target-size "$target_size" --query-config "$query_config" --algorithm-config "$algorithm_config" $model_flag $perplexity_only_flag $budget_flag $ignore_article_indices_flag

end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
