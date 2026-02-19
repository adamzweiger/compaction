#!/bin/bash
#SBATCH --job-name=g-lh5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-48
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

# perplexity_only="1"
MODEL=google/gemma-3-12b-it
DATASET=longhealth5
budget_path="head_budget_optimization/head_budgets/gemma-3-12b-it/optimized_agnostic.json"
chunking=longhealth

log_dir=logs/qa_evaluation/gemma-lh5

# Array: name n_articles start_article compute_stats methods target_size query_config algorithm_config use_budget
configs=(
  "t0.99_summarize      4  0 0 original,all 0.99 ss-plus-repeat summarize"
  "t0.99_orig2          4  0 0 original,no_context 0.99 ss-plus-repeat summarize"
  "t0.99_orig3          4  0 0 original,no_context 0.99 ss-plus-repeat summarize"
  #
  "t0.01_fastest             4  0 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.01 ss-plus-repeat best 1"
  "t0.02_fastest             4  0 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.02 ss-plus-repeat best 1"
  "t0.05_fastest             4  0 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.05 ss-plus-repeat best 1"
  "t0.1_fastest              4  0 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.1 ss-plus-repeat best 1"
  "t0.2_fastest              4  0 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.2 ss-plus-plus best 1"
  #
  "t0.01_fast                4  0 0 omp_nnls0_-inf_7_drop-7_lsq_k4int2_on-policy 0.01 ss-plus-repeat best 1"
  "t0.02_fast                4  0 0 omp_nnls0_-inf_7_drop-7_lsq_k4int2_on-policy 0.02 ss-plus-repeat best 1"
  "t0.05_fast                4  0 0 omp_nnls0_-inf_7_drop-7_lsq_k4int2_on-policy 0.05 ss-plus-repeat best 1"
  "t0.1_fast                 4  0 0 omp_nnls0_-inf_7_drop-7_lsq_k4int2_on-policy 0.1 ss-plus-repeat best 1"
  "t0.2_fast                 4  0 0 omp_nnls0_-inf_7_drop-7_lsq_k4int2_on-policy 0.2 ss-plus-repeat best 1"
  #
  "t0.01_best                4  0 0 omp_nnls0_-inf_7_drop-7_lsq_on-policy 0.01 ss-plus-repeat best 1"
  "t0.02_best                4  0 0 omp_nnls0_-inf_7_drop-7_lsq_on-policy 0.02 ss-plus-repeat best 1"
  "t0.05_best                4  0 0 omp_nnls0_-inf_7_drop-7_lsq_on-policy 0.05 ss-plus-repeat best 1"
  "t0.1_best                 1  0 0 omp_nnls0_-inf_7_drop-7_lsq_on-policy 0.1 ss-plus-repeat best 1"
  "t0.1_best                 1  1 0 omp_nnls0_-inf_7_drop-7_lsq_on-policy 0.1 ss-plus-repeat best 1"
  "t0.1_best                 1  2 0 omp_nnls0_-inf_7_drop-7_lsq_on-policy 0.1 ss-plus-repeat best 1"
  "t0.1_best                 1  3 0 omp_nnls0_-inf_7_drop-7_lsq_on-policy 0.1 ss-plus-repeat best 1"
  "t0.2_best                 1  0 0 omp_nnls0_-inf_7_drop-7_lsq_on-policy 0.2 ss-plus-repeat best 1"
  "t0.2_best                 1  1 0 omp_nnls0_-inf_7_drop-7_lsq_on-policy 0.2 ss-plus-repeat best 1"
  "t0.2_best                 1  2 0 omp_nnls0_-inf_7_drop-7_lsq_on-policy 0.2 ss-plus-repeat best 1"
  "t0.2_best                 1  3 0 omp_nnls0_-inf_7_drop-7_lsq_on-policy 0.2 ss-plus-repeat best 1"
  #
  "t0.01_fastest2            4  0 0 highest_attn_keys_rms_nnls2_-3_3_lsq 0.01 context-prefill best 1"
  "t0.02_fastest2            4  0 0 highest_attn_keys_rms_nnls2_-3_3_lsq 0.02 context-prefill best 1"
  "t0.05_fastest2            4  0 0 highest_attn_keys_rms_nnls2_-3_3_lsq 0.05 context-prefill best 1"
  "t0.1_fastest2             4  0 0 highest_attn_keys_rms_nnls2_-3_3_lsq 0.1 context-prefill best 1"
  "t0.2_fastest2             4  0 0 highest_attn_keys_rms_nnls2_-3_3_lsq 0.2 context-prefill best 1"
  #
  "t0.01_fastest3            4  0 0 highest_attn_keys_rms_nnls2_-3_3_lsq 0.01 repeat best 1"
  "t0.02_fastest3            4  0 0 highest_attn_keys_rms_nnls2_-3_3_lsq 0.02 repeat best 1"
  "t0.05_fastest3            4  0 0 highest_attn_keys_rms_nnls2_-3_3_lsq 0.05 repeat best 1"
  "t0.1_fastest3             4  0 0 highest_attn_keys_rms_nnls2_-3_3_lsq 0.1 repeat best 1"
  "t0.2_fastest3             4  0 0 highest_attn_keys_rms_nnls2_-3_3_lsq 0.2 repeat best 1"
  #
  "t0.01_kvzip               4  0 0 global_highest_attn_keys_max_nobeta_direct 0.01 repeat kvzip 0"
  "t0.02_kvzip               4  0 0 global_highest_attn_keys_max_nobeta_direct 0.02 repeat kvzip 0"
  "t0.05_kvzip               4  0 0 global_highest_attn_keys_max_nobeta_direct 0.05 repeat kvzip 0"
  "t0.1_kvzip                4  0 0 global_highest_attn_keys_max_nobeta_direct 0.1 repeat kvzip 0"
  "t0.2_kvzip                4  0 0 global_highest_attn_keys_max_nobeta_direct 0.2 repeat kvzip 0"
  #
  "t0.01_snapkv         4  0 0 snapkv_mean 0.01 ss-one-question snapkv 0"
  "t0.02_snapkv         4  0 0 snapkv_mean 0.02 ss-one-question snapkv 0"
  "t0.05_snapkv         4  0 0 snapkv_mean 0.05 ss-one-question snapkv 0"
  "t0.1_snapkv          4  0 0 snapkv_mean 0.1 ss-one-question snapkv 0"
  "t0.2_snapkv          4  0 0 snapkv_mean 0.2 ss-one-question snapkv 0"
  #
  "t0.01_h2o            4  0 0 highest_attn_keys_mean_nobeta_direct 0.01 context-prefill kvzip-uniform 0"
  "t0.02_h2o            4  0 0 highest_attn_keys_mean_nobeta_direct 0.02 context-prefill kvzip-uniform 0"
  "t0.05_h2o            4  0 0 highest_attn_keys_mean_nobeta_direct 0.05 context-prefill kvzip-uniform 0"
  "t0.1_h2o             4  0 0 highest_attn_keys_mean_nobeta_direct 0.1 context-prefill kvzip-uniform 0"
  "t0.2_h2o             4  0 0 highest_attn_keys_mean_nobeta_direct 0.2 context-prefill kvzip-uniform 0"
)

# Select configuration based on SLURM array task ID
config="${configs[$SLURM_ARRAY_TASK_ID]}"
read -r name n_articles start_article compute_stats methods target_size query_config algorithm_config use_budget <<< "$config"

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
  budget_flag="--precomputed-budget-path $budget_path --max-ratio-per-head 0.75"
fi

chunking_flag=""
batch_size_flag=""
if [ -n "$chunking" ]; then
  chunking_flag="--chunking $chunking"
  batch_size_flag="--batch-size 10"
fi

log_dir_flag=""
if [ -n "$log_dir" ]; then
  log_dir_flag="--log-dir $log_dir"
fi

methods_formatted=$(echo "$methods" | tr ',' ' ')

start_time=$(date +%s)
echo "Start time: $(date -d @$start_time)"
echo "Running on node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u -m evaluation.run_qa_evaluation --name "$name" $dataset_flag --n-articles "$n_articles" --start-article "$start_article" --compute-stats "$compute_stats" --methods $methods_formatted --target-size "$target_size" --query-config "$query_config" --algorithm-config "$algorithm_config" $model_flag $perplexity_only_flag $budget_flag $chunking_flag $batch_size_flag $log_dir_flag

end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
