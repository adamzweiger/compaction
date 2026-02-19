#!/bin/bash
#SBATCH --job-name=q-r
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-2
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=6:00:00
#SBATCH --requeue

# -------- Environment ------------------------------------------------ #
export HOME=/home/$USER
source ~/.bashrc
cd ~/compaction-release
conda activate compaction

MODEL=Qwen/Qwen3-4B
budget_path="head_budget_optimization/head_budgets/Qwen3-4B/optimized_agnostic.json"

log_dir=logs/reasoning_evaluation/qwen-reasoning

configs=()

# Baselines: 1024, 2048, 4096, 8192, 16384, 32768 with seeds 1-4
# for max_tokens in 1024 2048 4096 8192 16384 32768; do
#   for seed in 2 3 4; do
#     configs+=("baseline_${max_tokens}_s${seed}    baseline ${max_tokens} 0 0 0 ${seed} none none none 0")
#   done
# done

# Compaction with max_seq_len in {1024, 2048, 4096, 8192}
for max_seq_len in 2048; do
  for target_size in 0.2; do
    for max_compactions in 1 3 8; do
      for method in AM-HighestAttnKeys-basic; do
        for seed in 1; do
          configs+=("compact_${max_seq_len}_t${target_size}_c${max_compactions}_${method}_s${seed}    compaction ${max_seq_len} ${target_size} ${max_compactions} ${seed} ${method} repeat default 0")
        done
      done
    done
  done
done

# Select configuration based on SLURM array task ID
config="${configs[$SLURM_ARRAY_TASK_ID]}"
read -r name mode max_seq_len target_size max_compactions seed method query_config algorithm_config use_budget <<< "$config"

model_flag=""
if [ -n "$MODEL" ]; then
  model_flag="--model-name $MODEL"
fi

budget_flag=""
if [ -n "$budget_path" ] && [ "$use_budget" = "1" ]; then
  budget_flag="--precomputed-budget-path $budget_path --max-ratio-per-head 0.75" # currently doens't work with multiple compactions
fi

log_dir_flag=""
if [ -n "$log_dir" ]; then
  log_dir_flag="--log-dir $log_dir"
fi

start_time=$(date +%s)
echo "Start time: $(date -d @$start_time)"
echo "Running on node: $(hostname)"
echo "Config: $config"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ "$mode" = "baseline" ]; then
  python -u -m evaluation.run_reasoning_evaluation \
    --name "$name" \
    --mode baseline \
    --max-seq-len "$max_seq_len" \
    --seed "$seed" \
    $model_flag \
    $log_dir_flag

else
  # Compaction mode
  python -u -m evaluation.run_reasoning_evaluation \
    --name "$name" \
    --mode compaction \
    --max-seq-len "$max_seq_len" \
    --target-size "$target_size" \
    --max-compactions "$max_compactions" \
    --seed "$seed" \
    --method "$method" \
    --query-config "$query_config" \
    --algorithm-config "$algorithm_config" \
    $model_flag \
    $budget_flag \
    $log_dir_flag
fi

end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
