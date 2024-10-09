#!/bin/bash

MODEL_PATH="Qwen/Qwen2-7B-Instruct"
OUTPUT_DIR="results/leaderboard/qwen2_7b_instruct"
DEVICE="cuda:0"

# Ensure Hugging Face credentials are used
export HUGGINGFACE_TOKEN=$(cat ~/.cache/huggingface/token)

# Common arguments
COMMON_ARGS="--model hf \
    --model_args pretrained=$MODEL_PATH \
    --device $DEVICE \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --output_path $OUTPUT_DIR \
    --log_samples" 

# Function to run evaluation
run_eval() {
    local task=$1
    local batch_size=${3:-4}  # Default to 4 if not specified
    
    echo "Evaluating $task..."
    HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN lm_eval $COMMON_ARGS \
        --tasks $task \
        --batch_size $batch_size
}

# Run evaluations
run_eval "leaderboard_ifeval" 8
run_eval "leaderboard_musr" 8
run_eval "gpqa_main_zeroshot" 8
run_eval "gpqa_extended_zeroshot" 8
run_eval "gpqa_diamond_zeroshot" 8
run_eval "leaderboard_math_hard" 8 # 1324 samples, ~46GB, ~2h
run_eval "leaderboard_bbh" 8
run_eval "leaderboard_mmlu_pro" 8


# run_eval "winogrande" 5 8 # 2534 samples, ~18GB, ~2min
# run_eval "gsm8k" 5 8 # 1319 samples, ~20GB, ~40min
# run_eval "bbh_cot_fewshot_dyck_languages" 3 8 # 6511 samples, ~20GB, ~50min
# run_eval "mmlu" 5 4 # 56k samples, ~42GB, ~1h20min
# run_eval "hellaswag" 10 8 # 40k samples, ~38GB, ~1h50min
# run_eval "hendrycks_math" 4 8 # 5000 samples, ~20GB, ~2h30min
# run_eval "mmlu_pro" 5 8 # 12k samples, ~20GB, ~ 6h30min

# run_eval "gpqa" 5 4 # 9536 samples,            

echo "All evaluations completed."