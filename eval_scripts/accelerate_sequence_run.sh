#!/bin/bash

DEVICES=${1:-"0,1,2,3"}
OUTPUT_BASE_DIR="results/leaderboard"

# Ensure Hugging Face credentials are used
export HUGGINGFACE_TOKEN=$(cat ~/.cache/huggingface/token)

# Define models and their output directories
declare -A MODELS=(
    ["llava-hf/llama3-llava-next-8b-hf"]="llama3_llava_next_8b_hf"
    ["Qwen/Qwen2-VL-7B-Instruct"]="qwen2_vl_7b_instruct"
    ["Qwen/Qwen2-VL-2B-Instruct"]="qwen2_vl_2b_instruct"
    ["llava-hf/llava-onevision-qwen2-0.5b-si-hf"]="llava-onevision-qwen2-0.5b-si-hf"
    ["llava-hf/llava-onevision-qwen2-0.5b-ov-hf"]="llava-onevision-qwen2-0.5b-ov-hf"
    ["llava-hf/llava-onevision-qwen2-7b-si-hf"]="llava-onevision-qwen2-7b-si-hf"
    ["llava-hf/llava-onevision-qwen2-7b-ov-hf"]="llava-onevision-qwen2-7b-ov-hf"
    ["llava-hf/llava-onevision-qwen2-7b-ov-hf"]="llava-onevision-qwen2-7b-ov-chat-hf"  
    # Add more models as needed
)

# Function to run evaluation
un_eval() {
    local model_path=$1
    local output_dir=$2
    local task=$3
    local batch_size=${4:-4}  # Default to 4 if not specified
    
    echo "Evaluating $task for $(basename $model_path)..."
    COMMON_ARGS="--model hf \
        --model_args pretrained=$model_path \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --output_path $output_dir \
        --trust_remote_code \
        --gen_kwargs max_new_tokens=4096 \
        --log_samples"

    CUDA_VISIBLE_DEVICES=$DEVICES HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN accelerate launch -m lm_eval $COMMON_ARGS \
        --tasks $task \
        --batch_size $batch_size
}

# Loop through models and run evaluations
for model in "${!MODELS[@]}"; do
    MODEL_PATH="$model"
    OUTPUT_DIR="$OUTPUT_BASE_DIR/${MODELS[$model]}"
    
    echo "Processing model: $model"
    echo "Model path: $MODEL_PATH"
    echo "Output directory: $OUTPUT_DIR"
    
    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"
    
    # Run evaluations for each task
    run_eval "$MODEL_PATH" "$OUTPUT_DIR" "leaderboard_ifeval" 8
    run_eval "$MODEL_PATH" "$OUTPUT_DIR" "leaderboard_musr" 8
    run_eval "$MODEL_PATH" "$OUTPUT_DIR" "leaderboard_gpqa" 8
    run_eval "$MODEL_PATH" "$OUTPUT_DIR" "leaderboard_math_hard" 8 # 1324 samples, ~46GB, ~2h
    run_eval "$MODEL_PATH" "$OUTPUT_DIR" "leaderboard_bbh" 8
    run_eval "$MODEL_PATH" "$OUTPUT_DIR" "leaderboard_mmlu_pro" 8
    
    echo "Finished processing $model"
    echo "------------------------"
done