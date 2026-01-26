#!/bin/bash

# This script runs the evaluation pipeline on a user-defined range of model checkpoints.

# Check if exactly two arguments (start and end points) are provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <start_checkpoint> <end_checkpoint>"
    echo "Example: $0 400 1500 ./trained/mid_lr_pretrained_qwen_tcga_structured_revision/" "./configs/barrett/eval_config.yaml"
    exit 1
fi

# Assign the first and second arguments to variables
START_POINT=$1
END_POINT=$2
BASE_MODEL_PATH=$3
CONFIG=$4
INCREMENT=50 #$3

echo "Starting evaluation for checkpoints from ${START_POINT} to ${END_POINT}..."

# Loop from START_POINT to END_POINT with a step of 50
for i in $(seq ${START_POINT} ${INCREMENT} ${END_POINT})
do
  # Define the path to the specific checkpoint model file
  MODEL_PATH="${BASE_MODEL_PATH}/checkpoint-${i}/model.safetensors" #"./trained/mid_lr_pretrained_qwen_tcga_structured_revision/checkpoint-${i}/model.safetensors"

  # Check if the model file actually exists before trying to run
  if [ -f "$MODEL_PATH" ]; then
    echo "----------------------------------------------------"
    echo "Running evaluation for checkpoint-${i}"
    echo "----------------------------------------------------"
    
    python -m model_eval.run_eval_pipeline \
      --config $CONFIG \
      --model "$MODEL_PATH" \
      --gguf_model ./configs/barrett/gguf/Llama-3.3-70B-instruct-GGUF.yaml \
      --output_base_dir ../../data/results/${BASE_MODEL_PATH} \
      --source_report_labels ./configs/labels/reference_full_real_report_labels_revision_24_10_25.json
      
    echo "Finished evaluation for checkpoint-${i}"
  else
    echo "----------------------------------------------------"
    echo "Skipping checkpoint-${i}: Model file not found at ${MODEL_PATH}"
    echo "----------------------------------------------------"
  fi
done

echo "🎉 All evaluations are complete."
