#!/bin/bash

BASE_DIR="./"

for folder in "$BASE_DIR"/*/; do
    folder_name=$(basename "$folder")  # Extract folder name

    # if [[ "$folder_name" == *llama* ]]; then
    #     echo "Skipping folder: $folder_name (contains 'spectrum')"
    #     continue
    # fi
    
    timestamp=$(date +"%H:%M:%S")  # Generate timestamp

    jbsub -q nonstandard -cores 1x8+1 -mem 128G \
        -err "err1_${timestamp}.log" \
        -out "out_${timestamp}.log" \
        vllm serve "$folder" --port 5997 --gpu-memory-utilization 0.95 1>&2

    echo "Started job for folder: $folder_name"
    sleep 1
done