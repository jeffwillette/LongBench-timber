# #!/bin/bash

model=invalid-model-name
method=streaming_llm
while getopts m:d:g: flag
do
    case "${flag}" in
        m) model=${OPTARG};;
        d) method=${OPTARG};;
        g) gpu=${OPTARG};;
    esac
done

WINDOW=16384
CASCADES=(4 1)
SINKS=64
MODEL=$model
GPU=$gpu
ATTENTION_METHOD=$method
COMMENT=("quarter-ctx" "quarter-ctx")

if [ "$ATTENTION_METHOD" = "vanilla" ]; then
    CASCADES=(1 1)
    COMMENT=("vanilla-truncate" "vanilla-unconstrained")
fi


for i in "${!CASCADES[@]}";
do 
    PYTHONPATH=/c2/jeff/cascading_cache_2/ \
        ATTENTION_METHOD=$ATTENTION_METHOD \
        CUDA_VISIBLE_DEVICES=$GPU \
        python pred.py \
        --sinks $SINKS \
        --cascades ${CASCADES[$i]} \
        --window $WINDOW \
        --model $MODEL \
        --comment ${COMMENT[$i]}
done
