#!/bin/bash

model=invalid-model-name
attention_method=streaming_llm
port=63290
while getopts m:d:g:p: flag
do
    case "${flag}" in
        m) model=${OPTARG};;
        d) attention_method=${OPTARG};;
        g) gpu=${OPTARG};;
        p) port=${OPTARG};;
    esac
done

# WINDOW=16384
# CASCADES=(4 1)
# SINKS=64
# # COMMENT=("quarter-ctx" "quarter-ctx")
# COMMENT=("eighth-ctx" "eighth-ctx")

WINDOW=16384
CASCADES=(4)
SINKS=64
# COMMENT=("quarter-ctx-different-sized-layers-latterhalfpos")
# COMMENT=("quarter-ctx")
COMMENT=("half-ctx")

if [ "$attention_method" = "vanilla" ]; then
    # CASCADES=(1 1)
    # COMMENT=("vanilla-truncate" "vanilla-unconstrained")
    CASCADES=(1)
    COMMENT=("vanilla-truncate-retest")
elif [ "$attention_method" = "bigbird" ]; then
    CASCADES=(1)
    # COMMENT=("bigbird-quarter-ctx-test")
    COMMENT=("bigbird-quarter-ctx-msq-test")
elif [ "$attention_method" = "h2o" ]; then
    CASCADES=(1)
    # COMMENT=("h2o-linear-retest")
    COMMENT=("h2o-linear-75-25-half-ctx")
    # COMMENT=("none")
    # COMMENT=("truncate")
fi

source paths.sh

# CUDA_VISIBLE_DEVICES=$GPU \
for i in "${!CASCADES[@]}";
do 
    PYTHONPATH=$PYTHONPATH \
    ATTENTION_METHOD=$attention_method \
    deepspeed --include localhost:$gpu --master_port $port pred.py \
        --sinks $SINKS \
        --cascades ${CASCADES[$i]} \
        --window $WINDOW \
        --model $model \
        --comment ${COMMENT[$i]}
done

# ```
# cd cascading_cache_2
# git pull origin main
# cd third_party/LongBench-timber
# git pull origin cascade

# ./run.sh -m llama3.1-70b-instruct -d streaming_llm -g 0,1,2,3 -p 63290
# ./run.sh -m qwen2-72b-instruct -d streaming_llm -g 4,5,6,7 -p 63291

# ./run.sh -m llama3.1-70b-instruct -d vanilla -g 0,1,2,3 -p 63292
# ./run.sh -m qwen2-72b-instruct -d vanilla -g 4,5,6,7 -p 63293

# ./run.sh -m llama3.1-70b-instruct -d bigbird -g 0,1,2,3 -p 63294
# ./run.sh -m qwen2-72b-instruct -d bigbird -g 4,5,6,7 -p 63295
# ```
