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

# COMMENT="quarter-ctx"
# COMMENT="none"
COMMENT="truncate"

source paths.sh

PYTHONPATH=$PYTHONPATH \
ATTENTION_METHOD=$attention_method \
deepspeed --include localhost:$gpu --master_port $port pred.py \
    --sinks 1 \
    --cascades 1 \
    --window 1 \
    --model $model \
    --comment $COMMENT
