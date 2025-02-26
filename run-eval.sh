#!/bin/bash

# model=llama3.1-8b-instruct
MODEL=qwen2-7b-instruct

echo "vanilla unconstrained"
PYTHONPATH=/c2/jeff/cascading_cache_2 python eval.py --model $MODEL --method vanilla --comment vanilla-unconstrained

echo "vanilla truncated"
PYTHONPATH=/c2/jeff/cascading_cache_2 python eval.py --model $MODEL --method vanilla --comment vanilla-truncate

echo "cascade"
PYTHONPATH=/c2/jeff/cascading_cache_2 python eval.py --model $MODEL --method streaming_llm --comment quarter-ctx --cascades 4 --window 16384 --sinks 64

echo "streaming llm"
PYTHONPATH=/c2/jeff/cascading_cache_2 python eval.py --model $MODEL --method streaming_llm --comment quarter-ctx --cascades 1 --window 16384 --sinks 64
