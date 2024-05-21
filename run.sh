# HIP_K=1024 python pred.py --model llama2-7b-chat-4k --stride 16000
# python eval.py --model llama2-7b-chat-4k
# mv pred/llama2-7b-chat-4k pred/llama2-7b-chat-4k-hip-k1024-$RUN_NAME
# 
# HIP_K=2048 python pred.py --model llama2-7b-chat-4k --stride 16000
# python eval.py --model llama2-7b-chat-4k
# mv pred/llama2-7b-chat-4k pred/llama2-7b-chat-4k-hip-k2048-$RUN_NAME

# RUN_NAME=''
# PYTHONPATH=/c2/jeff/tree-attention/ \
#     ATTENTION_METHOD=streaming_llm \
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     python pred.py \
#     --sinks 4 \
#     --cascades 4 \
#     --window 1024 \
#     --model llama2-7b-chat-32k
#     # --model llama1.3b
#  
#  
# PYTHONPATH=/c2/jeff/tree-attention/ \
#     ATTENTION_METHOD=streaming_llm \
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     python pred.py \
#     --sinks 4 \
#     --cascades 1 \
#     --window 1024 \
#     --model llama2-7b-chat-32k
#     # --model llama1.3b

# PYTHONPATH=/c2/jeff/tree-attention/ \
#     ATTENTION_METHOD=streaming_llm \
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     python pred.py \
#     --sinks 4 \
#     --cascades 4 \
#     --window 1024 \
#     --model qwen2-7b-chat-32k
#     # --model qwen0.5b

# PYTHONPATH=/c2/jeff/tree-attention/ \
#     ATTENTION_METHOD=streaming_llm \
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     python pred.py \
#     --sinks 4 \
#     --cascades 1 \
#     --window 1024 \
#     --model qwen2-7b-chat-32k
#     # --model qwen0.5b
