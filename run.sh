export HIP_ROPE_METHOD=self_extend
export ATTENTION_BACKEND='hip'
export HIP_REFRESH_INTERVAL=8
export HIP_DENSE_LAYER=3
export CUDA_VISIBLE_DEVICES=0,1
export RUN_NAME='self_extend'
export MEASURE_PEAK_MEMORY=0

HIP_K=1024 python pred.py --model llama2-7b-chat-4k --stride 16000
python eval.py --model llama2-7b-chat-4k
mv pred/llama2-7b-chat-4k pred/llama2-7b-chat-4k-hip-k1024-$RUN_NAME

HIP_K=2048 python pred.py --model llama2-7b-chat-4k --stride 16000
python eval.py --model llama2-7b-chat-4k
mv pred/llama2-7b-chat-4k pred/llama2-7b-chat-4k-hip-k2048-$RUN_NAME