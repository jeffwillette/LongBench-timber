ATTENTION_BACKEND='hip' HIP_K=512 HIP_REFRESH_INTERVAL=8 HIP_DENSE_LAYER=3 CUDA_VISIBLE_DEVICES=0,1 python pred.py --model llama2-7b-chat-4k
python eval.py --model llama2-7b-chat-4k
mv pred/llama2-7b-chat-4k-hip-k512

ATTENTION_BACKEND='hip' HIP_K=1024 HIP_REFRESH_INTERVAL=8 HIP_DENSE_LAYER=3 CUDA_VISIBLE_DEVICES=0,1 python pred.py --model llama2-7b-chat-4k
python eval.py --model llama2-7b-chat-4k
mv pred/llama2-7b-chat-4k-hip-k1024

ATTENTION_BACKEND='hip' HIP_K=2048 HIP_REFRESH_INTERVAL=8 HIP_DENSE_LAYER=3 CUDA_VISIBLE_DEVICES=0,1 python pred.py --model llama2-7b-chat-4k
python eval.py --model llama2-7b-chat-4k
mv pred/llama2-7b-chat-4k-hip-k2048