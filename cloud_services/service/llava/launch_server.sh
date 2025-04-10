CUDA_VISIBLE_DEVICES=2 python llava_server.py \
    --host 0.0.0.0 \
    --worker http://localhost:40000 \
    --model-path liuhaotian/llava-v1.5-7b \
    --no-register