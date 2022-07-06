export NCCL_LL_THRESHOLD=0

# wavevit_s
python3 -m torch.distributed.launch \
   --nproc_per_node=8 \
   --nnodes=1 \
   --node_rank=0 \
   --master_addr="localhost" \
   --master_port=12346 \
   --use_env main.py --config configs/wavevit/wavevit_s.py --data-path /export/home/dataset/imagenet --epochs 310 --batch-size 128 \
   --token-label --token-label-size 7 --token-label-data /export/home/dataset/imagenet/label_top5_train_nfnet

# wavevit_b
# python3 -m torch.distributed.launch \
#    --nproc_per_node=8 \
#    --nnodes=1 \
#    --node_rank=0 \
#    --master_addr="localhost" \
#    --master_port=12346 \
#    --use_env main.py --config configs/wavevit/wavevit_b.py --data-path /export/home/dataset/imagenet --epochs 310 --batch-size 128 \
#    --token-label --token-label-size 7 --token-label-data /export/home/dataset/imagenet/label_top5_train_nfnet

# wavevit_l
# python3 -m torch.distributed.launch \
#    --nproc_per_node=8 \
#    --nnodes=1 \
#    --node_rank=0 \
#    --master_addr="localhost" \
#    --master_port=12346 \
#    --use_env main.py --config configs/wavevit/wavevit_l.py --data-path /export/home/dataset/imagenet --epochs 310 --batch-size 128 \
#    --token-label --token-label-size 7 --token-label-data /export/home/dataset/imagenet/label_top5_train_nfnet