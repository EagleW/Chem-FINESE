file="data/CHEMET/ck_6_data"
CUDA_VISIBLE_DEVICES=0 \
python self_validation_pretrain.py \
    --dataset_dir $file \
    --batch_size 16 \
    --valid_batch_size 16 \
    --output pretrain_checkpoint 