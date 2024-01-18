file="data/CHEMET/ck_6_data"
CUDA_VISIBLE_DEVICES=0 \
python trainer_cl.py \
    --output boxbart_cl_checkpoint \
    --load_cons pretrain_checkpoint/BEST \
    --dataset_dir $file \
    --epochs 100 \
    --lr 5e-5 \
    --patient 100 \
    --batch_size 16 \
    --valid_batch_size 12 \
    --wandb