export MODEL=64_35k_cl0.6_randomrandom && \
    python -u train.py \
    --dataset CASIA-B --resolution 64 --dataset_path /home2/ywc/workspace/noisy_output_pkl_6444 \
    --milestones 10000 20000 --total_iter 20000 --warmup False \
    --more_channels False --bin_num 16 --hidden_dim 256 \
    --encoder_entropy_weight 0.0 \
    --model_name $MODEL --gpu 0,1,2,3 \
    --AMP False --DDP False --reranking False\
    --mem_bank False --triplet_type 'hard-random'\
    2>&1 | tee $MODEL.log
cd /mnt/gait/Oracle_IDNoise_Gait_20211019
export MODEL=casia_b_idnoise44_rt64_train_base_bin16_entropy && \
    python -u train.py \
    --dataset CASIA-B --resolution 64 --dataset_path /home2/ywc/workspace/noisy_output_pkl_6444 \
    --milestones 10000 20000 30000 --total_iter 35000 --warmup False \
    --more_channels False --bin_num 16 --hidden_dim 256 \
    --encoder_entropy_weight 0.0 \
    --model_name $MODEL --gpu 0,1,2,3,4,5,6,7 \
    --AMP False --DDP False \
    2>&1 | tee $MODEL.log

#Eval
python -u test.py \
    --dataset CASIA-B --resolution 64 --dataset_path /home2/ywc/workspace/noisy_output_pkl_6444 \
    --more_channels False --bin_num 16 --hidden_dim 256 \
    --batch_size 1 --gpu 0,1,2,3 --test_set test \
    --reranking True --euc_or_cos_dist 'cos'\
    --resume False --ckp_prefix 