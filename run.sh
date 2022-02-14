export MODEL=64_35k_cl0.6_full && \
    python -u train.py \
    --dataset CASIA-B --resolution 64 --dataset_path /home/yuweichen/workspace/shared/casia_b_idnoise44/silhouettes_cut_pkl_idnoise44 \
    --milestones 10000 20000 --total_iter 20000 --warmup False --restore_iter=0\
    --more_channels False --bin_num 16 --hidden_dim 256 \
    --encoder_entropy_weight 0.1 --encoder_triplet_weight 1.0 --encoder_contrastive_weight 0.0\
    --model_name $MODEL --gpu 4,5,6,7 \
    --AMP False --DDP False --reranking False\
    --mem_bank False --triplet_type 'full'\
    2>&1 | tee $MODEL.log

cd /mnt/gait/Oracle_IDNoise_Gait_20211019
export MODEL=casia_b_idnoise44_rt64_train_base_bin16_entropy && \
    python -u train.py \
    --dataset CASIA-B --resolution 64 --dataset_path /home/yuweichen/workspace/shared/casia_b_idnoise44/silhouettes_cut_pkl_idnoise44 \
    --milestones 10000 20000 30000 --total_iter 35000 --warmup False \
    --more_channels False --bin_num 16 --hidden_dim 256 \
    --encoder_entropy_weight 0.0 \
    --model_name $MODEL --gpu 0,1,2,3,4,5,6,7 \
    --AMP False --DDP False \
    2>&1 | tee $MODEL.log

#Eval visualize_feature.py  test.py
python -u test.py \
    --dataset_path /home/yuweichen/workspace/shared/casia_b_idnoise44/silhouettes_cut_pkl_idnoise44 \
    --bin_num 16 \
    --batch_size 1 --gpu 4,5,6,7 --test_set test \
    --reranking False --euc_or_cos_dist 'euc' \
    --resume False --ckp_prefix 

python -u visualize_train_feature.py \
    --dataset_path /home/yuweichen/workspace/shared/casia_b_idnoise44/silhouettes_cut_pkl_idnoise44 \
    --bin_num 16 \
    --batch_size 1 --gpu 4,5,6,7 --test_set test --model_mix False\
    --reranking False --euc_or_cos_dist 'euc'\
    --resume False --ckp_prefix 

export MODEL=64_35k_cl0.6_lr0.5 && \
    python -u train.py \
    --dataset CASIA-B --resolution 64 --dataset_path /home/yuweichen/workspace/shared/casia_b_idnoise44/silhouettes_cut_pkl_idnoise44 \
    --milestones 10000 20000 --total_iter 20000 --warmup False --restore_iter 0\
    --more_channels False --bin_num 16 --hidden_dim 256 \
    --encoder_entropy_weight 0.1 --encoder_triplet_weight 1.0 --encoder_contrastive_weight 0.0 --encoder_triplet_margin 0\
    --model_name $MODEL --gpu 4,5,6,7 --lr 0.5\
    --AMP False --DDP False --reranking False\
    --mem_bank False --model_mix False\
    --dataset_augment False --contrastivelearning False\
    2>&1 | tee $MODEL.log

export MODEL=64_35k_cl0.6_clean_subset_noneview && \
    python -u train.py \
    --dataset CASIA-B --resolution 64 --dataset_path /home/yuweichen/workspace/dataset_clean_noisy/clean_subset \
    --milestones 10000 20000 --total_iter 20000 --warmup False --restore_iter 0\
    --more_channels False --bin_num 16 --hidden_dim 256 \
    --encoder_entropy_weight 0.1 --encoder_triplet_weight 1.0 \
    --model_name $MODEL --gpu 0,1,2,3 --lr 0.1\
    --AMP False --DDP False --reranking False\
    --mem_bank False --triplet_type 'full' --clean_subset True\
    --pid_num 30\
    2>&1 | tee ./log/$MODEL.log

export MODEL=64_35k_cl0.6_fullcl && \
    python -u train.py \
    --dataset_path /home/yuweichen/workspace/shared/casia_b_idnoise44/silhouettes_cut_pkl_idnoise44 \
    --milestones 10000 20000 30000 40000 --total_iter 20000 --warmup False \
    --bin_num 16 \
    --encoder_entropy_weight 0.1 --encoder_triplet_weight 1.0 \
    --model_name $MODEL --gpu 4,5,6,7 --lr 0.1\
    --AMP False --DDP False --cloth_loss_weight 0 --mixup_weight 0\
    --triplet_type 'full_cloth' --model_mix True\
    2>&1 | tee ./log/$MODEL.log


python -u train.py \
    --log_name 64_35k_cl0.6_clean_subset_ce0.1 --model_name 64_35k_cl0.6_clean_subset_ce0.1\
    --dataset_path /home/yuweichen/workspace/shared/casia_b_idnoise44/silhouettes_cut_pkl_idnoise44 \
    --milestones 10000 20000 --total_iter 30000 --warmup False  --restore_iter 20000 \
    --bin_num 16 --warmup_iter 20000 --static_iter 0 \
    --encoder_entropy_weight 0.1 --encoder_triplet_weight 1.0 \
    --gpu 0,1,2,3,4,5,6,7 --lr 0.1 \
    --AMP False --DDP False --spcloss_weight 1.0 \
    --triplet_type 'full' --model_mix True --mem_bank True --reranking_graph True --hiar_mem_bank True 

export MODEL=64_35k_cl0.6_clean_noneview_15 && \
    python -u train.py \
    --dataset CASIA-B --resolution 64 --dataset_path /home/yuweichen/workspace/data64pkl \
    --milestones 10000 20000 --total_iter 20000 --warmup False --restore_iter 0\
    --more_channels False --bin_num 16 --hidden_dim 256 \
    --encoder_entropy_weight 0.1 --encoder_triplet_weight 1.0 \
    --model_name $MODEL --gpu 0,1,2,3 --lr 0.1\
    --AMP False --DDP False --reranking False\
    --mem_bank False --triplet_type 'full' --clean_subset False\
    --pid_num 73\
    2>&1 | tee ./log/$MODEL.log

export MODEL=64_35k_cl0.6_for_test && \
    python -u train.py \
    --dataset CASIA-B --resolution 64 --dataset_path /home/yuweichen/workspace/shared/casia_b_idnoise44/silhouettes_cut_pkl_idnoise44 \
    --milestones 10000 20000 --total_iter 20000 --warmup False --restore_iter 0\
    --more_channels False --bin_num 16 --hidden_dim 256 \
    --encoder_entropy_weight 0.1 --encoder_triplet_weight 1.0 \
    --model_name $MODEL --gpu 0,1,2,3 --lr 0.1 \
    --AMP False --DDP False --reranking False \
    --mem_bank False --triplet_type 'full' --clean_subset False \
    --pid_num 73 \
    --dataset_augment True --self_supervised_weight 0.1 --infonce_git_weight 0.1 \
    2>&1 | tee ./log/$MODEL.log
    
export MODEL=64_35k_cl0.6_twostage && \
    python -u train.py \
    --dataset CASIA-B --resolution 64 --dataset_path /home/yuweichen/workspace/shared/casia_b_idnoise44/silhouettes_cut_pkl_idnoise44 \
    --milestones 10000 20000 30000 --total_iter 20000 --warmup False --restore_iter 20000\
    --more_channels False --bin_num 16 --hidden_dim 256 \
    --encoder_entropy_weight 0.1 --encoder_triplet_weight 1.0 \
    --model_name $MODEL --gpu 0,1,2,3 --lr 0.1 \
    --AMP False --DDP False --reranking False \
    --mem_bank False --triplet_type 'full' --clean_subset False \
    --pid_num 73 --restore_name  /home/yuweichen/workspace/noisy_gait/checkpoint/64_35k_cl0.6_baseline/64_35k_cl0.6_baseline_CASIA-B_73_False-20000-encoder.ptm\
    --dataset_augment False --self_supervised_weight 0.1 \
    2>&1 | tee ./log/$MODEL.log
    
