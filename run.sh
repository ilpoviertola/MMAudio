#!/bin/bash

MODEL_NAME=controlnet_small_44k_m_to_l_ec_sum_post_no_preconv

for i in {1..10}
do
    cd /home/ilpo/repos/MMAudio
    seed=$((14159265 + i))
    OMP_NUM_THREADS=8 torchrun --standalone --nproc_per_node=2 batch_eval.py duration_s=5 dataset=avssemantic model=$MODEL_NAME num_workers=8 batch_size=2 seed=$seed
    cd /home/ilpo/repos/av-benchmark
    python evaluate.py --gt_cache /home/hdd/ilpo/datasets/AVSSemantic/Single-source/s4_data/test_eval_cache --pred_audio /home/hdd/ilpo/logs/mmaudio/$MODEL_NAME/avssemantic --pred_cache /home/hdd/ilpo/logs/mmaudio/$MODEL_NAME/cache_$i --audio_length=5
    echo "Done with $i/10"
done
