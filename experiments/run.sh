SEGMENT_ANYTHING_FAST_USE_FLASH_4=0 python run_experiments.py 16 vit_h \
    ~/local/pytorch ~/local/segment-anything ~/local/sam_data \
    --run-experiments --local_fork_only \
    --num-workers 32  --capture_output False
