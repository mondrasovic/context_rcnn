#!/bin/bash

python -u -m src.main --test-only --checkpoint-file ./checkpoints_fasterrcnn/checkpoint_005.pth | tee eval_fasterrcnn_005.log

python -u -m src.main --test-only --checkpoint-file ./checkpoints_fasterrcnn/checkpoint_006.pth | tee eval_fasterrcnn_006.log

python -u -m src.main --test-only --checkpoint-file ./checkpoints_fasterrcnn/checkpoint_007.pth | tee eval_fasterrcnn_007.log

python -u -m src.main --test-only --checkpoint-file ./checkpoints_fasterrcnn/checkpoint_008.pth | tee eval_fasterrcnn_008.log
