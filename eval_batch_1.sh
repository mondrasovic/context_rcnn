#!/bin/bash

python -u -m src.main --test-only --checkpoint-file ./checkpoints_fasterrcnn/checkpoint_002.pth | tee eval_fasterrcnn_002.log

python -u -m src.main --test-only --checkpoint-file ./checkpoints_fasterrcnn/checkpoint_003.pth | tee eval_fasterrcnn_003.log

python -u -m src.main --test-only --checkpoint-file ./checkpoints_fasterrcnn/checkpoint_004.pth | tee eval_fasterrcnn_004.log
