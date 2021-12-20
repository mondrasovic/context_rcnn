#!/bin/bash

python -u -m src.main --test-only --checkpoint-file ./checkpoints_contextrcnn/checkpoint_002.pth | tee eval_contextrcnn_002.log

python -u -m src.main --test-only --checkpoint-file ./checkpoints_contextrcnn/checkpoint_003.pth | tee eval_contextrcnn_003.log

python -u -m src.main --test-only --checkpoint-file ./checkpoints_contextrcnn/checkpoint_004.pth | tee eval_contextrcnn_004.log

python -u -m src.main --test-only --checkpoint-file ./checkpoints_contextrcnn/checkpoint_001.pth | tee eval_contextrcnn_001.log
