#!/bin/bash

python -u -m src.main --test-only --checkpoints-dir ./checkpoints_tmp --checkpoint-file ./checkpoints_fasterrcnn/checkpoint_008.pth | tee tmp_log.log
