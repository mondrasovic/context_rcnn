#!/bin/bash

python -u -m src.main --checkpoints-dir ./checkpoints_tmp | tee tmp_log.log
