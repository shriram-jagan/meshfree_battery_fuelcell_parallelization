#!/bin/bash

export LEGATE_AUTO_CONFIG=0
export LEGATE_SHOW_CONFIG=1
#export LEGATE_SHOW_PROGRESS=1
#export LEGATE_MIN_GPU_CHUNK=10

legate --fbmem 45000 --sysmem 35000 --gpus 1 ./main.py >& out &
wait
