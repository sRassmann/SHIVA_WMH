#!/bin/bash
INPUT=$1
OUTPUT=$2

# Running Docker command with the file list
docker run --rm --user "$(id -u):$(id -g)" --ipc=host --ulimit memlock=-1 -it \
  --ulimit stack=67108864 --name "$USER"_GPU_0 --gpus 0 -v $(pwd):/workspace/shiva_wmh \
  -v $(realpath ../flairsyn/output):/workspace/flairsyn/output $USER/shiva_tf python \
  predict_one_file.py -i $INPUT -o $OUTPUT -s  --save_original
