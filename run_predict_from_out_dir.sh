#!/bin/bash

# Initialize variables
VERSION="v0"
INPUT=""
PATTERN="pred_flair.nii.gz"

# Parse optional arguments
while getopts "v:" opt; do
  case ${opt} in
    v )
      VERSION=$OPTARG
      ;;
    \? )
      echo "Usage: cmd [-v version] input_file [pattern]"
      exit 1
      ;;
  esac
done
shift $((OPTIND -1))

# The first positional argument after the options is the input file
INPUT=$1

# The second positional argument after the options is the pattern (optional)
if [ -n "$2" ]; then
  PATTERN=$2
fi

echo $VERSION

# Running Docker command with the file list
docker run --rm --user "$(id -u):$(id -g)" --ipc=host --ulimit memlock=-1 -it \
  --ulimit stack=67108864 --name "$USER"_GPU_0 --gpus 0 -v $(pwd):/workspace/shiva_wmh \
  -v $(realpath ../flairsyn/output):/workspace/flairsyn/output $USER/shiva_tf python \
  predict_from_out_dir.py -i $INPUT -f $PATTERN -s -v $VERSION

