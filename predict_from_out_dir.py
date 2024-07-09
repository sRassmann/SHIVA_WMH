# # predict something from one multi-modal nifti images
# Tested with Python 3.7, Tensorflow 2.7
# @author : Philippe Boutinaud - Fealinx

import gc
import os
import time
import numpy as np
from pathlib import Path
import argparse
import nibabel
import tensorflow as tf
import torch
from monai import transforms
from glob import glob
from tqdm import tqdm

size = (160, 214, 176)


from predict_one_file import *


def find_t1(input_image, args):
    if not args.t1 or "T1" not in args.version:
        return None

    t1 = input_image.replace(args.file, args.t1)
    if os.path.exists(t1):
        return t1
    else:
        subj = os.path.basename(os.path.dirname(input_image))
        for a in ["inference_wmh", "inference_pvs", "inference_t", "inference"]:
            f = os.path.join(
                "../flairsyn/output/original", a, subj, os.path.basename(t1)
            )
            if os.path.exists(f):
                return f


def main(args):
    predictor_files = get_weights(args)
    print(os.path.join(args.input, "*", args.file))
    files = glob(os.path.join(args.input, "*", args.file))
    print(f"Found {len(files)} files in {args.input}.")
    out_name = "wmh_seg_t1" if "T1" in args.version else "wmh_seg"
    for input_image in tqdm(files):
        predict_image(
            input_image,
            find_t1(input_image, args),
            input_image.replace(".nii.gz", f"_{out_name}.nii.gz"),
            os.path.join(os.path.dirname(input_image), "mask.nii.gz"),
            predictor_files,
            get_transforms(args),
            save_original=args.save_original,
            verbose=False,
        )


if __name__ == "__main__":
    # Script parameters
    parser = argparse.ArgumentParser(
        description="Run inference with tensorflow models(s) on an image that may be built from several modalities"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="input directory containing the subjects",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="name of input file within subject directory",
        default="pred_flair.nii.gz",
    )
    parser.add_argument(
        "-t1",
        "--t1",
        type=str,
        help="name of the t1 file within subject directory",
        default="t1_n4.nii.gz",
    )
    parser.add_argument(
        "-m", "--model", type=Path, action="append", help="(multiple) prediction models"
    )
    parser.add_argument(
        "--save_original",
        action="store_true",
        help="Save the original image to the output directory",
    )
    parser.add_argument(
        "-s",
        "--skull_strip",
        action="store_true",
        help="Skull strip the image before inference",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        help="Version of the model to use (v0, v1-T1)",
        default="v0",
    )

    args = parser.parse_args()

    main(args)
