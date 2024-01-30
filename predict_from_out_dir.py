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


def main(args):
    # The tf model files for the predictors, the prediction will be averaged
    predictor_files = args.model
    if not predictor_files:
        predictor_files = [
            f"WMH/v0-FLAIR.WMH/20220412-192541_Unet3Dv2-10.7.2-1.8-FLAIR.WMH_fold_WMH_1x5_2ndUnat_fold_{i}_model.h5"
            for i in range(5)
        ]
        print(
            f"Using default model ensemble from {os.path.dirname(predictor_files[0])}"
        )

    t = transforms.Compose(
        [
            transforms.LoadImaged(
                image_only=True, ensure_channel_first=True, keys=["image", "mask"]
            ),
            transforms.Lambdad(keys="mask", func=close_mask),
            transforms.MaskIntensityd(
                keys="image",
                mask_key="mask",
                allow_missing_keys=False,
                select_fn=is_greater_0,
            )
            if args.skull_strip
            else transforms.Identityd(keys="mask"),
            transforms.CropForegroundd(
                keys=("mask", "image"),
                source_key="mask",
                allow_smaller=True,
                margin=1,
            ),
            transforms.ResizeWithPadOrCropD(spatial_size=size, keys=["image", "mask"]),
        ]
    )

    print(os.path.join(args.input, "*", args.file))
    files = glob(os.path.join(args.input, "*", args.file))
    print(f"Found {len(files)} files in {args.input}.")
    for input_image in tqdm(files):
        predict_image(
            input_image,
            input_image.replace(".nii.gz", "_wmh_seg.nii.gz"),
            os.path.join(os.path.dirname(input_image), "mask.nii.gz"),
            predictor_files,
            t,
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

    args = parser.parse_args()

    main(args)
