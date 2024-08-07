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

size = (160, 214, 176)


@torch.no_grad()
def close_mask(mask, dilate_size=7, erode_size=5):
    org_shape, org_dtype = mask.shape, mask.dtype

    # dilate
    filter = (
        torch.ones((dilate_size, dilate_size, dilate_size)).unsqueeze(0).unsqueeze(0)
    )
    conv_res = torch.nn.functional.conv3d(
        mask.unsqueeze(0), filter, padding=dilate_size // 2
    )
    dil = conv_res > 0

    # erode
    filter = torch.ones((erode_size, erode_size, erode_size)).unsqueeze(0).unsqueeze(0)
    conv_res = torch.nn.functional.conv3d(
        dil.float(), filter, padding=erode_size // 2
    ).squeeze(dim=0)
    erode = conv_res == filter.sum()

    assert erode.shape == org_shape

    return erode.to(org_dtype)


def is_greater_0(x):
    return x > 0


def get_transforms(args):
    load_transforms = transforms.Compose(
        [
            transforms.LoadImaged(
                image_only=True,
                ensure_channel_first=True,
                keys=["image", "mask", "t1"],
                allow_missing_keys=True,
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
                keys=("mask", "image", "t1"),
                source_key="mask",
                allow_smaller=True,
                margin=1,
                allow_missing_keys=True,
            ),
            # based on this comment: https://github.com/pboutinaud/SHIVA_PVS/issues/2#issuecomment-1499029783 and what they mention in the paper
            transforms.Orientationd(
                keys=["image", "mask", "t1"], allow_missing_keys=True, axcodes="RAS"
            ),
            transforms.ResizeWithPadOrCropD(
                spatial_size=size, keys=["image", "mask", "t1"], allow_missing_keys=True
            ),
        ]
    )
    return load_transforms


def get_weights(args):
    # The tf model files for the predictors, the prediction will be averaged
    predictor_files = args.model
    if not predictor_files:
        if args.version == "v0":
            predictor_files = [
                f"WMH/v0-FLAIR.WMH/20220412-192541_Unet3Dv2-10.7.2-1.8-FLAIR.WMH_fold_WMH_1x5_2ndUnat_fold_{i}_model.h5"
                for i in range(5)
            ]
        elif args.version == "v1-T1":
            predictor_files = [
                f"WMH/v1-T1-FLAIR.WMH/20230427-215909_Unet3Dv3-10.7.2-1.8-T1_FLAIR.WMH-from15_fold_{i}_model.h5"
                for i in range(5)
            ]
        else:
            raise ValueError(f"Unknown version {args.version}")
        print(
            f"Using default model ensemble from {os.path.dirname(predictor_files[0])}"
        )
    return predictor_files


def main(args):
    args["version"] = "v0" if args.t1 is None else "v1-T1"
    predictor_files = get_weights(args)

    for input_image in args.input:
        print(f"Predicting {input_image}")

        if args.output is None:
            output = input_image.replace(".nii.gz", "_wmh_seg.nii.gz")
        else:
            output = args.output

        if args.brainmask is None:
            brainmask = os.path.join(os.path.dirname(input_image), "mask.nii.gz")
        else:
            brainmask = args.brainmask

        predict_image(
            input_image,
            output,
            brainmask,
            predictor_files,
            get_transforms(args),
            save_original=args.save_original,
        )


def preprocess_images(img):
    img = img.numpy()
    img /= np.percentile(img, 99)
    return np.reshape(img, (1, *size, 1))


def predict_image(
    input_path,
    t1_path,
    output_path,
    mask_path,
    predictor_files,
    t,
    save_original=False,
    verbose=True,
    threshold=0.2,  # according to README.md https://github.com/pboutinaud/SHIVA_WMH/tree/main/WMH/v0/FLAIR.WMH, 0.5 for newer models gives really bad results
):
    img = {"image": input_path, "mask": mask_path}
    if t1_path:
        img["t1"] = t1_path

    img = t(img)
    input_image = preprocess_images(img["image"])
    if t1_path:
        t1 = preprocess_images(img["t1"])
        input_image = np.concatenate([t1, input_image], axis=-1)
    predictions = []
    for predictor_file in predictor_files:
        tf.keras.backend.clear_session()
        gc.collect()
        try:
            model = tf.keras.models.load_model(
                predictor_file, compile=False, custom_objects={"tf": tf}
            )
        except Exception as err:
            print(f"\n\tWARNING : Exception loading model : {predictor_file}\n{err}")
            continue
        prediction = model.predict(input_image, batch_size=1)
        if verbose:
            print(prediction.sum())
        predictions.append(prediction)
    # Average all predictions
    mask = img["mask"].numpy()
    predictions = np.stack(predictions, axis=0)[..., 0]
    predictions = np.mean(predictions, axis=0) * (mask > 0)
    if threshold > 0:
        predictions = (predictions > threshold).astype(np.uint8)
    img["mask"] *= 0
    img["mask"] += torch.Tensor(predictions)
    inv = t.inverse(img)
    affine = inv["mask"].meta["affine"].numpy().squeeze().astype(np.float32)
    # Save prediction
    nifti = nibabel.Nifti1Image(
        inv["mask"].squeeze(dim=0).astype(np.uint8), affine=affine
    )
    nibabel.save(nifti, output_path)
    if save_original:
        nifti = nibabel.Nifti1Image(
            (inv["image"] * 255).squeeze(dim=0).astype(np.float32), affine=affine
        )
        org_out = output_path.replace(".nii.gz", "_input.nii.gz")
        print(f"Saving original image to {org_out}")
        nibabel.save(nifti, org_out)


if __name__ == "__main__":
    # Script parameters
    parser = argparse.ArgumentParser(
        description="Run inference with tensorflow models(s) on an image that may be built from several modalities"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        action="append",
        help="input image",
    )
    parser.add_argument(
        "-t1" "--t1",
        type=str,
        action="append",
        help="T1 image (optional)",
    )
    parser.add_argument(
        "-b",
        "--brainmask",
        type=str,
        help="brainmask image if None 'mask.nii.gz' in the same directory as the input image will be used",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="path for the output file (output of the inference from tensorflow model). If None, the output will be saved in the same directory as the input (with the suffix '_wmh_seg.nii.gz')",
        default=None,
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

    if args.t1 and len(args.t1) != len(args.input):
        raise ValueError("T1 images must be provided for each input image")

    if len(args.input) > 1 and (args.output is not None or args.brainmask is not None):
        raise ValueError(
            "Multiple input images are not supported with custom output or brainmask"
        )

    main(args)
