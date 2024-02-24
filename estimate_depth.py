import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

def depth_anything(image, encoder, pred_only=False, grayscale=False):
    """
    Process depth images based on the provided parameters.

    Parameters:
    img_path (str): Path to the image or directory of images.
    outdir (str): Output directory for the depth images.
    encoder (str): Encoder model to use ('vits', 'vitb', or 'vitl').
    pred_only (bool): If True, only display the prediction. Default is False.
    grayscale (bool): If True, do not apply colorful palette. Default is False.
    """

    margin_width = 50
    caption_height = 60


    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    depth_anything = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{encoder}14').to(DEVICE).eval()



    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE).eval()

    H, W = image.shape[1:]
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
            depth = depth_anything(image)


    depth = F.interpolate(depth.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False)


    return depth.squeeze(0)

    



# This part allows the script to be run as a standalone program or imported as a module
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str, required=False)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()

    depth_anything(image, args.encoder, args.pred_only, args.grayscale)