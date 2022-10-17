"""
Modified version of gym.utils.play.play
"""
from itertools import product

import cv2
import numpy as np


def _get_font_scale(text):
    return 1
    # if len(text) < 5:
    #     return 1.1
    # else:
    #     return 0.9

def add_img_text(img, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = _get_font_scale(text)
    thickness = 3
    textsize = cv2.getTextSize(text, font, fontscale, thickness)[0]
    # get coords based on boundary
    x = int((img.shape[1] - textsize[0]) / 2)
    y = int((img.shape[0] + textsize[1]) / 2)

    cv2.putText(img, text, (x, y), fontFace=font, fontScale=fontscale, color=(235, 235, 235),
                thickness=thickness)


def grid_random_position(size, n_samples=1, margin=0, exclude_pos: list = None):
    positions = list(product(range(margin, size-margin), range(margin, size-margin)))
    indices = np.arange(len(positions))
    pos_idx = np.random.choice(indices, n_samples, replace=False)
    sampled_pos = [positions[idx] for idx in pos_idx]

    if exclude_pos is not None:
        # resample if a sampled pos is in exclude_pos
        is_valid = True
        for p in sampled_pos:
            if p in exclude_pos:
                is_valid = False
                break
        if not is_valid:
            sampled_pos = grid_random_position(size, n_samples, margin, exclude_pos)

    return sampled_pos

