import numpy as np
import torch.nn.functional as F
import torch
import models as m

# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# Dice Score
# pred, mask : boolean mask tensors (batch_size x x_size x y_size)
def DiceScoreSum(pred, mask, smooth = 1e-7):
    intersection = torch.sum(pred * mask, dim = (1, 2))
    union = torch.sum(pred + mask, dim = (1, 2))
    
    return torch.sum((2.0 * intersection + smooth) / (union + smooth)).item()