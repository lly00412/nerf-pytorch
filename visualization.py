from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np

def gen_error_colormap():
    cols = np.array(
        [[0 / 3.0, 0.1875 / 3.0, 49, 54, 149],
         [0.1875 / 3.0, 0.375 / 3.0, 69, 117, 180],
         [0.375 / 3.0, 0.75 / 3.0, 116, 173, 209],
         [0.75 / 3.0, 1.5 / 3.0, 171, 217, 233],
         [1.5 / 3.0, 3 / 3.0, 224, 243, 248],
         [3 / 3.0, 6 / 3.0, 254, 224, 144],
         [6 / 3.0, 12 / 3.0, 253, 174, 97],
         [12 / 3.0, 24 / 3.0, 244, 109, 67],
         [24 / 3.0, 48 / 3.0, 215, 48, 39],
         [48 / 3.0, np.inf, 165, 0, 38]], dtype=np.float32)
    cols[:, 2: 5] /= 255.
    return cols


error_colormap = gen_error_colormap()


class color_error_image_func(nn.Module):
    def forward(self, D_err_tensor, D_gt_tensor=None, dilate_radius=1):
        D_err_np = D_err_tensor.detach().cpu().numpy()
        B, H, W = D_err_np.shape
        # error in percentage. When error <= 1, the pixel is valid since <= 3px & 5%
        # error = np.minimum(D_err_np / abs_thres, D_err_np / rel_thres)
        if D_gt_tensor is not None:
            D_gt_np = D_gt_tensor.detach().cpu().numpy()
            err_prob = D_err_np / (D_gt_np+1e-5)
        else:
            err_prob = D_err_np / (np.max(D_err_np)+1e-5)
        # get colormap
        cols = error_colormap
        # create error image
        error_image = np.zeros([B, H, W, 3], dtype=np.float32)
        for i in range(cols.shape[0]):
            error_image[np.logical_and(err_prob >= cols[i][0], err_prob < cols[i][1])] = cols[i, 2:]
        # TODO: imdilate
        # error_image = cv2.imdilate(D_err, strel('disk', dilate_radius));
        # show color tag in the top-left cornor of the image
        for i in range(cols.shape[0]):
            distance = 20
            error_image[:, :10, i * distance:(i + 1) * distance, :] = cols[i, 2:]

        return torch.from_numpy(np.ascontiguousarray(error_image))

    def backward(self, grad_output):
        return None
