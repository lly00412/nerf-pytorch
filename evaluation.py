import numpy as np
from tqdm import tqdm
from pytorch_msssim import ssim
import torch
from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity
from torch.nn.functional import normalize

img2mse = lambda x, y : np.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * np.log(x) / np.log(10.)
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def img2lpips(x,y):
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    x = x.permute(0, 3, 1, 2)
    y = y.permute(0, 3, 1, 2)
    x = normalize(x,dim=1)
    y = normalize(y,dim=1)
    lpips = learned_perceptual_image_patch_similarity(x, y, normalize=True)
    return lpips.detach().cpu().numpy()

def img2ssim(x, y, mask=None):
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    if mask is not None:
        x = mask.unsqueeze(-1) * x
        y = mask.unsqueeze(-1) * y

    x = x.permute(0, 3, 1, 2)
    y = y.permute(0, 3, 1, 2)
    ssim_ = ssim(x, y, data_range=1)
    return ssim_.cpu().numpy()

def roc(est_score,gt_score,n_value=21):
    # small uncert go first
     est_order = est_score.argsort()
     opt_order = gt_score.argsort()
     index = np.linspace(0, gt_score.shape[0], n_value)
     accum_opt = [0]
     accum_est = [0]
     samples = [0]
     for idx in tqdm(index[1:], leave=False):
         idx = np.int(idx)
         samples.append(idx)
         current_opt = gt_score[opt_order[:idx]]
         current_est = gt_score[est_order[:idx]]
         accum_opt.append(np.average(current_opt))
         accum_est.append(np.average(current_est))
     return {'pct':np.linspace(0, 100, n_value),
            'samples':np.array(samples),
            'roc_opt':np.array(accum_opt),
            'roc_est':np.array(accum_est)}

def auc(roc):
    n_key = len(roc) - 1
    auc = np.trapz(roc, dx=1. / n_key) * 100.
    return auc

