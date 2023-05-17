import numpy as np
from tqdm import tqdm

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