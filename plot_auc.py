import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import argparse
from evaluation import *
import os

def main():
    # model params
    parser = argparse.ArgumentParser(description='Evaluation Process')
    parser.add_argument("--expname", type=str,help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',help='where to store ckpts and logs')
    parser.add_argument("--iter", type=str, default=200000, help='compute the evaluation from which iteration')
    parser.add_argument('--n_key', type=int, default=21, help='number of points for roc curve')
    parser.add_argument('--gt', type=str, default='errors', help='ground truth baseline for optimal roc')
    parser.add_argument('--est', type=str, default='uncerts', help='sorted scores for estimated roc')
    parser.add_argument('--savedir', type=str, default='auc', help='where to save the logs')
    args = parser.parse_args()

    datasvaedir = os.path.join(args.basedir, args.expname, 'testset_{:06d}'.format(args.iter),'rawoutput')
    gt_file = os.path.join(datasvaedir,'test_{}.npy'.format(args.gt))
    test_gt = np.load(gt_file,allow_pickle=True)
    est_file = os.path.join(datasvaedir,'test_{}.npy'.format(args.est))
    test_est = np.load(est_file,allow_pickle=True)

    logdir = os.path.join(args.basedir, args.savedir)
    os.makedirs(logdir, exist_ok=True)

    ROC_opt = []
    ROC_est = []

    n_samples = test_gt.shape[0]
    for i in range(n_samples):
        gts = test_gt[i].flatten()
        ests = test_est[i].flatten()
        sample_roc = roc(ests,gts,n_value=21)
        ROC_opt.append(sample_roc['roc_opt'])
        ROC_est.append(sample_roc['roc_est'])

    ROC_opt = np.stack(ROC_opt,0)
    ROC_est = np.stack(ROC_est, 0)
    ROC_opt = ROC_opt.mean(axis=0)
    ROC_est = ROC_est.mean(axis=0)

    figdir = os.path.join(logdir,args.expname)
    os.makedirs(figdir, exist_ok=True)

    x_tickers = np.linspace(0, 100, args.n_key)
    plt.figure()
    plt.plot(x_tickers, ROC_opt, marker="^", color='blue', label='opt')
    plt.plot(x_tickers, ROC_est, marker="o", color='red', label='entropy')
    plt.xticks(x_tickers,fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Sample Size(%)',fontsize=25)
    plt.ylabel('Accumulative Errors',fontsize=25)
    plt.legend(fontsize=25)
    fig = plt.gcf()
    fig.set_size_inches(20, 8)
    fig.savefig(os.path.join(figdir,'ROC_opt_{}_vs_est_{}.png'.format(args.gt,args.est)))
    plt.close()

    table = PrettyTable()
    table.title = 'Global statistic'
    table.field_names = ['item', 'value']
    table.add_row(['Expname',args.expname])
    table.add_row(['Test Samples', n_samples])
    table.add_row(['{}(avg)'.format(args.gt), '{:.4f}'.format(np.mean(test_gt))])
    table.add_row(['{}(std)'.format(args.gt),'{:.4f}'.format(np.std(test_gt))])
    table.add_row(['{}(avg)'.format(args.est), '{:.4f}'.format(np.mean(test_est))])
    table.add_row(['{}(std)'.format(args.est), '{:.4f}'.format(np.std(test_est))])
    table.add_row(['AUC_opt', '{:.6f}'.format(auc(ROC_opt))])
    table.add_row(['AUC_est', '{:.6f}'.format(auc(ROC_est))])
    print(table)

    txt_file = os.path.join(logdir,'auc.txt')
    with open(txt_file,'a') as f:
        f.write(str(table))
        f.write('\n')
    f.close()

if __name__ == '__main__':
    main()