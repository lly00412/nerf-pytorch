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
    parser.add_argument('--est', type=str, default=['uncerts'],nargs='+', help='sorted scores for estimated roc')
    parser.add_argument('--savedir', type=str, default='auc', help='where to save the logs')
    args = parser.parse_args()

    datasvaedir = os.path.join(args.basedir, args.expname, 'testset_{:06d}'.format(args.iter),'rawoutput')
    gt_file = os.path.join(datasvaedir,'test_{}.npy'.format(args.gt))
    test_gt = np.load(gt_file,allow_pickle=True)

    logdir = os.path.join(args.basedir, args.savedir)
    os.makedirs(logdir, exist_ok=True)

    ROC_opt = {}
    ROC_est = {}
    means = {}
    stds = {}

    n_samples = test_gt.shape[0]
    for est_var in args.est:
        ROC_opt[est_var] = []
        ROC_est[est_var] = []
        for i in range(n_samples):
            gts = test_gt[i].flatten()
            est_file = os.path.join(datasvaedir, 'test_{}.npy'.format(est_var))
            test_est = np.load(est_file, allow_pickle=True)
            means[est_var] = np.mean(test_est)
            stds[est_var] = np.std(test_est)
            ests = test_est[i].flatten()
            sample_roc = roc(ests,gts,n_value=21)
            ROC_opt[est_var].append(sample_roc['roc_opt'])
            ROC_est[est_var].append(sample_roc['roc_est'])

        ROC_opt[est_var] = np.stack(ROC_opt[est_var],0)
        ROC_est[est_var] = np.stack(ROC_est[est_var], 0)
        ROC_opt[est_var] = ROC_opt[est_var].mean(axis=0)
        ROC_est[est_var] = ROC_est[est_var].mean(axis=0)

    figdir = os.path.join(logdir,args.expname)
    os.makedirs(figdir, exist_ok=True)

    x_tickers = np.linspace(0, 100, args.n_key)
    plt.figure()
    for est_var in args.est:
        plt.plot(x_tickers, ROC_est[est_var], marker="o", label=est_var)
    plt.plot(x_tickers, ROC_opt[est_var], marker="^", color='blue', label='opt')
    plt.xticks(x_tickers,fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Sample Size(%)',fontsize=25)
    plt.ylabel('Accumulative Errors',fontsize=25)
    plt.legend(fontsize=25)
    fig = plt.gcf()
    fig.set_size_inches(20, 8)
    fig.savefig(os.path.join(figdir,'ROC_opt_vs_est.png'.format(args.gt,args.est)))
    plt.close()

    table = PrettyTable()
    table.title = 'Global statistic'
    table.field_names = ['item', 'value']
    table.add_row(['Expname',args.expname])
    table.add_row(['Test Samples', n_samples])
    table.add_row(['{}(avg)'.format(args.gt), '{:.4f}'.format(np.mean(test_gt))])
    table.add_row(['{}(std)'.format(args.gt),'{:.4f}'.format(np.std(test_gt))])
    for est_var in args.est:
        table.add_row(['{}(avg)'.format(est_var), '{:.4f}'.format(means[est_var])])
        table.add_row(['{}(std)'.format(est_var), '{:.4f}'.format(stds[est_var])])
        table.add_row(['AUC_est', '{:.6f}'.format(auc(ROC_est[est_var]))])
    table.add_row(['AUC_opt', '{:.6f}'.format(auc(ROC_opt[est_var]))])
    print(table)

    txt_file = os.path.join(logdir,'auc_{}.txt'.format(args.expname))
    with open(txt_file,'a') as f:
        f.write(str(table))
        f.write('\n')
    f.close()

if __name__ == '__main__':
    main()