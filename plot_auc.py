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
    parser.add_argument("--iter", type=str, default=200001, help='compute the evaluation from which iteration')
    parser.add_argument('--n_key', type=int, default=21, help='number of points for roc curve')
    parser.add_argument('--savedir', type=str, default='auc', help='where to save the logs')
    args = parser.parse_args()

    datasvaedir = os.path.join(args.basedir, args.expname, 'testset_{:06d}'.format(args.iter),'rawoutput')
    errfile = os.path.join(datasvaedir,'test_errors.npy')
    test_errors = np.load(errfile,allow_pickle=True)
    entropy_file = os.path.join(datasvaedir,'test_entropys.npy')
    test_entorpys = np.load(entropy_file,allow_pickle=True)

    logdir = os.path.join(args.baseir, args.savedir)
    os.makedirs(logdir, exist_ok=True)

    ROC_opt = []
    ROC_est = []

    n_samples = test_errors.shape[0]
    for i in range(n_samples):
        errs = test_errors[i].flatten()
        entropys = test_entorpys[i].flatten()
        sample_roc = roc(entropys,errs,n_value=21)
        ROC_opt.append(sample_roc['roc_opt'])
        ROC_est.append(sample_roc['roc_est'])

    ROC_opt = np.stack(ROC_opt,0)
    ROC_est = np.stack(ROC_est, 0)
    ROC_opt = ROC_opt.mean(axis=0)
    ROC_est = ROC_est.mean(axis=0)

    AUC_opt = auc(ROC_opt)
    AUC_est = auc()

    x_tickers = np.linspace(0, 100, args.n_key)
    plt.figure()
    plt.plot(x_tickers, ROC_opt, marker="^", color='blue', label='opt')
    plt.plot(x_tickers, ROC_est, marker="o", color='red', label='entropy')
    plt.xticks(x_tickers)
    plt.xlabel('Sample Size(%)')
    plt.ylabel('Accumulative Errors')
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(20, 8)
    fig.savefig(os.path.join(logdir,args.expname,'ROC_opt_vs_entropy.png'))
    plt.close()

    table = PrettyTable()
    table.title = 'Global statistic'
    table.field_names = ['item', 'value']
    table.add_row(['Expname',args.expname])
    table.add_row(['Test Samples', n_samples])
    table.add_row(['Error','mean:{:.4f}\t std:{:.4f}'.format(np.mean(test_errors),np.std(test_errors))])
    table.add_row(['Entropy', 'mean:{:.4f}\t std:{:.4f}'.format(np.mean(test_entorpys),np.std(test_entorpys))])
    table.add_row(['AUC_opt', np.round(auc(ROC_opt), 6)])
    table.add_row(['AUC_est', np.round(auc(ROC_est), 6)])
    print(table)

    txt_file = os.path.join(logdir,'auc.txt')
    with open(txt_file,'a') as f:
        f.write(str(table))
        f.write('\n')
    f.close()

if __name__ == '__main__':
    main()