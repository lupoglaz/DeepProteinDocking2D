import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


class FIPlotter:
    def __init__(self, experiment=None, logfile_savepath='Log/losses/'):
        self.experiment = experiment
        self.logfile_savepath = logfile_savepath
        self.logtraindF_prefix = 'log_deltaF_TRAINset_epoch'
        self.logloss_prefix = 'log_loss_TRAINset_'
        if not experiment:
            print('no experiment name given')
            self.experiment = "NOTSET"

    def plot_loss(self):
        plt.close()
        #LOSS WITH ROTATION
        train = pd.read_csv(self.logfile_savepath+self.logloss_prefix+ self.experiment +'.txt', sep='\t', header=1, names=['Epoch', 'Loss'])
        num_epochs = len(train['Epoch'].to_numpy())

        fig, ax = plt.subplots(figsize=(20,10))

        train_loss = plt.plot(train['Epoch'].to_numpy(), train['Loss'].to_numpy())
        plt.title('log_loss_TRAINset_'+ self.experiment)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.grid(visible=True)
        plt.xticks(np.arange(0, num_epochs+1, num_epochs//10))
        plt.xlabel('Epochs')

        plt.savefig('Figs/FI_loss_plots/lossplot_'+self.experiment+'.png')
        plt.show()

    def plot_deltaF_distribution(self, filename=None, plot_epoch=None, show=False, xlim=None, binwidth=1):
        plt.close()
        # Plot free energy distribution of all samples across epoch
        if not filename:
            filename = self.logfile_savepath+self.logtraindF_prefix+str(plot_epoch)+ self.experiment +'.txt'
        train = pd.read_csv(filename, sep='\t', header=0, names=['F', 'F_0', 'Label'])

        fig, ax = plt.subplots(figsize=(10,10))
        plt.suptitle('deltaF distribution: epoch'+ str(plot_epoch) + ' ' + self.experiment)

        labels = sorted(train.Label.unique())
        F = train['F']
        bins = np.arange(min(F), max(F) + binwidth, binwidth)
        hist_data = [train.loc[train.Label == x, 'F'] for x in labels]
        y1, x1, _ = plt.hist(hist_data[0], label=labels, bins=bins, rwidth=binwidth, color=['r'], alpha=0.25)
        y2, x2, _ = plt.hist(hist_data[1], label=labels, bins=bins, rwidth=binwidth, color=['g'], alpha=0.25)

        if train['F_0'][0] != 'NA':
            plt.vlines(train['F_0'].to_numpy()[-1], ymin=0, ymax=max(y1.max(), y2.max())+1, linestyles='dashed', label='F_0', colors='k')
            plt.legend(('non-interaction (-)', ' interaction (+)', 'final F_0'), prop={'size': 10})
        else:
            plt.legend(('non-interaction (-)', ' interaction (+)'), prop={'size': 10})

        if xlim:
            ax.set_xlim([-xlim, 0])
        ax.set_ylabel('Training set counts')
        ax.set_xlabel('Free Energy (F)')
        ax.grid(visible=True)

        plt.savefig('Figs/FI_deltaF_distribution_plots/deltaFplot_epoch'+ str(plot_epoch) + '_' + self.experiment + '.png', format='png')
        if show:
            plt.show()
        plt.close()


if __name__ == "__main__":
    loadpath = '../Models/BruteForce/Log/losses/'
    testcase = 'scratch_'
    experiment = testcase+'BF_FI_NEWDATA_CHECK_400pool_5000ex30ep'
    Plotter = FIPlotter(experiment, logfile_savepath=loadpath)
    Plotter.plot_loss()
    Plotter.plot_deltaF_distribution(plot_epoch=3, show=True)
