import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from os.path import exists


class IPPlotter:
    def __init__(self, experiment=None, logfile_savepath='Log/losses/'):
        self.experiment = experiment
        self.logfile_savepath = logfile_savepath

        if not experiment:
            print('no experiment name given')
            sys.exit()

    def plot_loss(self, ylim=None, show=False):
        plt.close()
        #LOSS WITH ROTATION
        train = pd.read_csv(self.logfile_savepath+'log_loss_TRAINset_'+ self.experiment +'.txt', sep='\t', header=1, names=['Epoch', 'Loss', 'RMSD'])
        valid = pd.read_csv(self.logfile_savepath+'log_loss_VALIDset_'+ self.experiment +'.txt', sep='\t', header=1, names=['Epoch', 'Loss', 'RMSD'])
        test = pd.read_csv(self.logfile_savepath+'log_loss_TESTset_'+ self.experiment +'.txt', sep='\t', header=1, names=['Epoch', 'Loss', 'RMSD'])


        fig, ax = plt.subplots(2, figsize=(20,10))
        ax[0].plot(train['Epoch'].to_numpy(), train['RMSD'].to_numpy())
        ax[0].plot(valid['Epoch'].to_numpy(), valid['RMSD'].to_numpy())
        ax[0].plot(test['Epoch'].to_numpy(), test['RMSD'].to_numpy())
        ax[0].legend(('train RMSD', 'valid RMSD', 'test RMSD'))

        ax[0].set_title('Loss: ' + self.experiment)
        ax[0].set_ylabel('RMSD')
        ax[0].grid(visible=True)

        ax[1].plot(train['Epoch'].to_numpy(), train['Loss'].to_numpy())
        ax[1].plot(valid['Epoch'].to_numpy(), valid['Loss'].to_numpy())
        ax[1].plot(test['Epoch'].to_numpy(), test['Loss'].to_numpy())
        ax[1].legend(('train loss', 'valid loss', 'test loss'))

        ax[1].set_xlabel('epochs')
        ax[1].set_ylabel('loss')
        ax[1].grid(visible=True)

        # num_epochs = len(train['Epoch'].to_numpy())
        # ax[0].set_xticks(np.arange(0, num_epochs+1, num_epochs/10))
        # ax[1].set_xticks(np.arange(0, num_epochs+1, num_epochs/10))

        plt.xlabel('Epochs')
        if ylim:
            ax[0].set_ylim([0,ylim])
            ax[1].set_ylim([0,ylim])

        if not show:
            plt.savefig('Figs/IP_loss_plots/lossplot_'+self.experiment+'.png')
        else:
            plt.show()

    def plot_rmsd_distribution(self, plot_epoch=1, show=False, eval_only=True):
        plt.close()
        # Plot RMSD distribution of all samples across epoch
        train_log = self.logfile_savepath+'log_RMSDsTRAINset_epoch' + str(plot_epoch) + self.experiment + ".txt"
        valid_log = self.logfile_savepath+'log_RMSDsVALIDset_epoch' + str(plot_epoch) + self.experiment + ".txt"
        test_log = self.logfile_savepath+'log_RMSDsTESTset_epoch' + str(plot_epoch) + self.experiment + ".txt"
        train, valid, test, avg_validRMSD, avg_testRMSD = None, None, None, None, None
        if exists(train_log):
            train = pd.read_csv(train_log, sep='\t', header=0, names=['RMSD'])
        if exists(valid_log):
            valid = pd.read_csv(valid_log, sep='\t', header=0, names=['RMSD'])
            avg_validRMSD = str(valid['RMSD'].mean())[:4]
        if exists(test_log):
            test = pd.read_csv(test_log, sep='\t', header=0, names=['RMSD'])
            avg_testRMSD = str(test['RMSD'].mean())[:4]

        fig, ax = plt.subplots(3, figsize=(10, 30))
        plt.suptitle('RMSD distribution: epoch' + str(plot_epoch) + ' ' + self.experiment)
        plt.legend(('train rmsd', 'valid rmsd', 'test rmsd'))
        plt.xlabel('RMSD')
        binwidth=1
        bins = np.arange(0, 100 + binwidth, binwidth)

        if train is not None:
            ax[0].hist(train['RMSD'].to_numpy(), bins=bins, color='b')
            ax[0].set_ylabel('Training set counts')
            ax[0].grid(visible=True)
            ax[0].set_xticks(np.arange(0, 100, 10))
        if valid is not None:
            ax[1].hist(valid['RMSD'].to_numpy(), bins=bins, color='r')
            ax[1].set_ylabel('Valid set counts')
            ax[1].grid(visible=True)
            ax[1].set_xticks(np.arange(0, 100, 10))
        if test is not None:
            ax[2].hist(test['RMSD'].to_numpy(), bins=bins, color='g')
            ax[2].set_ylabel('Test set counts')
            ax[2].grid(visible=True)
            ax[2].set_xticks(np.arange(0, 100, 10))

        if not show:
            plt.savefig('Figs/IP_RMSD_distribution_plots/RMSDplot_epoch' + str(
                plot_epoch) + '_vRMSD' + avg_validRMSD + '_tRMSD' + avg_testRMSD + self.experiment + '.png')
        else:
            plt.show()


if __name__ == "__main__":
    loadpath = '../Models/ReducedSampling/Log/losses/'
    experiment = 'BS_IP_FINAL_DATASET_400pool_1000ex_5ep'
    Plotter = IPPlotter(experiment, logfile_savepath=loadpath)
    Plotter.plot_loss()
    Plotter.plot_rmsd_distribution(plot_epoch=4, show=True)
