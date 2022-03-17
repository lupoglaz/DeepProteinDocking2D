import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

class IPLossPlotter:
    def __init__(self, experiment=None):
        self.experiment = experiment
        if not experiment:
            print('no experiment name given')
            sys.exit()

    def plot_loss(self):
        plt.close()
        #LOSS WITH ROTATION
        train = pd.read_csv("Log/losses/log_train_"+ self.experiment +".txt", sep='\t', header=1, names=['Epoch',	'Loss',	'rmsd'])
        valid = pd.read_csv("Log/losses/log_valid_"+ self.experiment +".txt", sep='\t', header=1, names=['Epoch', 'Loss', 'rmsd'])
        test = pd.read_csv("Log/losses/log_test_"+ self.experiment +".txt", sep='\t', header=1, names=['Epoch', 'Loss', 'rmsd'])

        num_epochs = len(train['Epoch'].to_numpy())

        fig, ax = plt.subplots(2, figsize=(20,10))
        train_rmsd = ax[0].plot(train['Epoch'].to_numpy(), train['rmsd'].to_numpy())
        valid_rmsd = ax[0].plot(valid['Epoch'].to_numpy(), valid['rmsd'].to_numpy())
        test_rmsd = ax[0].plot(test['Epoch'].to_numpy(), test['rmsd'].to_numpy())
        ax[0].legend(('train rmsd', 'valid rmsd', 'test rmsd'))

        ax[0].set_title('Loss: ' + self.experiment)
        ax[0].set_ylabel('rmsd')
        ax[0].grid(visible=True)
        ax[0].set_xticks(np.arange(0, num_epochs+1, num_epochs/10))
        # ax[0].set_yticks(np.arange(0, max(train['Loss'].to_numpy())+1, 10))

        train_loss = ax[1].plot(train['Epoch'].to_numpy(), train['Loss'].to_numpy())
        valid_loss = ax[1].plot(valid['Epoch'].to_numpy(), valid['Loss'].to_numpy())
        test_loss = ax[1].plot(test['Epoch'].to_numpy(), test['Loss'].to_numpy())
        ax[1].legend(('train loss', 'valid loss', 'test loss'))

        # best_train_rmsd = train['rmsd'].min()
        # best_valid_rmsd = valid['rmsd'].min()
        # best_test_rmsd = test['rmsd'].min()

        ax[1].set_xlabel('epochs')
        ax[1].set_ylabel('loss')
        ax[1].grid(visible=True)
        ax[1].set_xticks(np.arange(0, num_epochs+1, num_epochs/10))
        # ax[0].set_yticks(np.arange(0, max(train['rmsd'].to_numpy())+1, 10))

        plt.xlabel('Epochs')
        ax[0].set_ylim([0,20])
        ax[1].set_ylim([0,20])

        plt.savefig('figs/BF_IP_loss_plots/Lossplot_'+self.experiment+'.png')
        plt.show()

    def plot_rmsd_distribution(self, plot_epoch=1, show=False, eval_only=True):
        plt.close()
        # Plot RMSD distribution of all samples across epoch
        if not eval_only:
            fig, ax = plt.subplots(3, figsize=(10,30))
            plt.suptitle('RMSD distribution: epoch' + str(plot_epoch) + ' ' + self.experiment)

            train = pd.read_csv("Log/losses/log_RMSDsTrainset_epoch" + str(plot_epoch) + self.experiment + ".txt", sep='\t', header=1, names=['RMSD'])
            valid = pd.read_csv("Log/losses/log_RMSDsValidset_epoch" + str(plot_epoch) + self.experiment + ".txt", sep='\t', header=1, names=['RMSD'])
            test = pd.read_csv("Log/losses/log_RMSDsTestset_epoch" + str(plot_epoch) + self.experiment + ".txt", sep='\t', header=1, names=['RMSD'])

            num_valid_examples = len(valid['RMSD'].to_numpy())
            num_test_examples = len(test['RMSD'].to_numpy())
            bins = int(min([num_valid_examples, num_test_examples])/2)

            train_rmsd = ax[0].hist(train['RMSD'].to_numpy(), bins=bins, color='b')
            valid_rmsd = ax[1].hist(valid['RMSD'].to_numpy(), bins=bins, color='r')
            test_rmsd = ax[2].hist(test['RMSD'].to_numpy(), bins=bins, color='g')
            plt.legend(('train rmsd', 'valid rmsd', 'test rmsd'))

            ax[0].set_ylabel('Training set counts')
            ax[0].grid(visible=True)
            ax[0].set_xticks(np.arange(0, max(train['RMSD'].to_numpy())+1, 10))

            ax[1].set_ylabel('Valid set counts')
            ax[1].grid(visible=True)
            ax[1].set_xticks(np.arange(0, max(valid['RMSD'].to_numpy())+1, 10))

            ax[2].set_ylabel('Test set counts')
            ax[2].grid(visible=True)
            ax[2].set_xticks(np.arange(0, max(test['RMSD'].to_numpy())+1, 10))
        else:
            fig, ax = plt.subplots(2, figsize=(10, 20))
            plt.suptitle('RMSD distribution: epoch' + str(plot_epoch) + ' ' + self.experiment)
            valid = pd.read_csv("Log/losses/log_RMSDsValidset_epoch" + str(plot_epoch) + self.experiment + ".txt",
                                sep='\t', header=1, names=['RMSD'])
            test = pd.read_csv("Log/losses/log_RMSDsTestset_epoch" + str(plot_epoch) + self.experiment + ".txt",
                               sep='\t', header=1, names=['RMSD'])

            num_valid_examples = len(valid['RMSD'].to_numpy())
            num_test_examples = len(test['RMSD'].to_numpy())
            bins = int(min([num_valid_examples, num_test_examples]) / 2)

            valid_rmsd = ax[0].hist(valid['RMSD'].to_numpy(), bins=bins, color='r')
            test_rmsd = ax[1].hist(test['RMSD'].to_numpy(), bins=bins, color='g')
            ax[0].legend(('valid rmsd'))
            ax[1].legend(('test rmsd'))

            ax[0].set_ylabel('Valid set counts')
            ax[0].grid(visible=True)
            ax[0].set_xticks(np.arange(0, max(valid['RMSD'].to_numpy()) + 1, 10))

            ax[1].set_ylabel('Test set counts')
            ax[1].grid(visible=True)
            ax[1].set_xticks(np.arange(0, max(test['RMSD'].to_numpy()) + 1, 10))

        plt.xlabel('RMSD')
        # ax[0].set_ylim([0, 20])
        avg_validRMSD = str(valid['RMSD'].mean())[:4]
        avg_testRMSD = str(test['RMSD'].mean())[:4]
        plt.savefig('figs/BF_IP_RMSD_distribution_plots/RMSDplot_epoch'+ str(plot_epoch) + '_vRMSD' + avg_validRMSD + '_tRMSD' + avg_testRMSD + self.experiment + '.png')
        if show:
            plt.show()

if __name__ == "__main__":
    # testcase = 'newdata_bugfix_docking_100epochs_'
    # testcase = 'test_datastream'
    # testcase = 'best_docking_model_epoch'
    # testcase = 'randinit_best_docking_model_epoch'
    # testcase = 'onesinit_lr4_best_docking_model_epoch'
    # testcase = '16scalar32vector_docking_epoch'
    # testcase = '1s4v_docking_epoch'

    # testcase = 'makefigs_IP_1s4v_docking_200epochs'
    # testcase = 'Checkgitmerge_IP_1s4v_docking_10epochs'
    # testcase = 'noRandseed_Checkgitmerge_IP_1s4v_docking_10epochs'
    # testcase = 'rep1_noRandseed_Checkgitmerge_IP_1s4v_docking_10epochs'
    # testcase = 'rep2_noRandseed_Checkgitmerge_IP_1s4v_docking_10epochs'
    # testcase = 'RECODE_CHECK_BFDOCKING'

    testcase = 'FINAL_CHECK_DOCKING'
    # IPLossPlotter(testcase).plot_loss()
    for epoch in range(1, 30):
        IPLossPlotter(testcase).plot_rmsd_distribution(plot_epoch=epoch, show=False)
