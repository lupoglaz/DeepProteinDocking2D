import torch
import random
from torch import optim
import sys
sys.path.append('/home/sb1638/') ## path for cluster

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from DeepProteinDocking2D.Utility.torchDataLoader import get_docking_stream
from DeepProteinDocking2D.Utility.torchDockingFFT import TorchDockingFFT
from DeepProteinDocking2D.Models.model_docking import Docking
from DeepProteinDocking2D.Utility.utility_functions import UtilityFuncs
from DeepProteinDocking2D.Utility.validation_metrics import RMSD
from DeepProteinDocking2D.Plotting.plot_IP_loss import IPLossPlotter


class BruteForceDockingTrainer:
    def __init__(self, cur_model, cur_optimizer, cur_experiment, debug=False, plotting=False):
        self.debug = debug
        self.plotting = plotting
        self.eval_freq = 1
        self.save_freq = 1
        self.model_savepath = 'Log/saved_models/'
        self.logfile_savepath = 'Log/losses/'
        self.plot_freq = Docking().plot_freq

        self.log_header = 'Epoch\tLoss\trmsd\n'
        self.log_format = '%d\t%f\t%f\n'

        self.dim = TorchDockingFFT().dim
        self.num_angles = TorchDockingFFT().num_angles

        self.model = cur_model
        self.optimizer = cur_optimizer
        self.experiment = cur_experiment

    def run_model(self, data, training=True, pos_idx=0, stream_name='trainset'):
        receptor, ligand, gt_rot, gt_txy = data

        receptor = receptor.to(device='cuda', dtype=torch.float).squeeze().unsqueeze(0)
        ligand = ligand.to(device='cuda', dtype=torch.float).squeeze().unsqueeze(0)
        gt_rot = gt_rot.to(device='cuda', dtype=torch.float).squeeze()
        gt_txy = gt_txy.to(device='cuda', dtype=torch.float).squeeze()

        if training:
            self.model.train()
        else:
            self.model.eval()

        ### run model and loss calculation
        ##### call model
        fft_score = self.model(receptor, ligand, training=training, plotting=self.plotting, plot_count=pos_idx, stream_name=stream_name)
        fft_score = fft_score.flatten()

        ### Encode ground truth transformation index into empty energy grid
        with torch.no_grad():
            target_flatindex = TorchDockingFFT().encode_transform(gt_rot, gt_txy)
            pred_rot, pred_txy = TorchDockingFFT().extract_transform(fft_score)
            rmsd_out = RMSD(ligand, gt_rot, gt_txy, pred_rot, pred_txy).calc_rmsd()

        #### Loss functions
        CE_loss = torch.nn.CrossEntropyLoss()
        loss = CE_loss(fft_score.squeeze().unsqueeze(0), target_flatindex.unsqueeze(0))

        ### check parameters and gradients
        ### if weights are frozen or updating
        if self.debug:
            self.check_model_gradients()

        if training:
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            self.model.eval()

        if self.plotting and not training:
            if pos_idx % self.plot_freq == 0:
                with torch.no_grad():
                    self.plot_pose(fft_score, receptor, ligand, gt_rot, gt_txy, pos_idx, stream_name)

        return loss.item(), rmsd_out.item()

    def train_model(self, train_epochs, train_stream=None, valid_stream=None, test_stream=None,
                    resume_training=False,
                    resume_epoch=0):
        if self.plotting:
            self.eval_freq = 1

        ### Continue training on existing model?
        start_epoch = self.resume_training_or_not(resume_training, resume_epoch)

        num_epochs = start_epoch + train_epochs

        for epoch in range(start_epoch, num_epochs):

            checkpoint_dict = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }

            if train_stream:
                ### Training epoch
                stream_name = 'TRAINset'
                self.run_epoch(train_stream, epoch, training=True, stream_name=stream_name)

                #### saving model while training
                if epoch % self.save_freq == 0:
                    model_savefile = self.model_savepath + self.experiment + str(epoch) + '.th'
                    self.save_checkpoint(checkpoint_dict, model_savefile)
                    print('saving model ' + model_savefile)

            ### Evaluation epoch
            if epoch % self.eval_freq == 0 or epoch == 1:
                if valid_stream:
                    stream_name = 'VALIDset'
                    self.run_epoch(valid_stream, epoch, training=False, stream_name=stream_name)

                if test_stream:
                    stream_name = 'TESTset'
                    self.run_epoch(test_stream, epoch, training=False, stream_name=stream_name)

    def run_epoch(self, data_stream, epoch, training=False, stream_name='train_stream'):
        stream_loss = []
        pos_idx = 0
        for data in tqdm(data_stream):
            train_output = [self.run_model(data, pos_idx=pos_idx, training=training, stream_name=stream_name)]
            stream_loss.append(train_output)
            with open(self.logfile_savepath + 'log_RMSDs'+stream_name+'_epoch' + str(epoch) + self.experiment + '.txt','a') as fout:
                fout.write('%f\n' % (train_output[0][-1]))
            pos_idx += 1

        avg_loss = np.average(stream_loss, axis=0)[0, :]
        print('\nEpoch', epoch, stream_name,':', avg_loss)
        with open(self.logfile_savepath + 'log_loss_' + stream_name + '_' + self.experiment + '.txt', 'a') as fout:
            fout.write(self.log_format % (epoch, avg_loss[0], avg_loss[1]))

    def save_checkpoint(self, state, filename):
        self.model.eval()
        torch.save(state, filename)

    def load_ckp(self, checkpoint_fpath):
        self.model.eval()
        checkpoint = torch.load(checkpoint_fpath)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return self.model, self.optimizer, checkpoint['epoch']

    def check_model_gradients(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                print('name', n, 'param', p, 'gradient', p.grad)

    ## Unused SE2 net has own Kaiming He weight initialization.
    def weights_init(self):
        if isinstance(self.model, torch.nn.Conv2d):
            print('updating convnet weights to kaiming uniform initialization')
            torch.nn.init.kaiming_uniform_(self.model.weight)
            # torch.nn.init.kaiming_normal_(model.weight)

    def resume_training_or_not(self, resume_training, resume_epoch):
        if resume_training:
            ckp_path = self.model_savepath + self.experiment + str(resume_epoch) + '.th'
            self.model, self.optimizer, start_epoch = self.load_ckp(ckp_path)
            start_epoch += 1

            # print(self.model)
            # print(list(self.model.named_parameters()))
            print('\nRESUMING TRAINING AT EPOCH', start_epoch, '\n')
            with open(self.logfile_savepath + 'log_RMSDsTRAINset_epoch' + str(start_epoch) + self.experiment + '.txt',
                      'w') as fout:
                fout.write('Training RMSD\n')
            with open(self.logfile_savepath + 'log_RMSDsVALIDset_epoch' + str(start_epoch) + self.experiment + '.txt',
                      'w') as fout:
                fout.write('Validation RMSD\n')
            with open(self.logfile_savepath + 'log_RMSDsTESTset_epoch' + str(start_epoch) + self.experiment + '.txt',
                      'w') as fout:
                fout.write('Testing RMSD\n')
        else:
            start_epoch = 1
            ### Loss log files
            with open(self.logfile_savepath + 'log_loss_TRAINset_' + self.experiment + '.txt', 'w') as fout:
                fout.write('Docking Training Loss:\n')
                fout.write(self.log_header)
            with open(self.logfile_savepath + 'log_loss_VALIDset_' + self.experiment + '.txt', 'w') as fout:
                fout.write('Docking Validation Loss:\n')
                fout.write(self.log_header)
            with open(self.logfile_savepath + 'log_loss_TESTset_' + self.experiment + '.txt', 'w') as fout:
                fout.write('Docking Testing Loss:\n')
                fout.write(self.log_header)
        return start_epoch

    @staticmethod
    def plot_pose(FFT_score, receptor, ligand, gt_rot, gt_txy, plot_count, stream_name):
        plt.close()
        plt.figure(figsize=(8, 8))
        pred_rot, pred_txy = TorchDockingFFT().extract_transform(FFT_score)
        print('extracted predicted indices', pred_rot, pred_txy)
        print('gt indices', gt_rot, gt_txy)
        rmsd_out = RMSD(ligand, gt_rot, gt_txy, pred_rot, pred_txy).calc_rmsd()
        print('RMSD', rmsd_out.item())

        pair = UtilityFuncs.plot_assembly(receptor.squeeze().detach().cpu().numpy(), ligand.squeeze().detach().cpu().numpy(),
                                          pred_rot.detach().cpu().numpy(),
                                          (pred_txy[0].detach().cpu().numpy(), pred_txy[1].detach().cpu().numpy()),
                                          gt_rot.squeeze().detach().cpu().numpy(), gt_txy.squeeze().detach().cpu().numpy())

        plt.imshow(pair.transpose())
        plt.title('Ground truth', loc='left')
        plt.title('Input')
        plt.title('Predicted pose', loc='right')
        plt.text(225, 110, "RMSD = " + str(rmsd_out.item())[:5], backgroundcolor='w')
        plt.grid(False)
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom
        plt.tick_params(
            axis='y',
            which='both',
            left=False,
            right=False,
            labelleft=False)
        plt.savefig('figs/rmsd_and_poses/'+stream_name+'_docking_pose_example' + str(plot_count) + '_RMSD' + str(rmsd_out.item())[:4] + '.png')
        # plt.show()

    def run_trainer(self, train_epochs, train_stream=None, valid_stream=None, test_stream=None, resume_training=False, resume_epoch=0):
        self.train_model(train_epochs, train_stream, valid_stream, test_stream,
                         resume_training=resume_training, resume_epoch=resume_epoch)

    def plot_evaluation_set(self, check_epoch, train_stream=None, valid_stream=None, test_stream=None):
        eval_epochs = 1
        self.train_model(eval_epochs, train_stream, valid_stream, test_stream,
                         resume_training=False, resume_epoch=check_epoch)

if __name__ == '__main__':
    #################################################################################
    # Datasets
    trainset = '../../Datasets/docking_train_set400pool'
    validset = '../../Datasets/docking_valid_set400pool'
    ### testing set
    testset = '../../Datasets/docking_test_set200pool'
    #########################
    #### initialization torch settings
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)
    # torch.autograd.set_detect_anomaly(True)
    ######################
    lr = 10 ** -4
    model = Docking().to(device=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    batch_size = 1
    max_size = None
    if batch_size > 1:
        raise NotImplementedError()
    train_stream = get_docking_stream(trainset + '.pkl', batch_size, shuffle=True, max_size=None)
    valid_stream = get_docking_stream(validset + '.pkl', batch_size=1, max_size=None)
    test_stream = get_docking_stream(testset + '.pkl', batch_size=1, max_size=None)

    ######################
    train_epochs = 30
    # experiment = 'SMALLDATA_100EXAMPLES' ## best test rmsd 5.1
    # experiment = 'BF_IP_NEWDATA_CHECK_100pool'
    experiment = 'BF_IP_NEWDATA_CHECK_400pool_30ep'

    ######################
    ### Train model from beginning
    BruteForceDockingTrainer(model, optimizer, experiment).run_trainer(train_epochs, train_stream, valid_stream=valid_stream, test_stream=test_stream)

    ### Resume training model at chosen epoch
    # BruteForceDockingTrainer(model, optimizer, experiment).run_trainer(
    #     train_epochs=1, train_stream=None, valid_stream=valid_stream, test_stream=test_stream,
    #     resume_training=True, resume_epoch=train_epochs-1)

    ### Evaluate model on chosen dataset only and plot at chosen epoch and dataset frequency
    BruteForceDockingTrainer(model, optimizer, experiment, plotting=False).plot_evaluation_set(
        check_epoch=30, train_stream=train_stream, valid_stream=valid_stream, test_stream=test_stream)

    ## Plot loss from current experiment
    IPLossPlotter(experiment).plot_loss()
    IPLossPlotter(experiment).plot_rmsd_distribution(plot_epoch=train_epochs, show=True)
