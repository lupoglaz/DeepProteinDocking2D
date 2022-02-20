import torch
import random
from torch import optim
import sys
sys.path.append('/home/sb1638/') ## path for cluster

import numpy as np
from tqdm import tqdm
from DeepProteinDocking2D.torchDataset import get_docking_stream
from DeepProteinDocking2D.Models.BruteForce.TorchDockingFFT import TorchDockingFFT
from DeepProteinDocking2D.Models.BruteForce.model_bruteforce_docking import BruteForceDocking
from DeepProteinDocking2D.Models.BruteForce.utility_functions import plot_assembly
from DeepProteinDocking2D.Models.BruteForce.validation_metrics import RMSD
import matplotlib.pyplot as plt
from plot_IP_loss import LossPlotter

class BruteForceDockingTrainer:
    def __init__(self, model, optimizer, train=True, debug=False, plotting=False):
        self.train = train
        self.debug = debug
        self.plotting = plotting
        self.test_freq = 1
        self.save_freq = 1

        self.dim = TorchDockingFFT().dim
        self.num_angles = TorchDockingFFT().num_angles

        self.model = model
        self.optimizer = optimizer

    def run_model(self, data, model):
        receptor, ligand, gt_txy, gt_rot, _ = data

        receptor = receptor.squeeze()
        ligand = ligand.squeeze()
        gt_txy = gt_txy.squeeze()
        gt_rot = gt_rot.squeeze()

        receptor = receptor.to(device='cuda', dtype=torch.float).unsqueeze(0)
        ligand = ligand.to(device='cuda', dtype=torch.float).unsqueeze(0)
        gt_rot = gt_rot.to(device='cuda', dtype=torch.float)
        gt_txy = gt_txy.to(device='cuda', dtype=torch.float)

        if self.train:
            model.train()

        ### run model and loss calculation
        ##### call model
        FFT_score = model(receptor, ligand, plotting=self.plotting)
        FFT_score = FFT_score.flatten()

        ### Encode ground truth transformation index into empty energy grid
        with torch.no_grad():
            target_flatindex = TorchDockingFFT().encode_transform(gt_rot, gt_txy)
            pred_rot, pred_txy = TorchDockingFFT().extract_transform(FFT_score)
            rmsd_out = RMSD(ligand, gt_rot, gt_txy, pred_rot, pred_txy).calc_rmsd()
            # print('extracted predicted indices', pred_rot, pred_txy)
            # print('gt indices', gt_rot, gt_txy)
            # print('RMSD', rmsd_out.item())
            # print('Lowest energy', -FFT_score.flatten()[target_flatindex])

        #### Loss functions
        CE_loss = torch.nn.CrossEntropyLoss()
        loss = CE_loss(FFT_score.squeeze().unsqueeze(0), target_flatindex.unsqueeze(0))

        ### check parameters and gradients
        ### if weights are frozen or updating
        if self.debug:
            self.check_model_gradients()

        if self.train:
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            self.model.eval()

        if self.plotting and eval:
            with torch.no_grad():
                self.plot_pose(FFT_score, receptor, ligand, gt_rot, gt_txy)

        return loss.item(), rmsd_out.item()

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
                print(n, p, p.grad)

    ## Unused SE2 net has own Kaiming He weight initialization.
    def weights_init(self):
        if isinstance(self.model, torch.nn.Conv2d):
            print('updating convnet weights to kaiming uniform initialization')
            torch.nn.init.kaiming_uniform_(self.model.weight)
            # torch.nn.init.kaiming_normal_(model.weight)

    def train_model(self, model, optimizer, experiment, train_epochs, train_stream, valid_stream, test_stream,
                    resume_training=False,
                    resume_epoch=0):

        if self.plotting:
            self.test_freq = 1

        log_header = 'Epoch\tLoss\trmsd\n'
        log_format = '%d\t%f\t%f\n'

        ### Continue training on existing model?
        if resume_training:
            ckp_path = 'Log/' + experiment + str(resume_epoch) + '.th'
            self.model, self.optimizer, start_epoch = self.load_ckp(ckp_path)
            start_epoch += 1

            print(self.model)
            print(list(self.model.named_parameters()))
            print('\nRESUMING TRAINING AT EPOCH', start_epoch, '\n')
        else:
            start_epoch = 1
            ### Loss log files
            with open('Log/losses/log_train_' + experiment + '.txt', 'w') as fout:
                fout.write('Docking Training Loss:\n')
                fout.write(log_header)
            with open('Log/losses/log_valid_' + experiment + '.txt', 'w') as fout:
                fout.write('Docking Validation Loss:\n')
                fout.write(log_header)
            with open('Log/losses/log_test_' + experiment + '.txt', 'w') as fout:
                fout.write('Docking Testing Loss:\n')
                fout.write(log_header)

        num_epochs = start_epoch + train_epochs

        for epoch in range(start_epoch, num_epochs):

            checkpoint_dict = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            ### Training epoch
            train_loss = []
            for data in tqdm(train_stream):
                train_output = [self.run_model(data, model)]
                train_loss.append(train_output)
                with open('Log/losses/log_RMSDsTrainset_epoch' + str(epoch) + experiment + '.txt', 'a') as fout:
                    fout.write('%f\n' % (train_output[0][-1]))

            avg_trainloss = np.average(train_loss, axis=0)[0, :]
            print('\nEpoch', epoch, 'Train Loss:', avg_trainloss)
            with open('Log/losses/log_train_' + experiment + '.txt', 'a') as fout:
                fout.write(log_format % (epoch, avg_trainloss[0], avg_trainloss[1]))

            ### Evaluation epoch
            if epoch % self.test_freq == 0 or epoch == 1:
                valid_loss = []
                for data in tqdm(valid_stream):
                    valid_output = [self.run_model(data, model)]
                    valid_loss.append(valid_output)
                    with open('Log/losses/log_RMSDsValidset_epoch' + str(epoch) + experiment + '.txt', 'a') as fout:
                        fout.write('%f\n' % (valid_output[0][-1]))

                avg_validloss = np.average(valid_loss, axis=0)[0, :]
                print('\nEpoch', epoch, 'VALID LOSS:', avg_validloss)
                with open('Log/losses/log_valid_' + experiment + '.txt', 'a') as fout:
                    fout.write(log_format % (epoch, avg_validloss[0], avg_validloss[1]))

                test_loss = []
                for data in tqdm(test_stream):
                    test_output = [self.run_model(data, model)]
                    test_loss.append(test_output)
                    with open('Log/losses/log_RMSDsTestset_epoch' + str(epoch) + experiment + '.txt', 'a') as fout:
                        fout.write('%f\n' % (test_output[0][-1]))

                avg_testloss = np.average(test_loss, axis=0)[0, :]
                print('\nEpoch', epoch, 'TEST LOSS:', avg_testloss)
                with open('Log/losses/log_test_' + experiment + '.txt', 'a') as fout:
                    fout.write(log_format % (epoch, avg_testloss[0], avg_testloss[1]))

            #### saving model while training
            if epoch % self.save_freq == 0:
                self.save_checkpoint(checkpoint_dict, 'Log/' + experiment + str(epoch) + '.th')
                print('saving model ' + 'Log/' + experiment + str(epoch) + '.th')
            if epoch == num_epochs - 1:
                self.save_checkpoint(checkpoint_dict, 'Log/' + experiment + 'end.th')
                print('saving LAST EPOCH model ' + 'Log/' + experiment + str(epoch) + '.th')

    @staticmethod
    def plot_pose(FFT_score, receptor, ligand, gt_rot, gt_txy):
        plt.close()
        plt.figure(figsize=(8, 8))
        pred_rot, pred_txy = TorchDockingFFT().extract_transform(FFT_score)
        print('extracted predicted indices', pred_rot, pred_txy)
        print('gt indices', gt_rot, gt_txy)
        rmsd_out = RMSD(ligand, gt_rot, gt_txy, pred_rot, pred_txy).calc_rmsd()
        print('RMSD', rmsd_out.item())

        pair = plot_assembly(receptor.squeeze().detach().cpu().numpy(), ligand.squeeze().detach().cpu().numpy(),
                             pred_rot.detach().cpu().numpy(),
                             (pred_txy[0].detach().cpu().numpy(), pred_txy[1].detach().cpu().numpy()),
                             gt_rot.squeeze().detach().cpu().numpy(), gt_txy.squeeze().detach().cpu().numpy())

        plt.imshow(pair.transpose())
        # plt.title('Ground truth'+' '*33+'Input'+' '*33+'Predicted pose')
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
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            right=False,  # ticks along the top edge are off
            labelleft=False)  # labels along the bottom
        plt.savefig('figs/docking_pose_RMSD' + str(rmsd_out.item())[:4] + '.png')
        # plt.show()

    def run_trainer(self, train_epochs, resume_training=False, resume_epoch=0):
        self.train_model(model, optimizer, experiment, train_epochs, train_stream, valid_stream,
                         test_stream,
                         resume_training=resume_training, resume_epoch=resume_epoch)

    def plot_evaluation_set(self, check_epoch):
        train_epochs = 1
        self.train_model(model, optimizer, experiment, train_epochs, train_stream, valid_stream,
                         test_stream,
                         resume_training=True, resume_epoch=check_epoch)

if __name__ == '__main__':
    #################################################################################
    trainset = 'toy_concave_data/docking_data_train'
    validset = 'toy_concave_data/docking_data_valid'
    ### testing set
    testset = 'toy_concave_data/docking_data_test'

    experiment = 'RECODE_CHECK_BFDOCKING'

    #########################
    #### initialization torch settings
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)

    torch.cuda.set_device(0)
    # torch.autograd.set_detect_anomaly(True)
    ######################
    lr = 10 ** -4
    model = BruteForceDocking().to(device=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_stream = get_docking_stream(trainset + '.pkl', batch_size=1)
    valid_stream = get_docking_stream(validset + '.pkl', batch_size=1)
    test_stream = get_docking_stream(testset + '.pkl', batch_size=1)

    ######################
    train_epochs = 10

    ######################
    ### Train model from beginning
    # epoch = train_epochs
    # BruteForceDockingTrainer(model, optimizer).run_trainer(train_epochs)

    ### Resume training model at chosen epoch
    # train(True, resume_epoch=100)
    # BruteForceDockingTrainer(model, optimizer).run_trainer(train_epochs, resume_training=True, resume_epoch=20)
    # BruteForceDockingTrainer(model, optimizer).run_trainer(train_epochs=1, resume_training=True, resume_epoch=30)

    ## Plot loss from current experiment
    # LossPlotter(experiment).plot_loss()
    LossPlotter(experiment).plot_rmsd_distribution(plot_epoch=31)

    ### Evaluate model only and plot, at chosen epoch
    # BruteForceDockingTrainer(model, optimizer, plotting=True).plot_evaluation_set(check_epoch=30)
