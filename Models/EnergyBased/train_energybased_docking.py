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
from DeepProteinDocking2D.Models.EnergyBased.model_energybased_sampling import EnergyBasedModel


class SampleBuffer:
    def __init__(self, num_samples, max_pos=100):
        self.num_samples = num_samples
        self.max_pos = max_pos
        self.buffer = {}
        for i in range(num_samples):
            self.buffer[i] = []

    def __len__(self, i):
        return len(self.buffer[i])

    def push(self, alphas, drs, index):
        alphas = alphas.clone().detach().to(device='cpu')
        drs = drs.clone().detach().to(device='cpu')

        for alpha, dr, idx in zip(alphas, drs, index):
            i = idx.item()
            self.buffer[i].append((alpha, dr))
            if len(self.buffer[i]) > self.max_pos:
                self.buffer[i].pop(0)

    def get(self, index, num_samples, device='cuda', training=True):
        alphas = []
        drs = []
        if not training:
            # print('EVAL rand init')
            alpha = torch.rand(num_samples, 1) * 2 * np.pi - np.pi
            dr = torch.rand(num_samples, 2) * 50.0 - 25.0
            alphas.append(alpha)
            drs.append(dr)
        else:
            for idx in index:
                i = idx.item()
                if len(self.buffer[i]) >= num_samples > 1:
                    # print('buffer if num_sampler > 1')
                    lst = random.choices(self.buffer[i], k=num_samples)
                    alpha = list(map(lambda x: x[0], lst))
                    dr = list(map(lambda x: x[1], lst))
                    alphas.append(torch.stack(alpha, dim=0))
                    drs.append(torch.stack(dr, dim=0))
                    # print('len buffer >= samples')
                elif len(self.buffer[i]) == num_samples == 1:
                    # print('buffer if num_sampler == 1')
                    lst = self.buffer[i]
                    alphas.append(lst[0][0])
                    drs.append(lst[0][1])
                else:
                    # print('else rand init')
                    # alpha = torch.rand(num_samples, 1) * 2 * np.pi - np.pi
                    # dr = torch.rand(num_samples, 2) * 50.0 - 25.0
                    # alphas.append(alpha)
                    # drs.append(dr)
                    alpha = torch.zeros(num_samples, 1)
                    dr = torch.zeros(num_samples, 2)
                    alphas.append(alpha)
                    drs.append(dr)

        # print('\nalpha', alpha)
        # print('dr', dr)

        alphas = torch.stack(alphas, dim=0).to(device=device)
        drs = torch.stack(drs, dim=0).to(device=device)

        return alphas, drs


class EnergyBasedDockingTrainer:
    def __init__(self, cur_model, cur_optimizer, cur_experiment, debug=False, plotting=False):
        self.debug = debug
        self.plotting = plotting
        self.eval_freq = 5
        self.save_freq = 1
        self.plot_freq = BruteForceDocking().plot_freq

        self.dim = TorchDockingFFT().dim
        self.num_angles = TorchDockingFFT().num_angles

        self.model = cur_model
        self.optimizer = cur_optimizer
        self.experiment = cur_experiment

    def run_model(self, data, training=True, plot_count=0, stream_name='trainset'):
        receptor, ligand, gt_txy, gt_rot, pos_idx = data

        receptor = receptor.squeeze()
        ligand = ligand.squeeze()
        gt_txy = gt_txy.squeeze()
        gt_rot = gt_rot.squeeze()

        receptor = receptor.to(device='cuda', dtype=torch.float).unsqueeze(0)
        ligand = ligand.to(device='cuda', dtype=torch.float).unsqueeze(0)
        gt_rot = gt_rot.to(device='cuda', dtype=torch.float)
        gt_txy = gt_txy.to(device='cuda', dtype=torch.float)

        if training:
            self.model.train()

        ### run model and loss calculation
        ##### call model
        neg_alpha, neg_dr = self.buffer.get(pos_idx, num_samples=1, training=training)
        pred_rot, pred_txy, FFT_score = self.model(neg_alpha, neg_dr, receptor, ligand, temperature='cold')
        self.buffer.push(neg_alpha, neg_dr, pos_idx)

        ### Encode ground truth transformation index into empty energy grid
        with torch.no_grad():
            target_flatindex = TorchDockingFFT(num_angles=1, angle=None).encode_transform(gt_rot, gt_txy)
            # pred_flatindex = TorchDockingFFT(num_angles=1, angle=None).encode_transform(pred_rot, pred_txy)
            rmsd_out = RMSD(ligand, gt_rot, gt_txy, pred_rot.squeeze(), pred_txy.squeeze()).calc_rmsd()
            # print(target_flatindex.shape, FFT_score.shape)
            # print(target_flatindex, pred_flatindex)

        if self.debug:
            print('\npredicted')
            print(pred_rot, pred_txy)
            print('\nground truth')
            print(gt_rot, gt_txy)
        #### Loss functions
        CE_loss = torch.nn.CrossEntropyLoss()
        # L1_loss = torch.nn.L1Loss()
        # print(FFT_score.flatten().squeeze().unsqueeze(0).shape, target_flatindex.unsqueeze(0))
        loss = CE_loss(FFT_score.flatten().unsqueeze(0), target_flatindex.unsqueeze(0)) #+ L1_loss(pred_rot.squeeze(), gt_rot)

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

        # if self.plotting and not train:
        #     if plot_count % self.plot_freq == 0:
        #         with torch.no_grad():
        #             self.plot_pose(FFT_score, receptor, ligand, gt_rot, gt_txy, plot_count, stream_name)

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

    def train_model(self, train_epochs, train_stream=None, valid_stream=None, test_stream=None,
                    resume_training=False,
                    resume_epoch=0):

        if self.plotting:
            self.eval_freq = 1
        self.buffer = SampleBuffer(len(train_stream))

        log_header = 'Epoch\tLoss\trmsd\n'
        log_format = '%d\t%f\t%f\n'

        ### Continue training on existing model?
        if resume_training:
            ckp_path = 'Log/' + self.experiment + str(resume_epoch) + '.th'
            self.model, self.optimizer, start_epoch = self.load_ckp(ckp_path)
            start_epoch += 1

            print(self.model)
            print(list(self.model.named_parameters()))
            print('\nRESUMING TRAINING AT EPOCH', start_epoch, '\n')
        else:
            start_epoch = 1
            ### Loss log files
            with open('Log/losses/log_train_' + self.experiment + '.txt', 'w') as fout:
                fout.write('Docking Training Loss:\n')
                fout.write(log_header)
            with open('Log/losses/log_valid_' + self.experiment + '.txt', 'w') as fout:
                fout.write('Docking Validation Loss:\n')
                fout.write(log_header)
            with open('Log/losses/log_test_' + self.experiment + '.txt', 'w') as fout:
                fout.write('Docking Testing Loss:\n')
                fout.write(log_header)

        num_epochs = start_epoch + train_epochs

        for epoch in range(start_epoch, num_epochs):

            checkpoint_dict = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }

            if train_stream:
                ### Training epoch
                train_loss = []
                for data in tqdm(train_stream):
                    train_output = [self.run_model(data, training=True)]
                    train_loss.append(train_output)
                    with open('Log/losses/log_RMSDsTrainset_epoch' + str(epoch) + self.experiment + '.txt', 'a') as fout:
                        fout.write('%f\n' % (train_output[0][-1]))

                avg_trainloss = np.average(train_loss, axis=0)[0, :]
                print('\nEpoch', epoch, 'Train Loss:', avg_trainloss)
                with open('Log/losses/log_train_' + self.experiment + '.txt', 'a') as fout:
                    fout.write(log_format % (epoch, avg_trainloss[0], avg_trainloss[1]))

            ### Evaluation epoch
            if epoch % self.eval_freq == 0 or epoch == 1:
                if valid_stream:
                    stream_name = 'validset'
                    plot_count = 0
                    valid_loss = []
                    for data in tqdm(valid_stream):
                        valid_output = [self.run_model(data, training=False, plot_count=plot_count, stream_name=stream_name)]
                        valid_loss.append(valid_output)
                        with open('Log/losses/log_RMSDsValidset_epoch' + str(epoch) + self.experiment + '.txt', 'a') as fout:
                            fout.write('%f\n' % (valid_output[0][-1]))
                        plot_count += 1

                    avg_validloss = np.average(valid_loss, axis=0)[0, :]
                    print('\nEpoch', epoch, 'VALID LOSS:', avg_validloss)
                    with open('Log/losses/log_valid_' + self.experiment + '.txt', 'a') as fout:
                        fout.write(log_format % (epoch, avg_validloss[0], avg_validloss[1]))

                if test_stream:
                    stream_name = 'testset'
                    plot_count = 0
                    test_loss = []
                    for data in tqdm(test_stream):
                        test_output = [self.run_model(data, training=False, plot_count=plot_count, stream_name=stream_name)]
                        test_loss.append(test_output)
                        with open('Log/losses/log_RMSDsTestset_epoch' + str(epoch) + self.experiment + '.txt', 'a') as fout:
                            fout.write('%f\n' % (test_output[0][-1]))
                        plot_count += 1

                    avg_testloss = np.average(test_loss, axis=0)[0, :]
                    print('\nEpoch', epoch, 'TEST LOSS:', avg_testloss)
                    with open('Log/losses/log_test_' + self.experiment + '.txt', 'a') as fout:
                        fout.write(log_format % (epoch, avg_testloss[0], avg_testloss[1]))

            #### saving model while training
            if epoch % self.save_freq == 0:
                self.save_checkpoint(checkpoint_dict, 'Log/' + self.experiment + str(epoch) + '.th')
                print('saving model ' + 'Log/' + self.experiment + str(epoch) + '.th')
            if epoch == num_epochs - 1:
                self.save_checkpoint(checkpoint_dict, 'Log/' + self.experiment + 'end.th')
                print('saving LAST EPOCH model ' + 'Log/' + self.experiment + str(epoch) + '.th')

    @staticmethod
    def plot_pose(FFT_score, receptor, ligand, gt_rot, gt_txy, plot_count, stream_name):
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

    def run_trainer(self, train_epochs, train_stream, valid_stream, test_stream, resume_training=False, resume_epoch=0):
        self.train_model(train_epochs, train_stream, valid_stream, test_stream,
                         resume_training=resume_training, resume_epoch=resume_epoch)

    def plot_evaluation_set(self, check_epoch, train_stream=None, valid_stream=None, test_stream=None):
        eval_epochs = 1
        self.train_model(eval_epochs, train_stream, valid_stream, test_stream,
                         resume_training=False, resume_epoch=check_epoch)

if __name__ == '__main__':
    #################################################################################
    # Datasets
    trainset = 'toy_concave_data/docking_data_train'
    validset = 'toy_concave_data/docking_data_valid'
    ### testing set
    testset = 'toy_concave_data/docking_data_test'
    #########################
    #### initialization torch settings
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)
    torch.autograd.set_detect_anomaly(True)
    ######################
    lr = 10 ** -4
    model = EnergyBasedModel().to(device=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    batch_size = 1
    if batch_size > 1:
        raise NotImplementedError()
    train_stream = get_docking_stream(trainset + '.pkl', batch_size=batch_size)
    valid_stream = get_docking_stream(validset + '.pkl', batch_size=1)
    test_stream = get_docking_stream(testset + '.pkl', batch_size=1)

    ######################
    train_epochs = 20
    # experiment = 'first_tests'
    # experiment = 'first_tests_clamp_LD1_sample1_step1'
    # experiment = 'first_tests_clamp_LD1_sample1_step1_NOL1rot'
    experiment = 'first_tests_clamp_LD1_sample1_step1_NOL1rot_sig2550'


    ######################
    ### Train model from beginning
    EnergyBasedDockingTrainer(model, optimizer, experiment).run_trainer(train_epochs, train_stream, valid_stream, test_stream)

    ### Resume training model at chosen epoch
    # EnergyBasedDockingTrainer(model, optimizer, experiment).run_trainer(
    #     train_epochs=10, train_stream=train_stream, valid_stream=valid_stream, test_stream=test_stream,
    #     resume_training=True, resume_epoch=5)

    ## Plot loss from current experiment
    LossPlotter(experiment).plot_loss()
    LossPlotter(experiment).plot_rmsd_distribution(plot_epoch=train_epochs)

    ### Evaluate model on chosen dataset only and plot at chosen epoch and dataset frequency
    # BruteForceDockingTrainer(model, optimizer, experiment, plotting=True).plot_evaluation_set(
    #     check_epoch=30, valid_stream=valid_stream, test_stream=test_stream)
