import torch
import random
from torch import optim
import sys
sys.path.append('/home/sb1638/') ## path for cluster

import numpy as np
from tqdm import tqdm
from DeepProteinDocking2D.torchDataset import get_docking_stream
from DeepProteinDocking2D.Models.BruteForce.TorchDockingFFT import TorchDockingFFT
from DeepProteinDocking2D.Models.BruteForce.train_bruteforce_docking import BruteForceDockingTrainer, BruteForceDocking
from DeepProteinDocking2D.Models.BruteForce.utility_functions import plot_assembly
from DeepProteinDocking2D.Models.BruteForce.validation_metrics import RMSD
import matplotlib.pyplot as plt
from DeepProteinDocking2D.Models.EnergyBased.plot_IP_loss import IPLossPlotter
from DeepProteinDocking2D.Models.EnergyBased.model_energybased_sampling import EnergyBasedModel


class SampleBuffer:
    def __init__(self, num_examples, max_pos=100):
        self.num_examples = num_examples
        self.max_pos = max_pos
        self.buffer = {}
        for i in range(num_examples):
            self.buffer[i] = []

    def __len__(self, i):
        return len(self.buffer[i])

    def push(self, alphas, index):
        alphas = alphas.clone().detach().float().to(device='cpu')
        for alpha, idx in zip(alphas, index):
            i = idx.item()
            self.buffer[i].append((alpha))
            if len(self.buffer[i]) > self.max_pos:
                self.buffer[i].pop(0)
            # print('buffer push\n', self.buffer[i])

    def get(self, index, samples_per_example, device='cuda'):
        alphas = []
        for idx in index:
            i = idx.item()
            buffer_idx_len = len(self.buffer[i])
            if buffer_idx_len < samples_per_example:
                # print('epoch 0 init')
                alpha = torch.zeros(samples_per_example, 1)
                alphas.append(alpha)
            else:
                # print('continuous LD picking previous rotation')
                # alpha = torch.rand(samples_per_example, 1) * 2 * np.pi - np.pi
                alpha = self.buffer[i][-1]
                alphas.append(alpha)
            # print('buffer get\n', self.buffer[i])

        # print('\nalpha', alpha)
        # print('dr', dr)

        alphas = torch.stack(alphas, dim=0).to(device=device)

        return alphas


class EnergyBasedDockingTrainer:
    def __init__(self, dockingFFT, cur_model, cur_optimizer, cur_experiment, debug=False, plotting=False):

        self.debug = debug
        self.plotting = plotting
        self.eval_freq = 1
        self.save_freq = 1
        self.plot_freq = BruteForceDocking().plot_freq

        # self.dockingFFT = TorchDockingFFT(num_angles=1, angle=None, swap_plot_quadrants=False, debug=debug)
        self.dockingFFT = dockingFFT
        self.dim = TorchDockingFFT().dim
        self.num_angles = TorchDockingFFT().num_angles

        self.model = cur_model
        self.optimizer = cur_optimizer
        self.experiment = cur_experiment

        num_examples = max(len(train_stream), len(valid_stream), len(test_stream))
        # self.trainbuffer = SampleBuffer(num_examples=num_examples)
        self.evalbuffer = SampleBuffer(num_examples=num_examples)

    def run_model(self, data, training=True, plot_count=0, stream_name='trainset'):
        receptor, ligand, gt_txy, gt_rot, pos_idx = data
        # print(pos_idx)
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
        else:
            self.model.eval()

        ### run model and loss calculation
        ##### call model
        if training:
            neg_energy, pred_rot, pred_txy, FFT_score = self.model(gt_rot, receptor, ligand, plot_count=pos_idx.item(), stream_name=stream_name, plotting=self.plotting)
        else:
            alpha = self.evalbuffer.get(pos_idx, samples_per_example=1)
            energy, pred_rot, pred_txy, FFT_score = self.model(alpha, receptor, ligand, plot_count=pos_idx.item(), stream_name=stream_name, plotting=self.plotting, training=False)
            self.evalbuffer.push(pred_rot, pos_idx)
            # print(neg_alpha, pred_rot)
            # pred_txy = gt_txy

        # neg_energy, pred_rot, pred_txy, FFT_score = self.model(gt_rot, receptor, ligand, plot_count=pos_idx.item(), stream_name=stream_name, plotting=self.plotting)

        ### Encode ground truth transformation index into empty energy grid
        with torch.no_grad():
            rmsd_out = RMSD(ligand, gt_rot, gt_txy, pred_rot.squeeze(), pred_txy.squeeze()).calc_rmsd()
            target_flatindex = self.dockingFFT.encode_transform(gt_rot, gt_txy)

        if self.debug:
            print('\npredicted')
            print(pred_rot, pred_txy)
            print('\nground truth')
            print(gt_rot, gt_txy)

        ### check parameters and gradients
        ### if weights are frozen or updating
        if self.debug:
            self.check_model_gradients()

        if training:
            #### Loss functions
            CE_loss = torch.nn.CrossEntropyLoss()
            loss = CE_loss(FFT_score.flatten().unsqueeze(0), target_flatindex.unsqueeze(0))
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            loss = torch.zeros(1)
            self.model.eval()
            if self.plotting and plot_count % self.plot_freq == 0:
                with torch.no_grad():
                    self.plot_pose(receptor, ligand, gt_rot, gt_txy, pred_rot.squeeze(), pred_txy.squeeze(), pos_idx.item(), stream_name)

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
                print('name', n, 'param', p, 'gradient', p.grad)

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

        log_header = 'Epoch\tLoss\trmsd\n'
        log_format = '%d\t%f\t%f\n'

        ### Continue training on existing model?
        start_epoch = self.resume_training_or_not(resume_training, resume_epoch, log_header)

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

                # scheduler.step()
                # print(scheduler.get_last_lr())

                avg_trainloss = np.average(train_loss, axis=0)[0, :]
                print('\nEpoch', epoch, 'Train Loss:', avg_trainloss)
                with open('Log/losses/log_train_' + self.experiment + '.txt', 'a') as fout:
                    fout.write(log_format % (epoch, avg_trainloss[0], avg_trainloss[1]))

                #### saving model while training
                if epoch % self.save_freq == 0:
                    self.save_checkpoint(checkpoint_dict, 'Log/' + self.experiment + str(epoch) + '.th')
                    print('saving model ' + 'Log/' + self.experiment + str(epoch) + '.th')
                if epoch == num_epochs - 1:
                    self.save_checkpoint(checkpoint_dict, 'Log/' + self.experiment + 'end.th')
                    print('saving LAST EPOCH model ' + 'Log/' + self.experiment + str(epoch) + '.th')

            ### Evaluation epoch
            if epoch % self.eval_freq == 0 or epoch == 1:
                if valid_stream:
                    stream_name = 'validset'
                    # for epoch in range(10):
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

    def resume_training_or_not(self, resume_training, resume_epoch, log_header, bf_path=None):
        if resume_training:
            ckp_path = 'Log/' + self.experiment + str(resume_epoch) + '.th'
            self.model, self.optimizer, start_epoch = self.load_ckp(ckp_path)
            start_epoch += 1

            # print(self.model)
            # print(list(self.model.named_parameters()))
            print('\nRESUMING TRAINING AT EPOCH', start_epoch, '\n')
            with open('Log/losses/log_RMSDsTrainset_epoch' + str(start_epoch) + self.experiment + '.txt', 'w') as fout:
                fout.write('Training RMSD\n')
            with open('Log/losses/log_RMSDsValidset_epoch' + str(start_epoch) + self.experiment + '.txt', 'w') as fout:
                fout.write('Validation RMSD\n')
            with open('Log/losses/log_RMSDsTestset_epoch' + str(start_epoch) + self.experiment + '.txt', 'w') as fout:
                fout.write('Testing RMSD\n')
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
        return start_epoch

    def plot_pose(self, receptor, ligand, gt_rot, gt_txy, pred_rot, pred_txy, plot_count, stream_name):
        plt.close()
        plt.figure(figsize=(8, 8))
        # pred_rot, pred_txy = self.dockingFFT.extract_transform(FFT_score)
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
    # torch.autograd.set_detect_anomaly(True)
    # ######################
    # lr = 10 ** -4
    # model = EnergyBasedModel(sample_steps=10).to(device=0)
    # optimizer = optim.Adam(model.parameters(), lr=lr)

    batch_size = 1
    if batch_size > 1:
        raise NotImplementedError()
    train_stream = get_docking_stream(trainset + '.pkl', batch_size)
    valid_stream = get_docking_stream(validset + '.pkl', batch_size=1)
    test_stream = get_docking_stream(testset + '.pkl', batch_size=1)

    ######################
    # experiment = 'BSmodel_check'
    # experiment = 'BSmodel_MHalgo_eval'
    # experiment = 'BSmodel_MHalgo_freeEeval'
    experiment = 'BSmodel_lr-2_5ep_refmodel' #
    # RMSD 2.82 both valid/test; MC 100step RMSD valid 6.49; 150 step RMSD 6.20; 200 step RMSD 5.27; 250 step RMSD 3.56

    # experiment = 'BSmodel_lr-2_5ep_test_unsignedScoring' # ## doesn't work...
    # experiment = 'BSmodel_lr-2_5ep_test_signedBULKScoring' #
    # experiment = 'BSmodel_lr-2_5ep_test_unsignedBULKScoring_minimizing' # doesn't work
    # experiment = 'BSmodel_lr-2_5ep_test_signedBULKScoring_minimizing' ## doesn't work
    experiment = 'BSmodel_lr-2_5ep_check_signedBULKScoring' # confirmed works

    ### For IP MC eval: sigma alpha 1 RMSD 10, 1.5 RMSD 7.81, 2 RMSD 6.59, 2.5 RMSD 7.44, 1.25, RMSD 6.82, pi/2 RMSD 8.79
    ######################
    lr = 10 ** -2
    sample_steps = 100
    debug = False
    # debug = True
    plotting = False
    # plotting = True
    show = False
    # show = True

    train_epochs = 5
    continue_epochs = 1
    ######################
    dockingFFT = TorchDockingFFT(num_angles=1, angle=None, swap_plot_quadrants=False, debug=debug)
    model = EnergyBasedModel(dockingFFT, num_angles=1, IP=True, sample_steps=sample_steps, debug=debug).to(device=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    ### Train model from beginning
    # EnergyBasedDockingTrainer(dockingFFT, model, optimizer, experiment, debug=debug).run_trainer(train_epochs, train_stream=train_stream)

    ## Brute force eval and plotting
    start = 1
    stop = train_epochs
    eval_angles = 360
    for epoch in range(start, stop):
        ### Evaluate model using all 360 angles (or less).
        if stop-1 == epoch:
            plotting = True
        eval_model = EnergyBasedModel(dockingFFT, num_angles=eval_angles, IP=True).to(device=0)
        EnergyBasedDockingTrainer(dockingFFT, eval_model, optimizer, experiment, plotting=plotting).run_trainer(
            train_epochs=1, train_stream=None, valid_stream=valid_stream, test_stream=test_stream,
            resume_training=True, resume_epoch=epoch)

    ## Plot loss from current experiment
    # IPLossPlotter(experiment).plot_loss(ylim=10)
    IPLossPlotter(experiment).plot_rmsd_distribution(plot_epoch=epoch+1, show=show, eval_only=True)

    ### Resume training model at chosen epoch
    # EnergyBasedDockingTrainer(dockingFFT, model, optimizer, experiment, plotting=True, debug=debug).run_trainer(
    #     train_epochs=1, train_stream=train_stream, valid_stream=None, test_stream=None,
    #     resume_training=True, resume_epoch=train_epochs)

    ### Resume training for validation sets
    # EnergyBasedDockingTrainer(dockingFFT, model, optimizer, experiment, plotting=plotting, debug=debug).run_trainer(
    #     train_epochs=1, train_stream=None, valid_stream=valid_stream, #test_stream=valid_stream,
    #     resume_training=True, resume_epoch=train_epochs)


    # # ########### Metropolis-Hastings eval on ideal learned energy surface
    # dockingFFT = TorchDockingFFT(num_angles=1, angle=None, swap_plot_quadrants=False, debug=debug)
    # model = EnergyBasedModel(dockingFFT, num_angles=1, IP=True, sample_steps=sample_steps, debug=debug).to(device=0)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    #
    # monte_carlo_eval = EnergyBasedModel(dockingFFT, num_angles=1, sample_steps=sample_steps, IP_MC=True).to(device=0)
    # EnergyBasedDockingTrainer(dockingFFT, monte_carlo_eval, optimizer, experiment, plotting=plotting).run_trainer(
    #     train_epochs=1, train_stream=None, valid_stream=valid_stream, test_stream=None,
    #     resume_training=True, resume_epoch=train_epochs)
    #
    # IPLossPlotter(experiment).plot_rmsd_distribution(plot_epoch=train_epochs + 1, show=show, eval_only=True)
    #
    # ## Sampling based eval and plotting
    # start = train_epochs - 1
    # stop = train_epochs
    # plot_BFeval_refplots = True
    # for epoch in range(start, stop):
    #     if stop - 1 == epoch:
    #         plotting = True
    #         if plot_BFeval_refplots:
    #             eval_model = EnergyBasedModel(dockingFFT, num_angles=360, sample_steps=sample_steps, IP=True).to(device=0)
    #             EnergyBasedDockingTrainer(dockingFFT, eval_model, optimizer, experiment, plotting=plotting).run_trainer(
    #                 train_epochs=1, train_stream=None, valid_stream=valid_stream, test_stream=test_stream,
    #                 resume_training=True, resume_epoch=epoch)
    #             # Plot loss from current experiment
    #             IPLossPlotter(experiment).plot_loss()
    #             IPLossPlotter(experiment).plot_rmsd_distribution(plot_epoch=epoch + 1, show=show, eval_only=True)
