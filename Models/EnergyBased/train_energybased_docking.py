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

    def get(self, index, samples_per_example, device='cuda', training=True):
        alphas = []
        if not training:
            # print('EVAL zeros init')
            alpha = torch.zeros(samples_per_example, 1)
            alphas.append(alpha)
        else:
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
        self.buffer = SampleBuffer(num_examples=num_examples)
        # self.buffer2 = SampleBuffer(880)

    # def encode_rotation(self, gt_rot, pred_rot):
    #     empty_1D_target = torch.zeros([360], dtype=torch.double).cuda()
    #     deg_index_rot = (((gt_rot * 180.0/np.pi) + 180.0) % 360).type(torch.long)
    #     empty_1D_target[deg_index_rot] = 1
    #     target_rotindex = torch.argmax(empty_1D_target.flatten()).cuda()
    #
    #     empty_1D_pred = torch.zeros([360], dtype=torch.double).cuda()
    #     deg_index_rot = (((pred_rot * 180.0/np.pi) + 180.0) % 360).type(torch.long)
    #     empty_1D_pred[deg_index_rot] = 1
    #     pred_rotindex = empty_1D_pred
    #     return target_rotindex, pred_rotindex

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
        neg_alpha = self.buffer.get(pos_idx, samples_per_example=1, training=training)
        pred_rot, pred_txy, FFT_score = self.model(neg_alpha, receptor, ligand, temperature='cold', plot_count=pos_idx.item(), stream_name=stream_name, plotting=self.plotting)
        self.buffer.push(pred_rot, pos_idx)

        ### Encode ground truth transformation index into empty energy grid
        with torch.no_grad():
            target_flatindex = self.dockingFFT.encode_transform(gt_rot, gt_txy)
            rmsd_out = RMSD(ligand, gt_rot, gt_txy, pred_rot.squeeze(), pred_txy.squeeze()).calc_rmsd()
            # target_rotindex, pred_rotindex = self.encode_rotation(gt_rot, pred_rot)


        if self.debug:
            print('\npredicted')
            print(pred_rot, pred_txy)
            print('\nground truth')
            print(gt_rot, gt_txy)

        #### Loss functions
        CE_loss = torch.nn.CrossEntropyLoss()
        # L1_loss = torch.nn.L1Loss()

        loss = CE_loss(FFT_score.flatten().unsqueeze(0), target_flatindex.unsqueeze(0))
        # loss = loss + CE_loss(pred_rotindex.unsqueeze(0), target_rotindex.unsqueeze(0))

        # loss = CE_loss(FFT_score_sum.flatten().unsqueeze(0), target_flatindex.unsqueeze(0))


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

    def resume_training_or_not(self, resume_training, resume_epoch, log_header):
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
    # experiment = 'cold_10LD_lr-2_10ep_contLD_Einv1e2' ## !!!! trained on 10 examples, 10 epochs lowest RMSD yet ep7 12.11, 11.18
    # experiment = 'cold_10LD_lr-2_10ep_contLD_Einv1e2_rep1' ## works
    # experiment = 'cold_10LD_lr-2_10ep_contLD_Einv1e1_10ex' !!!! RMSD yet ep8 9.87, 8.37
    # experiment = 'testing_sigmadecay_supergaussian_1LD_lr-2_testingcoeffs_step1_nosm_n4a10bpi'
    # experiment = 'testing_sigmadecay_supergaussian_10LD_lr-2_testingcoeffs_step1_nosm_n4a5b1'
    # experiment = 'testing_sigmadecay_supergaussian_10LD_lr-3_step1_withsm_a0p5b1n10' ### softmax is not the way, gets shrunken down to unpredictably small
    # experiment = 'testing_sigmadecay_supergaussian_10LD_lr-3_step1_withsm_a0p8b1n4'
    # experiment = 'testing_sigmadecay_supergaussian_10LD_lr-3_step1_NOsm_a2p5b1n2'
    # experiment = 'testing_sigmadecay_supergaussian_10LD_lr-3_step1_NOsm_a0p5b1n2_5ep_learnedA'
    # experiment = 'testing_10LD_lr-3_stepsched0p95'
    # experiment = 'testing_10LD_lr-2_step10sched0p8_sig0p05_10ep'
    # experiment = 'testing_10LD_lr-2_step10sched0p8_sigmaa0p2b0p5n2_5ep'
    # experiment = 'testing_10LD_lr-2_step10sched0p8_sigmaa0p5n2_5ep' # worth improving
    experiment = 'testing_10LD_lr-2_step10sched0p8_sigmaa0p5n2_5ep'

    ######################
    lr = 10 ** -2 # any lr != 1e-2 not as good
    LD_steps = 10
    debug = False
    # debug = True
    plotting = False
    # plotting = True
    show = False
    # show = True

    dockingFFT = TorchDockingFFT(num_angles=1, angle=None, swap_plot_quadrants=False, debug=debug)
    model = EnergyBasedModel(dockingFFT, num_angles=1, sample_steps=LD_steps, debug=debug).to(device=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_epochs = 5
    continue_epochs = 1
    ######################
    ### Train model from beginning
    EnergyBasedDockingTrainer(dockingFFT, model, optimizer, experiment, debug=debug).run_trainer(train_epochs, train_stream=train_stream, valid_stream=None, test_stream=None)

    ### Resume training model at chosen epoch
    # EnergyBasedDockingTrainer(dockingFFT, model, optimizer, experiment, plotting=True, debug=debug).run_trainer(
    #     train_epochs=1, train_stream=train_stream, valid_stream=None, test_stream=None,
    #     resume_training=True, resume_epoch=train_epochs)

    ### Plot poses and features of validation set
    # EnergyBasedDockingTrainer(dockingFFT, model, optimizer, experiment, plotting=True, debug=debug).run_trainer(
    #     train_epochs=1, train_stream=None, valid_stream=valid_stream, test_stream=valid_stream,
    #     resume_training=True, resume_epoch=5)

    for epoch in range(continue_epochs+1, train_epochs):
        ### Evaluate model using all 360 angles (or less).
        if train_epochs-1 == epoch:
            plotting = True
        eval_model = EnergyBasedModel(dockingFFT, num_angles=360).to(device=0)
        EnergyBasedDockingTrainer(dockingFFT, eval_model, optimizer, experiment, plotting=plotting).run_trainer(
            train_epochs=1, train_stream=None, valid_stream=valid_stream, test_stream=test_stream,
            resume_training=True, resume_epoch=epoch)

        ## Plot loss from current experiment
        # IPLossPlotter(experiment).plot_loss()
        IPLossPlotter(experiment).plot_rmsd_distribution(plot_epoch=epoch+1, show=show, eval_only=True)
