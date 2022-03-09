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
    def __init__(self, num_samples, max_pos=100):
        self.num_samples = num_samples
        self.max_pos = max_pos
        self.buffer = {}
        for i in range(num_samples):
            self.buffer[i] = []

    def __len__(self, i):
        return len(self.buffer[i])

    def push(self, alphas, drs, index):
        alphas = alphas.clone().detach().float().to(device='cpu')
        drs = drs.clone().detach().float().to(device='cpu')

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
            # dr = torch.rand(num_samples, 2) * 50.0 - 25.0
            alphas.append(alpha)
            # drs.append(dr)
            # alpha = torch.zeros(num_samples, 1)
            dr = torch.zeros(num_samples, 2)
            # alphas.append(alpha)
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
                    # drs.append(lst[0][1])
                    drs.append(torch.zeros(num_samples, 2))
                else:
                    # print('else rand init')
                    alpha = torch.rand(num_samples, 1) * 2 * np.pi - np.pi
                    # dr = torch.rand(num_samples, 2) * 50.0 - 25.0
                    alphas.append(alpha)
                    # drs.append(dr)
                    # alpha = torch.zeros(num_samples, 1)
                    dr = torch.zeros(num_samples, 2)
                    # alphas.append(alpha)
                    drs.append(dr)

        # print('\nalpha', alpha)
        # print('dr', dr)

        alphas = torch.stack(alphas, dim=0).to(device=device)
        drs = torch.stack(drs, dim=0).to(device=device)

        return alphas, drs


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

        self.buffer2 = SampleBuffer(880)
        self.buffer = SampleBuffer(880)

    def encode_rotation(self, gt_rot, pred_rot):
        empty_1D_target = torch.zeros([360], dtype=torch.double).cuda()
        deg_index_rot = (((gt_rot * 180.0/np.pi) + 180.0) % 360).type(torch.long)
        empty_1D_target[deg_index_rot] = 1
        target_rotindex = torch.argmax(empty_1D_target.flatten()).cuda()

        empty_1D_pred = torch.zeros([360], dtype=torch.double).cuda()
        deg_index_rot = (((pred_rot * 180.0/np.pi) + 180.0) % 360).type(torch.long)
        empty_1D_pred[deg_index_rot] = 1
        pred_rotindex = empty_1D_pred
        return target_rotindex, pred_rotindex

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
        else:
            self.model.eval()

        ### run model and loss calculation
        ##### call model
        # pos_rot, pos_txy, GT_FFT_score, GT_minE = self.model(gt_rot, gt_txy, receptor, ligand, temperature='cold')

        neg_alpha, neg_dr = self.buffer.get(pos_idx, num_samples=1, training=training)
        pred_rot, pred_txy, FFT_score, minE = self.model(neg_alpha, neg_dr, receptor, ligand, temperature='cold', plot_count=pos_idx.item(), stream_name=stream_name, plotting=self.plotting)
        neg_alpha, neg_dr = pred_rot, pred_txy
        self.buffer.push(neg_alpha, neg_dr, pos_idx)

        # neg_alpha2, neg_dr2 = self.buffer2.get(pos_idx, num_samples=1, training=training)
        # pred_rot2, pred_txy2, FFT_score2, minE2 = self.model(neg_alpha2, neg_dr2, receptor, ligand, temperature='hot')
        # neg_alpha2, neg_dr2 = pred_rot2, pred_txy2
        # self.buffer2.push(neg_alpha2, neg_dr2, pos_idx)
        #
        # # pred_rot = (pred_rot + pred_rot2)/2
        # # pred_txy = (pred_txy + pred_txy2)/2
        # FFT_score_avg = (FFT_score + FFT_score2)/2
        # FFT_score_sum = (FFT_score + FFT_score2)

        # FFT_score = -FFT_score
        # print(FFT_score)
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
        # L2_loss = torch.nn.MSELoss()
        # print(GT_minE.shape, minE.shape)
        # loss = (GT_minE**2) + -(minE**2)
        # print(loss.shape)
        # print(FFT_score.shape, target_flatindex.shape)
        # print(pred_rotindex.shape, target_rotindex.shape)
        # txy_loss = CE_loss(FFT_score.flatten().unsqueeze(0), target_flatindex.unsqueeze(0))
        # rot_loss = CE_loss(pred_rotindex.flatten().unsqueeze(0), target_rotindex.unsqueeze(0))
        # loss = txy_loss + rot_loss
        # loss = txy_loss
        # rot_loss = L1_loss(pred_rot.squeeze(), gt_rot)
        # loss = rot_loss

        loss = CE_loss(FFT_score.flatten().unsqueeze(0), target_flatindex.unsqueeze(0))
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
    train_stream = get_docking_stream(trainset + '.pkl', batch_size=batch_size)
    valid_stream = get_docking_stream(validset + '.pkl', batch_size=1)
    test_stream = get_docking_stream(testset + '.pkl', batch_size=1)

    ######################
    # experiment = 'coldonly_1LD_learnRotationLDonly_LDopt_passdockingFFT_rotindex_3ep_sig3_lr-2_NOrotlossonly'
    # experiment = 'coldp5hot5_1LD_learnRotationLDonly_3ep_lr-2_sumFFTscore'
    # experiment = 'coldp5hot5_1LD_learnRotationLDonly_lr-3_sumFFTscore_20ep'
    # experiment = 'coldp5hot5_1LD_learnRotationLDonly_lr-2_sumFFTscore_3ep_rotclamp' # 15rmsd test set, best so far
    # experiment = 'coldp5hot5_1LD_learnRotationLDonly_lr-2_sumFFTscore_3ep_rotclamp_doublestep'
    # experiment = 'coldp5hot5_1LD_learnRotationLDonly_lr-2_sumFFTscore_3ep_rotclamp'
    # experiment = 'coldp5hot5_10LD_learnRotationLDonly_lr-2_sumFFTscore_3ep_rotclamp'
    # experiment = 'coldp5hot5_1LD_learnRotationLDonly_lr-2_sumFFTscore_3ep_rotclamp_noLDeval' # best model so far, rmsds 18 valid, 14 test
    # experiment = 'coldp5hot5_10LD_learnRotationLDonly_lr-2_sumFFTscore_1ep_rotclamp_noLDeval' # feats are zeros
    # experiment = 'coldp5hot5_10LD_learnRotationLDonly_lr-2_sumFFTscore_1ep_rotclamp_withLDeval' # also feats are zeros
    # experiment = 'coldp5hot5_10LD_learnRotationLDonly_lr-2_sumFFTscore_1ep_rotclamp_withLDeval_gradcheck'
    # experiment = 'coldp5hot5_10LD_learnRotationLDonly_lr-2_sumFFTscore_1ep_rotclamp_noLDeval_gradcheck'
    # experiment = 'cold2hot10_1LD_learnRotationLDonly_lr-2_sumFFTscore_1ep_rotclamp_noLDeval'
    # experiment = 'cold2hot10_1LD_learnRotationLDonly_lr-3_sumFFTscore_1ep_rotclamp_noLDeval'
    # experiment = 'coldp5hot5_1LD_learnRotationLDonly_lr-2_sumFFTscore_3ep_rotclamp_noLDeval_rep' # best model so far, rmsds 18 valid, 14 test
    # experiment = 'coldonly_005sigma'
    # experiment = 'coldonly_2sigma'
    # experiment = 'coldonly_2sigma_10LD'
    # experiment = 'coldonly_005sigma_10LD'
    # experiment = 'coldonly_005sigma_10LD_step0p1_noclamp'
    # experiment = 'coldonly_005sigma_10LD_step0p1_noclamp_10ep_lr-3'
    # experiment = 'coldonly_005sigma_1LD_step0p1_noclamp_10ep_lr-3_contLD'
    # experiment = 'coldonly_005sigma_10LD_step0p1_noclamp_lr-2_contLD' # tied with best model so far, rmsds 18 valid, 14 test
    # experiment = 'coldonly_005sigma_1LD_step0p1_noclamp_lr-2_contLD_checkdebugplotting'
    # experiment = 'coldonly_005sigma_1LD_step0p1_noclamp_lr-2_contLD_plottingmaxvssoftmax'
    # experiment = 'coldonly_005sigma_1LD_step0p1_noclamp_lr-2_contLD_FFTsoftmax'
    experiment = 'coldonly_005sigma_10LD_step0p1_noclamp_contLD_lr-3_10ep'

    ######################
    lr = 10 ** -3
    LD_steps = 10
    debug = False
    plotting = False

    dockingFFT = TorchDockingFFT(num_angles=1, angle=None, swap_plot_quadrants=False, debug=debug)
    model = EnergyBasedModel(dockingFFT, num_angles=1, sample_steps=LD_steps, debug=debug).to(device=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_epochs = 10
    ######################
    ### Train model from beginning
    EnergyBasedDockingTrainer(dockingFFT, model, optimizer, experiment, debug=debug).run_trainer(train_epochs, train_stream=train_stream, valid_stream=None, test_stream=None)

    ### Resume training model at chosen epoch
    # EnergyBasedDockingTrainer(dockingFFT, model, optimizer, experiment, plotting=False, debug=True).run_trainer(
    #     train_epochs=1, train_stream=train_stream, valid_stream=valid_stream, test_stream=test_stream,
    #     resume_training=True, resume_epoch=train_epochs)

    ### Evaluate model using all 360 angles (or less).
    eval_model = EnergyBasedModel(dockingFFT, num_angles=360).to(device=0)
    EnergyBasedDockingTrainer(dockingFFT, eval_model, optimizer, experiment, plotting=plotting).run_trainer(
        train_epochs=1, train_stream=None, valid_stream=valid_stream, test_stream=test_stream,
        resume_training=True, resume_epoch=train_epochs)



    ## Plot loss from current experiment
    # IPLossPlotter(experiment).plot_loss()
    IPLossPlotter(experiment).plot_rmsd_distribution(plot_epoch=train_epochs+1, show=True, eval_only=True)
