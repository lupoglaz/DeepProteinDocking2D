import torch
import random
from torch import optim
import sys
sys.path.append('/home/sb1638/') ## path for cluster

import numpy as np
from tqdm import tqdm
from DeepProteinDocking2D.Utility.torchDataLoader import get_interaction_stream
from DeepProteinDocking2D.Utility.torchDockingFFT import TorchDockingFFT
from DeepProteinDocking2D.Models.BruteForce.train_bruteforce_interaction import Interaction

from DeepProteinDocking2D.Models.EnergyBased.model_energybased_sampling import EnergyBasedModel
from DeepProteinDocking2D.Utility.validation_metrics import APR
from DeepProteinDocking2D.Plotting.plot_FI_loss import FILossPlotter


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


class EnergyBasedInteractionTrainer:

    def __init__(self, docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment,
                 debug=False, plotting=False):
        # print("RUNNING INIT")
        self.debug = debug
        self.plotting = plotting

        # self.train_epochs = 6
        self.check_epoch = 1
        self.eval_freq = 1
        self.save_freq = 1

        self.model_savepath = 'Log/saved_models/'
        self.logfile_savepath = 'Log/losses/'

        self.loss_log_header = 'Epoch\tLoss\n'
        self.loss_log_format = '%d\t%f\n'

        self.deltaf_log_header = 'F\tF_0\tLabel\n'
        self.deltaf_log_format = '%f\t%f\t%d\n'

        self.docking_model = docking_model
        self.interaction_model = interaction_model
        self.docking_optimizer = docking_optimizer
        self.interaction_optimizer = interaction_optimizer
        self.experiment = experiment

        num_examples = max(len(train_stream), len(valid_stream), len(test_stream))
        self.buffer = SampleBuffer(num_examples=num_examples)

        self.sig_alpha = 3
        self.wReg = 10**-5

    def run_model(self, data, pos_idx=torch.tensor([0]), training=True, stream_name='trainset'):
        receptor, ligand, gt_interact = data

        receptor = receptor.to(device='cuda', dtype=torch.float).squeeze().unsqueeze(0)
        ligand = ligand.to(device='cuda', dtype=torch.float).squeeze().unsqueeze(0)
        gt_interact = gt_interact.to(device='cuda', dtype=torch.float).squeeze()

        if training:
            self.docking_model.train()
            self.interaction_model.train()
        else:
            self.docking_model.eval()
            self.interaction_model.eval()

        ### run model and loss calculation
        ##### call model
        alpha = self.buffer.get(pos_idx, samples_per_example=1)
        energy, pred_rot, pred_txy, FFT_score_stack = self.docking_model(alpha, receptor, ligand, sig_alpha=self.sig_alpha, plot_count=pos_idx.item(),
                                                           stream_name=stream_name, plotting=self.plotting,
                                                           training=training)
        self.buffer.push(pred_rot, pos_idx)
        pred_interact, deltaF, F, F_0 = self.interaction_model(FFT_score_stack.unsqueeze(0), plotting=self.plotting, debug=False)

        ### check parameters and gradients
        ### if weights are frozen or updating
        if self.debug:
            self.check_model_gradients(self.docking_model)
            self.check_model_gradients(self.interaction_model)

        #### Loss functions
        BCEloss = torch.nn.BCELoss()
        l1_loss = torch.nn.L1Loss()
        w = 10 ** -5  # * scheduler.get_last_lr()[0]
        L_reg = w * l1_loss(deltaF, torch.zeros(1).squeeze().cuda())
        loss = BCEloss(pred_interact, gt_interact) + L_reg

        if self.debug:
            print('\n predicted', pred_interact.item(), '; ground truth', gt_interact.item())

        if training:
            self.docking_model.zero_grad()
            self.interaction_model.zero_grad()
            loss.backward(retain_graph=True)
            self.docking_optimizer.step()
            self.interaction_optimizer.step()
        else:
            self.docking_model.eval()
            self.interaction_model.eval()
            with torch.no_grad():
                return self.classify(pred_interact, gt_interact)

        # if self.plotting and not training:
        #     # if plot_count % self.plot_freq == 0:
        #     with torch.no_grad():
        #         self.plot_pose(fft_score, receptor, ligand, gt_rot, gt_txy, plot_count, stream_name)

        return loss.item(), F.item(), F_0.item(), gt_interact.item()

    @staticmethod
    def classify(pred_interact, gt_interact):
        threshold = 0.5
        TP, FP, TN, FN = 0, 0, 0, 0
        p = pred_interact.item()
        a = gt_interact.item()
        if p >= threshold and a >= threshold:
            TP += 1
        elif p >= threshold and a < threshold:
            FP += 1
        elif p < threshold and a >= threshold:
            FN += 1
        elif p < threshold and a < threshold:
            TN += 1
        # print('returning', TP, FP, TN, FN)
        return TP, FP, TN, FN

    def train_model(self, train_epochs, train_stream, valid_stream, test_stream, resume_training=False,
                    resume_epoch=0):
        if self.plotting:
            self.eval_freq = 1

        ### Continue training on existing model?
        start_epoch = self.resume_training_or_not(resume_training, resume_epoch)

        num_epochs = start_epoch + train_epochs

        for epoch in range(start_epoch, num_epochs):

            docking_checkpoint_dict = {
                'epoch': epoch,
                'state_dict': self.docking_model.state_dict(),
                'optimizer': self.docking_optimizer.state_dict(),
            }
            interaction_checkpoint_dict = {
                'epoch': epoch,
                'state_dict': self.interaction_model.state_dict(),
                'optimizer': self.interaction_optimizer.state_dict(),
            }

            if train_stream:
                self.run_epoch(train_stream, epoch, training=True)
                scheduler.step()
                print('last learning rate', scheduler.get_last_lr())

                FILossPlotter(self.experiment).plot_deltaF_distribution(plot_epoch=epoch, show=False, xlim=None, binwidth=1)

            ### evaluate on training and valid set
            ### training set to False downstream in calcAPR() run_model()

            if epoch % self.eval_freq == 0:
                if valid_stream:
                    self.checkAPR(epoch, valid_stream, 'VALIDset')
                if test_stream:
                    self.checkAPR(epoch, test_stream, 'TESTset')

            #### saving model while training
            if epoch % self.save_freq == 0:
                docking_savepath =  self.model_savepath + 'docking_' + self.experiment + str(epoch) + '.th'
                self.save_checkpoint(docking_checkpoint_dict, docking_savepath, self.docking_model)
                print('saving docking model ' + docking_savepath)

                interaction_savepath = self.model_savepath + self.experiment + str(epoch) + '.th'
                self.save_checkpoint(interaction_checkpoint_dict, interaction_savepath, self.interaction_model)
                print('saving interaction model ' + interaction_savepath)

    def run_epoch(self, data_stream, epoch, training=False):
        stream_loss = []
        with open(self.logfile_savepath + 'log_deltaF_TRAINset_epoch' + str(epoch) + self.experiment + '.txt', 'w') as fout:
            fout.write(self.deltaf_log_header)
        for data in tqdm(data_stream):
            train_output = [self.run_model(data, training=training)]
            stream_loss.append(train_output)
            with open(self.logfile_savepath + 'log_deltaF_TRAINset_epoch' + str(epoch) + self.experiment + '.txt', 'a') as fout:
                fout.write(self.deltaf_log_format % (train_output[0][1], train_output[0][2], train_output[0][3]))

        avg_loss = np.average(stream_loss, axis=0)[0, :]
        print('\nEpoch', epoch, 'Train Loss: epoch, loss', avg_loss)
        with open(self.logfile_savepath + 'log_loss_TRAINset_' + self.experiment + '.txt', 'a') as fout:
            fout.write(self.loss_log_format % (epoch, avg_loss[0]))

    def checkAPR(self, check_epoch, datastream, stream_name=None):
        log_APRheader = 'Accuracy\tPrecision\tRecall\tF1score\tMCC\n'
        log_APRformat = '%f\t%f\t%f\t%f\t%f\n'
        print('Evaluating ', stream_name)
        Accuracy, Precision, Recall, F1score, MCC = APR().calcAPR(datastream, self.run_model, check_epoch)
        with open(self.logfile_savepath + 'log_validAPR_' + self.experiment + '.txt', 'a') as fout:
            fout.write('Epoch '+str(check_epoch)+'\n')
            fout.write(log_APRheader)
            fout.write(log_APRformat % (Accuracy, Precision, Recall, F1score, MCC))
        fout.close()

    def resume_training_or_not(self, resume_training, resume_epoch):
        if resume_training:
            print('Loading docking model at', str(resume_epoch))
            ckp_path = self.model_savepath+'docking_' + self.experiment + str(resume_epoch) + '.th'
            self.docking_model, self.docking_optimizer, _ = self.load_ckp(ckp_path, self.docking_model, self.docking_optimizer)
            print('Loading interaction model at', str(resume_epoch))
            ckp_path = self.model_savepath + self.experiment + str(resume_epoch) + '.th'
            self.interaction_model, self.interaction_optimizer, start_epoch = self.load_ckp(ckp_path, self.interaction_model, self.interaction_optimizer)

            start_epoch += 1

            print('\ndocking model:\n', self.docking_model)
            ## print model and params being loaded
            self.check_model_gradients(self.docking_model)
            print('\ninteraction model:\n', self.interaction_model)
            ## print model and params being loaded
            self.check_model_gradients(self.interaction_model)

            print('\nLOADING MODEL AT EPOCH', start_epoch, '\n')
        else:
            start_epoch = 1
            ### Loss log files
            with open(self.logfile_savepath + 'log_loss_TRAINset_' + self.experiment + '.txt', 'w') as fout:
                fout.write(self.loss_log_header)
            with open(self.logfile_savepath + 'log_deltaF_TRAINset_epoch' + str(start_epoch) + self.experiment + '.txt', 'w') as fout:
                fout.write(self.deltaf_log_header)

        return start_epoch

    @staticmethod
    def save_checkpoint(state, filename, model):
        model.eval()
        torch.save(state, filename)

    @staticmethod
    def load_ckp(checkpoint_fpath, model, optimizer):
        model.eval()
        checkpoint = torch.load(checkpoint_fpath)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer, checkpoint['epoch']

    @staticmethod
    def check_model_gradients(model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                print('Name', n, '\nParam', p, '\nGradient', p.grad)

    def run_trainer(self, train_epochs, train_stream=None, valid_stream=None, test_stream=None, resume_training=False, resume_epoch=0):
        self.train_model(train_epochs, train_stream, valid_stream, test_stream,
                         resume_training=resume_training, resume_epoch=resume_epoch)

if __name__ == '__main__':
    #################################################################################
    # Datasets
    trainset = '../../Datasets/interaction_train_set200pool'
    validset = '../../Datasets/interaction_valid_set200pool'
    ### testing set
    testset = '../../Datasets/interaction_test_set100pool'

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

    #########################
    max_size = 1000
    batch_size = 1
    if batch_size > 1:
        raise NotImplementedError()
    train_stream = get_interaction_stream(trainset + '.pkl', batch_size=batch_size, shuffle=True, max_size=max_size)
    valid_stream = get_interaction_stream(validset + '.pkl', batch_size=1, max_size=max_size)
    test_stream = get_interaction_stream(testset + '.pkl', batch_size=1, max_size=max_size)
    ######################
    # experiment = 'workingMCsampling_50steps_wregsched_g=0.50_modelEvalMCloop_100ex_sigalpha=3' ## 15ep MCC 0.40 valid/test
    # experiment = 'MC_FI_SMALLDATA_100EXAMPLES_50STEPS' ## 15ep MCC 0.40 valid/test
    # experiment = 'MC_FI_NEWDATA_CHECK_100pool_1000ex50steps'
    # experiment = 'MC_FI_NEWDATA_CHECK_400pool_2000ex10steps'
    experiment = 'MC_FI_NEWDATA_CHECK_400pool_1000ex10steps'

    lr_interaction = 10 ** 0
    lr_docking = 10 ** -4
    sample_steps = 10
    debug = False
    # debug = True
    plotting = False
    # plotting = True
    show = False
    # show = True

    interaction_model = Interaction().to(device=0)
    interaction_optimizer = optim.Adam(interaction_model.parameters(), lr=lr_interaction)

    scheduler = optim.lr_scheduler.ExponentialLR(interaction_optimizer, gamma=0.50)

    dockingFFT = TorchDockingFFT(num_angles=1, angle=None, swap_plot_quadrants=False, debug=debug)
    docking_model = EnergyBasedModel(dockingFFT, num_angles=1, sample_steps=sample_steps, FI=True, debug=debug).to(device=0)
    docking_optimizer = optim.Adam(docking_model.parameters(), lr=lr_docking)

    # sigma_optimizer = optim.Adam(docking_model.parameters(), lr=2)
    # scheduler = optim.lr_scheduler.ExponentialLR(sigma_optimizer, gamma=0.95)

    train_epochs = 40
    # continue_epochs = 1
    ######################
    ### Train model from beginning
    # EnergyBasedInteractionTrainer(docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment, debug=debug
    #                               ).run_trainer(train_epochs, train_stream=train_stream, valid_stream=None, test_stream=None)

    ### resume training model
    EnergyBasedInteractionTrainer(docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment, debug=debug
                                 ).run_trainer(resume_training=True, resume_epoch=13, train_epochs=27,
                                               train_stream=train_stream, valid_stream=None, test_stream=None)
    #
    ### Evaluate model at chosen epoch
    eval_model = EnergyBasedModel(dockingFFT, num_angles=360, sample_steps=1, FI=True, debug=debug).to(device=0)
    # # eval_model = EnergyBasedModel(dockingFFT, num_angles=1, sample_steps=sample_steps, FI=True, debug=debug).to(device=0) ## eval with monte carlo
    EnergyBasedInteractionTrainer(eval_model, docking_optimizer, interaction_model, interaction_optimizer, experiment, debug=False
                                  ).run_trainer(resume_training=True, resume_epoch=train_epochs, train_epochs=1,
                                                train_stream=None, valid_stream=valid_stream, test_stream=test_stream)

    ### Plot free energy distributions with learned F_0 decision threshold
    FILossPlotter(experiment).plot_loss()
    FILossPlotter(experiment).plot_deltaF_distribution(plot_epoch=train_epochs, show=True)
