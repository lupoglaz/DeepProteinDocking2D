import torch
import random
from torch import optim
import sys
sys.path.append('/home/sb1638/') ## path for cluster

import numpy as np
from tqdm import tqdm
from DeepProteinDocking2D.torchDataset import get_interaction_stream, get_interaction_stream_balanced
from DeepProteinDocking2D.Models.BruteForce.TorchDockingFFT import TorchDockingFFT
from DeepProteinDocking2D.Models.BruteForce.train_bruteforce_interaction import BruteForceInteraction
from DeepProteinDocking2D.Models.BruteForce.utility_functions import plot_assembly
from DeepProteinDocking2D.Models.BruteForce.validation_metrics import RMSD
import matplotlib.pyplot as plt
from DeepProteinDocking2D.Models.EnergyBased.model_energybased_sampling import EnergyBasedModel
from DeepProteinDocking2D.Models.BruteForce.validation_metrics import APR
from DeepProteinDocking2D.Models.BruteForce.plot_FI_loss import FILossPlotter


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
    ## run replicates from sbatch script args, if provided
    if len(sys.argv) > 1:
        replicate = str(sys.argv[1])
    else:
        replicate = 'single_rep'

    # @classmethod
    # def get_trainer(cls, docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment, training_case, path_pretrain):
    #     print("Creating instance")
    #     return cls(docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment, training_case, path_pretrain).__new__(BruteForceInteractionTrainer)

    def __init__(self, docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment,
                 debug=False, plotting=False):
        # print("RUNNING INIT")
        self.debug = debug
        self.plotting = plotting

        self.train_epochs = 6
        self.check_epoch = 1
        self.eval_freq = 1
        self.save_freq = 1

        self.docking_model = docking_model
        self.interaction_model = interaction_model
        self.docking_optimizer = docking_optimizer
        self.interaction_optimizer = interaction_optimizer
        self.experiment = experiment
        # self.training_case = training_case
        # self.path_pretrain = path_pretrain
        # self.set_docking_model_state()
        # self.freeze_weights()
        num_examples = max(len(train_stream), len(valid_stream), len(test_stream))
        self.buffer = SampleBuffer(num_examples=num_examples)

    def run_model(self, data, pos_idx=torch.tensor([0]), training=True, stream_name='trainset'):
        receptor, ligand, gt_interact = data

        receptor = receptor.squeeze()
        ligand = ligand.squeeze()
        gt_interact = gt_interact.squeeze()
        # print(gt_interact.shape, gt_interact)

        receptor = receptor.to(device='cuda', dtype=torch.float).unsqueeze(0)
        ligand = ligand.to(device='cuda', dtype=torch.float).unsqueeze(0)
        gt_interact = gt_interact.to(device='cuda', dtype=torch.float)

        if training:
            self.docking_model.train()
            self.interaction_model.train()

        ### run model and loss calculation
        ##### call model
        neg_alpha = self.buffer.get(pos_idx, samples_per_example=1, training=training)
        Energy, pred_rot, pred_txy, FFT_score = self.docking_model(neg_alpha, receptor, ligand, temperature='cold',
                                                           plot_count=pos_idx.item(), stream_name=stream_name,
                                                           plotting=self.plotting)
        neg_alpha = pred_rot
        self.buffer.push(neg_alpha, pos_idx)
        pred_interact, deltaF, F, F_0 = self.interaction_model(FFT_score.unsqueeze(0), plotting=self.plotting)

        ### check parameters and gradients
        ### if weights are frozen or updating
        if self.debug:
            self.check_model_gradients(self.docking_model)
            self.check_model_gradients(self.interaction_model)

        #### Loss functions
        BCEloss = torch.nn.BCELoss()
        l1_loss = torch.nn.L1Loss()
        w = 10**-5
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
        #         self.plot_pose(FFT_score, receptor, ligand, gt_rot, gt_txy, plot_count, stream_name)

        return loss.item(), L_reg.item(), deltaF.item(), F, F_0, gt_interact.item()

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
        ## Freeze weights as needed for training model cases
        # self.freeze_weights()

        if self.plotting:
            self.eval_freq = 1

        log_header = 'Epoch\tLoss\tLreg\tdeltaF\tF_0\n'
        log_format = '%d\t%f\t%f\t%f\t%f\n'

        ### Continue training on existing model?
        start_epoch = self.resume_training_or_not(resume_training, resume_epoch, log_header)

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

            train_loss = []
            pos_idx = torch.tensor([0])
            for data in tqdm(train_stream):
                train_output = [self.run_model(data, pos_idx, training=True)]
                train_loss.append(train_output)
                with open('Log/losses/log_deltaF_Trainset_epoch' + str(epoch) + self.experiment + '.txt', 'a') as fout:
                    fout.write('%f\t%f\t%f\t%d\n' % (train_output[0][2], train_output[0][3], train_output[0][4], train_output[0][5]))
                pos_idx+=1

            # FILossPlotter(self.experiment).plot_deltaF_distribution(plot_epoch=epoch, show=False)

            avg_trainloss = np.average(train_loss, axis=0)[0, :]
            print('\nEpoch', epoch, 'Train Loss: Loss, Lreg, deltaF, F_0', avg_trainloss)
            with open('Log/losses/log_train_' + self.experiment + '.txt', 'a') as fout:
                fout.write(log_format % (epoch, avg_trainloss[0], avg_trainloss[1], avg_trainloss[2], avg_trainloss[3]))

            ### evaluate on training and valid set
            ### training set to False downstream in calcAPR() run_model()
            if epoch % self.eval_freq == 0:
                self.checkAPR(epoch, valid_stream, 'valid set')
                self.checkAPR(epoch, test_stream, 'test set')

            #### saving model while training
            if epoch % self.save_freq == 0:
                self.save_checkpoint(docking_checkpoint_dict, 'Log/docking_' + self.experiment + str(epoch) + '.th', self.docking_model)
                print('saving docking model ' + 'Log/docking_' + self.experiment + str(epoch) + '.th')

                self.save_checkpoint(interaction_checkpoint_dict, 'Log/' + self.experiment + str(epoch) + '.th', self.interaction_model)
                print('saving interaction model ' + 'Log/' + self.experiment + str(epoch) + '.th')

    def checkAPR(self, check_epoch, datastream, stream_name=None):
        log_format = '%f\t%f\t%f\t%f\t%f\n'
        log_header = 'Accuracy\tPrecision\tRecall\tF1score\tMCC\n'
        print('Evaluating ', stream_name)
        # trainer = BruteForceInteractionTrainer()
        Accuracy, Precision, Recall, F1score, MCC = APR().calcAPR(datastream, self.run_model, check_epoch)
        # print(Accuracy, Precision, Recall)
        with open('Log/losses/log_validAPR_' + self.experiment + '.txt', 'a') as fout:
            fout.write('Epoch '+str(check_epoch)+'\n')
            fout.write(log_header)
            fout.write(log_format % (Accuracy, Precision, Recall, F1score, MCC))
        fout.close()

    # def freeze_weights(self):
    #     if not self.param_to_freeze:
    #         print('\nAll docking model params unfrozen\n')
    #         return
    #     for name, param in self.docking_model.named_parameters():
    #         if self.param_to_freeze == 'all':
    #             print('Freeze ALL Weights', name)
    #             param.requires_grad = False
    #         elif self.param_to_freeze in name:
    #             print('Freeze Weights', name)
    #             param.requires_grad = False
    #         else:
    #             print('Unfreeze docking model weights', name)
    #             param.requires_grad = True

    def resume_training_or_not(self, resume_training, resume_epoch, log_header):
        if resume_training:
            print('Loading docking model at', str(resume_epoch))
            ckp_path = 'Log/docking_' + self.experiment + str(resume_epoch) + '.th'
            self.docking_model, self.docking_optimizer, _ = self.load_ckp(ckp_path, self.docking_model, self.docking_optimizer)
            print('Loading interaction model at', str(resume_epoch))
            ckp_path = 'Log/' + self.experiment + str(resume_epoch) + '.th'
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
            with open('Log/losses/log_train_' + self.experiment + '.txt', 'w') as fout:
                fout.write(log_header)
            with open('Log/losses/log_deltaF_Trainset_epoch' + str(start_epoch) + self.experiment + '.txt', 'w') as fout:
                fout.write('deltaF\tF\tF_0\tLabel\n')

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

    # def set_docking_model_state(self):
    #     # CaseA: train with docking model frozen
    #     if self.training_case == 'A':
    #         print('Training expA')
    #         self.param_to_freeze = 'all'
    #         self.docking_model.load_state_dict(torch.load(path_pretrain)['state_dict'])
    #     # CaseB: train with docking model unfrozen
    #     if self.training_case == 'B':
    #         print('Training expB')
    #         lr_docking = 10 ** -5
    #         print('Docking learning rate changed to', lr_docking)
    #         # self.experiment = 'case' + self.training_case + '_lr5change_' + self.experiment
    #         self.docking_model = BruteForceDocking().to(device=0)
    #         self.docking_optimizer = optim.Adam(self.docking_model.parameters(), lr=lr_docking)
    #         self.param_to_freeze = None
    #         self.docking_model.load_state_dict(torch.load(path_pretrain)['state_dict'])
    #     # CaseC: train with docking model SE2 CNN frozen
    #     if self.training_case == 'C':
    #         print('Training expC')
    #         self.param_to_freeze = 'netSE2'  # leave "a" scoring coefficients unfrozen
    #         self.docking_model.load_state_dict(torch.load(path_pretrain)['state_dict'])
    #     # Case scratch: train everything from scratch
    #     if self.training_case == 'scratch':
    #         print('Training from scratch')
    #         self.param_to_freeze = None
    #         # self.experiment = self.training_case + '_' + self.experiment

    def run_trainer(self, train_epochs, train_stream=None, valid_stream=None, test_stream=None, resume_training=False, resume_epoch=0):
        self.train_model(train_epochs, train_stream, valid_stream, test_stream,
                         resume_training=resume_training, resume_epoch=resume_epoch)

    def plot_evaluation_set(self, check_epoch, train_stream=None, valid_stream=None, test_stream=None):
        eval_epochs = 1
        self.train_model(eval_epochs, train_stream, valid_stream, test_stream,
                         resume_training=False, resume_epoch=check_epoch)

    # @classmethod
    # def get_trainer(cls):
    #     return super(BruteForceInteractionTrainer, cls).__new__(cls)

if __name__ == '__main__':
    #################################################################################
    trainset = 'toy_concave_data/interaction_data_train'
    validset = 'toy_concave_data/interaction_data_valid'
    # ### testing set
    testset = 'toy_concave_data/interaction_data_test'

    #########################
    #### initialization torch settings
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)
    # CUDA_LAUNCH_BLOCKING = 1
    # torch.autograd.set_detect_anomaly(True)
    #########################

    lr_interaction = 10**0
    lr_docking = 10**-4
    LD_steps = 10
    debug = False
    # debug = True
    plotting = False
    # plotting = True
    show = False
    # show = True

    interaction_model = BruteForceInteraction().to(device=0)
    interaction_optimizer = optim.Adam(interaction_model.parameters(), lr=lr_interaction)

    dockingFFT = TorchDockingFFT(num_angles=1, angle=None, swap_plot_quadrants=False, debug=debug)
    docking_model = EnergyBasedModel(dockingFFT, num_angles=1, sample_steps=LD_steps, debug=debug).to(device=0)
    docking_optimizer = optim.Adam(docking_model.parameters(), lr=lr_docking)

    # max_size = 400
    # max_size = 50
    max_size = 25
    batch_size = 1
    if batch_size > 1:
        raise NotImplementedError()
    train_stream = get_interaction_stream_balanced(trainset + '.pkl', batch_size=batch_size, max_size=max_size)
    valid_stream = get_interaction_stream_balanced(validset + '.pkl', batch_size=1, max_size=max_size)
    test_stream = get_interaction_stream_balanced(testset + '.pkl', batch_size=1, max_size=max_size)
    ######################
    experiment = 'EBM_FI_23ex_1LD_10ep'

    ######################
    lr = 10 ** -3 # any lr != 1e-2 not as good
    LD_steps = 1
    debug = False
    # debug = True
    plotting = False
    # plotting = True
    show = False
    # show = True

    dockingFFT = TorchDockingFFT(num_angles=1, angle=None, swap_plot_quadrants=False, debug=debug)
    model = EnergyBasedModel(dockingFFT, num_angles=1, sample_steps=LD_steps, debug=debug).to(device=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_epochs = 10
    # continue_epochs = 1
    ######################
    ### Train model from beginning
    EnergyBasedInteractionTrainer(docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment, debug=debug).run_trainer(train_epochs, train_stream=train_stream, valid_stream=valid_stream, test_stream=test_stream)

    # EnergyBasedInteractionTrainer(docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment, debug=debug).plot_evaluation_set(check_epoch=train_epochs, valid_stream=valid_stream) ## also checks APR