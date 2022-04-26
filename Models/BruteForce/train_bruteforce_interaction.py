import random
import torch
from torch import optim

import sys
## path for cluster
sys.path.append('/home/sb1638/')

import numpy as np
from tqdm import tqdm
from DeepProteinDocking2D.Utility.torchDataLoader import get_interaction_stream
from DeepProteinDocking2D.Models.model_interaction import Interaction
from DeepProteinDocking2D.Utility.validation_metrics import APR
from DeepProteinDocking2D.Models.model_docking import Docking
from DeepProteinDocking2D.Plotting.plot_FI_loss import FILossPlotter

class BruteForceInteractionTrainer:
    ## run replicates from sbatch script args, if provided
    if len(sys.argv) > 1:
        replicate = str(sys.argv[1])
    else:
        replicate = 'single_rep'

    # @classmethod
    # def get_trainer(cls, docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment, training_case, path_pretrain):
    #     print("Creating instance")
    #     return cls(docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment, training_case, path_pretrain).__new__(BruteForceInteractionTrainer)

    def __init__(self, docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment, training_case, path_pretrain,
                 debug=False, plotting=False):
        # print("RUNNING INIT")
        self.debug = debug
        self.plotting = plotting

        self.check_epoch = 1
        self.eval_freq = 1
        self.save_freq = 1

        self.docking_model = docking_model
        self.interaction_model = interaction_model
        self.docking_optimizer = docking_optimizer
        self.interaction_optimizer = interaction_optimizer
        self.experiment = experiment
        self.training_case = training_case
        self.path_pretrain = path_pretrain
        self.set_docking_model_state()
        self.freeze_weights()

    def run_model(self, data, training=True):
        receptor, ligand, gt_interact = data

        receptor = receptor.to(device='cuda', dtype=torch.float)
        ligand = ligand.to(device='cuda', dtype=torch.float)
        gt_interact = gt_interact.to(device='cuda', dtype=torch.float).squeeze()

        if training:
            self.docking_model.train()
            self.interaction_model.train()

        ### run model and loss calculation
        ##### call model(s)
        fft_score = self.docking_model(receptor, ligand, plotting=self.plotting)
        pred_interact, deltaF, F, F_0 = self.interaction_model(fft_score, plotting=self.plotting)

        ### check parameters and gradients
        ### if weights are frozen or updating
        if self.debug:
            self.check_model_gradients(self.docking_model)
            self.check_model_gradients(self.interaction_model)

        #### Loss functions
        BCEloss = torch.nn.BCELoss()
        l1_loss = torch.nn.L1Loss()
        w = 10**-5 #* scheduler.get_last_lr()[0]
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

        return loss.item(), L_reg.item(), deltaF.item(), F.item(), F_0.item(), gt_interact.item()

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

            if train_stream:
                train_loss = []
                for data in tqdm(train_stream):
                    train_output = [self.run_model(data, training=True)]
                    train_loss.append(train_output)
                    with open('Log/losses/log_deltaF_Trainset_epoch' + str(epoch) + self.experiment + '.txt', 'a') as fout:
                        fout.write('%f\t%f\t%d\n' % (train_output[0][3], train_output[0][4], train_output[0][5]))

                FILossPlotter(self.experiment).plot_deltaF_distribution(plot_epoch=epoch, show=False)

                avg_trainloss = np.average(train_loss, axis=0)[0, :]
                print('\nEpoch', epoch, 'Train Loss: loss, L_reg, deltaF, F, F_0, gt_interact', avg_trainloss)
                with open('Log/losses/log_train_' + self.experiment + '.txt', 'a') as fout:
                    fout.write(log_format % (epoch, avg_trainloss[0], avg_trainloss[1], avg_trainloss[2], avg_trainloss[3]))

                scheduler.step()
                print(scheduler.get_last_lr())
            ### evaluate on training and valid set
            ### training set to False downstream in calcAPR() run_model()

            if epoch % self.eval_freq == 0:
                if valid_stream:
                    self.checkAPR(epoch, valid_stream, 'valid set')
                if test_stream:
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

    def freeze_weights(self):
        if not self.param_to_freeze:
            print('\nAll docking model params unfrozen\n')
            return
        for name, param in self.docking_model.named_parameters():
            if self.param_to_freeze == 'all':
                print('Freeze ALL Weights', name)
                param.requires_grad = False
            elif self.param_to_freeze in name:
                print('Freeze Weights', name)
                param.requires_grad = False
            else:
                print('Unfreeze docking model weights', name)
                param.requires_grad = True

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

    def set_docking_model_state(self):
        # CaseA: train with docking model frozen
        if self.training_case == 'A':
            print('Training expA')
            self.param_to_freeze = 'all'
            self.docking_model.load_state_dict(torch.load(path_pretrain)['state_dict'])
        # CaseB: train with docking model unfrozen
        if self.training_case == 'B':
            print('Training expB')
            lr_docking = 10 ** -5
            print('Docking learning rate changed to', lr_docking)
            # self.experiment = 'case' + self.training_case + '_lr5change_' + self.experiment
            self.docking_model = Docking().to(device=0)
            self.docking_optimizer = optim.Adam(self.docking_model.parameters(), lr=lr_docking)
            self.param_to_freeze = None
            self.docking_model.load_state_dict(torch.load(path_pretrain)['state_dict'])
        # CaseC: train with docking model SE2 CNN frozen
        if self.training_case == 'C':
            print('Training expC')
            self.param_to_freeze = 'netSE2'  # leave "a" scoring coefficients unfrozen
            self.docking_model.load_state_dict(torch.load(path_pretrain)['state_dict'])
        # Case scratch: train everything from scratch
        if self.training_case == 'scratch':
            print('Training from scratch')
            self.param_to_freeze = None
            # self.experiment = self.training_case + '_' + self.experiment

    def run_trainer(self, train_epochs, train_stream=None, valid_stream=None, test_stream=None, resume_epoch=0, resume_training=False):
        self.train_model(train_epochs, train_stream, valid_stream, test_stream,
                                                   resume_training=resume_training, resume_epoch=resume_epoch)


if __name__ == '__main__':
    #################################################################################
    # ##Datasets
    # trainset = '../../Datasets/interaction_train_set200pool'
    # validset = '../../Datasets/interaction_valid_set200pool'
    # # ### testing set
    # testset = '../../Datasets/interaction_test_set100pool'

    ##Datasets
    trainset = '../../Datasets/interaction_train_set100pool'
    validset = '../../Datasets/interaction_valid_set100pool'
    # ### testing set
    testset = '../../Datasets/interaction_test_set50pool'
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

    interaction_model = Interaction().to(device=0)
    interaction_optimizer = optim.Adam(interaction_model.parameters(), lr=lr_interaction)

    scheduler = optim.lr_scheduler.ExponentialLR(interaction_optimizer, gamma=0.95)

    docking_model = Docking().to(device=0)
    docking_optimizer = optim.Adam(docking_model.parameters(), lr=lr_docking)

    # max_size = 400
    # max_size = 1000
    # max_size = 50
    # max_size = 4500
    max_size = None
    batch_size = 1
    if batch_size > 1:
        raise NotImplementedError()
    train_stream = get_interaction_stream(trainset + '.pkl', batch_size=batch_size, max_size=max_size)
    valid_stream = get_interaction_stream(validset + '.pkl', batch_size=1, max_size=max_size)
    test_stream = get_interaction_stream(testset + '.pkl', batch_size=1, max_size=max_size)

    # experiment = 'BF_FI_SMALLDATA_100EXAMPLES'
    # experiment = 'BF_FI_NEWDATA_TEST1'
    # experiment = 'BF_FI_NEWDATA_TEST1_1000ex'
    # experiment = 'BF_FI_NEWDATA_TEST1_4500ex'

    experiment = 'BF_FI_NEWDATA_CHECK_100pool'
    # experiment = 'BF_FI_NEWDATA_CHECK_200pool_10ep'

    ##################### Load and freeze/unfreeze params (training, no eval)
    ### path to pretrained docking model
    # path_pretrain = 'Log/IP_1s4v_docking_epoch200.th'
    # path_pretrain = 'Log/RECODE_CHECK_BFDOCKING_30epochsend.th'
    path_pretrain = 'Log/FINAL_CHECK_DOCKING30.th'
    # training_case = 'A' # CaseA: train with docking model frozen
    # training_case = 'B' # CaseB: train with docking model unfrozen
    # training_case = 'C' # CaseC: train with docking model SE2 CNN frozen and scoring ("a") coeffs unfrozen
    training_case = 'scratch' # Case scratch: train everything from scratch
    experiment = 'FI_case' + training_case + '_' + experiment
    train_epochs = 20
    #####################
    ### Train model from beginning
    # BruteForceInteractionTrainer(docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment, training_case, path_pretrain
    #                              ).run_trainer(train_epochs, train_stream=train_stream, valid_stream=None, test_stream=None)

    ### Resume training model at chosen epoch
    # BruteForceInteractionTrainer(docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment, training_case, path_pretrain
    #                              ).run_trainer(resume_training=True, resume_epoch=32, train_epochs=8, train_stream=train_stream, valid_stream=None, test_stream=None)
    #
    ### Validate model at chosen epoch
    BruteForceInteractionTrainer(docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment, training_case, path_pretrain
                                 ).run_trainer(train_epochs=1, train_stream=None, valid_stream=valid_stream, test_stream=test_stream,
                                               resume_training=True, resume_epoch=40)

    ### Plot free energy distributions with learned F_0 decision threshold
    FILossPlotter(experiment).plot_deltaF_distribution(plot_epoch=40, show=True)
