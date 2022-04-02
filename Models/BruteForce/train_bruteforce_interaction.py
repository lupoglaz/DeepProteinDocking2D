import random
import torch
from torch import optim

import sys
## path for cluster
sys.path.append('/home/sb1638/')

import numpy as np
from tqdm import tqdm
from DeepProteinDocking2D.torchDataset import get_interaction_stream_balanced, get_interaction_stream
from DeepProteinDocking2D.Models.BruteForce.model_bruteforce_interaction import BruteForceInteraction
from DeepProteinDocking2D.Models.BruteForce.validation_metrics import APR
from DeepProteinDocking2D.Models.BruteForce.model_bruteforce_docking import BruteForceDocking
from DeepProteinDocking2D.Models.BruteForce.plot_FI_loss import FILossPlotter

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
        ##### call model(s)
        FFT_score = self.docking_model(receptor, ligand, plotting=self.plotting)
        pred_interact, deltaF, F, F_0 = self.interaction_model(FFT_score, plotting=self.plotting)

        ### check parameters and gradients
        ### if weights are frozen or updating
        if self.debug:
            self.check_model_gradients(self.docking_model)
            self.check_model_gradients(self.interaction_model)

        #### Loss functions
        BCEloss = torch.nn.BCELoss()
        l1_loss = torch.nn.L1Loss()
        w = 10**-5 * scheduler.get_last_lr()[0]
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
                        fout.write('%f\t%f\t%f\t%d\n' % (train_output[0][2], train_output[0][3], train_output[0][4], train_output[0][5]))

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
            self.docking_model = BruteForceDocking().to(device=0)
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

    def plot_evaluation_set(self, eval_stream=None, resume_epoch=1):
        self.plotting = True
        train_epochs = 1
        self.train_model(train_epochs, train_stream, eval_stream, test_stream,
                                                   resume_training=True, resume_epoch=resume_epoch)

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

    interaction_model = BruteForceInteraction().to(device=0)
    interaction_optimizer = optim.Adam(interaction_model.parameters(), lr=lr_interaction)

    scheduler = optim.lr_scheduler.ExponentialLR(interaction_optimizer, gamma=0.99)

    docking_model = BruteForceDocking().to(device=0)
    docking_optimizer = optim.Adam(docking_model.parameters(), lr=lr_docking)

    max_size = 400
    # max_size = 100
    # max_size = 50
    batch_size = 1
    if batch_size > 1:
        raise NotImplementedError()
    train_stream = get_interaction_stream_balanced(trainset + '.pkl', batch_size=batch_size, max_size=max_size)
    valid_stream = get_interaction_stream_balanced(validset + '.pkl', batch_size=1, max_size=max_size)
    test_stream = get_interaction_stream_balanced(testset + '.pkl', batch_size=1, max_size=max_size)

    # experiment = 'RECODE_CHECK_INTERACTION'
    # experiment = 'PLOT_FREE_ENERGY_HISTOGRAMS'
    # experiment = 'FINAL_CHECK_INTERACTION_FULLDATA'
    # experiment = 'FINAL_CHECK_INTERACTION_FULLDATA_LR-1'
    # experiment = 'checkhists_lf-0_and_lr-4_50ex_100ep' ## working 50 ep MCCs ~0.7
    ## attempting to improve brute force FI
    # experiment = 'baseline_expA_lr-0_and_lr-4_50ex' ## 1 epoch MCCs 0.74 and 0.81, then fluctuates
    # experiment = 'baseline_expB_lr-0_and_lr-4_50ex' ## 3 epoch MCCs 0.8, then fluctuates
    # experiment = 'baseline_expC_lr-0_and_lr-4_50ex' ## 1 epoch MCCs 0.74 and 0.78, then fluctuates
    # experiment = 'baseline_scratch_lr-0_and_lr-4_50ex' ## 3 epoch MCCs ~0.6, then fluctuates
    # experiment = 'F0schedulerg=0p5_scratch_lr-0_and_lr-4_50ex'
    # experiment = 'F0schedulerg=0p5_scratch_lr-0_and_lr-4_50ex_novalidortest'
    # experiment = 'F0schedulerg=0p5_scratch_lr-0_and_lr-4_50ex_novalidortest_noWreg' ## wreg required
    # experiment = 'F0schedulerg=0p25_scratch_lr-0_and_lr-3_50ex_novalidortest' ## doesn't learn
    # experiment = 'F0schedulerg=0p95_scratch_lr-0_and_lr-3_50ex_novalidortest'
    # experiment = 'F0schedulerg=0p25_scratch_lr-0_and_lr-4_50ex_novalidortest'
    # experiment = 'F0schedulerg=0p95_scratch_lr-0_and_lr-4_50ex_novalidortest' ## 100ep 0.70 and 0.80
    experiment = 'Wregsched_F0schedulerg=0p95_scratch_lr-0_and_lr-4_50ex_novalidortest' ## 200 ep 0.7 > MCC > 0.8
    # experiment = 'Wregsched_F0schedulerg=0p95_scratch_lr-0_and_lr-4_novalidortest_100ex'


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
    train_epochs = 100
    #####################
    ### Train model from beginning
    # BruteForceInteractionTrainer(docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment, training_case, path_pretrain
    #                              ).run_trainer(train_epochs, train_stream=train_stream, valid_stream=None, test_stream=None)

    ### Resume training model at chosen epoch
    # BruteForceInteractionTrainer(docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment, training_case, path_pretrain
    #                              ).run_trainer(train_epochs=100, train_stream=train_stream, valid_stream=None, test_stream=None, resume_training=True, resume_epoch=100)

    ### Validate model at chosen epoch
    BruteForceInteractionTrainer(docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment, training_case, path_pretrain
                                 ).run_trainer(train_epochs=1, valid_stream=valid_stream, test_stream=test_stream,
                                               resume_training=True, resume_epoch=200)

    ### Plot free energy distributions with learned F_0 decision threshold
    # FILossPlotter(experiment).plot_deltaF_distribution(plot_epoch=2, show=True)

    ### Evaluate model only and plot, at chosen epoch
    # resume_epoch = 5
    ### loads relevant pretrained model under resume_training condition
    # BruteForceInteractionTrainer().plot_evaluation_set(eval_stream=valid_stream, resume_epoch=resume_epoch) ## also checks APR
    #
    # BruteForceInteractionTrainer().plot_evaluation_set(eval_stream=test_stream, resume_epoch=resume_epoch)

    ##################### Resume training model
    # BruteForceInteractionTrainer().train(resume_epoch, load_models=True)


# BruteForceInteractionTrainer(docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment, training_case, path_pretrain).run_trainer(train_epochs)
# trainer = BruteForceInteractionTrainer.__new__(BruteForceInteractionTrainer)
# trainer = trainer.get_trainer(self.interaction_model, self.interaction_optimizer, self.docking_model, self.docking_optimizer, self.experiment, self.training_case, self.path_pretrain)
# trainer = self.get_trainer(self.interaction_model, self.interaction_optimizer, self.docking_model, self.docking_optimizer, self.experiment, self.training_case, self.path_pretrain)
# trainer = BruteForceInteractionTrainer(self.interaction_model, self.interaction_optimizer, self.docking_model, self.docking_optimizer, self.experiment, self.training_case, self.path_pretrain)
# trainer = BruteForceInteractionTrainer(self.interaction_model, self.interaction_optimizer, self.docking_model, self.docking_optimizer, self.experiment, self.training_case, self.path_pretrain).__new__(BruteForceInteractionTrainer)
# trainer = self.get_trainer(self.docking_model, self.docking_optimizer, self.interaction_model, self.interaction_optimizer, self.experiment, self.training_case, self.path_pretrain)
# BruteForceInteractionTrainer(self.interaction_model, self.interaction_optimizer, self.docking_model,
#                              self.docking_optimizer, self.experiment, self.training_case,
#                              self.path_pretrain).get_trainer(self.interaction_model, self.interaction_optimizer,
#                                                              self.docking_model, self.docking_optimizer,
#                                                              self.experiment, self.training_case,
#                                                              self.path_pretrain)
# Accuracy, Precision, Recall, F1score, MCC = APR().calcAPR(datastream, trainer, check_epoch)
