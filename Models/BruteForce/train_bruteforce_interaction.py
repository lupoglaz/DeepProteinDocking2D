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


class BruteForceInteractionTrainer:
    ## run replicates from sbatch script args
    if len(sys.argv) > 1:
        replicate = str(sys.argv[1])
    else:
        replicate = 'single_rep'

    # plotting = True
    plotting = False

    train_epochs = 6
    check_epoch = 1
    test_freq = 1
    save_freq = 1

    ##### load blank models and optimizers, once
    lr_interaction = 10**0
    lr_docking = 10**-5

    model = BruteForceInteraction().to(device=0)
    optimizer = optim.Adam(model.parameters(), lr=lr_interaction)

    pretrain_model = BruteForceDocking().to(device=0)
    optimizer_pretrain = optim.Adam(pretrain_model.parameters(), lr=lr_docking)

    print('SHOULD ONLY PRINT ONCE PER TRAINING')
    ##############################################################################
    # case = 'final'
    # case = 'final_ones'
    case = 'final_lr5_ones'


    # exp = 'A'
    # exp = 'B'
    # exp = 'C'
    exp = 'scratch'

    testcase = 'exp' + exp + '_' + case

    ###################### Load and freeze/unfreeze params (training, no eval)
    # path to pretrained docking model
    path_pretrain = 'Log/randinit_best_docking_model_epoch11.th'
    # train with docking model frozen
    if exp == 'A':
        print('Training expA')
        param_to_freeze = 'all'
        pretrain_model.load_state_dict(torch.load(path_pretrain)['state_dict'])
    # train with docking model unfrozen
    if exp == 'B':
        print('Training expB')
        # lr_docking = 10**-5
        # print('Docking learning rate changed to', lr_docking)
        pretrain_model = BruteForceDocking().to(device=0)
        optimizer_pretrain = optim.Adam(pretrain_model.parameters(), lr=lr_docking)
        param_to_freeze = None
        pretrain_model.load_state_dict(torch.load(path_pretrain)['state_dict'])
    # train with docking model SE2 CNN frozen
    if exp == 'C':
        print('Training expC')
        param_to_freeze = 'netSE2'  # leave "a" scoring coefficients unfrozen
        pretrain_model.load_state_dict(torch.load(path_pretrain)['state_dict'])
    # train everything from scratch
    if exp == 'scratch':
        print('Training from scratch')
        param_to_freeze = None
        testcase = exp + '_' + case

    def __init__(self):
        pass

    def run_model(self, data, train=True, debug=False):
        receptor, ligand, gt_interact = data

        receptor = receptor.squeeze()
        ligand = ligand.squeeze()
        gt_interact = gt_interact.squeeze()
        # print(gt_interact.shape, gt_interact)

        receptor = receptor.to(device='cuda', dtype=torch.float).unsqueeze(0)
        ligand = ligand.to(device='cuda', dtype=torch.float).unsqueeze(0)
        gt_interact = gt_interact.to(device='cuda', dtype=torch.float)

        if train:
            self.model.train()
            self.pretrain_model.train()

        ### run model and loss calculation
        ##### call model(s)
        FFT_score = self.pretrain_model(receptor, ligand, plotting=self.plotting)
        pred_interact, deltaF = self.model(FFT_score, plotting=self.plotting)

        ### check parameters and gradients
        ### if weights are frozen or updating
        if debug:
            BruteForceInteractionTrainer().check_gradients(self.pretrain_model)
            BruteForceInteractionTrainer().check_gradients(self.model)

        #### Loss functions
        BCEloss = torch.nn.BCELoss()
        l1_loss = torch.nn.L1Loss()
        w = 10**-5
        L_reg = w * l1_loss(deltaF, torch.zeros(1).squeeze().cuda())
        loss = BCEloss(pred_interact, gt_interact) + L_reg
        print('\n predicted', pred_interact.item(), '; ground truth', gt_interact.item())

        # with torch.no_grad():
        #     if str(pred_interact.item())[0] != '1' or str(pred_interact.item())[0] != '0':
        #         print('PROBLEM...')
        #         print('bad prediction', pred_interact.item())


        if train:
            self.pretrain_model.zero_grad()
            self.model.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer_pretrain.step()
            self.optimizer.step()
        else:
            self.pretrain_model.eval()
            self.model.eval()
            with torch.no_grad():
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

        return loss.item(), pred_interact.item()

    def train_model(self, model, testcase, train_epochs, train_stream, valid_stream, test_stream, load_models,
                    resume_epoch=0, plotting=False, debug=False):

        training = True

        if plotting:
            self.test_freq = 1
            training = False

        log_header = 'Epoch\tLoss\trmsd\n'
        log_format = '%d\t%f\t%f\n'

        ### freeze weights in pretrain model
        BruteForceInteractionTrainer().freeze_weights(self.pretrain_model, self.param_to_freeze)

        ### Continue training on existing model?
        if load_models:
            print('Loading interaction model at', str(resume_epoch))
            ckp_path = 'Log/' + self.testcase + str(resume_epoch) + '.th'
            self.model, self.optimizer, start_epoch = BruteForceInteractionTrainer().load_ckp(ckp_path, self.model, self.optimizer)

            print('Loading docking model at', str(resume_epoch))
            ckp_path = 'Log/docking_' + self.testcase + str(resume_epoch) + '.th'
            self.pretrain_model, self.optimizer_pretrain, _ = BruteForceInteractionTrainer().load_ckp(ckp_path, self.pretrain_model, self.optimizer_pretrain)

            start_epoch += 1

            print('\ndocking model:\n', model)
            ## print model and params being loaded
            BruteForceInteractionTrainer().check_gradients(self.pretrain_model)

            print('\ninteraction model:\n', model)
            ## print model and params being loaded
            BruteForceInteractionTrainer().check_gradients(self.model)

            print('\nLOADING MODEL AT EPOCH', start_epoch, '\n')
        else:
            start_epoch = 1
            ### Loss log files
            with open('Log/losses/log_train_' + self.testcase + '.txt', 'w') as fout:
                fout.write(log_header)
            with open('Log/losses/log_test_' + self.testcase + '.txt', 'w') as fout:
                fout.write(log_header)
        num_epochs = start_epoch + train_epochs

        for epoch in range(start_epoch, num_epochs):

            checkpoint_dict = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }

            pretrain_checkpoint_dict = {
                'epoch': epoch,
                'state_dict': self.pretrain_model.state_dict(),
                'optimizer': self.optimizer_pretrain.state_dict(),
            }

            if training:
                trainloss = []
                for data in tqdm(train_stream):
                    train_output = [BruteForceInteractionTrainer().run_model(data, self.model, debug=debug)]
                    trainloss.append(train_output)

                avg_trainloss = np.average(trainloss, axis=0)[0, :]
                print('\nEpoch', epoch, 'Train Loss:', avg_trainloss)
                with open('Log/losses/log_train_' + testcase + '.txt', 'a') as fout:
                    fout.write(log_format % (epoch, avg_trainloss[0], avg_trainloss[1]))

            ### evaluate on training and valid set
            if epoch % self.test_freq == 0:
                BruteForceInteractionTrainer().checkAPR(epoch, valid_stream)
                BruteForceInteractionTrainer().checkAPR(epoch, test_stream)

            #### saving model while training
            if epoch % self.save_freq == 0:
                BruteForceInteractionTrainer().save_checkpoint(pretrain_checkpoint_dict, 'Log/docking_' + testcase + str(epoch) + '.th', self.pretrain_model)
                print('saving docking model ' + 'Log/docking_' + testcase + str(epoch) + '.th')

                BruteForceInteractionTrainer().save_checkpoint(checkpoint_dict, 'Log/' + testcase + str(epoch) + '.th', self.model)
                print('saving interaction model ' + 'Log/' + testcase + str(epoch) + '.th')

    def checkAPR(self, check_epoch, datastream):
        log_format = '%f\t%f\t%f\t%f\t%f\n'
        log_header = 'Accuracy\tPrecision\tRecall\tF1score\tMCC\n'
        Accuracy, Precision, Recall, F1score, MCC = APR().calcAPR(datastream, BruteForceInteractionTrainer(), check_epoch)
        # print(Accuracy, Precision, Recall)
        with open('Log/losses/log_validAPR_' + self.testcase + '.txt', 'a') as fout:
            fout.write('Epoch '+str(check_epoch)+'\n')
            fout.write(log_header)
            fout.write(log_format % (Accuracy, Precision, Recall, F1score, MCC))
        fout.close()

    @staticmethod
    def freeze_weights(model, param_to_freeze=None):
        if not param_to_freeze:
            print('\nAll docking model params unfrozen\n')
            return
        for name, param in model.named_parameters():
            if param_to_freeze == 'all':
                print('Freeze ALL Weights', name)
                param.requires_grad = False
            elif param_to_freeze in name:
                print('Freeze Weights', name)
                param.requires_grad = False
            else:
                print('Unfreeze docking model weights', name)
                param.requires_grad = True

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
    def check_gradients(model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                print('Name', n, '\nParam', p, '\nGradient', p.grad)

    def train(self, resume_epoch=0, load_models=False, debug=False):
        BruteForceInteractionTrainer().train_model(self.model, self.testcase, self.train_epochs, train_stream, valid_stream, test_stream,
                                                   load_models=load_models, resume_epoch=resume_epoch, debug=debug
                                                   )

    def plot_evaluation_set(self, plotting=True, eval_stream=None, resume_epoch=1):
        BruteForceInteractionTrainer().train_model(self.model, self.testcase, self.train_epochs, train_stream, eval_stream, test_stream,
                                                   load_models=True, resume_epoch=resume_epoch, plotting=plotting, debug=False
                                                   )

if __name__ == '__main__':
    #################################################################################
    trainset = 'toy_concave_data/interaction_data_train'
    validset = 'toy_concave_data/interaction_data_valid'
    # ### testing set
    testset = 'toy_concave_data/interaction_data_test'

    #########################
    #### initialization torch settings
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.determininistic = True
    torch.cuda.set_device(0)
    # CUDA_LAUNCH_BLOCKING = 1
    # torch.autograd.set_detect_anomaly(True)

    train_stream = get_interaction_stream_balanced(trainset + '.pkl', batch_size=1)
    valid_stream = get_interaction_stream_balanced(validset + '.pkl', batch_size=1)
    test_stream = get_interaction_stream_balanced(testset + '.pkl', batch_size=1)

    ##################### Train model
    BruteForceInteractionTrainer().train(debug=False)

    ##################### Evaluate model
    # resume_epoch = 6
    ### loads relevant pretrained model under resume_training condition
    # BruteForceInteractionTrainer().plot_evaluation_set(eval_stream=valid_stream, resume_epoch=resume_epoch) ## also checks APR
    #
    # BruteForceInteractionTrainer().plot_evaluation_set(eval_stream=test_stream, resume_epoch=resume_epoch)

    ##################### Resume training model
    # BruteForceInteractionTrainer().train(resume_epoch, load_models=True)
