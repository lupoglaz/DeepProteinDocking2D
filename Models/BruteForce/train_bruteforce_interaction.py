import random
import torch
from torch import nn
from torch import optim

import sys
sys.path.append('/home/sb1638/')

import numpy as np
from tqdm import tqdm
from DeepProteinDocking2D.torchDataset import get_interaction_stream_balanced, get_interaction_stream
from DeepProteinDocking2D.Models.BruteForce.model_bruteforce_interaction import BruteForceInteraction
from DeepProteinDocking2D.Models.BruteForce.validation_metrics import APR
from DeepProteinDocking2D.Models.BruteForce.model_bruteforce_docking import BruteForceDocking

import matplotlib.pyplot as plt
from DeepProteinDocking2D.Models.BruteForce.TorchDockingFilter import TorchDockingFilter
from DeepProteinDocking2D.Models.BruteForce.utility_functions import plot_assembly


class BruteForceInteractionTrainer:
    if len(sys.argv) > 1:
        replicate = str(sys.argv[1])
    else:
        replicate = 'single_rep'

    # testcase = 'WM_f22_lr2_2ep_aW_unfrozen' #c exp
    # testcase = 'WM_f25_lr3_2ep_aW_unfrozen' #c exp

    testcase = 'WM_f25_lr4_1ep_allUnfrozen_expD' #d exp

    # testcase = 'WM_f50_expC' #d exp


    train_epochs = 1
    check_epoch = 1
    test_freq = 1
    save_freq = 1

    ##### load blank models and optimizers, oncewa
    lr = 10 ** -4
    model = BruteForceInteraction().to(device=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    pretrain_model = BruteForceDocking().to(device=0)
    optimizer_pretrain = optim.Adam(pretrain_model.parameters(), lr=lr)

    print('SHOULD ONLY PRINT ONCE')

    ###################### Load and freeze/unfreeze params (training no eval)
    ## for exp a,b,c
    # path_pretrain = 'Log/newdata_bugfix_docking_100epochs_19.th'
    # pretrain_model.load_state_dict(torch.load(path_pretrain)['state_dict'])

    # param_to_freeze = 'all'
    # param_to_freeze = 'W' ##freeze all but "a" weights
    param_to_freeze = None

    #### load (pretrained: IP CNN frozen, a00...a11 unfrozen) and retrain IP as unfrozen (d exp)
    # path_pretrain = 'Log/docking_ndp_simpleexp_eq15sigmoid_aW_unfrozen1.th' # pretrained on expC only
    path_pretrain = 'Log/docking_WM_f25_lr4_1ep_allUnfrozen_expD1.th' ### post training expD
    pretrain_model.load_state_dict(torch.load(path_pretrain)['state_dict'])

    # plotting = True
    plotting = False

    def __init__(self):
        pass

    def run_model(self, data, train=True):
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
        pred_interact = self.model(FFT_score, plotting=self.plotting)

        ### check if pretrain weights are frozen or updating
        # for n, p in self.pretrain_model.named_parameters():
        #     if p.requires_grad:
        #         print(n, p, p.grad)

        ### check if pretrain weights are frozen or updating
        # for n, p in self.model.named_parameters():
        #     if p.requires_grad:
        #         print(n, p, p.grad)


        #### Loss functions
        BCEloss = torch.nn.BCELoss()
        loss = BCEloss(pred_interact, gt_interact)
        # loss = SharpLoss().forward(pred_interact, gt_interact)
        print('\n', pred_interact.item(), gt_interact.item())

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

    def train_model(self, model, testcase, train_epochs, train_stream, valid_stream, test_stream, load_models=False,
                    resume_epoch=0, plotting=False):

        if plotting:
            self.test_freq = 1

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

            print(model)
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

            trainloss = []
            for data in tqdm(train_stream):
                train_output = [BruteForceInteractionTrainer().run_model(data, self.model)]
                trainloss.append(train_output)
                break

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
            fout.write(log_header)
            fout.write(log_format % (Accuracy, Precision, Recall, F1score, MCC))
        fout.close()

    @staticmethod
    def freeze_weights(model, param_to_freeze=None):
        if not param_to_freeze:
            print('All params unfrozen')
            return
        for name, param in model.named_parameters():
            if param_to_freeze is not None and param_to_freeze in name:
                print('Unfreezing Weights', name)
                param.requires_grad = True
            else:
                print('Freezing docking model weights', name)
                param.requires_grad = False

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
    def weights_init(model):
        if isinstance(model, torch.nn.Conv2d):
            print('updating convnet weights to kaiming uniform initialization')
            torch.nn.init.kaiming_uniform_(model.weight)
            # torch.nn.init.kaiming_normal_(model.weight)

    def train(self, resume_epoch=0):
        BruteForceInteractionTrainer().train_model(self.model, self.testcase, self.train_epochs, train_stream, valid_stream, test_stream,
                                                   load_models=False, resume_epoch=resume_epoch,
                                                   )

    def plot_validation_set(self, plotting=True, eval_stream=None):
        BruteForceInteractionTrainer().train_model(self.model, self.testcase, self.train_epochs, train_stream, eval_stream,
                                                   load_models=True, resume_epoch=self.check_epoch, plotting=plotting,
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
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.determininistic = True
    torch.cuda.set_device(0)
    # CUDA_LAUNCH_BLOCKING = 1
    # torch.autograd.set_detect_anomaly(True)

    train_stream = get_interaction_stream_balanced(trainset + '.pkl', batch_size=1)
    valid_stream = get_interaction_stream_balanced(validset + '.pkl', batch_size=1)
    test_stream = get_interaction_stream_balanced(testset + '.pkl', batch_size=1)

    #### model and pretrain model

    ##################### Train model
    BruteForceInteractionTrainer().train()
    #
    # give time to save models
    # time.sleep(60)

    ##################### Evaluate model
    ### loads relevant pretrained model under resume_training condition
    # BruteForceInteractionTrainer().plot_validation_set(eval_stream=valid_stream) ## also checks APR

    # BruteForceInteractionTrainer().plot_validation_set(eval_stream=test_stream)


    # BruteForceInteractionTrainer().train(1)
