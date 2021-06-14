import random
import torch
from torch import nn
from torch import optim

import numpy as np
from DeepProteinDocking2D.torchDataset import get_interaction_stream_balanced, get_interaction_stream
from tqdm import tqdm
from DeepProteinDocking2D.Models.BruteForce.TorchDockingFilter import TorchDockingFilter
from DeepProteinDocking2D.Models.BruteForce.model_bruteforce_interaction import BruteForceInteraction
from DeepProteinDocking2D.Models.BruteForce.utility_functions import plot_assembly
from DeepProteinDocking2D.Models.BruteForce.validation_metrics import APR

from DeepProteinDocking2D.Models.BruteForce.train_bruteforce_docking import BruteForceDockingTrainer
from DeepProteinDocking2D.Models.BruteForce.model_bruteforce_docking import BruteForceDocking


import matplotlib.pyplot as plt


class BruteForceInteractionTrainer:
    def __init__(self):
        pass
        # self.dim = TorchDockingFilter().dim
        # self.num_angles = TorchDockingFilter().num_angles

    def run_model(self, data, model, train=True, plotting=False, pretrain_model=None):
        receptor, ligand, gt_interact = data

        receptor = receptor.squeeze()
        ligand = ligand.squeeze()
        gt_interact = gt_interact.squeeze()
        # print(gt_interact.shape, gt_interact)

        receptor = receptor.to(device='cuda', dtype=torch.float).unsqueeze(0)
        ligand = ligand.to(device='cuda', dtype=torch.float).unsqueeze(0)
        gt_interact = gt_interact.to(device='cuda', dtype=torch.float)

        if train:
            model.train()

        ### run model and loss calculation
        ##### call model(s)
        FFT_score = pretrain_model(receptor, ligand, plotting=False)
        for n, p in pretrain_model.named_parameters():
            if p.requires_grad:
                print(n, p, p.grad)
        pred_interact = model(FFT_score, plotting=plotting)
        #### Loss functions
        BCEloss = torch.nn.BCELoss()

        loss = BCEloss(pred_interact, gt_interact)
        print(pred_interact.item(), gt_interact.item())


        if eval and plotting:
            with torch.no_grad():
                plt.close()
                pred_rot, pred_txy = TorchDockingFilter().extract_transform(FFT_score)
                # print(pred_txy, pred_rot)
                pair = plot_assembly(receptor.squeeze().detach().cpu().numpy(), ligand.squeeze().detach().cpu().numpy(),
                                     pred_rot.detach().cpu().numpy(),
                                     (pred_txy[0].detach().cpu().numpy(), pred_txy[1].detach().cpu().numpy()),
                                     pred_rot.squeeze().detach().cpu().numpy(),
                                     (pred_txy[0].detach().cpu().numpy(), pred_txy[1].detach().cpu().numpy()))
                plt.imshow(pair.transpose())
                plt.title('Ground Truth                      Input                       Predicted Pose')
                transform = str(pred_rot.item())+'_'+str(pred_txy)
                plt.text(10, 10, "Ligand transform=" + transform, backgroundcolor='w')
                plt.savefig('figs/Pose_BruteForce_Interaction_TorchFFT_SE2Conv2D_'+transform+'.png')
                plt.show()

        if train:
            model.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_pretrain.step()
        else:
            model.eval()
            with torch.no_grad():
                threshold = 0.5
                TP = 0
                FP = 0
                TN = 0
                FN = 0
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

    @staticmethod
    def save_checkpoint(state, filename):
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

    @staticmethod
    def train_model(model, optimizer, testcase, train_epochs, train_stream, valid_stream, resume_training=False,
                    resume_epoch=0, plotting=False, pretrain_model=None):

        test_freq = 1
        save_freq = 1

        if plotting:
            test_freq = 1

        log_header = 'Epoch\tLoss\trmsd\n'
        log_format = '%d\t%f\t%f\n'

        ### Continue training on existing model?
        if resume_training:
            ckp_path = 'Log/' + testcase + str(resume_epoch) + '.th'
            model, optimizer, start_epoch = BruteForceDockingTrainer().load_ckp(ckp_path, model, optimizer)
            start_epoch += 1

            print(model)
            print('\nRESUMING TRAINING AT EPOCH', start_epoch, '\n')
        else:
            start_epoch = 1
            ### Loss log files
            with open('Log/losses/log_train_' + testcase + '.txt', 'w') as fout:
                fout.write(log_header)
            with open('Log/losses/log_test_' + testcase + '.txt', 'w') as fout:
                fout.write(log_header)
        num_epochs = start_epoch + train_epochs

        for epoch in range(start_epoch, num_epochs):

            checkpoint_dict = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            if epoch % test_freq == 0 and epoch > 1:
                BruteForceInteractionTrainer().checkAPR(epoch, valid_stream, pretrain_model=pretrain_model)
                #
                # testloss = []
                # for data in tqdm(valid_stream):
                #     test_output = [BruteForceInteractionTrainer().run_model(data, model, train=False, plotting=plotting, pretrain_model=pretrain_model)]
                #     testloss.append(test_output)
                #
                # avg_testloss = np.average(testloss, axis=0)[0, :]
                # print('\nEpoch', epoch, 'TEST LOSS:', avg_testloss)
                # with open('Log/losses/log_test_' + testcase + '.txt', 'a') as fout:
                #     fout.write(log_format % (epoch, avg_testloss[0], avg_testloss[1]))

            trainloss = []
            for data in tqdm(train_stream):
                train_output = [BruteForceInteractionTrainer().run_model(data, model, train=True, pretrain_model=pretrain_model)]
                trainloss.append(train_output)

            avg_trainloss = np.average(trainloss, axis=0)[0, :]
            print('\nEpoch', epoch, 'Train Loss:', avg_trainloss)
            with open('Log/losses/log_train_' + testcase + '.txt', 'a') as fout:
                fout.write(log_format % (epoch, avg_trainloss[0], avg_trainloss[1]))

            #### saving model while training
            if epoch % save_freq == 0:
                BruteForceInteractionTrainer().save_checkpoint(checkpoint_dict, 'Log/' + testcase + str(epoch) + '.th')
                print('saving model ' + 'Log/' + testcase + str(epoch) + '.th')

        BruteForceInteractionTrainer().save_checkpoint(checkpoint_dict, 'Log/' + testcase + 'end.th')

    @staticmethod
    def checkAPR(check_epoch, datastream, pretrain_model):
        Accuracy, Precision, Recall = APR().calcAPR(datastream, BruteForceInteractionTrainer(), model, check_epoch, pretrain_model)
        # print(Accuracy, Precision, Recall)
        log_format = '%f\t%f\t%f\n'
        with open('Log/losses/log_validAPR_' + testcase + '.txt', 'a') as fout:
            fout.write(log_format % (Accuracy, Precision, Recall))

    @staticmethod
    def freeze_weights(pretrain_model):
        print('Freezing docking model CNN weights')
        for name, param in pretrain_model.named_parameters():
            if 'W' not in name:
                print('Freezing weights', name)
                param.requires_grad = False
            elif 'W' in name:
                param.requires_grad = False
                param.copy_(torch.rand(1))
                param.requires_grad = True

if __name__ == '__main__':
    #################################################################################
    # import sys
    # print(sys.path)
    trainset = 'toy_concave_data/interaction_data_train'
    testset = 'toy_concave_data/interaction_data_valid'
    # testcase = 'CHECK_reluBCEloss_statphys_interaction_balancedstream_dockingpretrain_BruteForce_training_'
    # testcase = 'revSigmoidB_statphys_interaction_balancedstream_dockingpretrain_BruteForce_training_'
    # testcase = 'conv3d3layers4feats_revSigmoidB_statphys_interaction_balancedstream_dockingpretrain_BruteForce_training_'
    testcase = 'learnedscoringW_conv3d2layers4feats_statphys_interaction_balancedstream_dockingpretrain_BruteForce_training_'

    #########################
    ### testing set
    testset = 'toy_concave_data/interaction_data_test'

    #### initialization torch settings
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.determininistic = True

    torch.cuda.set_device(0)
    # torch.autograd.set_detect_anomaly(True)
    ######################
    lr = 10 ** -4
    model = BruteForceInteraction().to(device=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    pretrain_model = BruteForceDocking().to(device=0)
    optimizer_pretrain = optim.Adam(pretrain_model.parameters(), lr=lr)
    path_pretrain = 'Log/docking_pretrain_bruteforce_allLearnedWs_10epochs_end.th'
    pretrain_model.load_state_dict(torch.load(path_pretrain)['state_dict'])
    BruteForceInteractionTrainer().freeze_weights(pretrain_model)

    train_stream = get_interaction_stream_balanced(trainset + '.pkl', batch_size=1)
    valid_stream = get_interaction_stream_balanced(testset + '.pkl', batch_size=1)

    ######################
    train_epochs = 1

    def train(resume_training=False, resume_epoch=0):
        BruteForceInteractionTrainer().train_model(model, optimizer, testcase, train_epochs, train_stream, valid_stream,
                                               resume_training=resume_training, resume_epoch=resume_epoch, pretrain_model=pretrain_model)


    def plot_validation_set(check_epoch, plotting=True):
        BruteForceInteractionTrainer().train_model(model, optimizer, testcase, train_epochs, train_stream, valid_stream,
                                               resume_training=True, resume_epoch=check_epoch, plotting=plotting, pretrain_model=pretrain_model)

    ######################
    train()

    # epoch = 1
    #
    # plot_validation_set(check_epoch=epoch) ## also checks APR
    #
    # train(True, epoch)
