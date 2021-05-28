import random
import torch
from torch import optim

import numpy as np
# from DeepProteinDocking2D.Models.BruteForce.torchDataset import get_dataset_stream
from DeepProteinDocking2D.torchDataset import get_interaction_stream_balanced, get_interaction_stream
from tqdm import tqdm
from DeepProteinDocking2D.Models.BruteForce.TorchDockingFilter import TorchDockingFilter
from DeepProteinDocking2D.Models.BruteForce.model_bruteforce_interaction import BruteForceInteraction
from DeepProteinDocking2D.Models.BruteForce.utility_functions import plot_assembly
from DeepProteinDocking2D.Models.BruteForce.grid_rmsd import RMSD

import matplotlib.pyplot as plt


class BruteForceDockingTrainer:
    def __init__(self):
        self.dim = TorchDockingFilter().dim
        self.num_angles = TorchDockingFilter().num_angles

    def run_model(self, data, model, train=True, plotting=False):
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
        ##### call model
        pred_interact = model(receptor, ligand, plotting=plotting)
        # print(pred_interact.shape, pred_interact)
        #### Loss functions
        # CE_loss = torch.nn.CrossEntropyLoss()
        l1_loss = torch.nn.L1Loss()
        loss = l1_loss(pred_interact.squeeze().unsqueeze(0), gt_interact.squeeze().unsqueeze(0))

        if train:
            model.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            model.eval()

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
                    resume_epoch=0, plotting=False):

        test_freq = 10
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

            trainloss = []
            for data in tqdm(train_stream):
                train_output = [BruteForceDockingTrainer().run_model(data, model, train=True)]
                trainloss.append(train_output)

            avg_trainloss = np.average(trainloss, axis=0)[0, :]
            print('\nEpoch', epoch, 'Train Loss:', avg_trainloss)
            with open('Log/losses/log_train_' + testcase + '.txt', 'a') as fout:
                fout.write(log_format % (epoch, avg_trainloss[0], avg_trainloss[1]))

            if epoch % test_freq == 0 or epoch == 1:
                testloss = []
                for data in tqdm(valid_stream):
                    test_output = [BruteForceDockingTrainer().run_model(data, model, train=False, plotting=plotting)]
                    testloss.append(test_output)

                avg_testloss = np.average(testloss, axis=0)[0, :]
                print('\nEpoch', epoch, 'TEST LOSS:', avg_testloss)
                with open('Log/losses/log_test_' + testcase + '.txt', 'a') as fout:
                    fout.write(log_format % (epoch, avg_testloss[0], avg_testloss[1]))

            #### saving model while training
            if epoch % save_freq == 0:
                BruteForceDockingTrainer().save_checkpoint(checkpoint_dict, 'Log/' + testcase + str(epoch) + '.th')
                print('saving model ' + 'Log/' + testcase + str(epoch) + '.th')

        BruteForceDockingTrainer().save_checkpoint(checkpoint_dict, 'Log/' + testcase + 'end.th')


if __name__ == '__main__':
    #################################################################################
    # import sys
    # print(sys.path)
    trainset = 'toy_concave_data/interaction_data_train'
    testset = 'toy_concave_data/interaction_data_valid'
    testcase = 'TEST_interaction_BruteForce_training_'


    #########################
    ### testing set
    # testset = 'toy_concave_data/interaction_data_test'

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

    train_stream = get_interaction_stream(trainset + '.pkl', batch_size=1)
    valid_stream = get_interaction_stream(testset + '.pkl', batch_size=1)
    # train_stream = get_interaction_stream_balanced(trainset + '.pkl', batch_size=1)
    # valid_stream = get_interaction_stream_balanced(testset + '.pkl', batch_size=1)

    ######################
    train_epochs = 30


    def train(resume_training=False, resume_epoch=0):
        BruteForceDockingTrainer().train_model(model, optimizer, testcase, train_epochs, train_stream, valid_stream,
                                               resume_training=resume_training, resume_epoch=resume_epoch)


    def plot_validation_set(check_epoch):
        BruteForceDockingTrainer().train_model(model, optimizer, testcase, train_epochs, train_stream, valid_stream,
                                               resume_training=True, resume_epoch=check_epoch, plotting=True)

    ######################
    train()

    epoch = 30

    # epoch = 'end'

    plot_validation_set(check_epoch=epoch)

    # train(True, epoch)
