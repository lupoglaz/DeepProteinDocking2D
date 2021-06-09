import random
import torch
from torch import nn
from torch import optim

import numpy as np
from DeepProteinDocking2D.Models.BruteForce.torchDataset import get_dataset_stream
from tqdm import tqdm
from DeepProteinDocking2D.Models.BruteForce.TorchDockingFilter import TorchDockingFilter
from DeepProteinDocking2D.Models.BruteForce.model_bruteforce_docking import BruteForceDocking
from DeepProteinDocking2D.Models.BruteForce.utility_functions import plot_assembly
from DeepProteinDocking2D.Models.BruteForce.validation_metrics import RMSD

import matplotlib.pyplot as plt


class BruteForceDockingTrainer:
    def __init__(self):
        self.dim = TorchDockingFilter().dim
        self.num_angles = TorchDockingFilter().num_angles

    def run_model(self, data, model, train=True, plotting=False):
        receptor, ligand, gt_rot, gt_txy = data

        receptor = receptor.squeeze()
        ligand = ligand.squeeze()
        gt_txy = gt_txy.squeeze()
        gt_rot = gt_rot.squeeze()

        receptor = receptor.to(device='cuda', dtype=torch.float).unsqueeze(0)
        ligand = ligand.to(device='cuda', dtype=torch.float).unsqueeze(0)
        gt_rot = gt_rot.to(device='cuda', dtype=torch.float)
        gt_txy = gt_txy.to(device='cuda', dtype=torch.float)

        if train:
            model.train()

        ### run model and loss calculation
        ##### call model
        FFT_score = model(receptor, ligand, plotting=plotting)
        FFT_score = FFT_score.flatten()
        with torch.no_grad():
            target_flatindex = TorchDockingFilter().encode_transform(gt_rot, gt_txy)
            pred_rot, pred_txy = TorchDockingFilter().extract_transform(FFT_score)
            rmsd_out = RMSD(ligand, gt_rot, gt_txy, pred_rot, pred_txy).calc_rmsd()
            # print('extracted predicted indices', pred_rot, pred_txy)
            # print('gt indices', gt_rot, gt_txy)
            # print('RMSD', rmsd_out.item())

        #### Loss functions
        CE_loss = torch.nn.CrossEntropyLoss()
        loss = CE_loss(FFT_score.squeeze().unsqueeze(0), target_flatindex.unsqueeze(0))

        if train:
            model.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            model.eval()

        if eval and plotting:
            with torch.no_grad():
                # plt.close()
                # maxind = torch.argmax(FFT_score)
                # plot_index = int(((maxind / self.dim ** 2) * np.pi / 180.0) - np.pi)
                # plotE = FFT_score.reshape(self.num_angles, self.dim, self.dim).squeeze()[plot_index, :, :].detach().cpu()
                # plt.imshow(plotE)
                # plt.title('FFT best rotation x, y')
                # plt.colorbar()
                # plt.show()

                plt.close()
                plt.figure(figsize=(8, 8))
                pred_rot, pred_txy = TorchDockingFilter().extract_transform(FFT_score)
                print('extracted predicted indices', pred_rot, pred_txy)
                print('gt indices', gt_rot, gt_txy)
                rmsd_out = RMSD(ligand, gt_rot, gt_txy, pred_rot, pred_txy).calc_rmsd()
                print('RMSD', rmsd_out.item())

                pair = plot_assembly(receptor.squeeze().detach().cpu().numpy(), ligand.squeeze().detach().cpu().numpy(),
                                     pred_rot.detach().cpu().numpy(),
                                     (pred_txy[0].detach().cpu().numpy(), pred_txy[1].detach().cpu().numpy()),
                                     gt_rot.squeeze().detach().cpu().numpy(), gt_txy.squeeze().detach().cpu().numpy())

                plt.imshow(pair.transpose())
                plt.title('Ground Truth                      Input                       Predicted Pose')
                plt.text(10, 10, "Ligand RMSD=" + str(rmsd_out.item()), backgroundcolor='w')
                plt.savefig('figs/Pose_BruteForceTorchFFT_SE2Conv2D_Ligand_RMSD_' + str(rmsd_out.item()) + '.png')
                plt.show()

        return loss.item(), rmsd_out.item()

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
            print(list(model.named_parameters()))
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
            if epoch % test_freq == 0 or epoch == 1:
                testloss = []
                for data in tqdm(valid_stream):
                    test_output = [BruteForceDockingTrainer().run_model(data, model, train=False, plotting=plotting)]
                    testloss.append(test_output)

                avg_testloss = np.average(testloss, axis=0)[0, :]
                print('\nEpoch', epoch, 'TEST LOSS:', avg_testloss)
                with open('Log/losses/log_test_' + testcase + '.txt', 'a') as fout:
                    fout.write(log_format % (epoch, avg_testloss[0], avg_testloss[1]))

            trainloss = []
            for data in tqdm(train_stream):
                train_output = [BruteForceDockingTrainer().run_model(data, model, train=True)]
                trainloss.append(train_output)

            avg_trainloss = np.average(trainloss, axis=0)[0, :]
            print('\nEpoch', epoch, 'Train Loss:', avg_trainloss)
            with open('Log/losses/log_train_' + testcase + '.txt', 'a') as fout:
                fout.write(log_format % (epoch, avg_trainloss[0], avg_trainloss[1]))

            #### saving model while training
            if epoch % save_freq == 0:
                BruteForceDockingTrainer().save_checkpoint(checkpoint_dict, 'Log/' + testcase + str(epoch) + '.th')
                print('saving model ' + 'Log/' + testcase + str(epoch) + '.th')

        BruteForceDockingTrainer().save_checkpoint(checkpoint_dict, 'Log/' + testcase + 'end.th')


if __name__ == '__main__':
    #################################################################################
    # import sys
    # print(sys.path)
    trainset = 'toy_concave_data/docking_data_train'
    testset = 'toy_concave_data/docking_data_valid'
    # testcase = 'newdata_BruteForce_training_check'
    # testcase = 'newdata_twoCTweights_alllearnedWs_BruteForce_training_check'

    # testcase = 'pretrain_bruteforcedocking_alllearnedWs_10epochs'
    # testcase = 'pretrain_shiftedorigin_bruteforcedocking_alllearnedWs_10epochs'

    testcase = 'docking_pretrain_bruteforce_allLearnedWs_10epochs_'


    #########################
    ### testing set
    # testset = 'toy_concave_data/docking_data_test'

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
    model = BruteForceDocking().to(device=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(list(model.parameters()))

    train_stream = get_dataset_stream(trainset + '.pkl', batch_size=1)
    valid_stream = get_dataset_stream(testset + '.pkl', batch_size=1)

    ######################
    train_epochs = 10


    def train(resume_training=False, resume_epoch=0):
        BruteForceDockingTrainer().train_model(model, optimizer, testcase, train_epochs, train_stream, valid_stream,
                                               resume_training=resume_training, resume_epoch=resume_epoch)


    def plot_validation_set(check_epoch):
        BruteForceDockingTrainer().train_model(model, optimizer, testcase, train_epochs, train_stream, valid_stream,
                                               resume_training=True, resume_epoch=check_epoch, plotting=True)

    ######################
    # train()

    epoch = 10

    # epoch = 'end'

    plot_validation_set(check_epoch=epoch)

    # train(True, epoch)
