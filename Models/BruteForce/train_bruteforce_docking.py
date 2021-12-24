import random
import torch
from torch import nn
from torch import optim

import sys
sys.path.append('/home/sb1638/')

import numpy as np
from tqdm import tqdm
from DeepProteinDocking2D.torchDataset import get_docking_stream
from DeepProteinDocking2D.Models.BruteForce.TorchDockingFFT import TorchDockingFFT
from DeepProteinDocking2D.Models.BruteForce.model_bruteforce_docking import BruteForceDocking
from DeepProteinDocking2D.Models.BruteForce.utility_functions import plot_assembly
from DeepProteinDocking2D.Models.BruteForce.validation_metrics import RMSD

import matplotlib.pyplot as plt


class BruteForceDockingTrainer:
    def __init__(self):
        self.dim = TorchDockingFFT().dim
        self.num_angles = TorchDockingFFT().num_angles

    def run_model(self, data, model, train=True, plotting=False, debug=False):
        receptor, ligand, gt_txy, gt_rot, _ = data

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
            target_flatindex = TorchDockingFFT().encode_transform(gt_rot, gt_txy)
            pred_rot, pred_txy = TorchDockingFFT().extract_transform(FFT_score)
            rmsd_out = RMSD(ligand, gt_rot, gt_txy, pred_rot, pred_txy).calc_rmsd()
            # print('extracted predicted indices', pred_rot, pred_txy)
            # print('gt indices', gt_rot, gt_txy)
            # print('RMSD', rmsd_out.item())
            # print('Lowest energy', -FFT_score.flatten()[target_flatindex])

        #### Loss functions
        CE_loss = torch.nn.CrossEntropyLoss()
        loss = CE_loss(FFT_score.squeeze().unsqueeze(0), target_flatindex.unsqueeze(0))

        ### check parameters and gradients
        ### if weights are frozen or updating
        if debug:
            BruteForceDockingTrainer().check_gradients(model)

        if train:
            model.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            model.eval()

        if eval and plotting:
            with torch.no_grad():
                # import matplotlib.colors as colors
                # cmap = colors.ListedColormap(['white', 'black', 'red'])
                plt.close()
                plt.figure(figsize=(8, 8))
                pred_rot, pred_txy = TorchDockingFFT().extract_transform(FFT_score)
                print('extracted predicted indices', pred_rot, pred_txy)
                print('gt indices', gt_rot, gt_txy)
                rmsd_out = RMSD(ligand, gt_rot, gt_txy, pred_rot, pred_txy).calc_rmsd()
                print('RMSD', rmsd_out.item())

                pair = plot_assembly(receptor.squeeze().detach().cpu().numpy(), ligand.squeeze().detach().cpu().numpy(),
                                     pred_rot.detach().cpu().numpy(),
                                     (pred_txy[0].detach().cpu().numpy(), pred_txy[1].detach().cpu().numpy()),
                                     gt_rot.squeeze().detach().cpu().numpy(), gt_txy.squeeze().detach().cpu().numpy())

                plt.imshow(pair.transpose())
                # plt.title('Ground truth'+' '*33+'Input'+' '*33+'Predicted pose')
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
                    axis='y',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    left=False,  # ticks along the bottom edge are off
                    right=False,  # ticks along the top edge are off
                    labelleft=False)  # labels along the bottom
                plt.savefig('figs/makefigs_pose' + str(rmsd_out.item()) + '.png')
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
    def check_gradients(model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(n, p, p.grad)

    @staticmethod
    ## Unused SE2 net has own Kaiming He weight initialization.
    def weights_init(model):
        if isinstance(model, torch.nn.Conv2d):
            print('updating convnet weights to kaiming uniform initialization')
            torch.nn.init.kaiming_uniform_(model.weight)
            # torch.nn.init.kaiming_normal_(model.weight)

    @staticmethod
    def train_model(model, optimizer, testcase, train_epochs, train_stream, valid_stream, test_stream, resume_training=False,
                    resume_epoch=0, plotting=False, debug=False):

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
            print(list(model.named_parameters()))
            print('\nRESUMING TRAINING AT EPOCH', start_epoch, '\n')
        else:
            start_epoch = 1
            ### Loss log files
            with open('Log/losses/log_train_' + testcase + '.txt', 'w') as fout:
                fout.write('Docking Training Loss:\n')
                fout.write(log_header)
            with open('Log/losses/log_valid_' + testcase + '.txt', 'w') as fout:
                fout.write('Docking Validation Loss:\n')
                fout.write(log_header)
            with open('Log/losses/log_test_' + testcase + '.txt', 'w') as fout:
                fout.write('Docking Testing Loss:\n')
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
                train_output = [BruteForceDockingTrainer().run_model(data, model, train=True, debug=debug)]
                trainloss.append(train_output)

            if epoch % test_freq == 0 or epoch == 1:
                valid_loss = []
                for data in tqdm(valid_stream):
                    valid_output = [
                        BruteForceDockingTrainer().run_model(data, model, train=False, plotting=plotting,
                                                             debug=False)]
                    valid_loss.append(valid_output)

                avg_validloss = np.average(valid_loss, axis=0)[0, :]
                print('\nEpoch', epoch, 'VALID LOSS:', avg_validloss)
                with open('Log/losses/log_valid_' + testcase + '.txt', 'a') as fout:
                    fout.write(log_format % (epoch, avg_validloss[0], avg_validloss[1]))

                test_loss = []
                for data in tqdm(test_stream):
                    test_output = [BruteForceDockingTrainer().run_model(data, model, train=False, plotting=plotting,
                                                                        debug=False)]
                    test_loss.append(test_output)

                avg_testloss = np.average(test_loss, axis=0)[0, :]
                print('\nEpoch', epoch, 'TEST LOSS:', avg_testloss)
                with open('Log/losses/log_test_' + testcase + '.txt', 'a') as fout:
                    fout.write(log_format % (epoch, avg_testloss[0], avg_testloss[1]))

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
    trainset = 'toy_concave_data/docking_data_train'
    validset = 'toy_concave_data/docking_data_valid'
    ### testing set
    testset = 'toy_concave_data/docking_data_test'

    # testcase = 'makefigs_IP_1s4v_docking_200epochs'

    # testcase = 'Checkgitmerge_IP_1s4v_docking_10epochs'

    # testcase = 'noRandseed_Checkgitmerge_IP_1s4v_docking_10epochs'

    # testcase = 'rep1_noRandseed_Checkgitmerge_IP_1s4v_docking_10epochs'
    testcase = 'rep2_noRandseed_Checkgitmerge_IP_1s4v_docking_10epochs'

    #########################
    #### initialization torch settings
    # random_seed = 42
    # np.random.seed(random_seed)
    # torch.manual_seed(random_seed)
    # random.seed(random_seed)
    # torch.cuda.manual_seed(random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.cuda.set_device(0)

    torch.cuda.set_device(0)
    # torch.autograd.set_detect_anomaly(True)
    ######################
    lr = 10 ** -4
    model = BruteForceDocking().to(device=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_stream = get_docking_stream(trainset + '.pkl', batch_size=1)
    valid_stream = get_docking_stream(validset + '.pkl', batch_size=1)
    test_stream = get_docking_stream(testset + '.pkl', batch_size=1)

    ######################
    train_epochs = 10

    def train(resume_training=False, resume_epoch=0, debug=False):
        BruteForceDockingTrainer().train_model(model, optimizer, testcase, train_epochs, train_stream, valid_stream, test_stream,
                                               resume_training=resume_training, resume_epoch=resume_epoch, debug=debug)


    def plot_evaluation_set(check_epoch, train_epochs=1, plotting=False):
        BruteForceDockingTrainer().train_model(model, optimizer, testcase, train_epochs, train_stream, valid_stream, test_stream,
                                               resume_training=True, resume_epoch=check_epoch, plotting=plotting, debug=False)

    ######################
    ### Train model from beginning
    # epoch = train_epochs
    train(debug=False)

    ### Resume training model at chosen epoch
    # train(True, resume_epoch=100)

    ### Evaluate model only and plot, at chosen epoch
    # plotting = True
    # plotting = False
    # epoch = '' # when loading FI trained docking model state_dict explicitly.
    # epoch = 11 # best epoch from 'randinit_best_docking_model_epoch'
    # epoch = 75 # best epoch from 'onesinit_lr4_best_docking_model_epoch'
    # epoch = 200 # best epoch from '16scalar32vector_docking_model_epoch'
    # plot_evaluation_set(check_epoch=epoch, plotting=plotting)
