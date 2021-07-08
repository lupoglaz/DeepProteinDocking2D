import random
import torch
from torch import nn
from torch import optim

import sys
sys.path.append('/home/sb1638/')

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
            pretrain_model.train()

        ### run model and loss calculation
        ##### call model(s)
        FFT_score = pretrain_model(receptor, ligand, plotting=False)

        pred_interact = model(FFT_score, plotting=False)

        ### check if pretrain weights are frozen or updating
        # for n, p in pretrain_model.named_parameters():
        #     if p.requires_grad:
        #         print(n, p, p.grad)
        #### Loss functions
        BCEloss = torch.nn.BCELoss()
        loss = BCEloss(pred_interact, gt_interact)
        # print(pred_interact.item(), gt_interact.item())


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
            pretrain_model.zero_grad()
            model.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_pretrain.step()
            optimizer.step()

        else:
            pretrain_model.eval()
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
                    resume_epoch=0, plotting=False, pretrain_model=None, optimizer_pretrain=None):

        test_freq = 1
        save_freq = 1

        if plotting:
            test_freq = 1

        log_header = 'Epoch\tLoss\trmsd\n'
        log_format = '%d\t%f\t%f\n'

        ### Continue training on existing model?
        if resume_training:
            print('Loading interaction model at', str(resume_epoch))
            ckp_path = 'Log/' + testcase + str(resume_epoch) + '.th'
            model, optimizer, start_epoch = BruteForceDockingTrainer().load_ckp(ckp_path, model, optimizer)

            print('Loading docking model at', str(resume_epoch))
            ckp_path = 'Log/docking_' + testcase + str(resume_epoch) + '.th'
            pretrain_model, optimizer_pretrain, start_epoch = BruteForceDockingTrainer().load_ckp(ckp_path, pretrain_model, optimizer_pretrain)

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

            pretrain_checkpoint_dict = {
                'epoch': epoch,
                'state_dict': pretrain_model.state_dict(),
                'optimizer': optimizer_pretrain.state_dict(),
            }


            if epoch % test_freq == 0 and epoch > 1:
                BruteForceInteractionTrainer().checkAPR(epoch, valid_stream, pretrain_model=pretrain_model)
                break

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
                BruteForceInteractionTrainer().save_checkpoint(pretrain_checkpoint_dict, 'Log/docking_' + testcase + str(epoch) + '.th')

                print('saving model ' + 'Log/' + testcase + str(epoch) + '.th')

        ### unecessary unless training > 1 epoch
        # BruteForceInteractionTrainer().save_checkpoint(checkpoint_dict, 'Log/' + testcase + 'end.th')
        # BruteForceInteractionTrainer().save_checkpoint(pretrain_checkpoint_dict, 'Log/docking_' + testcase + 'end.th')

    @staticmethod
    def checkAPR(check_epoch, datastream, pretrain_model):
        log_format = '%f\t%f\t%f\t%f\t%f\n'
        log_header = 'Accuracy\tPrecision\tRecall\tF1score\tMCC\n'
        Accuracy, Precision, Recall, F1score, MCC = APR().calcAPR(datastream, BruteForceInteractionTrainer(), model, check_epoch, pretrain_model)
        # print(Accuracy, Precision, Recall)
        with open('Log/losses/log_validAPR_' + testcase + '.txt', 'a') as fout:
            fout.write(log_header)
            fout.write(log_format % (Accuracy, Precision, Recall, F1score, MCC))
        fout.close()

    @staticmethod
    def freeze_weights(model, param_to_freeze=None):
        for name, param in model.named_parameters():
            if param_to_freeze is not None and param_to_freeze in name:
                print('Unfreezing Weights', name)
                param.requires_grad = True
            else:
                print('Freezing docking model weights', name)
                param.requires_grad = False


if __name__ == '__main__':
    #################################################################################
    # import sys
    # print(sys.path)
    import time

    trainset = 'toy_concave_data/interaction_data_train'
    validset = 'toy_concave_data/interaction_data_valid'
    ### testing set
    testset = 'toy_concave_data/interaction_data_test'

    ###### replicates
    # testcase = str(sys.argv[1])+'_pretrain_frozen_a,theta_'
    # testcase = str(sys.argv[1])+'_pretrain_unfrozen_a,theta_'

    ##### after thought checks
    # testcase = str(sys.argv[1])+'_bias=True_frozen'
    # testcase = str(sys.argv[1])+'_bias=True_unfrozen'
    # testcase = str(sys.argv[1])+'_bias=True_aW_unfrozen'

    # testcase = str(sys.argv[1])+'_bias=True_scratch'

    testcase = str(sys.argv[1])+'_dexpLOAD_bias=True_aW_unfrozen'


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
    ######################
    lr = 10 ** -4
    model = BruteForceInteraction().to(device=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    pretrain_model = BruteForceDocking().to(device=0)
    optimizer_pretrain = optim.Adam(pretrain_model.parameters(), lr=lr)

    # path_pretrain = 'Log/docking_pretrain_bruteforce_allLearnedWs_10epochs_end.th'
    # pretrain_model.load_state_dict(torch.load(path_pretrain)['state_dict'])
    #### freezing all weights in pretrain model
    # BruteForceInteractionTrainer().freeze_weights(pretrain_model, None)

    #### freezing weights except for "a" weights
    # BruteForceInteractionTrainer().freeze_weights(pretrain_model, 'W')

    #### load d experiment (pretrained: IP CNN frozen, a00...a11 unfrozen) and retrain IP as unfrozen
    path_pretrain = 'Log/docking_rep1_bias=True_aW_unfrozen1.th'
    pretrain_model.load_state_dict(torch.load(path_pretrain)['state_dict'])

    train_stream = get_interaction_stream_balanced(trainset + '.pkl', batch_size=1)
    valid_stream = get_interaction_stream_balanced(validset + '.pkl', batch_size=1)
    test_stream = get_interaction_stream_balanced(testset + '.pkl', batch_size=1)

    ######################
    train_epochs = 1

    def train(resume_training=False, resume_epoch=0):
        BruteForceInteractionTrainer().train_model(model, optimizer, testcase, train_epochs, train_stream, valid_stream,
                                               resume_training=resume_training, resume_epoch=resume_epoch,
                                                   pretrain_model=pretrain_model, optimizer_pretrain=optimizer_pretrain)

    def plot_validation_set(check_epoch, plotting=True, valid_stream=valid_stream, pretrain_model=pretrain_model):
        BruteForceInteractionTrainer().train_model(model, optimizer, testcase, train_epochs, train_stream, valid_stream,
                                               resume_training=True, resume_epoch=check_epoch, plotting=plotting,
                                                   pretrain_model=pretrain_model, optimizer_pretrain=optimizer_pretrain)

    #####################
    train()

    epoch = 1

    # give time to save models
    time.sleep(60)

    ### loads relevant pretrained model under resume_training condition
    plot_validation_set(check_epoch=epoch, valid_stream=valid_stream, pretrain_model=pretrain_model) ## also checks APR

    plot_validation_set(check_epoch=epoch, valid_stream=test_stream, pretrain_model=pretrain_model)

