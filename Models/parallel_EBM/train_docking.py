import os

import torch
from torch import optim
from pathlib import Path
import numpy as np
import argparse
import sys
sys.path.append('/home/sb1638/')
# print(sys.path)
sys.path.append('C:\\Users\\Sid\\PycharmProjects\\lamoureuxlab\\')
# print(sys.path)

import warnings
warnings.filterwarnings("ignore")

from DeepProteinDocking2D.Models import EQScoringModel, EQDockerGPU, CNNInteractionModel, EQRepresentation, EQInteraction
from DeepProteinDocking2D.torchDataset import get_docking_stream, get_interaction_stream_balanced
from tqdm import tqdm

from EBMTrainer import EBMTrainer
from DeepProteinDocking2D.SupervisedTrainer import SupervisedTrainer
from DeepProteinDocking2D.DockingTrainer import DockingTrainer

from DeepProteinDocking2D.DatasetGeneration import Protein, Complex
# from Logger import Logger
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pylab as plt
import seaborn as sea

sea.set_style("whitegrid")


def run_FI_eval(valid_stream, trainer):
    log_format = '%f\t%f\t%f\t%f\t%f\n'
    log_header = 'Accuracy\tPrecision\tRecall\tF1score\tMCC\n'
    TP, FP, TN, FN = 0, 0, 0, 0
    pos_idx = 0
    for data in tqdm(valid_stream):
        receptor, ligand, gt_interact = data
        data = (receptor, ligand, gt_interact, torch.tensor(pos_idx).unsqueeze(0).cuda())
        tp, fp, tn, fn, pred_minus_gt = trainer.step_parallel(data, epoch=epoch, train=False)
        # print(tp, fp, tn,fn)
        TP += tp
        FP += fp
        TN += tn
        FN += fn
        pos_idx += 1

    Accuracy = float(TP + TN) / float(TP + TN + FP + FN)
    if (TP + FP) > 0:
        Precision = float(TP) / float(TP + FP)
    else:
        Precision = 0.0
    if (TP + FN) > 0:
        Recall = float(TP) / float(TP + FN)
    else:
        Recall = 0.0
    F1score = TP / (TP + 0.5 * (FP + FN) + 1E-5)

    MCC = ((TP * TN) - (FP * FN)) / (np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) + 1E-5)

    # it_loss = MCC

    print(f'Epoch {epoch} Acc: {Accuracy} Prec: {Precision} Rec: {Recall} F1: {F1score} MCC: {MCC}')

    with open('Log/' + str(args.experiment) + '/log_EBM_FI_validAPR.txt', 'a') as fout:
        # fout.write('Epoch ' + str(check_epoch) + '\n')
        fout.write('Epoch' + str(epoch) + '\n')
        fout.write(log_header)
        fout.write(log_format % (Accuracy, Precision, Recall, F1score, MCC))
    fout.close()

    # return it_loss
    return {"Loss": pred_minus_gt.item()}

def run_docking_model(data, docker, iter, logger=None):
    receptor, ligand, translation, rotation, indexes = data
    receptor = receptor.to(device='cuda', dtype=torch.float).unsqueeze(dim=1)
    ligand = ligand.to(device='cuda', dtype=torch.float).unsqueeze(dim=1)
    translation = translation.to(device='cuda', dtype=torch.float)
    rotation = rotation.to(device='cuda', dtype=torch.float).unsqueeze(dim=1)
    docker.eval()
    pred_angles, pred_translations = docker(receptor, ligand)

    rec = Protein(receptor[0, 0, :, :].cpu().numpy())
    lig = Protein(ligand[0, 0, :, :].cpu().numpy())
    angle = rotation[0].item()
    pos = translation[0, :].cpu().numpy()
    angle_pred = pred_angles.item()
    pos_pred = pred_translations.cpu().numpy()
    rmsd = lig.rmsd(pos, angle, pos_pred, angle_pred)

    if not (logger is None):
        logger.add_image("DockIP/Docker/Translations", docker.top_translations, iter, dataformats='HW')
        fig = plt.figure()
        plt.plot(docker.angles.cpu(), docker.top_rotations)
        logger.add_figure("DockIP/Docker/Rotations", fig, iter)

        cell_size = 100
        plot_image = np.zeros((2 * cell_size, cell_size))
        cplx = Complex(Protein(receptor.squeeze().cpu().numpy()),
                       Protein(ligand.squeeze().cpu().numpy()),
                       rotation.cpu().item(),
                       translation.cpu().squeeze().numpy())
        plot_image[:cell_size, :] = cplx.get_canvas(cell_size=cell_size)

        cplx = Complex(Protein(receptor.squeeze().cpu().numpy()),
                       Protein(ligand.squeeze().cpu().numpy()),
                       angle_pred,
                       pos_pred)
        plot_image[cell_size:, :] = cplx.get_canvas(cell_size=cell_size)
        logger.add_image("DockIP/Docker/Dock", plot_image, iter, dataformats='HW')

    return float(rmsd)


def run_prediction_model(data, trainer, epoch=None):
    log_dict = trainer.eval(data)
    receptor, ligand, translation, rotation, _ = data
    log_data = {"receptors": receptor.unsqueeze(dim=1).cpu(),
                "ligands": ligand.unsqueeze(dim=1).cpu(),
                "rotation": rotation.squeeze().cpu(),
                "translation": translation.squeeze().cpu(),
                "pred_rotation": log_dict["Rotation"].squeeze().cpu(),
                "pred_translation": log_dict["Translation"].squeeze().cpu()}
    return log_dict["Loss"], log_data


def save_checkpoint(state, filename, model):
    model.eval()
    torch.save(state, filename)

def load_ckp(checkpoint_fpath, model, optimizer):
    model.eval()
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']

if __name__ == '__main__':
    #### initialization torch settings
    import random
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)
    # torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser(description='Train deep protein docking')
    parser.add_argument('-data_dir', default='Log', type=str)
    parser.add_argument('-experiment', default='DebugDocking', type=str)

    parser.add_argument('-train', action='store_const', const=lambda: 'train', dest='cmd')
    parser.add_argument('-test', action='store_const', const=lambda: 'test', dest='cmd')

    parser.add_argument('-resnet', action='store_const', const=lambda: 'resnet', dest='model')
    parser.add_argument('-ebm', action='store_const', const=lambda: 'ebm', dest='model')
    parser.add_argument('-docker', action='store_const', const=lambda: 'docker', dest='model')

    parser.add_argument('-gpu', default=1, type=int)
    parser.add_argument('-step_size', default=10.0, type=float)
    parser.add_argument('-num_samples', default=10, type=int)
    parser.add_argument('-batch_size', default=24, type=int)
    parser.add_argument('-num_epochs', default=100, type=int)
    parser.add_argument('-LD_steps', default=100, type=int)


    parser.add_argument('-no_global_step', action='store_const', const=lambda: 'no_global_step', dest='ablation')
    parser.add_argument('-no_pos_samples', action='store_const', const=lambda: 'no_pos_samples', dest='ablation')
    parser.add_argument('-default', action='store_const', const=lambda: 'default', dest='ablation')
    parser.add_argument('-parallel_noGSAP', action='store_const', const=lambda: 'parallel_noGSAP', dest='ablation')
    parser.add_argument('-FI', action='store_const', const=lambda: 'FI', dest='ablation')

    args = parser.parse_args()

    if (args.model is None):
        parser.print_help()
        sys.exit()

    if torch.cuda.device_count() > 1:
        for i in range(torch.cuda.device_count()):
            print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
        torch.cuda.set_device(args.gpu)

    train_stream = get_docking_stream('toy_concave_data/docking_data_train.pkl', batch_size=args.batch_size,
                                      max_size=None)
    valid_stream = get_docking_stream('toy_concave_data/docking_data_valid.pkl', batch_size=1, max_size=None)

    if args.model() == 'resnet':
        model = CNNInteractionModel().to(device='cuda')
        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.0, 0.999))
        trainer = SupervisedTrainer(model, optimizer, type='pos')

    elif args.model() == 'ebm':
        repr = EQRepresentation()
        model = EQScoringModel(repr=repr).to(device='cuda')
        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.0, 0.999))
        if args.ablation is None:
            print('My default')
            trainer = EBMTrainer(model, optimizer, num_samples=args.num_samples,
                                 num_buf_samples=len(train_stream) * args.batch_size, step_size=args.step_size,
                                 global_step=True, add_positive=True)
        elif args.ablation() == 'no_global_step':
            print('No global step')
            trainer = EBMTrainer(model, optimizer, num_samples=args.num_samples,
                                 num_buf_samples=len(train_stream) * args.batch_size, step_size=args.step_size,
                                 global_step=False, add_positive=True)
        elif args.ablation() == 'no_pos_samples':
            print('No positive samples')
            trainer = EBMTrainer(model, optimizer, num_samples=args.num_samples,
                                 num_buf_samples=len(train_stream) * args.batch_size, step_size=args.step_size,
                                 global_step=True, add_positive=False)
        elif args.ablation() == 'default':
            print('Default')
            trainer = EBMTrainer(model, optimizer, num_samples=args.num_samples,
                                 num_buf_samples=len(train_stream) * args.batch_size, step_size=args.step_size,
                                 global_step=False, add_positive=False)
        elif args.ablation() == 'parallel_noGSAP':
            print('Parallel, two different distribution sigmas, no GS, no AP')
            trainer = EBMTrainer(model, optimizer, num_samples=args.num_samples,
                                 num_buf_samples=len(train_stream) * args.batch_size, step_size=args.step_size,
                                 global_step=False, add_positive=False, sample_steps=args.LD_steps, experiment=args.experiment)

            # trainer.load_checkpoint(Path('Log') / Path('check_IP_LD10_randseed_rep1') / Path('model.th'))
            # print(Path('Log') / Path('check_IP_LD10_randseed_rep1') / Path('model.th'))

        elif args.ablation() == 'FI':
            # mkdir = 'mkdir -p ../../EBM_figs/'+args.experiment+'/'
            # os.system(mkdir)
            # print(mkdir)
            print('Fact of interaction: using parallel, different distribution sigmas, no GS, no AP')
            # max_size = 100
            # max_size = 200
            # max_size = 50
            max_size = 50
            train_stream = get_interaction_stream_balanced('../../DatasetGeneration/interaction_data_train.pkl',
                                                           batch_size=args.batch_size,
                                                           max_size=max_size
                                                           )
            valid_stream = get_interaction_stream_balanced('../../DatasetGeneration/interaction_data_valid.pkl', batch_size=1,
                                                           max_size=max_size//2
                                                           )
            trainer = EBMTrainer(model, optimizer, num_samples=args.num_samples,
                                 num_buf_samples=len(train_stream) * args.batch_size, step_size=args.step_size,
                                 global_step=False, add_positive=False, sample_steps=args.LD_steps, FI=True, experiment=args.experiment)

            # trainer.load_checkpoint(Path('Log') / Path('check_IP_LD10_randseed_rep1') / Path('model.th'))
            # print(Path('Log') / Path('check_IP_LD10_randseed_rep1') / Path('model.th'))



    elif args.model() == 'docker':
        repr = EQRepresentation()
        model = EQScoringModel(repr=repr).to(device='cuda')
        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.0, 0.999))
        trainer = DockingTrainer(model, optimizer, type='pos')

    ### TRAINING
    if args.cmd() == 'train':
        logger = SummaryWriter(Path(args.data_dir) / Path(args.experiment))
        min_loss = float('+Inf')
        min_dockerloss = float('+Inf')

        # if args.ablation() == 'FI':
        #     print('*' * 100)
        #     print('Loading pretrained model for EBM')
            # trainer.load_checkpoint(Path('Log') / Path('check_IP_LD10_randseed_rep1') / Path('model.th'))
            # print(Path('Log') / Path('check_IP_LD10_randseed_rep1') / Path('model.th'))
            # model, optimizer, start_epoch = load_ckp(Path(args.data_dir) / Path(args.experiment) / Path('model.th'), model, optimizer)


        # iter = 0
        for epoch in range(args.num_epochs):

            iter = 0
            for data in tqdm(train_stream):
                if args.ablation() == 'parallel_noGSAP':
                    log_dict = trainer.step_parallel(data, epoch=epoch, train=True)
                    logger.add_scalar("DockIP/Loss/Train", log_dict["Loss"], iter*(epoch+1))
                elif args.ablation() == 'FI':
                    print('\nFI parallel')
                    receptor, ligand, gt_interact = data

                    data = (receptor, ligand, gt_interact, torch.tensor(iter).unsqueeze(0).cuda())
                    log_dict = trainer.step_parallel(data, epoch=epoch, train=True)
                    logger.add_scalar("DockFI/Loss/Train/", log_dict["Loss"], iter*(epoch+1))
                else:
                    log_dict = trainer.step(data, epoch=epoch)
                    logger.add_scalar("DockIP/Loss/Train", log_dict["Loss"], iter)
                iter += 1

            if args.ablation() == 'FI':
                log_valid = run_FI_eval(valid_stream, trainer)
                logger.add_scalar("DockFI/Loss/Valid/", log_valid["Loss"], epoch)

            else:
                loss = []
                dockerloss = []
                log_data = []
                docker = EQDockerGPU(model.eval(), num_angles=360)
                iter = 0
                for i, data in tqdm(enumerate(valid_stream)):
                    if args.model() == 'resnet':
                        it_loss, it_log_data = run_prediction_model(data, trainer, epoch=epoch, train=False)
                    elif args.model() == 'ebm' or args.model() == 'docker':
                        if args.ablation() == 'parallel_noGSAP':
                            receptor, ligand, translation, rotation, indexes = data
                            rec = Protein(receptor[0, :, :].cpu().numpy())
                            lig = Protein(ligand[0, :, :].cpu().numpy())
                            angle = rotation[0].item()
                            pos = translation[0, :].cpu().numpy()

                            if i == 0:
                                log_dict, angle_pred, pos_pred = trainer.step_parallel(data, epoch=epoch, train=False)
                                angle_pred = angle_pred[0].item()
                                pos_pred = pos_pred[0].cpu().numpy()
                                it_loss = float(lig.rmsd(pos, angle, pos_pred, angle_pred))
                                docker_loss = run_docking_model(data, docker, iter, logger)
                            else:
                                log_dict, angle_pred, pos_pred = trainer.step_parallel(data, epoch=epoch, train=False)
                                angle_pred = angle_pred[0].item()
                                pos_pred = pos_pred[0].cpu().numpy()
                                it_loss = float(lig.rmsd(pos, angle, pos_pred, angle_pred))
                                docker_loss = run_docking_model(data, docker, iter)
                        else:
                            if i == 0:
                                log_dict = trainer.step(data, epoch=epoch)
                                it_loss = run_docking_model(data, docker, iter, logger)
                            else:
                                log_dict = trainer.step(data, epoch=epoch)
                                it_loss = run_docking_model(data, docker, iter)
                    iter += 1

                    loss.append(it_loss)
                    dockerloss.append(docker_loss)
                    logger.add_scalar("DockIP/Loss/Valid", it_loss, iter)
                    logger.add_scalar("DockIP/Loss/ValidDocker", docker_loss, iter)

                av_loss = np.average(loss, axis=0)
                # logger.add_scalar("DockIP/Loss/Valid", av_loss, iter)
                av_dockerloss = np.average(dockerloss, axis=0)
                # logger.add_scalar("DockIP/Loss/Valid", av_dockerloss, iter)

                print('Epoch', epoch, 'Valid Loss:', av_loss)
                if av_loss < min_loss:
                    # torch.save(model.state_dict(), Path('Log') / Path(args.experiment) / Path('model.th'))
                    print(f'Model saved: min_loss = {av_loss} prev = {min_loss}')
                    min_loss = av_loss
                print('Epoch', epoch, 'Valid Docker Loss:', av_dockerloss)
                if av_dockerloss < min_dockerloss:
                    torch.save(model.state_dict(), Path('Log') / Path(args.experiment) / Path('model.th'))
                    print(f'docker eval: min_loss = {av_dockerloss} prev = {min_dockerloss}')
                    min_dockerloss = av_dockerloss

    ### TESTING
    if args.cmd() == 'test':
        test_stream = get_docking_stream('DatasetGeneration/docking_data_test.pkl', batch_size=1, max_size=None)
        trainer.load_checkpoint(Path(args.data_dir) / Path(args.experiment) / Path('model.th'))
        docker = EQDockerGPU(model, num_angles=360)
        loss = []
        for data in tqdm(test_stream):
            if args.model() == 'resnet':
                it_loss, it_log_data = run_prediction_model(data, trainer, epoch=0)
            elif args.model() == 'ebm' or args.model() == 'docker':
                it_loss = run_docking_model(data, docker, 0)
            loss.append(it_loss)
        av_loss = np.average(loss, axis=0)
        print(f'Test result: {av_loss}')

        valid_stream = get_docking_stream('DatasetGeneration/docking_data_valid.pkl', batch_size=1, max_size=None)
        loss = []
        for data in tqdm(valid_stream):
            if args.model() == 'resnet':
                it_loss, it_log_data = run_prediction_model(data, trainer, epoch=0)
            elif args.model() == 'ebm' or args.model() == 'docker':
                it_loss = run_docking_model(data, docker, 0)
            loss.append(it_loss)
        av_loss = np.average(loss, axis=0)
        print(f'Valid result: {av_loss}')

    # if args.cmd == 'train':
    # 	ablation = "None"
    # 	if not(args.ablation is None): args.ablation()
    # 	logger.add_hparams(	{	'ModelType': args.model(),
    # 							'Ablation': ablation,
    # 							'StepSize': args.step_size, 'NumSamples': args.num_samples
    # 						},
    # 						{'hparam/valid_loss': min_loss, 'hparam/test_loss': av_loss})#, run_name=args.experiment)


###

#python train_docking.py -data_dir Log -experiment FI_WITHpretrain_BFfreeE_gitrestore_both -train -ebm -num_epochs 1 -batch_size 1 -num_samples 1 -gpu 0 -FI
