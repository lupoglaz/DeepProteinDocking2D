import torch
import numpy as np
import random
from DeepProteinDocking2D.Models.BruteForce.utility_functions import read_pkl, plot_assembly
import matplotlib.pyplot as plt
from tqdm import tqdm

class RMSD:
    def __init__(self, ligand, gt_rot, gt_txy, pred_rot, pred_txy):
        self.bulk = np.array(ligand.detach().cpu())
        self.size = self.bulk.shape[-1]
        self.gt_rot = gt_rot
        self.gt_txy = gt_txy
        self.pr_rot = pred_rot
        self.pr_txy = pred_txy
        self.epsilon = 1e-3

    def get_XC(self):
        """
        Analog of inertia tensor and center of mass for rmsd calc. Return 2/W * X and C
        """
        X = torch.zeros(2, 2)
        C = torch.zeros(2)
        x_i = (torch.arange(self.size).unsqueeze(dim=0) - self.size / 2.0).repeat(self.size, 1)
        y_i = (torch.arange(self.size).unsqueeze(dim=1) - self.size / 2.0).repeat(1, self.size)
        mask = torch.from_numpy(self.bulk > 0.5)
        W = torch.sum(mask.to(dtype=torch.float32))
        x_i = x_i.masked_select(mask)
        y_i = y_i.masked_select(mask)
        # Inertia tensor
        X[0, 0] = torch.sum(x_i * x_i)
        X[1, 1] = torch.sum(y_i * y_i)
        X[0, 1] = torch.sum(x_i * y_i)
        X[1, 0] = torch.sum(y_i * x_i)
        # Center of mass
        C[0] = torch.sum(x_i)
        C[1] = torch.sum(x_i)
        return 2.0 * X / (W + self.epsilon), C / (W + self.epsilon)

    def rmsd(self):
        rotation1, translation1, rotation2, translation2 = self.gt_rot, self.gt_txy, self.pr_rot, self.pr_txy
        X, C = self.get_XC()
        X = X.type(torch.float).cuda()
        C = C.type(torch.float).cuda()

        T1 = translation1.clone().detach().cuda()
        T2 = translation2.clone().detach().cuda()
        T = T1 - T2

        rotation1 = torch.tensor([rotation1], dtype=torch.float).cuda()
        rotation2 = torch.tensor([rotation2], dtype=torch.float).cuda()
        R1 = torch.zeros(2, 2, dtype=torch.float).cuda()
        R1[0, 0] = torch.cos(rotation1)
        R1[1, 1] = torch.cos(rotation1)
        R1[1, 0] = torch.sin(rotation1)
        R1[0, 1] = -torch.sin(rotation1)
        R2 = torch.zeros(2, 2, dtype=torch.float).cuda()
        R2[0, 0] = torch.cos(rotation2)
        R2[1, 1] = torch.cos(rotation2)
        R2[1, 0] = torch.sin(rotation2)
        R2[0, 1] = -torch.sin(rotation2)
        R = R2.transpose(0, 1) @ R1

        I = torch.diag(torch.ones(2, dtype=torch.float)).cuda()
        # RMSD
        rmsd = torch.sum(T * T)
        rmsd = rmsd + torch.sum((I - R) * X, dim=(0, 1))
        rmsd = rmsd + 2.0 * torch.sum(torch.sum(T.unsqueeze(dim=1) * (R1 - R2), dim=0) * C, dim=0) + self.epsilon

        return torch.sqrt(rmsd)

    def calc_rmsd(self):
        rmsd = RMSD.rmsd(self)
        return rmsd


class APR:
    def __init__(self):
        pass

    def calcAPR(self, stream, trainer, model, epoch=0, pretrain_model=None):
        print('Calculating Accuracy, Precision, Recall')
        TP, FP, TN, FN = 0, 0, 0, 0

        for data in tqdm(stream):
            tp, fp, tn, fn = trainer.run_model(data, model, train=False, pretrain_model=pretrain_model)
            # print(tp, fp, tn,fn)
            TP += tp
            FP += fp
            TN += tn
            FN += fn

        Accuracy = float(TP + TN) / float(TP + TN + FP + FN)
        if (TP + FP) > 0:
            Precision = float(TP) / float(TP + FP)
        else:
            Precision = 0.0
        if (TP + FN) > 0:
            Recall = float(TP) / float(TP + FN)
        else:
            Recall = 0.0
        F1score = TP / (TP + 0.5*(FP + FN))

        MCC = ((TP * TN) - (FP * FN)) / (np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))

        print(f'Epoch {epoch} Acc: {Accuracy} Prec: {Precision} Rec: {Recall} F1: {F1score} MCC: {MCC}')

        return Accuracy, Precision, Recall, F1score, MCC

if __name__ == '__main__':
    from DeepProteinDocking2D.Models.BruteForce.model_bruteforce_docking import BruteForceDocking

    def load_ckp(checkpoint_fpath, model):
        model.eval()
        checkpoint = torch.load(checkpoint_fpath)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        return model

    def extract_transform(pred_score):
        dim = 100
        # dim = 52
        pred_argmax = torch.argmax(pred_score)
        # print(pred_argmax)
        pred_rot = ((pred_argmax / dim ** 2) * np.pi / 180.0) - np.pi
        # print(pred_score_rot)
        XYind = torch.remainder(pred_argmax, dim ** 2)
        # print(XYind)
        pred_X = XYind // dim
        pred_Y = XYind % dim
        if pred_X > dim//2:
            pred_X = pred_X - dim
        if pred_Y > dim//2:
            pred_Y = pred_Y - dim
        # print(pred_X, plot_Y)
        return pred_rot, torch.stack((pred_X, pred_Y), dim=0)

    def check_RMSD(model, data, index, plotting):
        receptor, ligand, gt_txy, gt_rot = data
        # print(gt_rot)
        # gt_rot = gt_rot + np.pi
        receptor = torch.from_numpy(receptor).type(torch.float).unsqueeze(0).cuda()
        ligand = torch.from_numpy(ligand).type(torch.float).unsqueeze(0).cuda()
        gt_rot = torch.tensor(gt_rot, dtype=torch.float)
        gt_txy = torch.tensor(gt_txy, dtype=torch.float)
        pred_score = model(receptor, ligand, plotting=plotting)
        pred_rot, pred_txy = extract_transform(pred_score)
        # print('extracted predicted indices', pred_rot.item(), pred_txy)
        # print('ground truth indices', gt_rot, gt_txy)

        rmsd_out = RMSD(ligand, gt_rot, gt_txy, pred_rot, pred_txy).calc_rmsd()

        print('RMSD',rmsd_out.item())

        pair = plot_assembly(receptor.squeeze().detach().cpu().numpy(), ligand.squeeze().detach().cpu().numpy(),
                             pred_rot.detach().cpu().numpy(), (pred_txy[0].detach().cpu().numpy(), pred_txy[1].detach().cpu().numpy()),
                             gt_rot.squeeze().detach().cpu().numpy(), gt_txy.squeeze().detach().cpu().numpy())

        plt.imshow(pair.transpose())
        plt.title('Ground Truth                      Input                       Predicted Pose')
        plt.text(10,10, "Ligand RMSD="+str(rmsd_out.item()), backgroundcolor='w')
        if plotting:
            plt.savefig('figs/RMSDCheck_Ligand'+str(index)+'_RMSD_'+str(rmsd_out.item()) + '.png')
            plt.show()

        return rmsd_out


    #### initialization torch settings
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.determininistic = True
    torch.cuda.set_device(0)

    plotting = False

    testcase = 'docking_pretrain_bruteforce_allLearnedWs_10epochs_'
    dataset = 'docking'
    setid = 'test'
    testset = 'toy_concave_data/' + dataset + '_data_' + setid
    resume_epoch = 10

    # testcase = 'TEST_interactionL2loss_B*smax(B)relu()_BruteForce_training_'
    # dataset = 'interaction'
    # setid = 'valid'
    # testset = 'toy_concave_data/'+dataset+'_data_'+setid
    # resume_epoch = 5

    data = read_pkl(testset)
    model = BruteForceDocking().to(device=0)
    ckp_path = 'Log/' + testcase + str(resume_epoch) + '.th'
    model = load_ckp(ckp_path, model)

    log_header = 'Example\tRMSD\n'
    with open('Log/log_RMSD_'+dataset+'_'+setid+'_' + testcase + '.txt', 'w') as fout:
        fout.write(log_header)

        rmsd_list = []
        index_range = len(data)
        print(index_range)
        for index in range(index_range):
            rmsd_out = check_RMSD(model, data[index], index, plotting=plotting)
            fout.write(str(index)+'\t'+str(rmsd_out.item())+'\n')
            rmsd_list.append(rmsd_out)


        print()
        avg_rmsd = torch.mean(torch.as_tensor(rmsd_list))
        print('Avg RMSD', avg_rmsd.item())
        fout.write('Avg RMSD\t'+str(avg_rmsd.item())+'\n')
        fout.close()
