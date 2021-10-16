import torch
import numpy as np

def get_threshold(pred, target):
	sorted_pred, perm = torch.sort(pred)
	sorted_target = target[perm]
	target_true = (sorted_target == 1.0).to(dtype=torch.float32)
	target_false = (sorted_target == 0.0).to(dtype=torch.float32)
	
	cum_TP = torch.cumsum(target_true, dim=0)
	cum_FN = torch.cumsum(target_false, dim=0)

	cum_FP = torch.cumsum(target_true.flip(dims=(0,)), dim=0).flip(dims=(0,))
	cum_TN = torch.cumsum(target_false.flip(dims=(0,)), dim=0).flip(dims=(0,))
	
	acc = (cum_TP + cum_TN).to(dtype=torch.float)/(cum_TP+cum_TN+cum_FP+cum_FN).to(dtype=torch.float)
	Accuracy, idx = torch.max(acc, dim=0)
	TP = cum_TP[idx]
	FP = cum_FP[idx]
	TN = cum_TN[idx]
	FN = cum_FN[idx]
	if (TP+FP)>0:
		Precision = float(TP)/float(TP + FP)
	else:
		Precision = 0.0
	if (TP + FN)>0:
		Recall = float(TP)/float(TP + FN)
	else:
		Recall = 0.0
	F1 = Precision*Recall/(Precision + Recall+1E-5)
	MCC = (TP*TN - FP*FN)/torch.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)+1E-5).item()
	return Accuracy.item(), Precision, Recall, MCC


if __name__=='__main__':
	scores = torch.randn(10, dtype=torch.float32)
	target = (torch.randn(10, dtype=torch.float32)>0.0).to(dtype=torch.int)
	print(scores)
	print(target)
	print(get_threshold(scores, target))

