import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

fullplot = True
# fullplot = False

loss_components = True
# loss_components = False

# CB_color_cycle = ['#377EB8', '#FF7F00', '#4DAF4A', '#F781BF', '#A65628', '#984EA3', '#999999', '#E41A1C', '#DEDE00']

testcase = 'newdata_BruteForce_training_check'

#LOSS WITH ROTATION
train = pd.read_csv("Log/losses/log_train_"+ testcase +".txt", sep='\t', header=0, names=['Epoch',	'Loss',	'rmsd'])
valid = pd.read_csv("Log/losses/log_test_"+ testcase +".txt", sep='\t', header=0, names=['Epoch', 'Loss', 'rmsd'])

num_epochs = train['Epoch'].to_numpy()

loss = plt.plot(train['Epoch'].to_numpy(), train['Loss'].to_numpy())
test = plt.plot(valid['Epoch'].to_numpy(), valid['Loss'].to_numpy())

if loss_components:
    train_rmsd = plt.plot(train['Epoch'].to_numpy(), train['rmsd'].to_numpy())
    test_rmsd = plt.plot(valid['Epoch'].to_numpy(), valid['rmsd'].to_numpy())
    best_valid_rmsd = valid['rmsd'].min()
    plt.legend(('train', 'valid', 'train_rmsd', 'valid_rmsd='+str(best_valid_rmsd)))
else:
    plt.legend(('train', 'test'), framealpha=0.5)

plt.title(testcase)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
# plt.yticks(np.arange(0, 20, 5))
# plt.xticks(np.arange(0, 110, 10))
# plt.ylim(0,360)
# plt.margins(0)

if not fullplot:
    plt.ylim(0,20)

plt.savefig('figs/'+testcase+'.png')
plt.show()