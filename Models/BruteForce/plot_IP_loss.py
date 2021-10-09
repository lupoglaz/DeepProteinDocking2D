import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# testcase = 'newdata_bugfix_docking_100epochs_'
# testcase = 'test_datastream'
# testcase = 'best_docking_model_epoch'
# testcase = 'randinit_best_docking_model_epoch'
# testcase = 'onesinit_lr4_best_docking_model_epoch'
testcase = '16scalar32vector_docking_epoch'

#LOSS WITH ROTATION
train = pd.read_csv("Log/losses/log_train_"+ testcase +".txt", sep='\t', header=1, names=['Epoch',	'Loss',	'rmsd'])
valid = pd.read_csv("Log/losses/log_valid_"+ testcase +".txt", sep='\t', header=1, names=['Epoch', 'Loss', 'rmsd'])
test = pd.read_csv("Log/losses/log_test_"+ testcase +".txt", sep='\t', header=1, names=['Epoch', 'Loss', 'rmsd'])

num_epochs = len(train['Epoch'].to_numpy())

fig, ax = plt.subplots(2)
train_rmsd = ax[0].plot(train['Epoch'].to_numpy(), train['rmsd'].to_numpy())
valid_rmsd = ax[0].plot(valid['Epoch'].to_numpy(), valid['rmsd'].to_numpy())
test_rmsd = ax[0].plot(test['Epoch'].to_numpy(), test['rmsd'].to_numpy())
ax[0].legend(('train rmsd', 'valid rmsd', 'test rmsd'))

ax[0].set_title('Loss: '+testcase)
ax[0].set_ylabel('rmsd')
ax[0].grid()

train_loss = ax[1].plot(train['Epoch'].to_numpy(), train['Loss'].to_numpy())
valid_loss = ax[1].plot(valid['Epoch'].to_numpy(), valid['Loss'].to_numpy())
test_loss = ax[1].plot(test['Epoch'].to_numpy(), test['Loss'].to_numpy())
ax[1].legend(('train loss', 'valid loss', 'test loss'))

# best_train_rmsd = train['rmsd'].min()
# best_valid_rmsd = valid['rmsd'].min()
# best_test_rmsd = test['rmsd'].min()

# ax[1].set_title('RMSD: '+testcase)
ax[1].set_xlabel('epochs')
ax[1].set_ylabel('loss')
ax[1].grid()
# plt.yticks(np.arange(0, int(train['Loss'].max()) + 1, 1))
ax[1].set_xticks(np.arange(0, num_epochs, 10))

plt.xlabel('Epochs')
ax[0].set_ylim([0,20])
ax[1].set_ylim([0,20])
plt.savefig('figs/'+testcase+'.png')
plt.show()
