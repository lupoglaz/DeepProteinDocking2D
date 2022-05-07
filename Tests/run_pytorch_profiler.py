from torch import optim
import torch.autograd.profiler as profiler

from DeepProteinDocking2D.Models.BruteForce.train_bruteforce_docking import BruteForceDockingTrainer, Docking, get_docking_stream

experiment = 'BF_IP_FINAL_DATASET_400pool_1000ex_30ep'

######################
trainset = '../Datasets/docking_train_400pool'

train_epochs = 1
lr = 10 ** -4
model = Docking().to(device=0)
optimizer = optim.Adam(model.parameters(), lr=lr)
train_stream = get_docking_stream(trainset+'.pkl', max_size=10)

### warm-up
BruteForceDockingTrainer(model, optimizer, experiment).run_trainer(
    train_epochs=train_epochs, train_stream=train_stream, valid_stream=None, test_stream=None)

#### run profiler
with profiler.profile(with_stack=True, profile_memory=True) as prof:
    BruteForceDockingTrainer(model, optimizer, experiment).run_trainer(
        train_epochs=train_epochs, train_stream=train_stream, valid_stream=None, test_stream=None)

print(prof.key_averages().table(sort_by='self_cpu_time_total'))
print(prof.total_average())
