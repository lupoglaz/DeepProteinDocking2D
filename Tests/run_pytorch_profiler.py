from torch import optim
import torch.autograd.profiler as profiler

from DeepProteinDocking2D.Models.BruteForce.train_bruteforce_docking import BruteForceDockingTrainer, Docking, get_docking_stream

#### RUN PROFILER
# warm-up ^^^

experiment = 'BF_IP_FINAL_DATASET_400pool_1000ex_30ep'

######################
train_epochs = 30
lr = 10 ** -4
model = Docking().to(device=0)
optimizer = optim.Adam(model.parameters(), lr=lr)
trainset = '../../Datasets/docking_train_400pool'
train_stream = get_docking_stream(trainset+'.pkl', max_size=10)
with profiler.profile(with_stack=True, profile_memory=True) as prof:
    BruteForceDockingTrainer(model, optimizer, experiment).run_trainer(
        train_epochs=train_epochs, train_stream=train_stream, valid_stream=None, test_stream=None)

# print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
print(prof.key_averages().table())
