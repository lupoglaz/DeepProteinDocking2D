import torch
from torch import nn
from torch import optim

class ModelClass(nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.net = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
        )
        ### with cuda
        # self.learnedW = nn.Parameter(torch.rand(1)).cuda()
        ### no cuda
        self.learnedW = nn.Parameter(torch.rand(1))

    def forward(self, x):
        x = self.net(x.unsqueeze(0).unsqueeze(0))
        x = x * self.learnedW
        return x

    @staticmethod
    def save_checkpoint(state, filename):
        # model.eval()
        torch.save(state, filename)

    @staticmethod
    def load_ckp(checkpoint_fpath, model, optimizer):
        # model.eval()
        checkpoint = torch.load(checkpoint_fpath)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer

if __name__ == "__main__":
    torch.cuda.set_device(0)
    pretrain_model = ModelClass().to(device=0)
    optimizer_pretrain = optim.Adam(pretrain_model.parameters(), lr=1e-3)
    input = torch.rand(50,50).cuda()
    output = pretrain_model(input)
    checkpoint_dict = {
        'state_dict': pretrain_model.state_dict(),
        'optimizer': optimizer_pretrain.state_dict(),
    }
    ModelClass().save_checkpoint(checkpoint_dict, 'train.th')
    path_pretrain = 'train.th'
    # pretrain_model.load_state_dict(torch.load(path_pretrain)['state_dict'])
    pretrain_model, _, = ModelClass().load_ckp(path_pretrain, pretrain_model, optimizer_pretrain)

    for name, param in pretrain_model.named_parameters():
        print(name)
        print(param)