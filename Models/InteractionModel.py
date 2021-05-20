import torch
from torch import nn

class BasicBlock(nn.Module):
	def __init__(self, inplanes: int, planes: int, stride: int=1, downsample = None) -> None:
		super(BasicBlock, self).__init__()
		self.net = nn.Sequential(
			nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, bias=False, padding=1),
			nn.BatchNorm2d(planes),
			nn.ReLU(inplace=True),
			nn.Conv2d(inplanes, planes, kernel_size=3, bias=False, padding=1),
			nn.BatchNorm2d(planes)
		)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample

	def forward(self, x):
		identity = x
		out = self.net(x)
		if self.downsample is not None:
			identity = self.downsample(x)
		out += identity
		out = self.relu(out)
		return out


class CNNInteractionModel(nn.Module):
	def __init__(self, layers=[2, 2, 2, 2], type='int'):
		super(CNNInteractionModel, self).__init__()
		self.inplanes = 64
		self.type = type
		self.layer0 = nn.Sequential(
			nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
			nn.BatchNorm2d(self.inplanes),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		)
		self.layer1 = self._make_layer(128, 128, blocks=2)
		self.layer2 = self._make_layer(256, 256, blocks=2, stride=2)
		self.layer3 = self._make_layer(512, 512, blocks=2, stride=2)
		
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		if self.type == 'int':
			self.fc_int = nn.Linear(1024, 1)
		elif self.type == 'pos':
			self.fc_pos = nn.Linear(1024, 3)
		else:
			raise(Exception('Type unknown:', type))
		self.sigmoid = nn.Sigmoid()

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, inplanes: int, planes: int, blocks: int, stride: int=1):
		downsample = None
		if stride != 1 or inplanes != planes:
			downsample = nn.Sequential(
				nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, stride=stride),
				nn.BatchNorm2d(planes)
			)
		layers = []
		layers.append(BasicBlock(inplanes, planes, downsample=downsample, stride=stride))
		for _ in range(1, blocks):
			layers.append(BasicBlock(planes, planes))
		return nn.Sequential(*layers)

	def forward(self, receptor, ligand):
		rec = self.layer0(receptor)
		lig = self.layer0(ligand)
		
		rec_cat = torch.cat([rec,lig], dim=1)
		lig_cat = torch.cat([lig, rec], dim=1)
		rec = self.layer1(rec_cat)
		lig = self.layer1(lig_cat)

		rec_cat = torch.cat([rec,lig], dim=1)
		lig_cat = torch.cat([lig, rec], dim=1)
		rec = self.layer2(rec_cat)
		lig = self.layer2(lig_cat)

		rec_cat = torch.cat([rec,lig], dim=1)
		lig_cat = torch.cat([lig, rec], dim=1)
		rec = self.layer3(rec_cat)
		rec = self.avgpool(rec)
		rec = torch.flatten(rec, 1)

		lig = self.layer3(lig_cat)
		lig = self.avgpool(lig)
		lig = torch.flatten(lig, 1)
		
		if self.type == 'int':
			interaction = self.sigmoid(self.fc_int(torch.cat([rec,lig], dim=1)))
			return interaction
		else:
			position = self.fc_pos(torch.cat([rec,lig], dim=1))
			return position
		
		