import torch
from Models import SharpLoss

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


if __name__=='__main__':
	loss = SharpLoss()
	y = torch.arange(0, 1.1, 0.1).unsqueeze(dim=0)
	x = torch.arange(-10, 10, 0.5).unsqueeze(dim=1)
	y = y.repeat(x.size(0), 1)
	x = x.repeat(1, y.size(1))
	x1 = x.clone()
	x1.requires_grad_()
	z = loss(x1.flatten(), y.flatten()).view(x.size(0), x.size(1))
	loss = z.mean()
	loss.backward()
	print(x1.grad)

	fig = plt.figure()
	ax1 = fig.add_subplot(121, projection='3d')
	ax1.plot_wireframe(x, y, z.clone().detach())

	ax2 = fig.add_subplot(122, projection='3d')
	ax2.plot_wireframe(x, y, x1.grad)

	plt.show()