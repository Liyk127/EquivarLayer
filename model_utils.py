import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from rbf.basis import get_rbf  # If you have trouble installing rbf, you can comment out this line and load precomputed kernels from local files.


def normalize(inv):
    batch = inv.shape[0]
    max_values = inv.abs().view(batch, -1).max(dim=-1)[0]
    max_values[max_values==0] = 1
    inv = inv / max_values.view(batch, 1, 1, 1)
    return inv


class Diff(nn.Module):
    def __init__(self, ord=2):
        super(Diff, self).__init__()
        self.ord = ord
        # By default, generate Gaussian derivative kernels using rbf.
        kernel1 = self.make_gauss(1, 7).cuda().reshape(2, 1, 1, 7, 7)
        kernel2 = self.make_gauss(2, 7).cuda().reshape(3, 1, 1, 7, 7)
        '''
        If you have trouble installing rbf, comment out the above two lines,
        and use the following two lines to load precomputed kernels from local files:
        kernel1 = torch.load("./kernel1.pt", weights_only=True).cuda().reshape(2, 1, 1, 7, 7)
        kernel2 = torch.load("./kernel2.pt", weights_only=True).cuda().reshape(3, 1, 1, 7, 7)
        '''
        conv_weights = torch.cat([-kernel1[1], kernel1[0], kernel2[2], kernel2[0]], dim=0)
        self.conv_weights = nn.Parameter(conv_weights.unsqueeze(1), requires_grad=False)
    
    def make_coord(self, kernel_size):
        x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1)
        coord = torch.meshgrid([-x, x])
        coord = torch.stack([coord[1], coord[0]], -1)
        return coord.reshape(kernel_size ** 2, 2)

    def make_gauss(self, order, kernel_size):
        diff = []
        coord = self.make_coord(kernel_size)
        gauss = get_rbf('ga')
        for i in range(order + 1):
            w = gauss(coord, torch.zeros(1, 2), eps=0.99, diff=[i, order - i]).reshape(kernel_size, kernel_size)
            w = torch.tensor(w)
            diff.append(w)
        tensor = torch.stack(diff, 0)
        return tensor.to(torch.float32)

    def dx(self, u):
        _,c,_,_ = u.shape
        weights = torch.cat([self.conv_weights[0]]*c, 0)
        u = F.pad(u, (3, 3, 3, 3), mode='replicate')
        return F.conv2d(u, weights, bias=None, stride=1, groups=c)

    def dy(self, u):
        _,c,_,_ = u.shape
        weights = torch.cat([self.conv_weights[1]]*c, 0)
        u = F.pad(u, (3, 3, 3, 3), mode='replicate')
        return F.conv2d(u, weights, bias=None, stride=1, groups=c)
    
    def dxx(self, u):
        _,c,_,_ = u.shape
        weights = torch.cat([self.conv_weights[2]]*c, 0)
        u = F.pad(u, (3, 3, 3, 3), mode='replicate')
        return F.conv2d(u, weights, bias=None, stride=1, groups=c)

    def dyy(self, u):
        _,c,_,_ = u.shape
        weights = torch.cat([self.conv_weights[3]]*c, 0)
        u = F.pad(u, (3, 3, 3, 3), mode='replicate')
        return F.conv2d(u, weights, bias=None, stride=1, groups=c)
    
    def forward(self, u):
        ux = self.dx(u)
        uy = self.dy(u)
        if self.ord == 1:
            return ux, uy
        elif self.ord == 2:
            uxx = self.dxx(u)
            uyy = self.dyy(u)
            uxy = self.dx(uy)
            return ux, uy, uxx, uyy, uxy


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)