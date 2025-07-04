import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import _weights_init, normalize, Diff, LambdaLayer


# Computing invariants of the scale group
def compute_scale_invariants(u):
    diff = Diff()
    ux, uy, uxx, uyy, uxy = diff(u)
    inv0 = normalize(u)
    inv1 = normalize(torch.cat((ux, uy), 1))
    inv2 = normalize(torch.cat((uxx, uxy, uyy), 1))
    return torch.cat((inv0, inv1, inv2), 1)


# EquivarLayer for the scale group
class EquivarLayer(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels=None, type=["0", "0"], stride=1):
        super(EquivarLayer, self).__init__()
        num_inv = 6 * in_channels
        if hid_channels == None:
            hid_channels = max(in_channels, out_channels)
        self.conv1 = torch.nn.Conv2d(num_inv, hid_channels, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(hid_channels, out_channels, kernel_size=1)
        self.stride = stride
        self.type = type

    def forward(self, x):
        inv = compute_scale_invariants(x)
        inv = self.conv1(inv)
        inv = F.relu(inv)
        if self.stride == 2:
            inv = F.max_pool2d(inv, (2, 2))
        inv = self.conv2(inv)

        if self.type[1] == "0":
            return inv
        elif self.type[1] == "c":
            b, _, h, w = inv.shape
            inv_matrix = inv.view(b, 2, 2, h, w)

            diff = Diff(ord=1)
            ux, _ = diff(x)
            Su = ux.abs().view(x.shape[0], -1).max(dim=-1)[0]
            out_matrix = Su.view(b, 1, 1, 1, 1) * inv_matrix
            return out_matrix


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = EquivarLayer(in_planes, planes, type=["0", "0"], stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = EquivarLayer(planes, planes, type=["0", "0"], stride=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class EquivarLayer_scale_resnet32(nn.Module):
    def __init__(self,
                 in_shape,
                 beta = 10,
                 block = BasicBlock,
                 num_blocks=[5, 5, 5],
                 ):
        super(EquivarLayer_scale_resnet32, self).__init__()
        self.beta = beta
        self.in_planes = 16
        self.conv1 = EquivarLayer(in_shape[0], 16, type=["0", "0"], stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.outlayer = EquivarLayer(64, 4, type=["0", "c"])
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def det_pool(self, x):
        b, _, h, w = x.shape
        det = (x[:,0,:,:] * x[:,3,:,:] - x[:,1,:,:] * x[:,2,:,:]).abs()
        det_onehot = torch.nn.functional.one_hot(
            torch.argmax(det.view(b, -1), dim=-1), h*w
        ).view(b, h, w).float()
        det_soft = torch.nn.functional.softmax(
            self.beta * det.view(b, -1), dim=-1
        ).view(b, h, w)
        det_weight = det_onehot + det_soft - det_soft.detach()
        pool_x = torch.sum(det_weight.unsqueeze(1) * x, dim=(-2,-1))
        return pool_x.view(b, 2, 2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        b, _, h, w = out.shape
        out_matrix = self.outlayer(out)
        out_matrix = self.det_pool(out_matrix.view(b, 4, h, w))

        return out_matrix.transpose(-2, -1)