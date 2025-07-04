import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import _weights_init, normalize, Diff, LambdaLayer


# Computing invariants of the affine group
def compute_affine_invariants(u):
    diff = Diff()
    ux, uy, uxx, uyy, uxy = diff(u)

    uxx_uy = uxx * uy
    uxy_ux = uxy * ux
    uxy_uy = uxy * uy
    uyy_ux = uyy * ux

    inv2 = uxx * uyy - uxy * uxy
    inv3 = uxx_uy * uy - 2 * uxy_ux * uy + uyy_ux * ux
    inv4 = uxx_uy[:,:-1,:,:] * uy[:,1:,:,:] - uxy_ux[:,:-1,:,:] * uy[:,1:,:,:] - uxy_uy[:,:-1,:,:] * ux[:,1:,:,:] + uyy_ux[:,:-1,:,:] * ux[:,1:,:,:]
    inv5 = uxx_uy[:,1:,:,:] * uy[:,:-1,:,:] - uxy_ux[:,1:,:,:] * uy[:,:-1,:,:] - uxy_uy[:,1:,:,:] * ux[:,:-1,:,:] + uyy_ux[:,1:,:,:] * ux[:,:-1,:,:]
    inv6 = ux[:,:-1,:,:] * uy[:,1:,:,:] - uy[:,:-1,:,:] * ux[:,1:,:,:]

    inv1 = normalize(u)
    inv2 = normalize(inv2)
    inv345 = normalize(torch.cat((inv3, inv4, inv5), 1))
    if u.shape[1] > 1:
        inv6 = normalize(inv6)         

    return torch.cat((inv1, inv2, inv345, inv6), 1)


# EquivarLayer for the affine group
class EquivarLayer(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels=None, type=["0", "0"], stride=1):
        super(EquivarLayer, self).__init__()
        num_inv = 6 * in_channels - 3
        if hid_channels == None:
            hid_channels = max(in_channels, out_channels)
        self.conv1 = torch.nn.Conv2d(num_inv, hid_channels, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(hid_channels, out_channels, kernel_size=1)
        self.stride = stride
        self.type = type
        if self.type[1] == "c":
            self.eq_matrix_layer = EqMatrixLayer(in_channels, out_channels)

    def forward(self, x):
        inv = compute_affine_invariants(x)
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
            eq_matrix = self.eq_matrix_layer(x).view(b, 2, 2, h, w)
            out_matrix = torch.einsum('bijkl,bjnkl->binkl', eq_matrix, inv_matrix)
            return out_matrix


# Computing an equivariant matrix for the affine group
class EqMatrixLayer(nn.Module):
    def __init__(self, in_channels, out_channels=4):
        super(EqMatrixLayer, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, bias=False)

    def forward(self, x):
        diff = Diff()
        ux, uy, uxx, uyy, uxy = diff(x)

        relative_inv = uxx * uyy - uxy * uxy
        Su = relative_inv.abs().view(x.shape[0], -1).max(dim=-1)[0]
        Su[Su==0] = 1
        Su = Su.view(x.shape[0], 1, 1, 1) ** 0.5

        eq_matrix11 = self.conv1((uxx * uy - uxy * ux) / Su)
        eq_matrix12 = self.conv1(ux)
        eq_matrix21 = self.conv1((uxy * uy - uyy * ux) / Su)
        eq_matrix22 = self.conv1(uy)
        
        return torch.cat(
                (eq_matrix11, eq_matrix12, eq_matrix21, eq_matrix22),
                dim=1)


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


class EquivarLayer_affine_resnet32(nn.Module):
    def __init__(self,
                 in_shape,
                 beta = 10,
                 block = BasicBlock,
                 num_blocks=[5, 5, 5],
                 ):
        super(EquivarLayer_affine_resnet32, self).__init__()
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