import sys
import math
import numpy as np
import kornia as K
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import affine, _get_inverse_affine_matrix


class Logger(object):
    def __init__(self, fileN="default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def compute_param(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params



class CanonicalizationTrainer:
    def __init__(self, network, train_loader, optimizer, scheduler, path_name, in_shape, args):
        self.network = network
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.path_name = path_name
        self.in_shape = in_shape
        self.log = args.log
        self.log_interval = int(
            round(len(train_loader.dataset) / 100 / args.batch_size * args.log_interval)
            )
        self.batch_size = args.batch_size
        self.ss_transform = args.ss_transform
        self.args = args
        if args.dataset == "fashion":
            self.pad = self.fashion_constant_padding
        elif args.dataset == "mnist":
            self.pad = self.mnist_constant_padding
        self.crop = transforms.CenterCrop((self.in_shape[-2], self.in_shape[-1]))

    def mnist_constant_padding(self, x):
        padding = (math.ceil(self.in_shape[-1] * 0.5), math.ceil(self.in_shape[-1] * 0.5),
                   math.ceil(self.in_shape[-1] * 0.5), math.ceil(self.in_shape[-1] * 0.5))
        value1 = -0.1307 / 0.3081
        x_pad = F.pad(x[:, 0:1, :, :], padding, "constant", value1)
        return x_pad

    def fashion_constant_padding(self, x):
        padding = (math.ceil(self.in_shape[-1] * 0.5), math.ceil(self.in_shape[-1] * 0.5),
                   math.ceil(self.in_shape[-1] * 0.5), math.ceil(self.in_shape[-1] * 0.5))
        value1 = -0.2860 / 0.3530
        x_pad = F.pad(x[:, 0:1, :, :], padding, "constant", value1)
        return x_pad


    def affine_transform(self, x, ss_transform):
        x_pad = self.pad(x)
        affine_matrices = torch.empty((x.shape[0], 2, 2)).cuda()

        if ss_transform == "scale":
            for i in range(x.shape[0]):
                angle = 0
                scale = np.random.rand() * 0.8 + 0.8
                shear, translate = [0., 0.], [0., 0.]
                x_pad[i] = affine(img=x_pad[i], translate=translate, angle=angle, scale=scale, shear=shear)
                affine_matrices[i] = torch.linalg.inv(
                    torch.tensor(_get_inverse_affine_matrix([0.0, 0.0], angle, translate, scale, shear)).reshape(2, 3)[:, :2])
        elif ss_transform == "RS":
            for i in range(x.shape[0]):
                angle = np.random.rand() * 360 - 180
                scale = np.random.rand() * 0.4 + 0.8
                shear, translate = [0., 0.], [0., 0.]
                x_pad[i] = affine(img=x_pad[i], translate=translate, angle=angle, scale=scale, shear=shear)
                affine_matrices[i] = torch.linalg.inv(
                    torch.tensor(_get_inverse_affine_matrix([0.0, 0.0], angle, translate, scale, shear)).reshape(2, 3)[:, :2])
        elif ss_transform == "GL2":
            for i in range(x.shape[0]):
                angle = np.random.rand() * 360 - 180
                scale = np.random.rand() * 0.4 + 0.8
                shear = [10 * (2 * np.random.rand() - 1), 10 * (2 * np.random.rand() - 1)]
                translate = [0., 0.]
                x_pad[i] = affine(img=x_pad[i], translate=translate, angle=angle, scale=scale, shear=shear)
                affine_matrices[i] = torch.linalg.inv(
                    torch.tensor(_get_inverse_affine_matrix([0.0, 0.0], angle, translate, scale, shear)).reshape(2, 3)[:, :2])
                
        x_aff = self.crop(x_pad)
        return x_aff, affine_matrices


    def gl2_to_affine(self, img_shape, matrix):
        a11, a12, a21, a22 = matrix[:, 0, 0], matrix[:, 0, 1], \
                                matrix[:, 1, 0], matrix[:, 1, 1]
        cx, cy = img_shape[-2] // 2, img_shape[-1] // 2
        affine_part = torch.stack(
            [(1 - a11) * cx - a12 * cy, - a21 * cx + (1 - a22) * cy], dim=1
        )
        affine_matrices = torch.cat(
            [matrix, affine_part.unsqueeze(-1)], dim=-1
        )
        return affine_matrices


    def train_one_epoch(self, epoch):
        self.network.train()

        avg_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.type(torch.FloatTensor).cuda(), target.cuda()
            self.optimizer.zero_grad()
            data, affine_matrices = self.affine_transform(data, self.ss_transform)
            output = self.network(data)
            loss = get_alignment_loss(output, affine_matrices)
            loss.backward()
            self.optimizer.step()

            avg_loss += loss.item()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader), loss.item()))
                assert not torch.isnan(loss), "Loss is NaN"

        avg_loss /= len(self.train_loader)
        print("Average Loss: {:.6f}".format(avg_loss))
        if self.log > 0:
            torch.save(self.network.state_dict(), self.path_name + 'model.pth')
            torch.save(self.optimizer.state_dict(), self.path_name + 'optimizer.pth')
            torch.save(self.scheduler.state_dict(), self.path_name + 'scheduler.pth')

    
    def test_acc(self, network, test_loader):
        network.eval()
        
        prediction_network = get_prediction_network(use_pretrained = True)
        prediction_network_state_dict = torch.load(
            "./experiments/checkpoint/" + self.args.predict_checkpoint + "/model.pth", weights_only=True
            )
        prediction_network.load_state_dict(prediction_network_state_dict)
        prediction_network.cuda()
        prediction_network.eval()

        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.type(torch.FloatTensor).cuda(), target.cuda()
                output_matrix = network(data)                
                x = self.pad(data)
                affine_matrices = self.gl2_to_affine(x.shape, output_matrix)
                x = K.geometry.warp_affine(x, affine_matrices, dsize=(x.shape[-2], x.shape[-1]))
                x = self.crop(x)
                logits = prediction_network(x)
                correct += (logits.argmax(dim=-1) == target).float().sum()
        
        print('acc: {}/{} ({:.2f}%)'.format(
                correct,
                len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)
                )
            )



class PredictionTrainer:
    def __init__(self, prediction_network, train_loader, optimizer, scheduler, path_name, args):
        self.prediction_network = prediction_network
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log = args.log
        self.log_interval = int(
            round(len(train_loader.dataset) / 100 / args.batch_size * args.log_interval)
            )
        self.batch_size = args.batch_size
        self.path_name = path_name


    def train_one_epoch(self, epoch):
        self.prediction_network.train()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            data, target = data.type(torch.FloatTensor).cuda(), target.cuda()
            logits = self.prediction_network(data)
            loss = torch.nn.CrossEntropyLoss()(logits, target)
            preds = logits.argmax(dim=-1)
            acc = (preds == target).float().mean().item()
            loss.backward()
            self.optimizer.step()

            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader), loss.item()) \
                    + "\t Acc: {:.2f}".format(acc))

                assert not torch.isnan(loss), "Loss is NaN"

        if self.log:
            torch.save(self.prediction_network.state_dict(), self.path_name + 'model.pth')
            torch.save(self.optimizer.state_dict(), self.path_name + 'optimizer.pth')
            torch.save(self.scheduler.state_dict(), self.path_name + 'scheduler.pth')


    def test_acc(self, prediction_network, test_loader):
        prediction_network.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.type(torch.FloatTensor).cuda(), target.cuda()
                logits = prediction_network(data)
                correct += (logits.argmax(dim=-1) == target).float().sum()
        
        print('acc: {}/{} ({:.2f}%)'.format(
                correct, 
                len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)
            )
        )



def get_alignment_loss(output, matrix):
    identity_matrix = (
            torch.eye(output.shape[-1])
            .repeat(output.shape[0], 1, 1)
            .cuda()
        )
    prod = torch.bmm(output, matrix)
    return torch.nn.MSELoss()(prod, identity_matrix)


class PredictionNetwork(nn.Module):
    def __init__(self, encoder: torch.nn.Module, feature_dim: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.predictor = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reps = self.encoder(x)
        reps = reps.view(x.shape[0], -1)
        return self.predictor(reps)
    

def get_prediction_network(
    use_pretrained: bool = False,
    num_classes: int = 10,
) -> torch.nn.Module:
    
    weights = "DEFAULT" if use_pretrained else None
    encoder = torchvision.models.resnet50(weights=weights)

    encoder.conv1 = nn.Conv2d(
        1, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    encoder.maxpool = nn.Identity()

    feature_dim = encoder.fc.in_features
    encoder.fc = nn.Identity()
    prediction_network = PredictionNetwork(encoder, feature_dim, num_classes)

    return prediction_network