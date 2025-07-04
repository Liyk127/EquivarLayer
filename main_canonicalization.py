import os
import sys
import numpy
import torch
import datetime
import argparse

from prepare_data_loader import prepare_data_loader
from train_utils import compute_param, Logger, CanonicalizationTrainer
from EquivaLayer_affine import EquivarLayer_affine_resnet32
from EquivarLayer_RS import EquivarLayer_RS_resnet32
from EquivarLayer_scale import EquivarLayer_scale_resnet32


parser = argparse.ArgumentParser(description='net')
parser.add_argument('--mode', default='train')
parser.add_argument('--log', default=1, type=int, help='log or not')
parser.add_argument('--model', default='EquivarLayer_affine')
parser.add_argument('--dataset', default='mnist', help='dataset')
parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training ')
parser.add_argument('--batch_size_test', type=int, default=128, help='input batch size for testing ')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--train_aug', type=str,  default='vanilla')
parser.add_argument('--test_aug', type=str,  default='GL2')
parser.add_argument('--ss_transform', default='GL2')
parser.add_argument('--log_interval', default=2, type=int, help='log interval')
parser.add_argument('--test_interval', default=5, type=int, help='test interval')
parser.add_argument('--checkpoint', default=None, help='load model')
parser.add_argument('--predict_checkpoint', default=None, help='load prediction model')
args = parser.parse_args()


date_time = datetime.datetime.now().strftime('%m%d-%H%M%S')
path_name = "./experiments/checkpoint/" + date_time + "/"
file_name = "./experiments/" + "{} {} {} {}".format(args.dataset, date_time, args.model, args.ss_transform)


os.environ['PYTHONHASHSEED'] = str(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
numpy.random.seed(args.seed)


if args.dataset == "mnist":
    in_shape = (1, 28, 28)
elif args.dataset == "fashion":
    in_shape = (1, 40, 40)

model_dict = {
    "EquivarLayer_affine": EquivarLayer_affine_resnet32,
    "EquivarLayer_RS": EquivarLayer_RS_resnet32,
    "EquivarLayer_scale": EquivarLayer_scale_resnet32,
}
network = model_dict[args.model](in_shape)
network.cuda()

if args.checkpoint:
    network.load_state_dict(torch.load('./experiments/checkpoint/' + args.checkpoint + "/model.pth"))


if args.log:
    os.makedirs(path_name)
    sys.stdout = Logger(file_name)
print(args)
print("\n" + "Number of parameters: ", compute_param(network))


optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
train_loader, test_loader, test_aug_loader = prepare_data_loader(args)


trainer = CanonicalizationTrainer(network, train_loader, optimizer, scheduler, path_name, in_shape, args)

if args.mode == "train":
    for epoch in range(1, args.epochs + 1):
        trainer.train_one_epoch(epoch)
        trainer.scheduler.step()
        print("")
    print("Test on original dataset")
    trainer.test_acc(trainer.network, test_loader)
    print("Test on transformed dataset")
    trainer.test_acc(trainer.network, test_aug_loader)

elif args.mode == "test":
    print("checkpoint:" + args.checkpoint)
    print("predict_checkpoint:" + args.predict_checkpoint)
    print("Test on original dataset")
    trainer.test_acc(network, test_loader)
    print("Test on transformed dataset")
    trainer.test_acc(network, test_aug_loader)