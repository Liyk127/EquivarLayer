import os
import sys
import numpy
import torch
import datetime
import argparse
from prepare_data_loader import prepare_data_loader
from train_utils import compute_param, get_prediction_network, Logger, PredictionTrainer


parser = argparse.ArgumentParser(description='net')
parser.add_argument('--model', default='resnet50')
parser.add_argument('--use_pretrained', type=int, default=1)
parser.add_argument('--dataset', default='mnist', help='dataset')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight_decay')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training ')
parser.add_argument('--batch_size_test', type=int, default=128, help='input batch size for testing ')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--train_aug', type=str,  default='vanilla')
parser.add_argument('--test_aug', type=str,  default='vanilla')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--mode', default='train')
parser.add_argument('--log', default=1, type=int, help='log or not')
parser.add_argument('--log_interval', default=2, type=int, help='log interval')
parser.add_argument('--test_interval', default=5, type=int, help='test interval')
parser.add_argument('--predict_checkpoint', default=None, help='load prediction model')
args = parser.parse_args()


date_time = datetime.datetime.now().strftime('%m%d-%H%M%S')
path_name = "./experiments/checkpoint/" + date_time + "/"
file_name = "./experiments/" + "{} {} {} {}".format(args.dataset, date_time, args.model, args.train_aug)


os.environ['PYTHONHASHSEED'] = str(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
numpy.random.seed(args.seed)


if args.model == "resnet50":
    prediction_network = get_prediction_network(use_pretrained = args.use_pretrained)
if args.predict_checkpoint:
    prediction_network.load_state_dict(torch.load('./experiments/checkpoint/' + args.predict_checkpoint + "/model.pth", weights_only=True))
prediction_network.cuda()


if args.log:
    os.makedirs(path_name)
    sys.stdout = Logger(file_name)
print(args)
print("\n" + "Number of parameters: ", compute_param(prediction_network))


optimizer = torch.optim.SGD(prediction_network.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
milestones = [args.epochs // 3, args.epochs // 2]
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
train_loader, test_loader, test_aug_loader = prepare_data_loader(args)


trainer = PredictionTrainer(prediction_network, train_loader, optimizer, scheduler, path_name, args)

if args.mode == "train":
    for epoch in range(1, args.epochs + 1):
        trainer.train_one_epoch(epoch)
        trainer.scheduler.step()
        print("")
    print("Test on original dataset")
    trainer.test_acc(trainer.prediction_network, test_loader)
    print("Test on transformed dataset")
    trainer.test_acc(trainer.prediction_network, test_aug_loader)

elif args.mode == "test":
    print("Test on original dataset")
    trainer.test_acc(trainer.prediction_network, test_loader)
    print("Test on transformed dataset")
    trainer.test_acc(trainer.prediction_network, test_aug_loader)