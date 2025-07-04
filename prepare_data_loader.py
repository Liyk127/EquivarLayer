import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST


def prepare_data_loader(args):
    data_name, batch_size_train, batch_size_test = args.dataset, args.batch_size, args.batch_size_test
    data_path = "./experiments/data/" + data_name

    if data_name == "mnist":
        transform_dict = {
            "vanilla": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]),
            "mild": transforms.Compose([
                transforms.RandomAffine(degrees=10, scale=(0.9, 1.1), shear=(-5, 5, -5, 5)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]),
            "GL2": transforms.Compose([
                transforms.RandomAffine(degrees=180, scale=(0.8, 1.2), shear=(-10, 10, -10, 10)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]),
            "RS": transforms.Compose([
                transforms.RandomAffine(degrees=180, scale=(0.8, 1.2), shear=0),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]),
            "scale": transforms.Compose([
                transforms.RandomAffine(degrees=0, scale=(0.8, 1.6), shear=0),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        }

        train_transform = transform_dict[args.train_aug]
        test_transform = transform_dict["vanilla"]
        test_aug_transform = transform_dict[args.test_aug]

        print("train_transform", train_transform)
        print("test_transform", test_transform)
        print("test_aug_transform", test_aug_transform)

        train_dataset = MNIST(root=data_path, train=True, transform=train_transform, download=True)
        test_dataset = MNIST(root=data_path, train=False, transform=test_transform, download=True)
        test_aug_dataset = MNIST(root=data_path, train=False, transform=test_aug_transform, download=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size_train,
            shuffle=True,
            num_workers=4,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size_test,
            shuffle=False,
            num_workers=4,
        )

        test_aug_loader = DataLoader(
            test_aug_dataset,
            batch_size_test,
            shuffle=False,
            num_workers=4,
        )

        return train_loader, test_loader, test_aug_loader


    elif data_name == "fashion":
        transform_dict = {
            "vanilla": transforms.Compose([
                transforms.Pad(6),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ]),
            "mild": transforms.Compose([
                transforms.Pad(6),
                transforms.RandomAffine(degrees=10, scale=(0.9, 1.1), shear=(-5, 5, -5, 5)),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ]),
            "GL2": transforms.Compose([
                transforms.Pad(6),
                transforms.RandomAffine(degrees=180, scale=(0.8, 1.2), shear=(-10, 10, -10, 10)),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ]),
            "RS": transforms.Compose([
                transforms.Pad(6),
                transforms.RandomAffine(degrees=180, scale=(0.8, 1.2), shear=0),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ]),
            "scale": transforms.Compose([
                transforms.Pad(6),
                transforms.RandomAffine(degrees=0, scale=(0.8, 1.6), shear=0),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])
        }

        train_transform = transform_dict[args.train_aug]
        test_transform = transform_dict["vanilla"]
        test_aug_transform = transform_dict[args.test_aug]

        print("train_transform", train_transform)
        print("test_transform", test_transform)
        print("test_aug_transform", test_aug_transform)

        train_dataset = FashionMNIST(root=data_path, train=True, transform=train_transform, download=True)
        test_dataset = FashionMNIST(root=data_path, train=False, transform=test_transform, download=True)
        test_aug_dataset = FashionMNIST(root=data_path, train=False, transform=test_aug_transform, download=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size_train,
            shuffle=True,
            num_workers=4,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size_test,
            shuffle=False,
            num_workers=4,
        )

        test_aug_loader = DataLoader(
            test_aug_dataset,
            batch_size_test,
            shuffle=False,
            num_workers=4,
        )

        return train_loader, test_loader, test_aug_loader